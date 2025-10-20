from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Tuple, Optional, Sequence
import numpy as np
import json


@dataclass(frozen=True)
class NodeSpec:
    """Metadata for a node in the graph."""
    idx: int
    name: str
    arity: int  # number of possible values (v for variables, 2 for reward)


class Env:
    """
    Causal bandit environment with categorical variables and a binary reward.

    Nodes:
      - Variables: indices 0..n-1, each taking values in {1, 2, ..., v}.
      - Reward: index n, taking values in {0, 1}. Non-intervenable.

    Parameters
    ----------
    n : int
        Number of non-reward variables.
    k : int
        Number of parents of the reward node (chosen among the n variables).
    v : int
        Domain size for each variable, i.e., values in {1, ..., v}.
    dag_type : str
        Type of DAG to create over the variables. Supported: 'random'.
    dist_type : str
        Type of conditional distributions. Supported: 'random'.
    p_edge : float
        Edge probability for the random DAG among variables (i -> j in a random topological order).
    parent_effect : float
        Degree to which a node's distribution depends on its parents:
          0.0 -> ignore parents; use the node's base distribution for all parent assignments.
          1.0 -> for each parent assignment, draw an independent random distribution (previous behavior).
          (0,1) -> convex mix (1-α)*base + α*independent_random for each assignment.
        Note: reward CPDs ignore this and are always independent random per parent assignment.
    seed : Optional[int]
        Seed for reproducible randomness.

    Attributes (key ones)
    ---------------------
    parents : Dict[int, List[int]]
        parents[i] is a sorted list of parent indices of node i.
    cpds : Dict[int, Dict[Tuple[int, ...], np.ndarray]]
        For each node i, maps tuples of parent values (ordered to match parents[i])
        to a probability vector of length v (variables) or 2 (reward).
    base_cpds : Dict[int, np.ndarray]
        Base distribution for each variable node i (0..n-1), independent of parents.
    nodes : List[NodeSpec]
        Per-node metadata including arity.
    topo_order : List[int]
        A topological order over all n+1 nodes (reward appears after its parents).
    reward_idx : int
        The index of the reward node (always n).
    rng : np.random.Generator
        Random number generator used throughout.
    """

    def __init__(
        self,
        n: int,
        k: int,
        v: int,
        dag_type: str = "random",
        dist_type: str = "random",
        p_edge: float = 0.2,
        parent_effect: float = 1.0,
        seed: Optional[int] = None,
        load_path: Optional[str] = None,
    ):
        """
        If load_path is provided, the environment will be loaded from the JSON file
        at that path and all other constructor arguments (n,k,v,...) will be ignored.
        """

        self.rng = np.random.default_rng(seed)

        if load_path is not None:
            self._load_from_file(load_path)
            return

        # ---- basic validation ----
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer.")
        if not isinstance(k, int) or not (0 <= k <= n):
            raise ValueError("k must be an integer with 0 <= k <= n.")
        if not isinstance(v, int) or v <= 0:
            raise ValueError("v must be a positive integer.")
        if dag_type.lower() != "random":
            raise NotImplementedError(f"dag_type '{dag_type}' is not supported yet (only 'random').")
        if dist_type.lower() != "random":
            raise NotImplementedError(f"dist_type '{dist_type}' is not supported yet (only 'random').")
        if not (0.0 <= p_edge <= 1.0):
            raise ValueError("p_edge must be in [0, 1].")
        if not (0.0 <= float(parent_effect) <= 1.0):
            raise ValueError("parent_effect must be in [0, 1].")

        self.n = n
        self.k = k
        self.v = v
        self.dag_type = dag_type.lower()
        self.dist_type = dist_type.lower()
        self.p_edge = float(p_edge)
        self.parent_effect = float(parent_effect)
        self.reward_idx = n
        

        # node specs
        self.nodes: List[NodeSpec] = [
            NodeSpec(idx=i, name=f"X{i}", arity=v) for i in range(n)
        ] + [NodeSpec(idx=self.reward_idx, name="R", arity=2)]

        # will be filled by builders
        self.parents: Dict[int, List[int]] = {i: [] for i in range(n + 1)}
        self.cpds: Dict[int, Dict[Tuple[int, ...], np.ndarray]] = {}
        self.base_cpds: Dict[int, np.ndarray] = {}  # only for variable nodes

        # construct graph + cpds
        self._build_graph()
        self.topo_order = self._compute_topological_order()
        self._build_distributions()

        # caches
        self._mean_cache: Dict[int, Dict[Tuple[Tuple[int, int], ...], float]] = {}
        self._ancestors_of_reward: set[int] = self._compute_ancestors(self.reward_idx) - {self.reward_idx}
    # -------------------------------------------------------------------------
    # Graph construction
    # -------------------------------------------------------------------------
    def _build_graph(self) -> None:
        """Create the DAG over variables and attach reward parents."""
        if self.dag_type == "random":
            self._build_graph_random()
        else:
            raise NotImplementedError(f"Unsupported dag_type '{self.dag_type}'.")

    def _build_graph_random(self) -> None:
        """Random DAG among variables via a random order; sample edges forward with prob p_edge."""
        n = self.n
        order = list(range(n))
        self.rng.shuffle(order)

        # edges only from earlier -> later in this random order to ensure acyclicity
        for i in range(n):
            self.parents[i] = []

        for i in range(n):
            for j in range(i + 1, n):
                u = order[i]
                v = order[j]
                if self.rng.random() < self.p_edge:
                    # u -> v
                    self.parents[v].append(u)

        # sort parent lists for consistent key ordering in CPDs
        for i in range(n):
            self.parents[i].sort()

        # choose reward parents
        reward_parents = self.rng.choice(np.arange(n), size=self.k, replace=False).tolist()
        reward_parents.sort()
        self.parents[self.reward_idx] = reward_parents

    # -------------------------------------------------------------------------
    # Topological order
    # -------------------------------------------------------------------------
    def _compute_topological_order(self) -> List[int]:
        """Kahn's algorithm using the parents[] lists."""
        n_nodes = self.n + 1
        indeg = [0] * n_nodes
        children: Dict[int, List[int]] = {i: [] for i in range(n_nodes)}
        for child, ps in self.parents.items():
            indeg[child] = len(ps)
            for p in ps:
                children[p].append(child)

        queue = [i for i in range(n_nodes) if indeg[i] == 0]
        order: List[int] = []
        head = 0
        # Use a deterministic FIFO behavior
        while head < len(queue):
            u = queue[head]
            head += 1
            order.append(u)
            for v in children[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    queue.append(v)

        if len(order) != n_nodes:
            raise RuntimeError("Cycle detected in the constructed graph (should not happen).")

        return order

    # -------------------------------------------------------------------------
    # CPD construction
    # -------------------------------------------------------------------------
    def _build_distributions(self) -> None:
        """Assign CPDs to each node for every parent assignment."""
        if self.dist_type == "random":
            self._build_distributions_random()
        else:
            raise NotImplementedError(f"Unsupported dist_type '{self.dist_type}'.")

    def _build_distributions_random(self) -> None:
        """
        For each node i and each combination of its parents' values:
          - Variables (0..n-1):
              Draw a base distribution b_i ~ Dir(1), independent of parents.
              For each parent assignment pa, draw r_{i,pa} ~ Dir(1) and set:
                CPD(i | pa) = normalize( (1 - α) * b_i + α * r_{i,pa} ), α=parent_effect.
              Edge cases: α=0 -> CPD(i|pa) = b_i for all pa.
                          α=1 -> CPD(i|pa) = r_{i,pa} (independent per pa).
          - Reward (n):
              For each parent assignment pa, CPD(R|pa) ~ Dir(1,1) (independent of α).
        """
        # Helper for domains (variables in 1..v, reward in {0,1})
        def parent_domains_of(node: int) -> List[List[int]]:
            doms: List[List[int]] = []
            for p in self.parents[node]:
                if p == self.reward_idx:
                    doms.append([0, 1])
                else:
                    doms.append(list(range(1, self.v + 1)))
            return doms

        # Variables
        for node in range(self.n):
            ps = self.parents[node]
            parent_assignments = list(product(*parent_domains_of(node))) if ps else [()]
            out_arity = self.nodes[node].arity

            # Base distribution for the node (independent of parents)
            base = self.rng.dirichlet(np.ones(out_arity, dtype=float))
            base = base / base.sum()
            self.base_cpds[node] = base

            self.cpds[node] = {}
            alpha = np.ones(out_arity, dtype=float)  # for random component

            pe = self.parent_effect
            for pa in parent_assignments:
                rnd = self.rng.dirichlet(alpha)
                probs = (1.0 - pe) * base + pe * rnd
                probs = probs / probs.sum()
                self.cpds[node][tuple(pa)] = probs

        # Reward (always independent random per parent assignment; ignore parent_effect)
        node = self.reward_idx
        ps = self.parents[node]
        parent_assignments = list(product(*parent_domains_of(node))) if ps else [()]
        self.cpds[node] = {}
        alpha_r = np.ones(self.nodes[node].arity, dtype=float) 
        for pa in parent_assignments:
            probs = self.rng.dirichlet(alpha_r)
            probs = probs / probs.sum()
            self.cpds[node][tuple(pa)] = probs

    # -------------------------------------------------------------------------
    # Sampling with interventions
    # -------------------------------------------------------------------------
    def sample(self, intervention: Sequence[int]) -> np.ndarray:
        """
        Draw a sample given an intervention vector.

        Parameters
        ----------
        intervention : Sequence[int]
            A length-(n+1) vector. For variable indices 0..n-1:
              - 0 means no intervention on that variable (let the env sample it).
              - A value in {1, ..., v} fixes that variable to that value.
            For the reward index n:
              - Ignored if non-zero; reward is always sampled from its CPD.
              - 0 is fine (no-op).

        Returns
        -------
        x : np.ndarray
            A (n+1,) array with realized values. Variables are in {1..v},
            reward is in {0,1}.
        """
        if len(intervention) != self.n + 1:
            raise ValueError(f"intervention must have length {self.n + 1}.")
        x = np.zeros(self.n + 1, dtype=int)

        # Validate interventions for variables
        for i in range(self.n):
            val = intervention[i]
            if val != 0 and not (1 <= val <= self.v):
                raise ValueError(f"Intervention at X{i} must be 0 or in 1..{self.v}, got {val}.")
            if val != 0:
                x[i] = val  # set intervened value

        # Reward intervention is ignored by design, but we validate type
        if not isinstance(intervention[self.reward_idx], (int, np.integer)):
            raise ValueError("Reward entry in intervention must be an int (0 or ignored).")

        # Sample variables in topological order (excluding reward), respecting interventions
        for node in self.topo_order:
            if node == self.reward_idx:
                continue  # sample reward after all variables
            if x[node] != 0:
                continue  # already intervened

            pa_vals = tuple(x[p] for p in self.parents[node])
            if any(v == 0 for v in pa_vals):
                # This should not happen in a correct topological order.
                raise RuntimeError(f"Parent values for node {node} not resolved before sampling.")

            probs = self._get_cpd(node, pa_vals)
            # Variables take values in {1..v}
            choices = np.arange(1, self.v + 1, dtype=int)
            x[node] = self.rng.choice(choices, p=probs)
            #print(f"Sampled X{node} = {x[node]} with parents {pa_vals} and probs {probs}")

        # Finally sample reward from its CPD using realized parents
        r_parents = self.parents[self.reward_idx]
        r_pa_vals = tuple(x[p] for p in r_parents)
        if any(v == 0 for v in r_pa_vals):
            raise RuntimeError("Reward parent values not resolved before sampling.")
        r_probs = self._get_cpd(self.reward_idx, r_pa_vals)
        x[self.reward_idx] = self.rng.choice(np.array([0, 1], dtype=int), p=r_probs)

        return x
    

    # -------------------------------------------------------------------------
    # Mean reward computation and Best action
    # -------------------------------------------------------------------------
    def mean_reward(self, intervention: Sequence[int]) -> float:
        """
        Compute E[R | do(intervention)] exactly from CPDs.

        The intervention is a length-(n+1) vector as in `sample()`. Non-zero entries in 0..n-1 fix variables.
        The reward entry is ignored.

        Uses exact enumeration over the (non-intervened) ancestors of the reward only.
        """
        if len(intervention) != self.n + 1:
            raise ValueError(f"intervention must have length {self.n + 1}.")

        # Check cache first (based on number of intervened variables among X0..X{n-1})
        key = self._canonical_intervention_key(intervention)
        m_key = len(key)
        cached = self._mean_cache.get(m_key)
        if cached is not None and key in cached:
            return cached[key]

        # Build fixed assignments from intervention
        fixed: Dict[int, int] = {i: int(intervention[i]) for i in range(self.n) if intervention[i] != 0}
        S = set(fixed.keys())  # intervened variables (reward never here)

        # Only variables that can affect R matter: ancestors of R (excluding R itself)
        U = list(self._ancestors_of_reward - S)  # non-intervened ancestors
        # Use a topological order restricted to U for stable iteration
        topo_pos = {node: i for i, node in enumerate(self.topo_order)}
        U.sort(key=lambda x: topo_pos[x])

        # Domains for enumeration
        domains = [list(range(1, self.v + 1)) for _ in U]

        # Helper to get a parent assignment tuple given a partial (fixed+assign) assignment
        def pa_tuple(node: int, assign: Dict[int, int]) -> Tuple[int, ...]:
            return tuple(assign[p] for p in self.parents[node])

        mean = 0.0

        if not U:
            # No free ancestors: probability mass 1 on the fixed assignment
            assign = dict(fixed)
            # Reward parents must be determined by fixed
            r_pa = pa_tuple(self.reward_idx, assign)
            r_probs = self._get_cpd(self.reward_idx, r_pa)
            mean = float(r_probs[1])  # P(R=1 | parents)
        else:
            from itertools import product

            for values in product(*domains):
                # Build a full assignment over ancestors (only U plus fixed S)
                assign: Dict[int, int] = dict(fixed)
                assign.update({U[i]: values[i] for i in range(len(U))})

                # Probability of this assignment under do(intervention): product of CPDs
                prob = 1.0
                for node in U:
                    pa = pa_tuple(node, assign)
                    probs = self._get_cpd(node, pa)
                    val = assign[node]
                    prob *= probs[val - 1]  # variables domain is {1..v}

                    if prob == 0.0:
                        break  # small speedup

                if prob == 0.0:
                    continue

                # Now evaluate reward conditional on its parents
                r_pa = pa_tuple(self.reward_idx, assign)
                r_probs = self._get_cpd(self.reward_idx, r_pa)
                mean += prob * float(r_probs[1])  # add P(assignment) * P(R=1 | parents)

        # Store in cache bucket for this m if appropriate
        self._mean_cache.setdefault(m_key, {})[key] = mean
        return mean

    def best_action(self, m: int) -> Tuple[np.ndarray, float]:
        """
        Search over all interventions on exactly m variables (values in {1..v}) and
        return (best_intervention_vector, best_mean_reward).

        The returned intervention is a length-(n+1) vector; reward entry is 0.
        Results are cached per m to speed up repeated calls.
        """
        if not isinstance(m, int) or not (0 <= m <= self.n):
            raise ValueError(f"m must be an integer in [0, {self.n}].")

        from itertools import combinations, product

        best_mean = -1.0
        best_interv = np.zeros(self.n + 1, dtype=int)

        # Initialize cache bucket for this m
        cache_m = self._mean_cache.setdefault(m, {})

        # Iterate over all choices of m variables and all value assignments
        for idxs in combinations(range(self.n), m):
            if m == 0:
                # Single "empty" intervention case handled by the loop below with product([]) == [()]
                value_products = [()]
            else:
                value_products = product(range(1, self.v + 1), repeat=m)

            for vals in value_products:
                interv = np.zeros(self.n + 1, dtype=int)
                for i, val in zip(idxs, vals):
                    interv[i] = int(val)

                key = self._canonical_intervention_key(interv)

                # Use cache if available, otherwise compute and store
                if key in cache_m:
                    mean = cache_m[key]
                else:
                    mean = self.mean_reward(interv)  # this also populates cache under its |key|
                    cache_m[key] = mean

                if mean > best_mean:
                    best_mean = mean
                    best_interv = interv.copy()

        return best_interv, float(best_mean)


    # -------------------------------------------------------------------------
    # Helpers / Inspectors
    # -------------------------------------------------------------------------
    def _get_cpd(self, node: int, pa_vals: Tuple[int, ...]) -> np.ndarray:
        """Fetch CPD probabilities for a node given a tuple of parent values."""
        try:
            return self.cpds[node][pa_vals]
        except KeyError as e:
            raise KeyError(
                f"Missing CPD for node {node} and parent assignment {pa_vals}. "
                "This indicates a bug in CPD construction or incorrect parent value encoding."
            ) from e

    def seed(self, seed: Optional[int]) -> None:
        """Reset the RNG seed (affects future random operations)."""
        self.rng = np.random.default_rng(seed)

    def describe(self) -> str:
        """Human-readable summary of the environment."""
        lines = [
            f"Env(n={self.n}, k={self.k}, v={self.v}, dag_type='{self.dag_type}', dist_type='{self.dist_type}', p_edge={self.p_edge}, parent_effect={self.parent_effect})",
            "Parents:",
        ]
        for i in range(self.n):
            lines.append(f"  X{i} <- {self.parents[i]}")
        lines.append(f"  R  <- {self.parents[self.reward_idx]}")
        lines.append(f"Topological order: {self.topo_order}")
        return "\n".join(lines)

    def _compute_ancestors(self, node: int) -> set[int]:
        """Return the set of ancestors of `node` using the parents[] relation."""
        anc: set[int] = set()
        stack = list(self.parents[node])
        while stack:
            p = stack.pop()
            if p not in anc:
                anc.add(p)
                stack.extend(self.parents[p])
        return anc

    def _canonical_intervention_key(self, intervention: Sequence[int]) -> Tuple[Tuple[int, int], ...]:
        """
        Canonical, compact key for caching: sorted tuple of (var_idx, value) for nonzero interventions.
        Reward entry is ignored.
        """
        items = [(i, int(intervention[i])) for i in range(self.n) if intervention[i] != 0]
        items.sort()
        return tuple(items)



    ###################################
    # Saving / Loading to/from JSON
    ###################################
    def save(self, path: str) -> None:
        """
        Save the entire environment (graph, CPDs, base dists, caches, RNG state) to a JSON file.
        """
        def to_list(x: np.ndarray) -> List[float]:
            return [float(z) for z in np.asarray(x, dtype=float).tolist()]

        # serialize CPDs as: node -> list of {"pa": [...], "probs": [...]}
        cpds_ser: Dict[str, List[Dict[str, List[int]]]] = {}
        for node, table in self.cpds.items():
            rows = []
            for pa, probs in table.items():
                rows.append({"pa": [int(u) for u in pa], "probs": to_list(probs)})
            cpds_ser[str(node)] = rows

        # base_cpds for variable nodes only
        base_cpds_ser = {str(node): to_list(probs) for node, probs in self.base_cpds.items()}

        # parents as list of lists
        parents_ser = {str(i): [int(p) for p in ps] for i, ps in self.parents.items()}

        data = {
            "n": int(self.n),
            "k": int(self.k),
            "v": int(self.v),
            "dag_type": self.dag_type,
            "dist_type": self.dist_type,
            "p_edge": float(self.p_edge),
            "parent_effect": float(self.parent_effect),
            "reward_idx": int(self.reward_idx),
            "parents": parents_ser,
            "cpds": cpds_ser,
            "base_cpds": base_cpds_ser,
            "topo_order": [int(x) for x in self.topo_order],
            "_ancestors_of_reward": [int(a) for a in getattr(self, "_ancestors_of_reward", set())],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)


    def _load_from_file(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._load_from_dict(data)


    def _load_from_dict(self, data: Dict) -> None:
        # basic fields
        self.n = int(data["n"])
        self.k = int(data["k"])
        self.v = int(data["v"])
        self.dag_type = str(data.get("dag_type", "random"))
        self.dist_type = str(data.get("dist_type", "random"))
        self.p_edge = float(data.get("p_edge", 0.2))
        self.parent_effect = float(data.get("parent_effect", 1.0))
        self.reward_idx = int(data.get("reward_idx", self.n))
        self._mean_cache: Dict[int, Dict[Tuple[Tuple[int, int], ...], float]] = {}

        # nodes
        self.nodes = [NodeSpec(idx=i, name=f"X{i}", arity=self.v) for i in range(self.n)] + \
                    [NodeSpec(idx=self.reward_idx, name="R", arity=2)]

        # parents
        self.parents = {int(i): [int(p) for p in ps] for i, ps in data["parents"].items()}

        # cpds
        self.cpds = {}
        for node_str, rows in data["cpds"].items():
            node = int(node_str)
            table: Dict[Tuple[int, ...], np.ndarray] = {}
            for row in rows:
                pa = tuple(int(u) for u in row["pa"])
                probs = np.asarray(row["probs"], dtype=float)
                probs = probs / probs.sum()  # guard
                table[pa] = probs
            self.cpds[node] = table

        # base_cpds (variable nodes)
        self.base_cpds = {}
        for node_str, probs in data.get("base_cpds", {}).items():
            node = int(node_str)
            arr = np.asarray(probs, dtype=float)
            self.base_cpds[node] = arr / arr.sum()

        # topo order (validate/recompute)
        saved_topo = [int(x) for x in data.get("topo_order", [])]
        self.topo_order = self._compute_topological_order()
        if saved_topo and len(saved_topo) == len(self.topo_order) and set(saved_topo) == set(self.topo_order):
            # keep recomputed one; both should be valid; no strict equality enforce
            pass

        # ancestors of reward
        anc_loaded = set(int(a) for a in data.get("_ancestors_of_reward", []))
        if anc_loaded:
            self._ancestors_of_reward = anc_loaded
        else:
            self._ancestors_of_reward = self._compute_ancestors(self.reward_idx) - {self.reward_idx}


