import math
import random
from itertools import combinations, product
from typing import Dict, List, Tuple, Optional


class known_k_UCB_plus:
    """
    UCB+ learner for causal interventions with known k.

    - Nodes: n observed variables X_0,...,X_{n-1} and a binary reward node R at index n.
    - Each action is an (n+1)-vector of ints where exactly m entries among [0..n-1] are set to values in {1,...,v},
      others are 0, and the reward entry (index n) is always 0 (non-intervened).
    - On each round, the environment returns a length-(n+1) sample vector (values of all variables and the reward).

    Algorithm:
      * If m >= k:
          - Randomly choose a subset `A_sel` of size `n_actions` from all size-m interventions, where
              n_actions = min( C(n,m)*v^m , ceil( v^k * C(n,k) / C(m,k) * log(sqrt(T)) ) )
            (Using `min` avoids selecting more actions than exist. If T is None or <= 1, the second term is 0 and
             we default to the full action set.)
          - Run UCB over only `A_sel` with the UCB+ update-sharing rule below.
      * If m < k:
          - Run UCB over the full action set (all size-m interventions). In the update step, we efficiently
            update all actions that match the realized values via the UCB+ sharing rule.

    UCB+ sharing rule (the "plus" modification):
      When a sample x ∈ {0,1,...,v}^{n} × {0,1} (reward is x[n]) is observed after playing some action a,
      we *also* attribute this sample to any other action whose m intervened entries equal the realized values
      in x at those indices. In other words, if an action's non-zero coordinates match x exactly, it receives
      this (reward) sample too.

    Notes:
      - Reward is Bernoulli; UCB is clipped to 1.0.
      - We lazily store per-action statistics using the full (n+1)-tuple as the key, exactly as the project prefers.
    """

    def __init__(
        self,
        n: int,
        k: int,
        m: int,
        v: int,
        T: Optional[int] = None,
        seed: int = random.randint(1, 10**6),
    ):
        """
        Args:
            n: number of non-reward variables.
            k: number of parents of the reward node.
            m: intervention size (exactly m variables are intervened in each action).
            v: arity of non-reward variables (values 1..v; 0 means 'not intervened' / 'free').
            T: optional time horizon used only to size the random subset when m >= k.
               If None or <= 1, we default to using the full action set for m >= k.
            seed: optional RNG seed for reproducibility when sampling action subsets / tie-breaking.
        """
        assert n >= 1 and v >= 1 and 0 <= k <= n and 1 <= m <= n, "Invalid parameters."
        self.n = n
        self.k = k
        self.m = m
        self.v = v
        self.T = T 
        self.reward_idx = n
        self.ucb_alpha = 0.1 / math.comb(n,m)  

        self.rng = random.Random(seed)

        # Per-action stats, keyed by the full action tuple (length n+1)
        self.counts: Dict[Tuple[int, ...], int] = {}
        self.rew_sums: Dict[Tuple[int, ...], float] = {}

        # Total number of environment pulls (rounds played)
        self.t: int = 0

        # Candidate action set depending on regime:
        #   * If m >= k: random subset (size n_actions) from all actions
        #   * If m <  k: full action set
        if self.m >= self.k:
            total_actions = self._num_all_actions()
            if self.T is None:
                target = 0
            else:
                denom = math.comb(self.m, self.k) 
                # Guard against denom==0 (only possible if k=0 which makes sense: then v^k=1 and denom=1).
                denom = max(denom, 1)
                # log(sqrt(T)) = 0.5 * log(T) for T>1
                target = (self.v ** self.k) * math.comb(self.n, self.k) / denom * (0.5 * math.log(self.T))

            n_actions = min(total_actions, max(1, math.ceil(target)))

            if n_actions == total_actions:
                self.candidate_actions = self._enumerate_all_actions()
            else:
                self.candidate_actions = self._sample_unique_actions(n_actions)

            # Precompute sparse (idx, val) representation for quick matching on updates
            self._cand_sparse: Dict[Tuple[int, ...], List[Tuple[int, int]]] = {
                a: [(i, val) for i, val in enumerate(a[:self.n]) if val != 0]
                for a in self.candidate_actions
            }
        else:
            # m < k => full action set
            self.candidate_actions = self._enumerate_all_actions()
            # Sparse map is not needed here because we update from the sample by constructing all matches.

    # ------------------------- Public API -------------------------

    def action(self) -> List[int]:
        """
        Choose the next action according to UCB with the plus-sharing rule considered in updates.
        Returns:
            An (n+1)-length list representing the intervention to play.
        """
        self.t = max(self.t, 0)  # ensure t is initialized
        t_eff = max(self.t, 1)   # to avoid log(0)

        best_key = None
        best_ucb = -float("inf")

        # Iterate over candidate set; compute UCB for each
        for a in self.candidate_actions:
            c = self.counts.get(a, 0)
            if c == 0:
                ucb = float("inf")  # force initial exploration within the candidate set
            else:
                mean = self.rew_sums.get(a, 0.0) / c
                bonus = math.sqrt((self.ucb_alpha * math.log(self.T)) / c)
                #ucb = min(1.0, mean + bonus)  # clip for Bernoulli reward
                ucb = mean + bonus  # no clipping, as per UCB+ definition

            if ucb > best_ucb:
                best_ucb = ucb
                best_key = a

        # Safety: if for some reason candidate_actions is empty (shouldn't happen), sample one at random.
        if best_key is None:
            best_key = self._random_action_tuple()

        return list(best_key)

    def update(self, action, sample: List[int]) -> None:
        """
        Update statistics with the observed (n+1)-dimensional sample from the environment.

        Args:
            action: the action that was played (length-(n+1) list/array of ints). (Not used here.)
            sample: length-(n+1) list/array of realized values (X_0,...,X_{n-1}, R).
                    Reward is binary and located at index n.
        """
        assert len(sample) == self.n + 1, "Sample length must be n+1."
        r = sample[self.reward_idx]
        self.t += 1

        if self.m >= self.k:
            # Decide which loop is cheaper:
            #   S = number of selected actions we track (size of self._cand_sparse)
            #   C = number of m-combinations from the sample = comb(n, m)
            S = len(self._cand_sparse)
            C = math.comb(self.n, self.m)

            if C <= S:
                # Cheaper to enumerate all m-combinations of the sample, but
                # only update if that action is among the selected ones.
                for idx_tuple in combinations(range(self.n), self.m):
                    a = [0] * (self.n + 1)
                    for i in idx_tuple:
                        a[i] = sample[i]
                    a_key = tuple(a)
                    if a_key in self._cand_sparse:
                        self._accumulate(a_key, r)
            else:
                # Cheaper to scan the selected actions and test for a match.
                observed_pairs = {(i, sample[i]) for i in range(self.n)}
                for a_key, sparse in self._cand_sparse.items():  # sparse is list/iter of (i, val)
                    # Match iff all intervened pairs of the action appear in the sample
                    if all((i, val) in observed_pairs for (i, val) in sparse):
                        self._accumulate(a_key, r)

        else:
            # m < k: Update ALL actions (from the full set) that match the sample by construction.
            for idx_tuple in combinations(range(self.n), self.m):
                a = [0] * (self.n + 1)
                for i in idx_tuple:
                    a[i] = sample[i]
                a_key = tuple(a)
                self._accumulate(a_key, r)

    # ------------------------- Helpers -------------------------

    def _accumulate(self, a_key: Tuple[int, ...], reward: float) -> None:
        """Accumulate (count, sum of rewards) for the given action key."""
        self.counts[a_key] = self.counts.get(a_key, 0) + 1
        self.rew_sums[a_key] = self.rew_sums.get(a_key, 0.0) + reward

    def _num_all_actions(self) -> int:
        """Total number of size-m interventions: C(n, m) * v^m."""
        return math.comb(self.n, self.m) * (self.v ** self.m)

    def _enumerate_all_actions(self) -> List[Tuple[int, ...]]:
        """Enumerate all size-m interventions as full (n+1)-tuples (reward entry 0)."""
        actions: List[Tuple[int, ...]] = []
        for idxs in combinations(range(self.n), self.m):
            for vals in product(range(1, self.v + 1), repeat=self.m):
                a = [0] * (self.n + 1)
                for i, val in zip(idxs, vals):
                    a[i] = val
                # reward index stays 0 (non-intervened)
                actions.append(tuple(a))
        return actions

    def _random_action_tuple(self) -> Tuple[int, ...]:
        """Sample a single random size-m action as a full (n+1)-tuple."""
        idxs = self.rng.sample(range(self.n), self.m)
        a = [0] * (self.n + 1)
        for i in idxs:
            a[i] = self.rng.randint(1, self.v)
        return tuple(a)

    def _sample_unique_actions(self, k: int) -> List[Tuple[int, ...]]:
        """Sample k unique random actions (without enumerating the entire space)."""
        seen = set()
        while len(seen) < k:
            seen.add(self._random_action_tuple())
        return list(seen)

    # ------------------------- Optional Introspection -------------------------

    def get_action_stats(self) -> List[Tuple[Tuple[int, ...], int, float]]:
        """Return [(action, count, mean, ucb), ...] for inspection."""
        stats: List[Tuple[Tuple[int, ...], int, float, float]] = []
        for a in self.candidate_actions:
            count = self.counts.get(a, 0)
            mean = self.rew_sums.get(a, 0.0) / count if count > 0 else 0.0
            ucb = mean + math.sqrt((self.ucb_alpha * math.log(max(self.T, 1))) / (count)) if count > 0 else float("inf")
            stats.append((a, count, mean, ucb))
        return stats

    def get_ucb_scores(self) -> Dict[Tuple[int, ...], float]:
        """Return current UCB scores over the candidate set (for debugging/inspection)."""
        t_eff = max(self.t, 1)
        scores: Dict[Tuple[int, ...], float] = {}
        for a in self.candidate_actions:
            c = self.counts.get(a, 0)
            if c == 0:
                scores[a] = float("inf")
            else:
                mean = self.rew_sums.get(a, 0.0) / c
                bonus = math.sqrt((2.0 * math.log(t_eff)) / c)
                scores[a] = min(1.0, mean + bonus)
        return scores
