import math
import random
from itertools import combinations, product
from typing import Dict, List, Tuple, Set, Optional

class unknown_k_UCB_plus:
    """
    Unknown-k UCB-plus with phase schedule and mixture actions.

    Interaction pattern:
        a = learner.action()        # returns a concrete atomic action (tuple of length n+1)
        x = env.step(a)             # env returns sample, x[-1] in {0,1}
        learner.update(x)           # updates both the chosen arm's stats and (if mixture) the mixture stats

    Notes:
    - The learner never needs k.
    - UCB is standard Hoeffding-style on binary rewards; stats of atomic arms carry over across phases.
    - Mixture arms have their own UCB stats; when a mixture is selected, we ALSO update the underlying atomic arm stats.
    """

    def __init__(
        self,
        n: int,
        m: int,
        v: int,
        T: int,
        seed: int = random.randint(1, 10**6),
    ):
        assert 0 < m <= n, "m must be in {1,...,n}"
        assert v >= 2, "arity v must be >= 2"
        assert T >= 1, "horizon T must be >= 1"

        self.n, self.m, self.v, self.T = n, m, v, T
        self.alpha = 0.1 / math.comb(n,m) 
        self.rng = random.Random(seed)

        # ----- Phase schedule -----
        self.i_f = math.ceil(math.log2(math.sqrt(T * m / (v * n))))
        r = math.ceil(math.log2(math.sqrt(T)))
        self.q = lambda i: 2 ** (r - i + 1)
        self.Ti = lambda i: 2 ** (r + i) * math.ceil(v * n / m)

        # Running counters
        self.t_global = 0                # total pulls across all phases
        self.i_phase = 1                 # current phase index (1-based)
        self.t_in_phase = 0              # pulls within current phase (so far)
        self.current_Ti = self.Ti(1)     # rounds budget for current phase

        # ----- Sets C_i = S_i ∪ M -----
        # S_i: set of atomic actions in current phase
        self.S_i: List[Tuple[int, ...]] = self._sample_actions_uniform(self.q(1))
        # M: mixture arm ids available in current phase; each id is an int j with 1 <= j < i_phase
        self.M: List[int] = []

        # ----- Stats -----
        # Global stats for atomic actions (carried over phases)
        self.atomic_counts: Dict[Tuple[int, ...], int] = {a: 0 for a in self.S_i}
        self.atomic_sums: Dict[Tuple[int, ...], float] = {a: 0.0 for a in self.S_i}

        # Per-mixture stats (indexed by phase j)
        self.mixture_counts: Dict[int, int] = {}
        self.mixture_sums: Dict[int, float] = {}

        # History of actual atomic actions played *per phase* (for defining mixtures)
        # phase_actions[j] is a list of atomic actions (length will be exactly T_j unless truncated by T)
        self.phase_actions: Dict[int, List[Tuple[int, ...]]] = {}

        # Working list of actions played in the current phase (atomic actions actually sent to env)
        self._current_phase_playlog: List[Tuple[int, ...]] = []

        # Bookkeeping for the last action() (so update() knows what we chose)
        self._last_selected_kind: Optional[str] = None     # 'atomic' or 'mixture'
        self._last_selected_atomic: Optional[Tuple[int, ...]] = None  # the concrete atomic action sent to env
        self._last_selected_mixture_id: Optional[int] = None          # j if a mixture was selected

    # ---------- Public API ----------

    def action(self) -> Tuple[int, ...]:
        """
        Choose next action according to standard UCB over current C_i = S_i ∪ M.
        Returns a *concrete atomic* action (length n+1 tuple) to pass to the environment.
        """
        # If the current phase is done, advance
        if self.t_in_phase >= self.current_Ti:
            self._end_phase_and_advance()

        # Build candidate set for UCB indexes: atomic in S_i and mixtures in M
        candidates = []
        for a in self.S_i:
            c = self.atomic_counts.get(a, 0)
            s = self.atomic_sums.get(a, 0.0)
            mu = (s / c) if c > 0 else 0.0
            candidates.append(('atomic', a, c, mu))
        for j in self.M:
            c = self.mixture_counts.get(j, 0)
            s = self.mixture_sums.get(j, 0.0)
            mu = (s / c) if c > 0 else 0.0
            candidates.append(('mixture', j, c, mu))

        # Standard UCB index with optimistic init (ucb=1 for unseen), clip to 1 for binary rewards
        self.t_global += 1  # for log term progression
        log_term = max(1.0, math.log(self.t_global))
        best = None
        best_ucb = -1.0
        for kind, ident, c, mu in candidates:
            if c == 0:
                ucb = 1.0
            else:
                bonus = math.sqrt(self.alpha * log_term / (c))
                #ucb = min(1.0, mu + bonus)
                ucb = mu + bonus
            if ucb > best_ucb:
                best_ucb = ucb
                best = (kind, ident)

        # Resolve the choice to a concrete atomic action for the environment
        if best[0] == 'atomic':
            a = best[1]
            self._stash_last_choice('atomic', a, None)
            return a
        else:
            j = best[1]  # mixture id
            # Draw s uniformly from {1,.., T_j} (or observed length if truncated)
            hist = self.phase_actions[j]
            if not hist:
                # Defensive: if somehow empty, fall back to first S_i action
                a = self.S_i[0] if self.S_i else self._any_action()
            else:
                s_idx = self.rng.randrange(len(hist))  # 0..len-1
                a = hist[s_idx]
            self._stash_last_choice('mixture', a, j)
            return a

    def update(self, action, sample: Tuple[int, ...]) -> None:
        """
        Receive a sample (length n+1) and update stats for:
          - the concrete atomic action that was just executed, and
          - the mixture arm (if a mixture was selected).
        Also appends to the current phase's play log and handles phase transitions.
        """
        assert len(sample) == self.n + 1, "sample must be length n+1"
        reward = float(sample[-1])

        # Update atomic stats
        a = self._last_selected_atomic
        if a not in self.atomic_counts:
            self.atomic_counts[a] = 0
            self.atomic_sums[a] = 0.0
        self.atomic_counts[a] += 1
        self.atomic_sums[a] += reward

        # If this step came from a mixture, update its stats too
        if self._last_selected_kind == 'mixture':
            j = self._last_selected_mixture_id
            self.mixture_counts[j] = self.mixture_counts.get(j, 0) + 1
            self.mixture_sums[j] = self.mixture_sums.get(j, 0.0) + reward

        # Append to current phase's played atomic actions
        self._current_phase_playlog.append(a)
        self.t_in_phase += 1

        # If we've exhausted the phase budget, advance the phase (unless we're out of global horizon)
        if self.t_in_phase >= self.current_Ti and self.t_global < self.T:
            self._end_phase_and_advance()

    # ---------- Internals ----------

    def _end_phase_and_advance(self):
        """
        Finalize current phase i: freeze its action history (for future mixture),
        add its mixture arm to M, then build S_{i+1} and move to next phase.
        """
        # Freeze phase i history
        self.phase_actions[self.i_phase] = list(self._current_phase_playlog)

        # Future mixture gets the concrete actions of this phase
        if len(self._current_phase_playlog) > 0:
            self.M.append(self.i_phase)           # add mixture id for this phase
            self.mixture_counts.setdefault(self.i_phase, 0)
            self.mixture_sums.setdefault(self.i_phase, 0.0)

        # Advance phase
        self.i_phase += 1
        self.t_in_phase = 0
        self._current_phase_playlog = []

        # If exceeded i_f, keep using the last available (S_i with i=i_f) and all mixtures;
        # but we still cap by global T.
        if self.i_phase > self.i_f:
            # Keep S_i as the best q_{i_f} atomic arms so far (no change to M)
            self.S_i = self._select_top_atomic(self.q(self.i_f))
            self.current_Ti = self.T - self.t_global  # spend whatever is left
            return

        # Build S_i for the new phase: best q_i empirical atomic arms so far
        self.S_i = self._select_top_atomic(self.q(self.i_phase))
        # Ensure we have at least something to pull
        if not self.S_i:
            # Bootstrap by sampling fresh random actions if we somehow have none
            self.S_i = self._sample_actions_uniform(self.q(self.i_phase))

        # Update the phase budget
        remaining = self.T - self.t_global
        self.current_Ti = min(self.Ti(self.i_phase), max(0, remaining))

    def _select_top_atomic(self, q: int) -> List[Tuple[int, ...]]:
        """
        Select top-q atomic arms by empirical mean among all that have been tried so far.
        If fewer than q have nonzero counts, fill the remainder by random unseen actions.
        """
        tried = [(a, self.atomic_sums[a] / self.atomic_counts[a])
                 for a in self.atomic_counts.keys() if self.atomic_counts[a] > 0]
        tried.sort(key=lambda t: t[1], reverse=True)
        top = [a for a, _ in tried[:q]]
        if len(top) < q:
            # Fill with fresh random unseen actions
            need = q - len(top)
            fillers = self._sample_actions_uniform(need, exclude=set(self.atomic_counts.keys()))
            # initialize their stats
            for a in fillers:
                if a not in self.atomic_counts:
                    self.atomic_counts[a] = 0
                    self.atomic_sums[a] = 0.0
            top.extend(fillers)
        return top

    def _any_action(self) -> Tuple[int, ...]:
        """Return any valid atomic action (used only as a defensive fallback)."""
        for a in self._enumerate_all_actions():
            return a
        # Should never happen if inputs are valid
        return tuple([0] * (self.n + 1))

    def _stash_last_choice(self, kind: str, atomic_action: Tuple[int, ...], mixture_id: Optional[int]):
        self._last_selected_kind = kind              # 'atomic' or 'mixture'
        self._last_selected_atomic = atomic_action   # concrete atomic action sent to env
        self._last_selected_mixture_id = mixture_id  # j if mixture

    # ---------- Action space helpers ----------

    def _enumerate_all_actions(self):
        """
        Generator for all actions (n+1 tuple) with exactly m interventions among indices 0..n-1 (values 1..v).
        """
        zeros = [0] * (self.n + 1)
        for idxs in combinations(range(self.n), self.m):
            for vals in product(range(1, self.v + 1), repeat=self.m):
                a = list(zeros)
                for j, i in enumerate(idxs):
                    a[i] = vals[j]
                yield tuple(a)

    def _sample_actions_uniform(self, num: int, exclude: Optional[Set[Tuple[int, ...]]] = None) -> List[Tuple[int, ...]]:
        """
        Uniformly sample 'num' distinct actions without replacement.
        If the space is small, enumerate and sample; otherwise do rejection sampling.
        """
        exclude = exclude or set()
        total_actions = math.comb(self.n, self.m) * (self.v ** self.m)
        target = max(0, num)
        if target == 0:
            return []

        # If tractable, enumerate and sample
        if total_actions <= 200_000:
            pool = [a for a in self._enumerate_all_actions() if a not in exclude]
            self.rng.shuffle(pool)
            return pool[:target]

        # Otherwise rejection-sample using random index sets and values
        seen_keys: Set[frozenset] = set()
        results: List[Tuple[int, ...]] = []
        zeros = [0] * (self.n + 1)
        attempts = 0
        max_attempts = target * 100
        while len(results) < target and attempts < max_attempts:
            attempts += 1
            idxs = self.rng.sample(range(self.n), self.m)
            vals = [self.rng.randint(1, self.v) for _ in range(self.m)]
            a = list(zeros)
            for j, i in enumerate(idxs):
                a[i] = vals[j]
            a = tuple(a)
            if a in exclude:
                continue
            key = frozenset((i, a[i]) for i in range(self.n) if a[i] != 0)
            if key not in seen_keys:
                seen_keys.add(key)
                results.append(a)
        # Fallback fill by partial enumeration if rejection didn't reach target
        if len(results) < target:
            need = target - len(results)
            existing = set(results) | exclude
            for a in self._enumerate_all_actions():
                if a not in existing:
                    results.append(a)
                    existing.add(a)
                    if len(results) == target:
                        break
        return results

    # ---------- Convenience ----------

    def get_atomic_stats(self) -> List[Tuple[Tuple[int, ...], int, float]]:
        """[(action, count, mean), ...] for atomic arms with any count."""
        out = []
        for a, c in self.atomic_counts.items():
            mu = (self.atomic_sums[a] / c) if c > 0 else 0.0
            out.append((a, c, mu))
        return out

    def get_mixture_stats(self) -> List[Tuple[int, int, float]]:
        """[(mixture_phase_id, count, mean), ...]"""
        out = []
        for j, c in self.mixture_counts.items():
            mu = (self.mixture_sums[j] / c) if c > 0 else 0.0
            out.append((j, c, mu))
        return out

    def current_candidates(self) -> Tuple[List[Tuple[int, ...]], List[int]]:
        """Return (S_i, M) for the current phase."""
        return list(self.S_i), list(self.M)
