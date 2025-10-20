import math
import random
from typing import Dict, List, Tuple, Optional, Set

class RAPS:
    """
    RAPS (2-phase) for environments where the reward node has exactly one parent.
    Phase 1: identify the single parent via descendant tests.
    Phase 2: run UCB on that parent across its v values.
    Phase-2 is warm-started using *all* Phase-1 samples that intervened on the discovered parent.

    Interaction with Env:
        a = learner.action()   # one-hot single-node intervention (n+1 tuple)
        x = env.step(a)        # sample vector, x[-1] in {0,1}
        learner.update(x)

    Parameters
    ----------
    n : int
        Number of non-reward variables (reward is at index n).
    v : int
        Arity of non-reward variables (values 1..v). Reward is binary.
    T : int
        Horizon (total number of pulls planned).
    epsilon : float
        Threshold for descendant detection; uses N=ceil(1/epsilon^2) samples per value.
    alpha : float
        UCB exploration parameter (Hoeffding bonus).
    seed : int
        RNG seed for Phase-1 random choices.
    """

    def __init__(self, n: int, m: int, v: int, T: int, epsilon: float, alpha: float = 0.2, seed: int = random.randint(1, 10**6)):
        assert n >= 1 and v >= 2 and T >= 1
        assert epsilon > 0.0
        self.n, self.v, self.T = n, v, T
        self.epsilon = float(epsilon)
        self.N_per_value = math.ceil(1.0 / (self.epsilon ** 2) * math.log(10))
        self.alpha = alpha
        self.rng = random.Random(seed)

        # Phase control
        self.phase = 1
        self.t_global = 0

        # Phase 1 state
        self.C: Set[int] = set(range(n))      # candidate parents (indices 0..n-1)
        self.current_X: Optional[int] = None  # currently probing node
        self.current_a: int = 1               # value of X we are collecting now
        self.count_a: int = 0                 # how many samples collected for current (X, a)

        # For current X: counts per a for each Z (including reward)
        # counts[X][a][Z][z_val] = count ; we keep only current X to limit memory
        self.cur_counts: Dict[int, Dict[int, Dict[int, Dict[int, int]]]] = {}

        # Phase-1 pull log for warm-starting Phase 2: list of (node_index, value, reward)
        self.phase1_log: List[Tuple[int, int, int]] = []

        # Bookkeeping for Phase 1 decisions
        self._last_reward_descendant_X: Optional[int] = None  # fallback if C becomes empty

        # Phase 2 (once parent discovered)
        self.parent_idx: Optional[int] = None
        self.ucb_counts: List[int] = [0] * v      # per value a=1..v
        self.ucb_sums: List[float] = [0.0] * v
        self.ucb_means: List[float] = [0.0] * v

        # Last chosen action (so update() knows what we played)
        self._last_action: Optional[Tuple[int, ...]] = None
        self._last_phase2_a: Optional[int] = None  # the value set on the parent in phase 2

    # -------------------- public API --------------------
    def action(self) -> Tuple[int, ...]:
        """Return the next single-node intervention (n+1 tuple)."""
        if self.t_global >= self.T:
            a = [0] * (self.n + 1)
            self._last_action = tuple(a)
            return self._last_action

        if self.phase == 1:
            a_vec = self._phase1_next_action()
        else:
            a_vec = self._phase2_next_action()

        self._last_action = a_vec
        self.t_global += 1
        return a_vec

    def update(self, action, sample: Tuple[int, ...]) -> None:
        """Consume the environment sample; update Phase-1/Phase-2 statistics accordingly."""
        assert len(sample) == self.n + 1
        if self.phase == 1:
            self._phase1_update(sample)
        else:
            self._phase2_update(sample)

    # -------------------- Phase 1: discovery --------------------
    def _phase1_next_action(self) -> Tuple[int, ...]:
        if self.current_X is None:
            if not self.C:
                # Fallback: choose a parent using the last X that affected reward, or random.
                if self._last_reward_descendant_X is not None:
                    self._enter_phase2(self._last_reward_descendant_X)
                else:
                    self._enter_phase2(self.rng.randrange(self.n))
                return self._phase2_next_action()

            self.current_X = self.rng.choice(list(self.C))
            self.current_a = 1
            self.count_a = 0
            self._init_cur_counts(self.current_X)

        if self.count_a >= self.N_per_value:
            if self.current_a < self.v:
                self.current_a += 1
                self.count_a = 0
            else:
                # Completed all a in 1..v for this X: run descendant test and prune C
                descendants = self._identify_descendants(self.current_X)
                reward_is_desc = (self.n in descendants)
                if reward_is_desc:                # Make new C = descendants - {X, reward}
                    self._last_reward_descendant_X = self.current_X
                    self.C = descendants.copy()
                    self.C.discard(self.current_X)
                    self.C.discard(self.n)

                else:
                    # Remove X and all its descendants from C
                    to_remove = set(descendants)
                    to_remove.discard(self.n)  # reward isn't in C
                    to_remove.add(self.current_X)
                    self.C -= to_remove

                if len(self.C) == 1:
                    parent = next(iter(self.C))
                    self._enter_phase2(parent)
                    return self._phase2_next_action()

                self.current_X = None
                return self._phase1_next_action()

        a_vec = [0] * (self.n + 1)
        a_vec[self.current_X] = self.current_a
        return tuple(a_vec)

    def _phase1_update(self, sample: Tuple[int, ...]) -> None:
        """Record one sample for the current (X, a) and log it for Phase-2 warm-start."""
        X = self.current_X
        a = self.current_a
        assert X is not None and 1 <= a <= self.v

        # Update counts for each Z (0..n-1 plus reward n)
        for Z in range(self.n):
            z_val = int(sample[Z])
            if z_val == 0:
                continue
            self.cur_counts[X][a][Z][z_val] = self.cur_counts[X][a][Z].get(z_val, 0) + 1

        # Reward node
        r = int(sample[-1])  # in {0,1}
        self.cur_counts[X][a][self.n][r] = self.cur_counts[X][a][self.n].get(r, 0) + 1

        # ---- Warm-start log ----
        # Since all Phase-1 interventions are single-node do(X=a), we can log (X, a, reward).
        self.phase1_log.append((X, a, r))

        self.count_a += 1

    def _init_cur_counts(self, X: int) -> None:
        self.cur_counts[X] = {}
        for a in range(1, self.v + 1):
            self.cur_counts[X][a] = {}
            for Z in range(self.n + 1):  # includes reward at index n
                self.cur_counts[X][a][Z] = {}

    def _identify_descendants(self, X: int) -> Set[int]:
        """
        Z is a descendant of X iff ∃ a1≠a2 and value z such that
        | P(Z=z | do(X=a1)) - P(Z=z | do(X=a2)) | ≥ epsilon.
        """
        descendants: Set[int] = set()
        for Z in range(self.n + 1):
            domain = [0, 1] if Z == self.n else list(range(1, self.v + 1))

            def p_a_z(a: int, z_val: int) -> float:
                cnt = self.cur_counts[X][a][Z].get(z_val, 0)
                denom = max(1, self.N_per_value)
                return cnt / denom

            found = False
            for a1 in range(1, self.v + 1):
                for a2 in range(a1 + 1, self.v + 1):
                    for z_val in domain:
                        if abs(p_a_z(a1, z_val) - p_a_z(a2, z_val)) >= self.epsilon:
                            descendants.add(Z)
                            found = True
                            break
                    if found: break
                if found: break
        return descendants

    def _enter_phase2(self, parent_idx: int) -> None:
        """Initialize Phase 2 with the discovered parent index and warm-start UCB using Phase-1 samples."""
        self.phase = 2
        self.parent_idx = int(parent_idx)
        # Reset Phase-2 stats
        self.ucb_counts = [0] * self.v
        self.ucb_sums = [0.0] * self.v
        self.ucb_means = [0.0] * self.v

        # ---- Warm-start from Phase-1 log ----
        # Use ALL samples from Phase 1 where we intervened on this parent with value a.
        for (node, val, rew) in self.phase1_log:
            if node == self.parent_idx:
                i = val - 1
                self.ucb_counts[i] += 1
                self.ucb_sums[i] += float(rew)

        # Compute means (avoid div by zero)
        for i in range(self.v):
            if self.ucb_counts[i] > 0:
                self.ucb_means[i] = self.ucb_sums[i] / self.ucb_counts[i]

        # Clear Phase-1 working buffers (keep phase1_log for debugging if you prefer)
        self.current_X = None
        self.cur_counts.clear()

    # -------------------- Phase 2: UCB on the discovered parent --------------------
    def _phase2_next_action(self) -> Tuple[int, ...]:
        if self.parent_idx is None:
            fallback = self._last_reward_descendant_X if self._last_reward_descendant_X is not None else 0
            self._enter_phase2(fallback)

        # Standard UCB over a in {1..v} with binary rewards (warm-started counts/means)
        total_pulls = sum(self.ucb_counts)
        log_term = max(1.0, math.log(total_pulls + 1))
        best_a, best_ucb = 1, -1.0
        for a in range(1, self.v + 1):
            i = a - 1
            if self.ucb_counts[i] == 0:
                ucb = 1.0
            else:
                bonus = math.sqrt(self.alpha * log_term / (2.0 * self.ucb_counts[i]))
                #ucb = min(1.0, self.ucb_means[i] + bonus)
                ucb = self.ucb_means[i] + bonus
            if ucb > best_ucb:
                best_ucb, best_a = ucb, a

        self._last_phase2_a = best_a
        vec = [0] * (self.n + 1)
        vec[self.parent_idx] = best_a
        return tuple(vec)

    def _phase2_update(self, sample: Tuple[int, ...]) -> None:
        r = float(sample[-1])
        a = self._last_phase2_a
        if a is None:
            return
        i = a - 1
        self.ucb_counts[i] += 1
        self.ucb_sums[i] += r
        self.ucb_means[i] = self.ucb_sums[i] / self.ucb_counts[i]

    # -------------------- Convenience --------------------
    def discovered_parent(self) -> Optional[int]:
        """Return the discovered parent index (once Phase 2 starts)."""
        return self.parent_idx

    def phase1_candidates(self) -> Set[int]:
        """Return the current candidate set C (useful for monitoring)."""
        return set(self.C)

    def phase2_value_stats(self) -> List[Tuple[int, int, float]]:
        """Return [(a, count, mean)] for a in 1..v during Phase 2."""
        return [(a, self.ucb_counts[a-1], self.ucb_means[a-1]) for a in range(1, self.v + 1)]
