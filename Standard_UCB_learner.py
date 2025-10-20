import math
from itertools import combinations, product
from typing import List, Tuple

class standard_UCB:
    """
    Standard UCB learner (no sample sharing).
    
    Action representation:
      - Tuple of length n+1 (indices 0..n-1 are non-reward nodes, index n is reward node).
      - Exactly m entries among indices 0..n-1 are in {1,2,...,v}; others are 0.
      - The last entry (reward node) is always 0.

    Usage:
      learner = standard_UCB(n, m, v, T)
      a = learner.action()        # choose next action
      x = env.step(a)             # sample from Env
      learner.update(a, x)        # update stats using reward of action a
    """

    def __init__(self, n: int, m: int, v: int, T: int, alpha: float = 0.1):
        assert 0 < m <= n, "m must be in {1,...,n}"
        assert v >= 2, "arity v must be >= 2"
        assert T >= 1, "horizon T must be >= 1"

        self.n = n
        self.m = m
        self.v = v
        self.T = T
        self.ucb_alpha = alpha

        # Build full action set
        self.actions = list(self._enumerate_all_actions())
        self.num_actions = len(self.actions)
        #print(self.num_actions)

        # Stats
        self.counts = [0] * self.num_actions
        self.sum_rewards = [0.0] * self.num_actions
        self.means = [0.0] * self.num_actions
        self.t = 0

        # Action lookup: action tuple → index
        self.action_to_idx = {a: i for i, a in enumerate(self.actions)}

    # ---------- Core API ----------

    def action(self) -> Tuple[int, ...]:
        """
        Select next action according to standard UCB on binary rewards.
        """
        self.t += 1
        log_term = max(1.0, math.log(self.T))
        best_idx, best_ucb = 0, -1.0
        for i in range(self.num_actions):
            if self.counts[i] == 0:
                ucb = 1.0
            else:
                bonus = math.sqrt(self.ucb_alpha * log_term / (self.counts[i]))
                #ucb = min(1.0, self.means[i] + bonus)
                ucb = self.means[i] + bonus
            if ucb > best_ucb:
                best_ucb = ucb
                best_idx = i
        return self.actions[best_idx]

    def update(self, action: Tuple[int, ...], sample: Tuple[int, ...]) -> None:
        """
        Update stats using reward of the given action.
        - action: the action that was actually played (tuple of length n+1).
        - sample: full sample vector (length n+1), with sample[-1] = reward ∈ {0,1}.
        """
        reward = float(sample[-1])
        idx = self.action_to_idx[action]
        self.counts[idx] += 1
        self.sum_rewards[idx] += reward
        self.means[idx] = self.sum_rewards[idx] / self.counts[idx]

    # ---------- Helpers ----------

    def _enumerate_all_actions(self):
        """
        Generator for all actions with exactly m interventions over 0..n-1, values in 1..v.
        """
        zeros = [0] * (self.n + 1)
        for idxs in combinations(range(self.n), self.m):
            for vals in product(range(1, self.v + 1), repeat=self.m):
                a = list(zeros)
                for j, i in enumerate(idxs):
                    a[i] = vals[j]
                yield tuple(a)

    # ---------- Convenience ----------

    def get_action_stats(self) -> List[Tuple[Tuple[int, ...], int, float, float]]:
        """Return [(action, count, mean, ucb), ...] for inspection."""
        return [(self.actions[i], self.counts[i], self.means[i], 
                 self.means[i] + math.sqrt((self.ucb_alpha * math.log(max(self.T, 1))) / (self.counts[i])) if self.counts[i] > 0 else float("inf")) for i in range(self.num_actions)]

    def best_empirical_action(self) -> Tuple[Tuple[int, ...], float]:
        """Return (argmax_by_mean, mean)."""
        if self.num_actions == 0:
            return (tuple([0]*(self.n+1)), 0.0)
        i = max(range(self.num_actions), key=lambda j: self.means[j])
        return self.actions[i], self.means[i]
