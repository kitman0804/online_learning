"""
Q-Learning in Python - Path Finding

Ref:
http://mnemstudio.org/path-finding-q-learning-tutorial.htm
"""
import numpy as np


class Path(object):
    def __init__(self, path):
        self.path = path
    
    def __repr__(self):
        return " -> ".join([str(x) for x in self.path])
    
    def __str__(self):
        return " -> ".join([str(x) for x in self.path])


class QLearnPath(object):
    """
    c: list or numpy.ndarray
        A connection/adjacency matrix of each state.
    reward: list or numpy.ndarray
        The reward of each state.
    """
    def __init__(self, c, reward):
        assert isinstance(c, (list, np.ndarray)), "c must be a list or numpy.ndarray."
        assert c.shape[0] == c.shape[1], "c must be a square matrix."
        assert c.shape[0] == len(reward), "There must be a reward value for each state."
        assert np.prod(reward >= 0), "reward must be positive."
        self.c = np.array(c)
        self.reward = np.array(reward)
        self.r = None
        self.q = None
    
    def learn(self, gamma=0.8, epoch=10000, random_state=0):
        num_state = self.c.shape[0]
        self.r = -1 * (self.c == 0) + (self.c * self.reward)
        self.q = np.zeros_like(self.r)
        
        np.random.seed(random_state)
        for i in range(epoch):
            state = np.random.choice(num_state)
            possible_action = np.argwhere(self.r[state, :] >= 0).reshape(-1)
            a = np.random.choice(possible_action)
            self.q[state, a] = self.r[state, a] + gamma * self.q[a, self.r[a, :] >= 0].max()
    
    def suggest_path(self, start):
        if self.q is None:
            return None
        q = self.q / self.q.max()
        target = np.argwhere(self.reward == self.reward.max()).reshape(-1)
        path = [start]
        while path[-1] not in target:
            path.append(q[path[-1], :].argmax())
        
        return Path(path)


def main():
    # Connectivity matrix
    c_mtx = np.array([
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 1, 1],
    ])
    r_vec = np.zeros(6)
    r_vec[5] = 100
    # Q learning
    l = QLearnPath(c_mtx, reward=r_vec)
    l.learn(gamma=0.8, epoch=10000)
    print("Matrix R:")
    print(l.r)
    print("Matrix Q:")
    print(l.q)
    print("=" * 17)
    print("Suggested routes:")
    print("=" * 17)
    for s in range(c_mtx.shape[0]):
        print(l.suggest_path(s))


if __name__ == "__main__":
    main()

