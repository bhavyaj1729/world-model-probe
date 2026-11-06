import numpy as np

S = 5          # grid size
T = 20         # moves per trajectory
ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}   # UP, DOWN, LEFT, RIGHT
NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]
VOCAB = 4 + S * S      # 4 actions + 25 start cells = 29


def valid_moves(r, c):
    return [a for a, (dr, dc) in ACTIONS.items()
            if 0 <= r + dr < S and 0 <= c + dc < S]


def make_traj(rng):
    r, c = rng.integers(S), rng.integers(S)
    tokens = [4 + r * S + c]
    pos = [(r, c)]
    for _ in range(T):
        a = rng.choice(valid_moves(r, c))
        dr, dc = ACTIONS[a]
        r, c = r + dr, c + dc
        tokens.append(a)
        pos.append((r, c))
    return tokens, pos


def generate_dataset(N=30000, seed=0):
    rng = np.random.default_rng(seed)
    seen, X, P = set(), [], []
    while len(X) < N:
        t, p = make_traj(rng)
        if tuple(t) in seen:
            continue
        seen.add(tuple(t))
        X.append(t)
        P.append(p)
    return np.array(X), np.array(P)


def split(X, P, tr=0.8, va=0.1):
    n_tr, n_va = int(tr * len(X)), int(va * len(X))
    return {
        "X_train": X[:n_tr],            "P_train": P[:n_tr],
        "X_val":   X[n_tr:n_tr + n_va], "P_val":   P[n_tr:n_tr + n_va],
        "X_test":  X[n_tr + n_va:],     "P_test":  P[n_tr + n_va:],
    }
