import argparse
import numpy as np
from foienv import FOIEnv, Action
from itertools import product, combinations
from cvxopt import solvers, matrix
from copy import deepcopy

solvers.options['show_progress'] = False


def generate_phi(env_shape, action_space, n_drones):
    X, Y, Z = env_shape
    state_dim = (X + Y + Z) * n_drones

    drone_actions = np.arange(action_space)
    actions = {x: i for i, x in enumerate(product(*((drone_actions,) * n_drones)))}
    def phi(S, A):
        states = []
        for i in range(n_drones):
            x, y, z = S[i]
            arr = np.zeros(X)
            arr[x] = 1
            states.append(arr)

            arr = np.zeros(Y)
            arr[y] = 1
            states.append(arr)

            arr = np.zeros(Z)
            arr[z] = 1
            states.append(arr)
        states = np.concatenate(states)
        if A is None:
            state = np.repeat(states, action_space ** n_drones)
        else:
            state = np.zeros(len(states) * (action_space ** n_drones))
            action_slot = actions[A] * len(states)
            state[action_slot: action_slot + len(states)] = states
        return state.astype(int).reshape(-1, 1)
    return phi, state_dim * action_space ** n_drones

def generate_pi(env_shape, action_space, n_drones):
    def pi(phi, theta, S, eps=0.9):
        sample = np.random.random()
        if sample < eps:
            # uniform distribution over actions
            return {drone: np.random.choice(action_space) for drone in range(n_drones)}

        A = []
        b = []
        G = []
        h = []
        c = np.zeros(n_drones * action_space)

        # x = [P0(0), P0(1), ..., PM(N)]
        for i in range(n_drones):
            for A_ in product(*((np.arange(action_space),) * n_drones)):
                c[i * n_drones + A_[i]] -= phi(S, A_).T.dot(theta[i])

        # nonnegativity of variables
        G.append(-1 * np.identity(n_drones * action_space))
        h.extend([0] * (n_drones * action_space))

        # probability of single drone actions sum to 1
        for i in range(n_drones):
            arr = np.zeros(n_drones * action_space)
            start = action_space * i
            arr[start: start + action_space] = 1
            A.append(arr)
            b.append(1)

        # expected difference in Q-value > 0
        for i in range(n_drones):
            for A_ in product(*((np.arange(action_space),) * n_drones)):
                arr = np.zeros(n_drones * action_space)
                for a in range(action_space):
                    A__prime = deepcopy(A_)
                    A__prime = tuple(x if j != i else a for j, x in enumerate(A__prime))
                    arr[i * n_drones + A_[i]] -= phi(S, A_).T.dot(theta[i]) - phi(S, A__prime).T.dot(theta[i])
                G.append(arr)
                h.append(0)

        A = matrix(np.stack(A).astype(float))
        b = matrix(np.stack(b).astype(float))
        c = matrix(np.array(c).flatten().astype(float).reshape(-1, 1))
        G = matrix(np.vstack(G).astype(float))
        h = matrix(np.array(h).astype(float).reshape(-1, 1))

        solved = solvers.lp(c, G, h, A=A, b=b)
        sol = np.array(solved['x'])
        if solved['x'] is None:
            import pdb; pdb.set_trace()
            # return {drone: np.random.choice(action_space) for drone in range(n_drones)}


        actions = list(product(*((np.arange(action_space),) * n_drones)))
        values = []
        for A_ in actions:
            value = 0
            for i in range(n_drones):
                value += phi(S, A_).T.dot(theta[i]) * sol[i * action_space + A_[i]]
            values.append(value)
        A = actions[np.argmax(values)]
        return {drone: A[drone] for drone in range(n_drones)}
            
    return pi

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    n_drones = 3
    env_shape = (7, 7, 5)

    env = FOIEnv(env_shape, n_drones=n_drones)
    action_space = env.action_space.n

    gamma = 0.99
    alpha = 0.01
    L = 100
    episodes = 10
    decay_base = 0.99

    phi, phi_dim = generate_phi(env_shape, action_space, n_drones)
    theta = np.zeros((n_drones, phi_dim))

    pi = generate_pi(env_shape, action_space, n_drones)

    episode_rewards = np.zeros(episodes)
    steps = 0
    for episode in range(episodes):
        state = env.reset()
        done = False
        for k in range(2000):
            for i in range(n_drones):
                pi_A = pi(phi, theta, state, eps=decay_base ** steps)
                next_state, reward, done, _ = env.step(pi_A)
                print(env._map[:,:,0])
                for d in env._drones.values():
                    print(d.view.astype(int))
                print()
                A = tuple([pi_A[drone] for drone in range(n_drones)])
                q_next = 0
                for action in product(*((np.arange(action_space),) * n_drones)):
                    q = phi(next_state, action).T.dot(theta[i])
                    if q > q_next:
                        q_next = q
                theta[i] = theta[i] + alpha * (reward + gamma * q_next - phi(state, A).T.dot(theta[i])) * phi(state, A).flatten()
                episode_rewards[episode] += reward
            if done:
                break
            steps += 1
    print(episode_rewards)
    import pdb
    pdb.set_trace()

if __name__ == '__main__':
    main()