import argparse
import numpy as np
from foienv import FOIEnv, Action
from itertools import product, combinations
from cvxopt import solvers, matrix
from copy import deepcopy
import math

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
            # random joint action
            return {drone: np.random.choice(action_space) for drone in range(n_drones)}

        actions = list(product(*((np.arange(action_space),) * n_drones)))
        action_values = []
        for action in actions:
            A = []
            b = []
            G = []
            h = []
            c = np.zeros(n_drones * action_space)

            for i in range(n_drones):
                c[i * action_space + action[i]] -= phi(S, action).T.dot(theta[i])

            G.append(-1 * np.identity(n_drones * action_space))
            h.extend([0] * (n_drones * action_space))

            for i in range(n_drones):
                arr = np.zeros(n_drones * action_space)
                arr[i * action_space: (i + 1) * action_space] = 1
                A.append(arr)
                b.append(1)

            for i in range(n_drones):
                arr = np.zeros(n_drones * action_space)
                for a in range(action_space):
                    action_ = tuple(x if j != i else a for j, x in enumerate(action))
                    arr[i * action_space + action[i]] -= phi(S, action).T.dot(theta[i]) - phi(S, action_).T.dot(theta[i])
                G.append(arr)
                h.append(0)


            A = matrix(np.stack(A).astype(float))
            b = matrix(np.stack(b).astype(float))
            c = matrix(np.array(c).flatten().astype(float).reshape(-1, 1))
            G = matrix(np.vstack(G).astype(float))
            h = matrix(np.array(h).astype(float).reshape(-1, 1))

            solved = solvers.lp(c, G, h, A=A, b=b)
            if solved['x'] is None:
                import pdb
                pdb.set_trace()
            sol = np.array(solved['x'])
            action_values.append(np.sum([phi(S, action).T.dot(theta[i]) * sol[i * action_space + action[i]] for i in range(n_drones)]))
        action_values = np.array(action_values)
        # random tiebreaking among max value actions
        A = actions[np.random.choice(np.flatnonzero(action_values == action_values.max()))]
        return {drone: A[drone] for drone in range(n_drones)}
    return pi

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    n_drones = 2
    env_shape = (7, 7, 4)

    foi = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ])

    foi2 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ])

    foi3 = np.array([
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ])

    env = FOIEnv(env_shape, foi=foi, fov=30, n_drones=n_drones)
    action_space = env.action_space.n

    gamma = 0.9
    alpha = 0.1
    L = 2000
    episodes = 500
    max_eps = 0.95
    min_eps = 0.05
    decay = 10000
    map_seed = 1
    drone_seed = None

    phi, phi_dim = generate_phi(env_shape, action_space, n_drones)
    theta = np.zeros((n_drones, phi_dim))

    pi = generate_pi(env_shape, action_space, n_drones)

    episode_rewards = np.zeros(episodes)
    episode_steps = np.zeros(episodes).astype(int)
    steps = 0

    try:
        for episode in range(episodes):
            if episode == 250:
                pass
                # env = FOIEnv(env_shape, foi=np.roll(foi, shift=-2, axis=1), n_drones=n_drones)
                # env = FOIEnv(env_shape, foi=foi, fov=45, n_drones=n_drones)
                # env = FOIEnv(env_shape, foi=foi2, fov=45, n_drones=n_drones)
                env = FOIEnv(env_shape, foi=foi3, fov=45, n_drones=n_drones)

            state = env.reset(map_seed=map_seed, drone_seed=drone_seed)
            done = False
            for k in range(L):
                for i in range(n_drones):
                    eps_ = min_eps + (max_eps - min_eps) * math.exp(-1. * steps / decay)
                    pi_A = pi(phi, theta, state, eps=eps_)
                    next_state, reward, done, meta = env.step(pi_A)
                    print(env._map[:,:,0].astype(int))
                    for d in env._drones.values():
                        print(d.view.astype(int))
                    print(episode, k, eps_)
                    print()
                    A = tuple([pi_A[drone] for drone in range(n_drones)])
                    q_next = 0.
                    for action in product(*((np.arange(action_space),) * n_drones)):
                        q = phi(next_state, action).T.dot(theta[i])
                        if q > q_next:
                            q_next = q
                    theta[i] = theta[i] + alpha * (reward + gamma * q_next - phi(state, A).T.dot(theta[i])) * phi(state, A).flatten()
                    episode_rewards[episode] += reward
                    # if done:
                    #     import pdb
                    #     pdb.set_trace()
                    state = next_state
                    if done:
                        break
                episode_steps[episode] += 1
                steps += 1
                if done:
                    break
    except KeyboardInterrupt:
        pass
    print(episode_rewards)
    print(episode_steps)
    import matplotlib.pyplot as plt
    plt.plot(episode_steps)
    plt.savefig('foo.png')
    import pickle
    pickle.dump(episode_steps, open('episode_steps.pkl', 'wb'))

if __name__ == '__main__':
    main()