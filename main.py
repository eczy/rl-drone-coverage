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
        A = actions[np.argmax(action_values)]
        return {drone: A[drone] for drone in range(n_drones)}



        # A = []
        # b = []
        # G = []
        # h = []
        # c = np.zeros(n_drones * action_space)

        # # x = [P0(0), P0(1), ..., PM(N)]
        # for i in range(n_drones):
        #     for A_ in product(*((np.arange(action_space),) * n_drones)):
        #         c[i * action_space + A_[i]] -= phi(S, A_).T.dot(theta[i])

        # # nonnegativity of variables
        # G.append(-1 * np.identity(n_drones * action_space))
        # h.extend([0] * (n_drones * action_space))

        # # probability of single drone actions sum to 1
        # for i in range(n_drones):
        #     arr = np.zeros(n_drones * action_space)
        #     start = action_space * i
        #     arr[start: start + action_space] = 1
        #     A.append(arr)
        #     b.append(1)

        # # expected difference in Q-value >= 0
        # for i in range(n_drones):
        #     arr = np.zeros(n_drones * action_space)
        #     for A_ in product(*((np.arange(action_space),) * n_drones)):
        #         for a in range(action_space):
        #             A__prime = deepcopy(A_)
        #             A__prime = tuple(x if j != i else a for j, x in enumerate(A__prime))
        #             arr[i * action_space + A_[i]] -= phi(S, A_).T.dot(theta[i]) - phi(S, A__prime).T.dot(theta[i])
        #     G.append(arr)
        #     h.append(0)

        # A = matrix(np.stack(A).astype(float))
        # b = matrix(np.stack(b).astype(float))
        # c = matrix(np.array(c).flatten().astype(float).reshape(-1, 1))
        # G = matrix(np.vstack(G).astype(float))
        # h = matrix(np.array(h).astype(float).reshape(-1, 1))

        # solved = solvers.lp(c, G, h, A=A, b=b)
        # sol = np.array(solved['x'])

        # actions = list(product(*((np.arange(action_space),) * n_drones)))
        # values = []
        # for A_ in actions:
        #     value = 0
        #     for i in range(n_drones):
        #         value += phi(S, A_).T.dot(theta[i]) * sol[i * action_space + A_[i]]
        #     values.append(value[0])
        # A = actions[np.argmax(values)]
        # return {drone: A[drone] for drone in range(n_drones)}
    return pi

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    n_drones = 2
    env_shape = (7, 7, 4)

    # foi = np.array([
    #     [0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 0, 0],
    #     [0, 0, 1, 1, 1, 0, 0],
    #     [0, 0, 1, 1, 1, 1, 0],
    #     [0, 1, 1, 1, 1, 0, 0],
    #     [0, 0, 1, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0],
    # ])

    foi = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
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

    def show_values(phi, theta, env, action_space, n_drones):
        X, Y, Z = env.env_shape
        values = np.zeros_like(env._map)
        for x in np.arange(X):
            for y in np.arange(Y):
                for z in np.arange(Z):
                    for action in product(*((np.arange(action_space),) * n_drones)):
                        values[x, y, z] = max(values[x, y, z], phi(((x, y, z),), action).T.dot(theta[0]))
        return np.round(values[:, :, -1], 3)


    try:
        for episode in range(episodes):
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
    # print(show_values(phi, theta, env, action_space, n_drones))
    print(episode_rewards)
    print(episode_steps)
    import matplotlib.pyplot as plt
    plt.plot(episode_steps)
    plt.savefig('foo.png')
    # import pdb
    # pdb.set_trace()

if __name__ == '__main__':
    main()