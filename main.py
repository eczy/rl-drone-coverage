import argparse
import numpy as np
from field_coverage_env import FieldCoverageEnv
from itertools import product, combinations
from cvxopt import solvers, matrix
from copy import deepcopy
import math
import matplotlib.pyplot as plt
import pickle
import shutil
import os

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
            sol = np.array(solved['x'])
            action_values.append(np.sum([phi(S, action).T.dot(theta[i]) * sol[i * action_space + action[i]] for i in range(n_drones)]))
        action_values = np.array(action_values)
        # random tiebreaking among max value actions
        A = actions[np.random.choice(np.flatnonzero(action_values == action_values.max()))]
        return {drone: A[drone] for drone in range(n_drones)}
    return pi

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('foi', type=str, help='File containing FOI data.')
    parser.add_argument('output_dir', type=str, help='Output directory.')
    parser.add_argument('-f', action='store_true', help='Overwrite output directory if it already exists.')
    parser.add_argument('--fov', type=float, default=np.radians(30), help='Drone field of vision.')
    parser.add_argument('--env_dim', default=None, nargs=3, type=int, metavar=('X', 'Y', 'Z'), help='Environment dimensions. Will be inferred from FOI if not specified.')
    parser.add_argument('--n_drones', default=1, type=int, help='Number of drones to simulate.')
    parser.add_argument('--gamma', default=0.9, type=float, help='Discount factor.')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate.')
    parser.add_argument('--n_episodes', default=500, type=int, help='Number of episodes to simulate.')
    parser.add_argument('--episode_max_steps', default=2000, type=int, help='Maximum number of steps per episode.')
    parser.add_argument('--max_eps', default=0.95, type=float, help='Max epsilon for epsilon-greedy policy.')
    parser.add_argument('--min_eps', default=0.05, type=float, help='Min epsilon for epsilon-greedy policy.')
    parser.add_argument('--eps_decay', default=10000, type=float, help='Epsilon decay rate for epsilon-greedy policy.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument('--perturb_foi', default=None, nargs=2, metavar=('new_foi', 'episode'), help='Substitute original foi with new_foi at specific episode.')
    parser.add_argument('--perturb_fov', default=None, nargs=2, metavar=('new_fov', 'episode'), help='Substitute original fov with new_fov at specific episode.')
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        if args.f:
            shutil.rmtree(args.output_dir)
        else:
            raise FileExistsError(f'Output directory {args.output_dir} already exists.')
    os.makedirs(args.output_dir)

    np.random.seed(args.seed)

    foi = np.genfromtxt(args.foi, delimiter=',')
    env_dim = args.env_dim if args.env_dim is not None else tuple([x for x in foi.shape] + [max(foi.shape)])

    env = FieldCoverageEnv(env_dim, foi=foi, fov=args.fov, n_drones=args.n_drones)
    action_space = env.action_space.n

    phi, phi_dim = generate_phi(env_dim, action_space, args.n_drones)
    theta = np.zeros((args.n_drones, phi_dim))

    pi = generate_pi(env_dim, action_space, args.n_drones)

    episode_rewards = np.zeros(args.n_episodes)
    episode_steps = np.zeros(args.n_episodes).astype(int)
    steps = 0

    try:
        for episode in range(args.n_episodes):
            if args.perturb_foi is not None and episode == int(args.perturb_foi[1]):
                print('FOI perturbation applied.')
                env.foi = np.genfromtxt(args.perturb_foi[0], delimiter=',')
            if args.perturb_fov is not None and episode == int(args.perturb_fov[1]):
                print('FOV perturbation applied.')
                env.fov = float(args.perturb_fov[0])
            state = env.reset()
            done = False
            for k in range(args.episode_max_steps):
                for i in range(args.n_drones):
                    epsilon = args.min_eps + (args.max_eps - args.min_eps) * math.exp(-1. * steps / args.eps_decay)
                    pi_A = pi(phi, theta, state, eps=epsilon)
                    next_state, reward, done, meta = env.step(pi_A)
                    A = tuple([pi_A[drone] for drone in range(args.n_drones)])
                    actions = product(*((np.arange(action_space),) * args.n_drones))
                    q_next = np.max([phi(next_state, action).T.dot(theta[i]) for action in actions])
                    theta[i] = theta[i] + args.lr * (reward + args.gamma * q_next - phi(state, A).T.dot(theta[i])) * phi(state, A).flatten()
                    episode_rewards[episode] += reward
                    state = next_state
                    if done:
                        break
                episode_steps[episode] += 1
                steps += 1
                if done:
                    break
            print(f'Episode {episode}: {k + 1} steps.')
    except KeyboardInterrupt:
        pass
    plt.ylim(0, args.episode_max_steps)
    plt.plot(episode_steps)
    plt.savefig(os.path.join(args.output_dir, 'episode_steps.png'))
    pickle.dump(episode_steps, open(os.path.join(args.output_dir, 'episode_steps.pkl'), 'wb'))

if __name__ == '__main__':
    main()