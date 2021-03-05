import argparse
import os
import time
import torch
import uuid
import numpy as np

from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
from rlkit.core import logger
from data_generator.maze.utils import plot_problem_path, plot_problem_paths

filename = str(uuid.uuid4())


def simulate_policy(args):
    data = torch.load(args.file)
    fig_dir = os.path.dirname(args.file)
    policy = data['evaluation/policy']
    env = data['evaluation/env']
    print("Policy loaded")
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()

    # Obtain path for each risk bound.
    risk_bounds = args.risk_bounds
    paths = []
    times = []
    for risk_bound in risk_bounds:
        t0 = time.time()
        # Roll out path in test mode (without risk computation, fixed random seed).
        path = rollout(
            env,
            policy,
            max_path_length=args.H,
            render=True,
            risk_bound=risk_bound,
            test_mode=True,
        )
        time_diff = time.time() - t0
        times.append(time_diff)
        paths.append(path)
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        logger.dump_tabular()

    # Generate paths at different initial locations.
    start_locations = [[4.0, 4.0], [15.0, 10.0], [17.0, 17.0], [16.0, 5.0]]
    paths_s = []
    for start in start_locations:
        env.set_start_state(np.array(start))
        path = rollout(
            env,
            policy,
            max_path_length=args.H,
            render=True,
            risk_bound=0.1,
            test_mode=True,
        )
        paths_s.append(path)

    start_locations = [[4.0, 4.0], [15.0, 10.0], [17.0, 17.0], [16.0, 5.0]]
    paths = []
    for start in start_locations:
        env.set_start_state(np.array(start))
        path = rollout(
            env,
            policy,
            max_path_length=args.H,
            render=True,
            risk_bound=0.3,
            test_mode=True,
        )
        paths.append(path)
    env.set_start_state(np.array([4.0, 4.0]))

    plot_problem_paths(env, paths, risk_bounds, times, fig_dir, show_fig=args.visualize, show_baseline=args.baseline,
                       extra_paths=paths_s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('-v', '--visualize', action='store_true',
                        help='Whether to pop visualized paths.')
    parser.add_argument('--risk-bounds',
                        nargs='+',
                        type=float,
                        default=[0.1, 0.2, 0.3],
                        help="Set of risk bounds to plot.")
    args = parser.parse_args()
    simulate_policy(args)
