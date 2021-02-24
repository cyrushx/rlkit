import argparse
import os
import torch
import uuid

from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
from rlkit.core import logger
from data_generator.maze.utils import plot_problem_path, plot_problem_paths

filename = str(uuid.uuid4())


def simulate_policy(args):
    data = torch.load(args.file)
    fig_dir = os.path.dirname(args.file)
    print(fig_dir)
    policy = data['evaluation/policy']
    env = data['evaluation/env']
    print("Policy loaded")
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()
    paths = []
    risk_bounds = [0.1, 0.2, 0.29]
    for risk_bound in risk_bounds:
        path = rollout(
            env,
            policy,
            max_path_length=args.H,
            render=True,
            risk_bound=risk_bound,
        )
        paths.append(path)
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        logger.dump_tabular()

    if args.visualize:
        plot_problem_paths(env, paths, risk_bounds, fig_dir, show_fig=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('-n', '--num-path', type=int, default=10,
                        help='Number of paths to simulate.')
    parser.add_argument('-v', '--visualize', action='store_true',
                        help='Whether to visualize paths.')
    args = parser.parse_args()

    simulate_policy(args)
