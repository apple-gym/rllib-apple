import argparse
from pathlib import Path
import os
import torch
import ray
from ray import tune
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

import apple_gym.env
from apple_gym.env import apple_pick_env
from rllib_apple.models import GRConv
from rllib_apple.callbacks import AppleCallbacks


# logging
from rich import print
import logging
from rich.logging import RichHandler
logging.basicConfig(level=logging.INFO, handlers=[RichHandler(rich_tracebacks=True, markup=True)])
# logger = logging.getLogger('apple_gym')
# logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("--num-cpus", type=int, default=4)
parser.add_argument('-e', "--eval", action="store_true", help='Test only')
parser.add_argument('-r', "--render", action='store_true', help='gui (much slower)')
parser.add_argument('-l', "--load", type=str, default=None, help='set to auto or point to a .tune_metadata file')
parser.add_argument('-b', "--buffer_size", type=int, default=50000)
parser.add_argument("--stop-iters", type=int, default=1000000)
parser.add_argument("--stop-timesteps", type=int, default=10000000)
parser.add_argument("--stop-reward", type=float, default=80000.0)
parser.add_argument("--demonstrations", type=str, default=None)
parser.add_argument('-d', "--debug", action="store_true", help='single threaded, for use with pdb')



if __name__ == "__main__":

    args = parser.parse_args()

    # register custom stuff
    register_env("ApplePick-v0", lambda c: apple_pick_env(render=args.render))
    ModelCatalog.register_custom_model("grconv", GRConv)

    if args.load == 'auto':
        checkpoints_metadata = (Path.home() / 'ray_results' / 'SAC/').glob('*/checkpoint_*/checkpoint-*.tune_metadata')
        checkpoints = [c.parent / c.stem for c in checkpoints_metadata]
        checkpoints = sorted(checkpoints, key=lambda s:float(s.stem.split('-')[1]))
        args.load = str(checkpoints[-1])
        print(f'auto restore {args.load}')

    DEBUG = args.debug or args.eval
    workers = 0 if DEBUG else (args.num_cpus-1)
    print(f'DEBUG {DEBUG}')
    print(f'workers {workers}')

    ray.init(num_cpus=args.num_cpus or None, local_mode=DEBUG, num_gpus=1)

    # see https://docs.ray.io/en/master/rllib-algorithms.html#sac
    # https://docs.ray.io/en/master/rllib-training.html#common-parameters
    
    config = {
            "framework": "torch",
            "env": "ApplePick-v0",
            # Size of a batched sampled from replay buffer for training.
            "train_batch_size": 256,
            # Number of workers for collecting samples with. This only makes sense
            # to increase if your environment is particularly slow to sample, or if
            # you're using the Async or Ape-X optimizers.
            "num_workers": workers,
            "num_envs_per_worker": 1,
            "num_gpus": 0.4,
            "num_gpus_per_worker": 0.1,
            # "Q_model": {
            #     "custom_model": "grconv",
            # },
            # "policy_model": {
            #     "custom_model": "grconv",
            # },
            # "model": {
            #     "custom_model": "grconv",
            # },
            # Unsquash actions to the upper and lower bounds of env's action space
            "normalize_actions": True,
            # Target entropy lower bound. If "auto", will be set to -|A| (e.g. -2.0 for
            # Discrete(2), -3.0 for Box(shape=(3,))).
            # This is the inverse of reward scale, and will be optimized automatically.
            "target_entropy": "auto",
            # Size of the replay buffer (in time steps).
            "buffer_size": args.buffer_size,
            # If True prioritized replay buffer will be used.
            "prioritized_replay": True,
            # If not None, clip gradients during optimization at this value.
            "grad_clip": 20,
            "batch_mode": "complete_episodes",
            "no_done_at_end": True,
            "log_level": "INFO",
            # Observation compression. Note that compression makes simulation slow in
            # MPE.
            "compress_observations": True,
            # Whether to attempt to continue training if a worker crashes. The number
            # of currently healthy workers is reported as the "num_healthy_workers"
            # metric.
            "ignore_worker_failures": True,
            # Whether to compute priorities on workers.
            "worker_side_prioritization": True,
            # "input_evaluation": [],
            # Callbacks that will be run during various phases of training. See the
            # `DefaultCallbacks` class and `examples/custom_metrics_and_callbacks.py`
            # for more usage information.
            "callbacks": AppleCallbacks,
            # Whether to write episode stats and videos to the agent log dir. This is
            # typically located in ~/ray_results.   
            "monitor": args.render,
            # "record_env": args.render,
            # "render_env": args.render
        }
    if args.eval:
        config['evaluation_config'] = {'explore': False}

    # if args.demonstrations:
    #     config['input'] = {
    #         "sampler": 0.8,
    #         args.demonstrations: 0.2,
    #     }



    if args.eval:
        # # https://docs.ray.io/en/master/tune/user-guide.html#tune-debugging
        import ray.rllib.agents.sac as sac
        from ray.tune.logger import pretty_print
        trainer = sac.SACTrainer(config=config)
        if args.load is not None:
            trainer.restore(args.load)

        # summary
        policy = trainer.get_policy()
        print(policy.model)

        from torchsummaryX import summary
        summary(policy.model, {'obs':torch.zeros(2, policy.observation_space.shape[0])})

        for i in range(2):
            result = trainer.train()
            print(pretty_print(result))

        # if i % 100 == 0:
        #     checkpoint = trainer.save()
        #     print("checkpoint saved at", checkpoint)

        print(trainer.logdir)
    else:
        stop = {
            "training_iteration": args.stop_iters,
            "timesteps_total": args.stop_timesteps,
            "episode_reward_mean": args.stop_reward,
        }
        results = tune.run(
            "SAC",
            config=config,
            stop=stop,
            verbose=2 if DEBUG else 1,
            fail_fast=DEBUG,  # DEBUG
            checkpoint_freq=10,
            checkpoint_at_end=True,
            restore=args.load, # e.g. restore="~/ray_results/Original/PG_<xxx>/checkpoint_5/checkpoint-5",
        )

        if args.eval:
            check_learning_achieved(results, args.stop_reward)

        chkpt_path = tune.save()
        print('chkpt_path', chkpt_path)
    ray.shutdown()
