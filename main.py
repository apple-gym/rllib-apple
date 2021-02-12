import argparse
import os

from ray.rllib.utils.test_utils import check_learning_achieved

from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
import apple_gym.env
from pathlib import Path
from apple_gym.env import apple_pick_env
from models import GRConv
register_env("ApplePick-v0", lambda c: apple_pick_env())
ModelCatalog.register_custom_model(
        "grconv", GRConv)

parser = argparse.ArgumentParser()
parser.add_argument("--num-cpus", type=int, default=6)
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--use-prev-action", action="store_true")
parser.add_argument("--use-prev-reward", action="store_true")
parser.add_argument("--stop-iters", type=int, default=1000000)
parser.add_argument("--stop-timesteps", type=int, default=1000000)
parser.add_argument("--stop-reward", type=float, default=50000.0)
parser.add_argument("--buffer_size", type=int, default=300000)
parser.add_argument("--demonstrations", type=str, default=None)
parser.add_argument("--restore", type=str, default=None)
parser.add_argument('-d', "--debug", action="store_true")

from rich import print
import logging
from rich.logging import RichHandler
logging.basicConfig(level=logging.INFO, handlers=[RichHandler(rich_tracebacks=True, markup=True)])


if __name__ == "__main__":
    import ray
    from ray import tune

    args = parser.parse_args()

    if args.restore == 'auto':
        checkpoints = sorted(Path('/home/wassname/ray_results/SAC/').glob('*/checkpoint_*/checkpoint-*.tune_metadata'))
        print(f'checkpoints found {checkpoints}')
        args.restore = str(checkpoints[-1])
        print(f'auto restore {args.restore}')

    DEBUG = args.debug
    workers = 0 if DEBUG else (args.num_cpus-1)
    print(f'DEBUG {DEBUG}')
    print(f'workers {workers}')

    ray.init(num_cpus=args.num_cpus or None, local_mode=DEBUG, num_gpus=1)

    # see https://docs.ray.io/en/master/rllib-algorithms.html#sac
    # https://docs.ray.io/en/master/rllib-training.html#common-parameters
    
    config = {
            "env": "ApplePick-v0",
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": 0.4,  #int(os.environ.get("RLLIB_NUM_GPUS", "0")),=
            "num_workers": workers,
            "train_batch_size": 256,
            "num_envs_per_worker": 1,
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
            "normalize_actions": True,
            "framework": "torch",
            "target_entropy": "auto",
            "buffer_size": args.buffer_size,
            "prioritized_replay": True,
            "grad_clip": 20,
            "batch_mode": "complete_episodes",
            "no_done_at_end": True,
            "log_level": "INFO",
            "compress_observations": True,
            "ignore_worker_failures": True,
            "worker_side_prioritization": True,
            # "input_evaluation": [],
        }

    # if args.demonstrations:
    #     config['input'] = {
    #         "sampler": 0.8,
    #         args.demonstrations: 0.2,
    #     }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    # # https://docs.ray.io/en/master/tune/user-guide.html#tune-debugging
    # # TODO replace with trainer?
    # import ray.rllib.agents.sac as sac
    # from ray.tune.logger import pretty_print
    # trainer = sac.SACTrainer(config=config)
    # if args.restore is not None:
    #     trainer.restore(args.restore)

    # policy = trainer.get_policy()
    # policy.model.base_model.summary()
    # for i in range(500000):
    #     result = trainer.train()
    #     print(pretty_print(result))

    # if i % 100 == 0:
    #     checkpoint = trainer.save()
    #     print("checkpoint saved at", checkpoint)

    results = tune.run(
        "SAC",
        config=config,
        stop=stop,
        verbose=2 if DEBUG else 1,
        fail_fast=DEBUG,  # DEBUG
        checkpoint_freq=10,
        checkpoint_at_end=True,
        restore=args.restore, # e.g. restore="~/ray_results/Original/PG_<xxx>/checkpoint_5/checkpoint-5",
    )

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    chkpt_path = tune.save()
    print('chkpt_path', chkpt_path)
    ray.shutdown()
