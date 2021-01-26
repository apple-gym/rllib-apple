import argparse
import os

from ray.rllib.utils.test_utils import check_learning_achieved

from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
import apple_gym.env
from apple_gym.env import apple_pick_env
from models import GRConv
register_env("ApplePick-v0", lambda c: apple_pick_env())
ModelCatalog.register_custom_model(
        "grconv", GRConv)

parser = argparse.ArgumentParser()
parser.add_argument("--num-cpus", type=int, default=0)
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--use-prev-action", action="store_true")
parser.add_argument("--use-prev-reward", action="store_true")
parser.add_argument("--stop-iters", type=int, default=1)
parser.add_argument("--stop-timesteps", type=int, default=100000)
parser.add_argument("--stop-reward", type=float, default=10000.0)


if __name__ == "__main__":
    import ray
    from ray import tune

    args = parser.parse_args()


    ray.init(num_cpus=args.num_cpus or None)

    config = {
            "env": "ApplePick-v0",
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "num_workers": 0,
            "num_envs_per_worker": 1,
            "model": {
                "custom_model": "grconv",
            },
            "framework": "torch",
        }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    results = tune.run("SAC", config=config, stop=stop, verbose=2)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)
    ray.shutdown()
