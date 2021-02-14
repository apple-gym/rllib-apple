
from typing import Dict
import numpy as np
from rich import print
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks



class AppleCallbacks(DefaultCallbacks):
    keys_to_monitor=[
        'env_reward/apple_pick/tree/min_fruit_dist_reward',
        'env_reward/apple_pick/tree/gripping_fruit_reward',
        # 'env_reward/apple_pick/tree/force_tree_reward',
        # 'env_reward/apple_pick/tree/force_fruit_reward',
        'env_obs/apple_pick/tree/picks'
        ]
    def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, **kwargs):
        print("episode {} started".format(episode.episode_id))
        for k in self.keys_to_monitor:
            episode.user_data[k] = []
            episode.hist_data[k] = []

    def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, **kwargs):
        info = episode.last_info_for()
        if info is not None:
            for k in self.keys_to_monitor:
                episode.user_data[k].append(info[k])

    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       **kwargs):
        for k in self.keys_to_monitor:
            episode.custom_metrics[f'episode_sum_{k}'] = np.sum(episode.user_data[k])
        print(f"episode {episode.episode_id} l={episode.length} ended with {episode.custom_metrics}")
        for k in self.keys_to_monitor:
            episode.hist_data[k] = episode.user_data[k]

    # def on_sample_end(self, worker: RolloutWorker, samples: SampleBatch,
    #                   **kwargs):
    #     print("returned sample batch of size {}".format(samples.count))

    # def on_train_result(self, trainer, result: dict, **kwargs):
    #     print("trainer.train() result: {} -> {} episodes".format(
    #         trainer, result["episodes_this_iter"]))
    #     # you can mutate the result dict to add new fields to return
    #     result["callback_ok"] = True

    # def on_postprocess_trajectory(
    #         self, worker: RolloutWorker, episode: MultiAgentEpisode,
    #         agent_id: str, policy_id: str, policies: Dict[str, Policy],
    #         postprocessed_batch: SampleBatch,
    #         original_batches: Dict[str, SampleBatch], **kwargs):
    #     print("postprocessed {} steps".format(postprocessed_batch.count))
    #     if "num_batches" not in episode.custom_metrics:
    #         episode.custom_metrics["num_batches"] = 0
    #     episode.custom_metrics["num_batches"] += 1
