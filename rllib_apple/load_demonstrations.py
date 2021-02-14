# see https://docs.ray.io/en/master/rllib-offline.html#example-converting-external-experiences-to-batch-format
import gym
import numpy as np
import os

import ray.utils

from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter

from loguru import logger
from gym_recording_modified.playback import get_recordings
from tqdm.auto import tqdm
from pathlib import Path

def load_demonstrations(recordings: Path, outpath: Path = Path('data/rllib_demo')):

    batch_builder = SampleBatchBuilder()
    writer = JsonWriter(str(outpath))

    records = get_recordings(str(recordings))
    logger.info('picks in recordings %s' % sum(records['reward']>10))
    ends=records["episodes_end_point"]
    for i in range(len(ends)-1):
        a = ends[i]
        b = ends[i + 1]
        if b - a < 4:
            continue
        for s in range(a + 1, b):
            rewards = records['reward'][s]
            obs = records['observation'][s-1]
            actions = records['action'][s]
            prev_action = records['action'][s - 1]
            prev_reward = records['reward'][s-1]
            new_obs = records['observation'][s]
            dones = s == b

            batch_builder.add_values(
                t=s,
                eps_id=i,
                agent_index=0,
                obs=obs,
                actions=actions,
                action_prob=1.0,  # put the true action probability here
                action_logp=0.0,
                rewards=rewards,
                prev_actions=prev_action,
                prev_rewards=prev_reward,
                dones=dones,
                infos=None,
                new_obs=new_obs)
        writer.write(batch_builder.build_and_reset())


if __name__ == "__main__":
    load_demonstrations(Path('data/demonstrations'))
