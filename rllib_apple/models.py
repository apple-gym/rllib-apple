import logging
import numpy as np
import gym

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, \
    normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.modelv2 import ModelV2
from rllib_apple.process_obs import ProcessObservation

class GRConv(FullyConnectedNetwork):
    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        # reduce obs space
        obs_space2 = gym.spaces.Box(obs_space.low.min(), obs_space.high.max(), (obs_space.shape[0]-ProcessObservation.reduce_obs_space,))
        super().__init__(obs_space2, action_space, num_outputs, model_config, name)
        self.pre = ProcessObservation()
        ModelV2.__init__(self, obs_space, action_space,
                                            num_outputs, model_config, name, framework='torch')
        # self.pre.first_half()

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        x = input_dict["obs_flat"].float()
        x = self.pre(x)
        input_dict["obs_flat"] = x
        return super().forward(input_dict, state, seq_lens)
        

