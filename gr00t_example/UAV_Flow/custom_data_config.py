# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from gr00t.data.transform.base import ComposedModalityTransform, ModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform import InvertibleModalityTransform
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from gr00t.experiment.data_config import BaseDataConfig
from gr00t.model.transforms import GR00TTransform
import numpy as np
from pydantic import Field
from typing import Any

class ToZeroStateTransform(InvertibleModalityTransform):
    apply_to: list[str] = Field(..., description="The keys in the modality to load and transform.")
    
    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            data[key] = np.array([[0.0]])
        return data
    
    def unapply(self, data):
        # not doing anything in unapply since we don't have the original data
        for key in self.apply_to:
            data[key] = np.array([[0.0]])
        return data
    

class UAVFlowDataConfig(BaseDataConfig):
    video_keys = [
        "video.ego_view",
    ]
    state_keys = [
        "state.drone",
    ]
    action_keys = [
        "action.delta_pose",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def transform(self, action_norm: str = "min_max") -> ModalityTransform:
        transforms=[
            # video transforms
            VideoToTensor(apply_to=self.video_keys, backend="torchvision"),
            VideoCrop(apply_to=self.video_keys, scale=0.95, backend="torchvision"),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear", backend="torchvision" ),
            VideoColorJitter(apply_to=self.video_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08, backend="torchvision"),
            VideoToNumpy(apply_to=self.video_keys),

            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(apply_to=self.state_keys, normalization_modes={
                "state.drone": "min_max",
            }),

            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(apply_to=self.action_keys, normalization_modes={
                "action.delta_pose": "min_max",
            }),

            # ConcatTransform
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class UAVFlowZeroStateDataConfig(BaseDataConfig):
    video_keys = [
        "video.ego_view",
    ]
    state_keys = [
        "state.drone",
    ]
    action_keys = [
        "action.delta_pose",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def transform(self, action_norm: str = "min_max") -> ModalityTransform:
        transforms=[
            # video transforms
            VideoToTensor(apply_to=self.video_keys, backend="torchvision"),
            VideoCrop(apply_to=self.video_keys, scale=0.95, backend="torchvision"),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear", backend="torchvision" ),
            VideoColorJitter(apply_to=self.video_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08, backend="torchvision"),
            VideoToNumpy(apply_to=self.video_keys),

            # state transforms
            # NOTE: need to change modality.json. state.drone start=0, end=1
            ToZeroStateTransform(apply_to=self.state_keys),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(apply_to=self.state_keys, normalization_modes={
                "state.drone": "min_max",
            }),

            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(apply_to=self.action_keys, normalization_modes={
                "action.delta_pose": "min_max",
            }),

            # ConcatTransform
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)



class UAVFlowFirstLastFrameStateDataConfig(BaseDataConfig):
    video_keys = [
        "video.ego_view",
    ]
    state_keys = [
        "state.drone",
    ]
    action_keys = [
        "action.delta_pose",
    ]
    language_keys = ["annotation.human.action.task_description"]
    video_indices = [-9999, 0] # the first frame (-9999) and the current frame (0)
    observation_indices = [0]
    action_indices = list(range(16))
    
    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.video_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        return {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

    def transform(self, action_norm: str = "min_max") -> ModalityTransform:
        transforms=[
            # video transforms
            VideoToTensor(apply_to=self.video_keys, backend="torchvision"),
            VideoCrop(apply_to=self.video_keys, scale=0.95, backend="torchvision"),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear", backend="torchvision" ),
            VideoColorJitter(apply_to=self.video_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08, backend="torchvision"),
            VideoToNumpy(apply_to=self.video_keys),

            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(apply_to=self.state_keys, normalization_modes={
                "state.drone": "min_max",
            }),

            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(apply_to=self.action_keys, normalization_modes={
                "action.delta_pose": "min_max",
            }),

            # ConcatTransform
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)
