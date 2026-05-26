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


import os
from abc import ABC, abstractmethod
from typing import Optional

import torch

from pid._ext.imaginaire.utils.env_parsers.cred_env_parser import CRED_ENVS


class VideoTokenizerInterface(ABC):
    def __init__(self, s3_credential_path: Optional[str] = None):
        assert s3_credential_path is None or isinstance(s3_credential_path, str)
        if s3_credential_path is None:
            self.backend_args = None
        elif os.path.exists(s3_credential_path) or CRED_ENVS.APP_ENV in ["prod", "dev", "stg"]:
            self.backend_args = {
                "backend": "s3",
                "path_mapping": None,
                "s3_credential_path": s3_credential_path,
            }
        else:
            raise FileNotFoundError(f"Invalid s3_credential_path: {s3_credential_path} and APP_ENV is not prod/dev/stg")

    @abstractmethod
    def reset_dtype(self):
        """
        Reset the dtype of the model to the dtype its weights were trained with or quantized to.
        """
        pass

    @abstractmethod
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        pass

    @abstractmethod
    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        pass

    @property
    @abstractmethod
    def spatial_compression_factor(self):
        pass

    @property
    @abstractmethod
    def temporal_compression_factor(self):
        pass

    @property
    @abstractmethod
    def spatial_resolution(self):
        pass

    @property
    @abstractmethod
    def pixel_chunk_duration(self):
        pass

    @property
    @abstractmethod
    def latent_chunk_duration(self):
        pass

    @property
    @abstractmethod
    def latent_ch(self) -> int:
        pass

    @property
    def is_chunk_overlap(self):
        return False

    @property
    def is_causal(self):
        return True
