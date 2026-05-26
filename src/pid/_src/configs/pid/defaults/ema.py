# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import attrs
from hydra.core.config_store import ConfigStore


@attrs.define(slots=False)
class EMAConfig:
    enabled: bool = True
    rate: float = 0.1
    iteration_shift: int = 0


PowerEMAConfig: EMAConfig = EMAConfig(
    enabled=True,
    rate=0.10,
    iteration_shift=0,
)


def register_ema():
    cs = ConfigStore.instance()
    cs.store(group="ema", package="model.config.ema", name="power", node=PowerEMAConfig)
