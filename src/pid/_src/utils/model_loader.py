# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import os

import torch
import torch.distributed.checkpoint as dcp

from pid._ext.imaginaire.checkpointer.dcp import DefaultLoadPlanner, DistributedCheckpointer, ModelWrapper
from pid._ext.imaginaire.lazy_config import instantiate
from pid._ext.imaginaire.utils import log, misc
from pid._ext.imaginaire.utils.config_helper import get_config_module, override
from pid._ext.imaginaire.utils.easy_io import easy_io


def load_model_from_checkpoint(
    experiment_name,
    checkpoint_path,
    config_file="pid/_src/configs/pid/config.py",
    enable_fsdp=False,
    instantiate_ema=True,
    load_ema_to_reg=False,
    seed=0,
    experiment_opts: list[str] = [],
    strict=True,
):
    config_module = get_config_module(config_file)
    config = importlib.import_module(config_module).make_config()
    config = override(config, ["--", f"experiment={experiment_name}"] + experiment_opts)

    if instantiate_ema is False and hasattr(config.model.config, "ema") and config.model.config.ema.enabled:
        config.model.config.ema.enabled = False

    config.validate()
    config.freeze()  # type: ignore
    misc.set_random_seed(seed=seed, by_rank=True)
    torch.backends.cudnn.deterministic = config.trainer.cudnn.deterministic
    torch.backends.cudnn.benchmark = config.trainer.cudnn.benchmark
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True

    if not enable_fsdp and hasattr(config.model.config, "fsdp_shard_size"):
        config.model.config.fsdp_shard_size = 1

    with misc.timer("instantiate model"):
        model = instantiate(config.model).cuda()
        model.on_train_start()

    if checkpoint_path.endswith(".pth"):
        log.info(f"Loading model from consolidated checkpoint {checkpoint_path}")
        model.load_state_dict(easy_io.load(checkpoint_path), strict=strict)
    else:
        log.info(f"Loading model from dcp checkpoint {checkpoint_path}")
        checkpointer = DistributedCheckpointer(config.checkpoint, config.job, callbacks=None, disable_async=True)
        cur_key_ckpt_full_path = os.path.join(checkpoint_path, "model")
        storage_reader = checkpointer.get_storage_reader(cur_key_ckpt_full_path)

        _model_wrapper = ModelWrapper(model, load_ema_to_reg=load_ema_to_reg)
        _state_dict = _model_wrapper.state_dict()
        dcp.load(
            _state_dict,
            storage_reader=storage_reader,
            planner=DefaultLoadPlanner(allow_partial_load=True),
        )
        _model_wrapper.load_state_dict(_state_dict)

    if not enable_fsdp:
        model = model.to(dtype=model.precision)

    torch.cuda.empty_cache()

    return model, config
