# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Stub. The Imaginaire trainer callbacks were stripped from the inference subset;
# `imaginaire/config.py` still references EMAModelCallback / ProgressBarCallback /
# WandBCallback in its default `trainer.callbacks` LazyDict, but inference never
# composes that field. These stubs keep the import path alive.


class EMAModelCallback:
    pass


class ProgressBarCallback:
    pass


class WandBCallback:
    pass


class Callback:
    pass


class CallBackGroup:
    pass
