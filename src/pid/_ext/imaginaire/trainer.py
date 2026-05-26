# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Stub. The full Imaginaire trainer was stripped from the inference subset; the
# pixel-diffusion config still references `ImaginaireTrainer` as a type marker
# (`c.trainer.type = Trainer`) but never instantiates it.


class ImaginaireTrainer:
    """No-op type marker for c.trainer.type. Training is not supported in the inference subset."""

    pass
