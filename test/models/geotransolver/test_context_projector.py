# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from physicsnemo.models.geotransolver.context_projector import (
    ContextProjector,
)
from test.common import duplicate_first_half_tokens

# =============================================================================
# ContextProjector Tests
# =============================================================================


def test_context_projector_forward(device):
    """Test ContextProjector forward pass."""
    torch.manual_seed(42)

    dim = 64
    heads = 4
    dim_head = 16
    slice_num = 8
    batch_size = 2
    n_tokens = 100

    projector = ContextProjector(
        dim=dim,
        heads=heads,
        dim_head=dim_head,
        dropout=0.0,
        slice_num=slice_num,
        use_te=False,
        plus=False,
    ).to(device)

    x = torch.randn(batch_size, n_tokens, dim).to(device)

    slice_tokens = projector(x)

    # Output shape: [Batch, Heads, Slice_num, dim_head]
    assert slice_tokens.shape == (batch_size, heads, slice_num, dim_head)
    assert not torch.isnan(slice_tokens).any()


def test_context_projector_measure_weights():
    """Measure-weighted geometry tokens are invariant to splitting a point's measure.

    Ones reproduce the unweighted tokens bitwise; duplicating the first half
    of the points with each copy carrying half its measure leaves the
    weighted tokens unchanged (fp64) while the stock pooling moves them.
    """
    torch.manual_seed(0)
    n_tokens = 16384
    projector = ContextProjector(
        dim=8, heads=2, dim_head=8, dropout=0.0, slice_num=2, use_te=False, plus=False
    ).double()
    x = torch.randn(1, n_tokens, 8, dtype=torch.float64)
    measure = torch.rand(1, n_tokens, dtype=torch.float64) + 0.5
    x_dup, measure_dup = duplicate_first_half_tokens(x, measure=measure)

    with torch.no_grad():
        tokens_u = projector(x)
        tokens_ones = projector(x, torch.ones_like(measure))
        tokens_w = projector(x, measure)
        tokens_w_dup = projector(x_dup, measure_dup)
        tokens_u_dup = projector(x_dup)

    assert torch.equal(tokens_ones, tokens_u)
    torch.testing.assert_close(tokens_w_dup, tokens_w, rtol=1e-6, atol=1e-6)
    with pytest.raises(AssertionError):  # the test has teeth
        torch.testing.assert_close(tokens_u_dup, tokens_u, rtol=1e-6, atol=1e-6)
