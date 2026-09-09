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

"""Tests for the transfer program's campaign-C hooks: the constant condition
field and weight-only initialization from another run's checkpoint."""

import torch
from domain_transforms import SetConstantCellField
from tensordict import TensorDict
from utils import initialize_from_checkpoint

from physicsnemo.experimental.nn.isla import ISLA
from physicsnemo.mesh import Mesh
from physicsnemo.utils.checkpoint import save_checkpoint


def _patch() -> Mesh:
    pts = torch.tensor([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    cells = torch.tensor([[0, 1, 2], [0, 2, 3]])
    return Mesh(points=pts, cells=cells, global_data=TensorDict({}, batch_size=[]))


def test_constant_cell_field_shape_value_and_dtype():
    m = SetConstantCellField(field_name="cond", value=1.0)(_patch())
    f = m.cell_data["cond"]
    assert f.shape == (2, 1) and f.dtype == m.points.dtype
    assert torch.equal(f, torch.ones(2, 1))
    m0 = SetConstantCellField(field_name="cond", value=0.0)(_patch())
    assert torch.equal(m0.cell_data["cond"], torch.zeros(2, 1))


def test_constant_cell_field_survives_cell_slicing():
    m = SetConstantCellField(field_name="cond", value=1.0)(_patch()).slice_cells(torch.tensor([1]))
    assert torch.equal(m.cell_data["cond"], torch.ones(1, 1))


def test_initialize_from_checkpoint_loads_weights_only(tmp_path):
    torch.manual_seed(0)
    src = ISLA(hidden=32, n_layers=2, n_slices=8)
    save_checkpoint(str(tmp_path), models=src, epoch=7)
    saved = sorted(tmp_path.glob("*.mdlus"))
    assert saved, "save_checkpoint wrote no .mdlus file"
    torch.manual_seed(1)
    dst = ISLA(hidden=32, n_layers=2, n_slices=8)
    before = {k: v.clone() for k, v in dst.state_dict().items()}
    assert any(not torch.equal(before[k], v) for k, v in src.state_dict().items())
    report = initialize_from_checkpoint(dst, str(tmp_path))
    assert report["epoch"] == 7
    for k, v in src.state_dict().items():
        assert torch.equal(dst.state_dict()[k], v)


def test_initialize_from_checkpoint_rejects_architecture_mismatch(tmp_path):
    src = ISLA(hidden=32, n_layers=2, n_slices=8)
    save_checkpoint(str(tmp_path), models=src, epoch=1)
    other = ISLA(hidden=48, n_layers=2, n_slices=8)
    try:
        initialize_from_checkpoint(other, str(tmp_path))
    except Exception:
        return
    raise AssertionError("mismatched architecture loaded without error")
