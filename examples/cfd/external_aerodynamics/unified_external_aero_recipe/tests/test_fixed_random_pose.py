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

"""FixedRandomPose (POSE-BENCH): a per-case deterministic SO(3) pose."""

import torch
from domain_transforms import FixedRandomPose, _pose_rotation_for_key
from merge_global_data import MeshReaderWithGlobalData
from tensordict import TensorDict

from physicsnemo.mesh import Mesh


def _mesh(key: int = 12345) -> Mesh:
    torch.manual_seed(0)
    pts = torch.randn(40, 3, dtype=torch.float64)
    cells = torch.stack([torch.arange(0, 38), torch.arange(1, 39), torch.arange(2, 40)], dim=1)
    return Mesh(
        points=pts,
        cells=cells,
        cell_data=TensorDict({"wss": torch.randn(38, 3, dtype=torch.float64), "pressure": torch.randn(38, dtype=torch.float64)}, batch_size=[38]),
        global_data=TensorDict(
            {"U_inf": torch.tensor([30.0, 0.0, 0.0], dtype=torch.float64), "p_inf": torch.tensor(0.5, dtype=torch.float64),
             "case_key": torch.tensor(key, dtype=torch.int64)},
            batch_size=[],
        ),
    )


def test_rotation_is_deterministic_in_key_and_salt():
    R1 = _pose_rotation_for_key(7, 0)
    assert torch.allclose(R1, _pose_rotation_for_key(7, 0))
    assert not torch.allclose(R1, _pose_rotation_for_key(8, 0))
    assert not torch.allclose(R1, _pose_rotation_for_key(7, 1))
    assert torch.allclose(R1 @ R1.T, torch.eye(3, dtype=torch.float64), atol=1e-12)
    assert torch.det(R1) > 0


def test_pose_rotates_every_vector_field_together_and_is_reproducible():
    m = _mesh()
    t = FixedRandomPose(salt=0)
    a = t(m)
    b = t(_mesh())
    R = _pose_rotation_for_key(12345, 0)
    assert torch.allclose(a.points, m.points @ R.T, atol=1e-12)
    assert torch.allclose(a.cell_data["wss"], m.cell_data["wss"] @ R.T, atol=1e-12)
    assert torch.allclose(a.cell_data["pressure"], m.cell_data["pressure"])
    assert torch.allclose(a.global_data["U_inf"], m.global_data["U_inf"] @ R.T, atol=1e-12)
    assert torch.allclose(a.global_data["p_inf"], m.global_data["p_inf"])
    assert int(a.global_data["case_key"]) == 12345
    assert torch.allclose(a.points, b.points) and torch.allclose(a.global_data["U_inf"], b.global_data["U_inf"])


def test_reader_stores_case_key_from_case_directory(tmp_path):
    for name in ("run_1", "run_2"):
        d = tmp_path / name / "domain.pdmsh" / "_tensordict" / "boundaries"
        d.mkdir(parents=True)
        Mesh(points=torch.randn(6, 3), cells=torch.tensor([[0, 1, 2], [3, 4, 5]])).save(d / "vehicle")
        TensorDict({"U_inf": torch.tensor([30.0, 0.0, 0.0])}, batch_size=[]).save(tmp_path / name / "domain.pdmsh" / "_tensordict" / "global_data")
    r = MeshReaderWithGlobalData(tmp_path, pattern="run_*/*.pdmsh/_tensordict/boundaries/vehicle",
                                 merge_global_data_from="../../global_data", store_case_key=True)
    k0, k1 = int(r._load_sample(0).global_data["case_key"]), int(r._load_sample(1).global_data["case_key"])
    assert k0 != k1 and k0 >= 0 and k1 >= 0
    assert int(r._load_sample(0).global_data["case_key"]) == k0  # stable across loads
    assert torch.allclose(r._load_sample(0).global_data["U_inf"], torch.tensor([30.0, 0.0, 0.0]))
