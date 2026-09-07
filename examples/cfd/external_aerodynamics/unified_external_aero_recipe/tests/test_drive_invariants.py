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

"""Tests for ComputeDriveInvariants (the INV transplant feature)."""

import torch
from domain_transforms import ComputeDriveInvariants, ComputeFreestreamDirection
from tensordict import TensorDict

from physicsnemo.datapipes.transforms.mesh import ComputeSurfaceNormals
from physicsnemo.mesh import Mesh


def _patch(rotation: torch.Tensor | None = None) -> Mesh:
    pts = torch.tensor(
        [[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0]]
    )
    cells = torch.tensor([[0, 1, 2], [0, 2, 3], [1, 4, 5], [1, 5, 2]])
    u = torch.tensor([30.0, 0.0, 30.0])
    if rotation is not None:
        pts = pts @ rotation.T
        u = u @ rotation.T
    m = Mesh(points=pts, cells=cells, global_data=TensorDict({"U_inf": u}, batch_size=[]))
    m = ComputeFreestreamDirection()(m)
    m = ComputeSurfaceNormals(store_as="cell_data", field_name="normals")(m)
    return ComputeDriveInvariants(reference_length=8.0)(m)


def test_values_on_flat_patch():
    f = _patch().cell_data["drive_inv"]
    assert f.shape == (4, 3)
    # +z face normals against a 45-degree drive: |n.d| = 1/sqrt(2)
    assert torch.allclose(f[:, 0].abs(), torch.full((4,), 2**-0.5), atol=1e-6)
    # radius is |centroid - mean centroid| / L, non-negative and bounded here
    assert (f[:, 2] >= 0).all() and (f[:, 2] < 1.0).all()


def test_rotation_invariance():
    torch.manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(q) < 0:
        q[:, 0] *= -1
    a = _patch().cell_data["drive_inv"]
    b = _patch(q).cell_data["drive_inv"]
    assert torch.allclose(a, b, atol=1e-5)


def test_requires_normals_and_direction():
    pts = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    m = Mesh(points=pts, cells=torch.tensor([[0, 1, 2]]),
             global_data=TensorDict({"U_inf": torch.tensor([1.0, 0, 0])}, batch_size=[]))
    try:
        ComputeDriveInvariants()(m)
    except KeyError as e:
        assert "normals" in str(e)
    else:
        raise AssertionError("expected KeyError without normals")
