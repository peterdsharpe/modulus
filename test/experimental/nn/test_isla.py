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

"""Contract tests for ISLA (Invariant Slice Attention): every guarantee is exact by construction."""

import pytest
import torch

from physicsnemo.experimental.nn.isla import ISLA


@pytest.fixture
def setup():
    torch.manual_seed(0)
    m = ISLA(hidden=64, n_layers=3, n_slices=32).double().eval()
    n = 500
    ### Anisotropic cloud: the principal-axis frame is exactly covariant
    ### only where the covariance spectrum is non-degenerate (generic for
    ### vehicle geometry; an isotropic cloud is the degenerate corner).
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n, 3, dtype=torch.float64), dim=-1
    )
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    with torch.no_grad():
        base = m(pts, nrm, drv, w)
    return m, pts, nrm, drv, w, base


def _split(o):
    return o[..., :1], o[..., 1:4]


def test_rotation_equivariance(setup):
    m, pts, nrm, drv, w, base = setup
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    with torch.no_grad():
        rot = m(pts @ q.T, nrm @ q.T, drv @ q.T, w)
    p0, v0 = _split(base)
    p1, v1 = _split(rot)
    assert torch.allclose(p1, p0, atol=1e-10)
    assert torch.allclose(v1, v0 @ q.T, atol=1e-10)


def test_translation_invariance(setup):
    m, pts, nrm, drv, w, base = setup
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=torch.float64)
    with torch.no_grad():
        tr = m(pts + shift, nrm, drv, w)
    assert torch.allclose(tr, base, atol=1e-10)


@pytest.mark.parametrize("k", [0.5, 2.0, 4.0])
def test_drive_degree_one(setup, k):
    m, pts, nrm, drv, w, base = setup
    with torch.no_grad():
        sc = m(pts, nrm, drv * k, w)
    assert torch.allclose(sc, base * k, atol=1e-10)


def test_measure_weight_scale_invariance(setup):
    m, pts, nrm, drv, w, base = setup
    with torch.no_grad():
        ws = m(pts, nrm, drv, w * 137.0)
    assert torch.allclose(ws, base, atol=1e-9)


def test_collated_input_shapes(setup):
    m, pts, nrm, drv, w, base = setup
    with torch.no_grad():
        out = m(pts, nrm, drv[:, None, :], w[..., None].squeeze(-1))
    assert torch.allclose(out, base, atol=1e-12)


@pytest.fixture
def setup_local():
    torch.manual_seed(0)
    m = (
        ISLA(
            hidden=64, n_layers=2, n_slices=16,
            use_local_features=True, local_radii=(0.5, 1.5),
        )
        .double()
        .eval()
    )
    n = 400
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n, 3, dtype=torch.float64), dim=-1
    )
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    with torch.no_grad():
        base = m(pts, nrm, drv, w)
    return m, pts, nrm, drv, w, base


def test_local_rotation_equivariance(setup_local):
    m, pts, nrm, drv, w, base = setup_local
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    with torch.no_grad():
        rot = m(pts @ q.T, nrm @ q.T, drv @ q.T, w)
    p0, v0 = _split(base)
    p1, v1 = _split(rot)
    assert torch.allclose(p1, p0, atol=1e-10)
    assert torch.allclose(v1, v0 @ q.T, atol=1e-10)


def test_local_drive_degree_one(setup_local):
    m, pts, nrm, drv, w, base = setup_local
    with torch.no_grad():
        sc = m(pts, nrm, drv * 2.0, w)
    assert torch.allclose(sc, base * 2.0, atol=1e-10)


@pytest.fixture
def setup_qi():
    torch.manual_seed(0)
    m = (
        ISLA(
            hidden=64, n_layers=2, n_slices=16,
            query_independent=True, n_decoder_layers=2,
        )
        .double()
        .eval()
    )
    n = 400
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n, 3, dtype=torch.float64), dim=-1
    )
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    return m, pts, nrm, drv, w


def test_query_set_independence(setup_qi):
    """THE v5a contract: same source, different query companions ->
    bitwise-identical predictions at shared queries."""
    m, pts, nrm, drv, w = setup_qi
    qa = pts[:, :50]
    na = nrm[:, :50]
    q_big = torch.cat([qa, pts[:, 200:300]], dim=1)
    n_big = torch.cat([na, nrm[:, 200:300]], dim=1)
    with torch.no_grad():
        out_small = m(pts, nrm, drv, w, query_points=qa, query_normals=na)
        out_big = m(pts, nrm, drv, w, query_points=q_big, query_normals=n_big)
    ### Mathematically exact; allclose(1e-12) rather than bitwise because
    ### GEMM tiling reorders reductions when the query count changes.
    assert torch.allclose(out_small, out_big[:, :50], atol=1e-12, rtol=0.0)


def test_qi_rotation_equivariance(setup_qi):
    m, pts, nrm, drv, w = setup_qi
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    with torch.no_grad():
        base = m(pts, nrm, drv, w)
        rot = m(pts @ q.T, nrm @ q.T, drv @ q.T, w)
    p0, v0 = _split(base)
    p1, v1 = _split(rot)
    assert torch.allclose(p1, p0, atol=1e-10)
    assert torch.allclose(v1, v0 @ q.T, atol=1e-10)


def test_qi_drive_degree_one(setup_qi):
    m, pts, nrm, drv, w = setup_qi
    with torch.no_grad():
        base = m(pts, nrm, drv, w)
        sc = m(pts, nrm, drv * 2.0, w)
    assert torch.allclose(sc, base * 2.0, atol=1e-10)


def test_boundary_scalar_channel_contracts():
    torch.manual_seed(0)
    m = (
        ISLA(
            hidden=64, n_layers=2, n_slices=16, n_boundary_scalars=2
        )
        .double()
        .eval()
    )
    n = 300
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n, 3, dtype=torch.float64), dim=-1
    )
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    bs = torch.randn(1, n, 2, dtype=torch.float64)
    with torch.no_grad():
        base = m(pts, nrm, drv, w, boundary_scalars=bs)
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    with torch.no_grad():
        rot = m(pts @ q.T, nrm @ q.T, drv @ q.T, w, boundary_scalars=bs)
    assert torch.allclose(rot[..., :1], base[..., :1], atol=1e-10)
    assert torch.allclose(rot[..., 1:4], base[..., 1:4] @ q.T, atol=1e-10)


def test_scale_conditioning_rotation_equivariance():
    """M1 arm: the log-size scalar breaks scale equivariance by design but
    must leave rotation equivariance and translation invariance exact."""
    torch.manual_seed(0)
    m = (
        ISLA(hidden=64, n_layers=2, n_slices=16, scale_conditioning=True)
        .double()
        .eval()
    )
    n = 400
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n, 3, dtype=torch.float64), dim=-1
    )
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=torch.float64)
    with torch.no_grad():
        base = m(pts, nrm, drv, w)
        rot = m(pts @ q.T + shift, nrm @ q.T, drv @ q.T, w)
        double = m(pts * 2.0, nrm, drv, w)
    p0, v0 = _split(base)
    p1, v1 = _split(rot)
    assert torch.allclose(p1, p0, atol=1e-10)
    assert torch.allclose(v1, v0 @ q.T, atol=1e-10)
    ### the flag must actually break scale equivariance (else it is inert)
    assert not torch.allclose(double, base, atol=1e-6)


def test_anchor_conditioned_decode_query_independence():
    """v5a4 contract: with a fixed source cloud, the interacting core runs on
    a deterministic anchor subset at eval, so predictions at shared queries
    must not depend on the companion query set."""
    torch.manual_seed(0)
    m = (
        ISLA(
            hidden=64, n_layers=2, n_slices=16,
            query_independent=True, n_decoder_layers=2, n_anchors=100,
        )
        .double()
        .eval()
    )
    n = 400
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n, 3, dtype=torch.float64), dim=-1
    )
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    qa, na = pts[:, :50], nrm[:, :50]
    q_big = torch.cat([qa, pts[:, 200:300]], dim=1)
    n_big = torch.cat([na, nrm[:, 200:300]], dim=1)
    with torch.no_grad():
        out_small = m(pts, nrm, drv, w, query_points=qa, query_normals=na)
        out_big = m(pts, nrm, drv, w, query_points=q_big, query_normals=n_big)
    assert torch.allclose(out_small, out_big[:, :50], atol=1e-12, rtol=0.0)

    ### rotation equivariance must survive the anchor subset
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    with torch.no_grad():
        base = m(pts, nrm, drv, w)
        rot = m(pts @ q.T, nrm @ q.T, drv @ q.T, w)
    p0, v0 = _split(base)
    p1, v1 = _split(rot)
    assert torch.allclose(p1, p0, atol=1e-10)
    assert torch.allclose(v1, v0 @ q.T, atol=1e-10)


def test_parity_fix_reflection_equivariance():
    """M3 audit fix: with parity_fix=True the full output must be exactly
    reflection-covariant (scalars invariant, vectors mirrored) -- the parity
    covariance of Navier-Stokes. Without the fix the e_phi pseudovector
    channels break this; the test also asserts the defect is real so the
    fix cannot silently become inert."""
    torch.manual_seed(0)
    n = 400
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n, 3, dtype=torch.float64), dim=-1
    )
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    M = torch.diag(torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64))  # mirror

    torch.manual_seed(1)
    fixed = ISLA(
        hidden=64, n_layers=2, n_slices=16, parity_fix=True, parity_gate_scale=0.1
    )
    fixed = fixed.double().eval()
    torch.manual_seed(1)
    broken = ISLA(hidden=64, n_layers=2, n_slices=16).double().eval()

    with torch.no_grad():
        base = fixed(pts, nrm, drv, w)
        mirr = fixed(pts @ M.T, nrm @ M.T, drv @ M.T, w)
        base_b = broken(pts, nrm, drv, w)
        mirr_b = broken(pts @ M.T, nrm @ M.T, drv @ M.T, w)
    p0, v0 = _split(base)
    p1, v1 = _split(mirr)
    assert torch.allclose(p1, p0, atol=1e-10)
    assert torch.allclose(v1, v0 @ M.T, atol=1e-10)
    ### the unfixed head must violate reflection covariance on vectors
    _, v0b = _split(base_b)
    _, v1b = _split(mirr_b)
    assert not torch.allclose(v1b, v0b @ M.T, atol=1e-6)


@pytest.mark.parametrize("basis", ["true5", "true7"])
def test_true_vector_basis_reflection_and_rotation(basis):
    """L2 arms: the all-true-vector bases must be exactly reflection-covariant
    (no gate) and rotation-equivariant."""
    torch.manual_seed(0)
    n = 400
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n, 3, dtype=torch.float64), dim=-1
    )
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    m = ISLA(hidden=64, n_layers=2, n_slices=16, vector_basis=basis)
    m = m.double().eval()
    M = torch.diag(torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64))
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    with torch.no_grad():
        base = m(pts, nrm, drv, w)
        mirr = m(pts @ M.T, nrm @ M.T, drv @ M.T, w)
        rot = m(pts @ q.T, nrm @ q.T, drv @ q.T, w)
    p0, v0 = _split(base)
    p1, v1 = _split(mirr)
    p2, v2 = _split(rot)
    assert torch.allclose(p1, p0, atol=1e-10) and torch.allclose(v1, v0 @ M.T, atol=1e-10)
    assert torch.allclose(p2, p0, atol=1e-10) and torch.allclose(v2, v0 @ q.T, atol=1e-10)


def test_odd_head_reflection_rotation_and_translation():
    """W2 arm: odd-coefficient head must be exactly O(3)-covariant and
    translation-invariant, and must actually use the e_phi channels."""
    torch.manual_seed(0)
    n = 400
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n, 3, dtype=torch.float64), dim=-1
    )
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    m = ISLA(hidden=64, n_layers=2, n_slices=16, odd_head=True).double().eval()
    M = torch.diag(torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64))
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=torch.float64)
    with torch.no_grad():
        base = m(pts, nrm, drv, w)
        mirr = m(pts @ M.T, nrm @ M.T, drv @ M.T, w)
        rot = m(pts @ q.T + shift, nrm @ q.T, drv @ q.T, w)
    p0, v0 = _split(base)
    p1, v1 = _split(mirr)
    p2, v2 = _split(rot)
    assert torch.allclose(p1, p0, atol=1e-10) and torch.allclose(v1, v0 @ M.T, atol=1e-10)
    assert torch.allclose(p2, p0, atol=1e-10) and torch.allclose(v2, v0 @ q.T, atol=1e-10)
    ### the odd channels must be live once the (zero-initialized) gate is
    ### non-zero, and must remain exactly reflection-covariant
    with torch.no_grad():
        m.odd_gate.weight.normal_(0.0, 0.1)
        base2 = m(pts, nrm, drv, w)
        mirr2 = m(pts @ M.T, nrm @ M.T, drv @ M.T, w)
    _, v0b = _split(base2)
    _, v1b = _split(mirr2)
    assert not torch.allclose(v0b, v0, atol=1e-6)
    assert torch.allclose(v1b, v0b @ M.T, atol=1e-10)


@pytest.mark.parametrize("kw", [{"odd_head": True}, {"parity_fix": True}, {"vector_basis": "true7"}])
def test_head_variants_run_under_bf16_autocast(kw):
    """Mixed-precision smoke: every head variant must survive bf16 autocast
    (the odd-coefficient head once failed with a dtype mismatch at step 0)."""
    torch.manual_seed(0)
    m = ISLA(hidden=32, n_layers=1, n_slices=8, **kw)
    pts = torch.randn(1, 128, 3)
    nrm = torch.nn.functional.normalize(torch.randn(1, 128, 3), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3), dim=-1)
    w = torch.rand(1, 128) + 0.5
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = m(pts, nrm, drv, w)
    out.float().sum().backward()
    assert torch.isfinite(out.float()).all()


def test_similarity_gauge_geometric_scale_equivariance():
    """S1: with the measure-weighted gauge, a geometric rescale (points x k,
    areas x k^2) must leave the output exactly unchanged; the default gauge
    must NOT (documenting that the original model is not scale-equivariant)."""
    torch.manual_seed(0)
    n = 400
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n, 3, dtype=torch.float64), dim=-1
    )
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    torch.manual_seed(1)
    mg = ISLA(hidden=64, n_layers=2, n_slices=16, similarity_gauge=True).double().eval()
    torch.manual_seed(1)
    m0 = ISLA(hidden=64, n_layers=2, n_slices=16).double().eval()
    k = 2.7
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=torch.float64)
    with torch.no_grad():
        a = mg(pts, nrm, drv, w)
        bsc = mg(k * pts + shift, nrm, drv, k * k * w)
        a0 = m0(pts, nrm, drv, w)
        b0 = m0(k * pts, nrm, drv, k * k * w)
    assert torch.allclose(bsc, a, atol=1e-10)
    assert not torch.allclose(b0, a0, atol=1e-3)


def test_raw_coord_channel_breaks_equivariance_by_design():
    """Branch-B discriminator flag: must run, must change the output, and must
    NOT be rotation-equivariant (that is the point); default stays exact."""
    torch.manual_seed(0)
    n = 300
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    nrm = torch.nn.functional.normalize(torch.randn(1, n, 3, dtype=torch.float64), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    m = ISLA(hidden=64, n_layers=2, n_slices=16, similarity_gauge=True,
                         raw_coord_channel=True).double().eval()
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    with torch.no_grad():
        base = m(pts, nrm, drv, w)
        rot = m(pts @ q.T, nrm @ q.T, drv @ q.T, w)
    p0, _ = _split(base)
    p1, _ = _split(rot)
    assert torch.isfinite(base).all()
    assert not torch.allclose(p1, p0, atol=1e-3)


def test_interior_queries_contracts():
    """V0 boundary->interior mode: off-surface queries without normals must be
    exactly rotation/translation-equivariant, query-set independent, and
    finite; the derived proxy normal must not be degenerate."""
    torch.manual_seed(0)
    n, nq = 400, 150
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    nrm = torch.nn.functional.normalize(torch.randn(1, n, 3, dtype=torch.float64), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    qpts = torch.randn(1, nq, 3, dtype=torch.float64) * 4.0  # interior/exterior points, no normals
    m = ISLA(hidden=64, n_layers=2, n_slices=16, query_independent=True,
                         n_decoder_layers=2, interior_queries=True, similarity_gauge=True,
                         out_scalars=1, out_vectors=1).double().eval()
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=torch.float64)
    with torch.no_grad():
        base = m(pts, nrm, drv, w, query_points=qpts)
        rot = m(pts @ q.T + shift, nrm @ q.T, drv @ q.T, w, query_points=qpts @ q.T + shift)
        sub = m(pts, nrm, drv, w, query_points=qpts[:, :50])
    assert torch.isfinite(base).all()
    p0, v0 = _split(base)
    p1, v1 = _split(rot)
    assert torch.allclose(p1, p0, atol=1e-10)
    assert torch.allclose(v1, v0 @ q.T, atol=1e-10)
    assert torch.allclose(sub, base[:, :50], atol=1e-12, rtol=0.0)


def test_latent_volume_tokens_contracts():
    """Branch V: latent volume tokens keep exact SE(3) covariance and query
    independence for interior queries, and are live (change the output)."""
    torch.manual_seed(0)
    n, nq = 400, 120
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    nrm = torch.nn.functional.normalize(torch.randn(1, n, 3, dtype=torch.float64), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    qpts = torch.randn(1, nq, 3, dtype=torch.float64) * 4.0
    kw = dict(hidden=64, n_layers=2, n_slices=8, query_independent=True, n_decoder_layers=2,
              interior_queries=True, similarity_gauge=True, out_scalars=1, out_vectors=1)
    torch.manual_seed(1)
    m = ISLA(latent_volume_tokens=True, **kw).double().eval()
    torch.manual_seed(1)
    m0 = ISLA(latent_volume_tokens=False, **kw).double().eval()
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=torch.float64)
    with torch.no_grad():
        base = m(pts, nrm, drv, w, query_points=qpts)
        rot = m(pts @ q.T + shift, nrm @ q.T, drv @ q.T, w, query_points=qpts @ q.T + shift)
        sub = m(pts, nrm, drv, w, query_points=qpts[:, :40])
        plain = m0(pts, nrm, drv, w, query_points=qpts)
    assert torch.isfinite(base).all()
    p0, v0 = _split(base); p1, v1 = _split(rot)
    assert torch.allclose(p1, p0, atol=1e-10) and torch.allclose(v1, v0 @ q.T, atol=1e-10)
    assert torch.allclose(sub, base[:, :40], atol=1e-12, rtol=0.0)
    assert not torch.allclose(base, plain, atol=1e-6)


def test_a35b_ablation_flags_run_and_differ():
    """A35b: raw seeds break equivariance by design; no-relational-geo stays
    exactly equivariant; both run, are finite, and change the output."""
    torch.manual_seed(0)
    n = 300
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    nrm = torch.nn.functional.normalize(torch.randn(1, n, 3, dtype=torch.float64), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    torch.manual_seed(1)
    ref = ISLA(hidden=64, n_layers=2, n_slices=16).double().eval()
    torch.manual_seed(1)
    nogeo = ISLA(hidden=64, n_layers=2, n_slices=16, use_relational_geo=False).double().eval()
    raw = ISLA(hidden=64, n_layers=2, n_slices=16, seed_mode="raw").double().eval()
    with torch.no_grad():
        o_ref = ref(pts, nrm, drv, w)
        o_ng = nogeo(pts, nrm, drv, w)
        o_ng_rot = nogeo(pts @ q.T, nrm @ q.T, drv @ q.T, w)
        o_raw = raw(pts, nrm, drv, w)
        o_raw_rot = raw(pts @ q.T, nrm @ q.T, drv @ q.T, w)
    assert torch.isfinite(o_ng).all() and torch.isfinite(o_raw).all()
    assert not torch.allclose(o_ng, o_ref, atol=1e-6)
    p0, v0 = _split(o_ng); p1, v1 = _split(o_ng_rot)
    assert torch.allclose(p1, p0, atol=1e-10) and torch.allclose(v1, v0 @ q.T, atol=1e-10)
    pr0, _ = _split(o_raw); pr1, _ = _split(o_raw_rot)
    assert not torch.allclose(pr1, pr0, atol=1e-3)


@pytest.mark.parametrize("kw", [{}, {"use_relational_geo": False}, {"seed_mode": "raw"},
                                {"odd_head": True}, {"similarity_gauge": True}])
def test_all_parameters_receive_gradients(kw):
    """DDP requires every parameter to take part in the loss; a module built
    but skipped in forward crashes distributed training (A35b nogeo incident)."""
    torch.manual_seed(0)
    m = ISLA(hidden=32, n_layers=2, n_slices=8, **kw)
    pts = torch.randn(1, 128, 3)
    nrm = torch.nn.functional.normalize(torch.randn(1, 128, 3), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3), dim=-1)
    w = torch.rand(1, 128) + 0.5
    m(pts, nrm, drv, w).sum().backward()
    unused = [n for n, p in m.named_parameters() if p.grad is None]
    assert not unused, unused


def test_geo_pool_then_project_is_exact():
    """Pooling the 8 relational invariants over slices and then projecting is
    exactly the old project-then-pool (the projection is affine and the slice
    mix is a softmax over slices, so the bias passes through unchanged). This
    is the identity behind the memory saving: the saved activation is
    (B, N, 8) instead of (B, N, S, hidden/2)."""
    from physicsnemo.experimental.nn.isla.model import _ReadBlock, _SliceBlock

    torch.manual_seed(0)
    for blk in (_SliceBlock(64, 32).double(), _ReadBlock(64, 32).double()):
        geo = torch.randn(2, 50, 32, blk.N_GEO, dtype=torch.float64)
        mix = torch.softmax(torch.randn(2, 50, 32, dtype=torch.float64), dim=-1)
        old = torch.einsum("bns,bnsg->bng", mix, blk.geo_feat(geo))
        new = blk.geo_feat(torch.einsum("bns,bnsg->bng", mix, geo))
        assert torch.allclose(old, new, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("extra", [{}, {"query_independent": True, "interior_queries": True}])
def test_geo_checkpoint_is_exact(extra):
    """Recomputing the per-slice invariants in backward (geo_checkpoint=True)
    changes what autograd stores, not what it computes: identical forward
    output and gradients (to roundoff) against the stored-activation path,
    for the encoder blocks and the passive decoder blocks."""
    torch.manual_seed(0)
    kw = dict(out_scalars=1, out_vectors=1, hidden=32, n_layers=2, n_slices=8, **extra)
    ref = ISLA(**kw).double()
    ckp = ISLA(geo_checkpoint=True, **kw).double()
    ckp.load_state_dict(ref.state_dict())
    pts = torch.randn(1, 40, 3, dtype=torch.float64)
    nrm = torch.nn.functional.normalize(torch.randn(1, 40, 3, dtype=torch.float64), dim=-1)
    drive = torch.tensor([[1.0, 0.2, 0.0]], dtype=torch.float64)
    w = torch.rand(1, 40, dtype=torch.float64) + 0.5
    fk = {}
    if extra:
        fk = dict(query_points=torch.randn(1, 17, 3, dtype=torch.float64), query_normals=torch.nn.functional.normalize(torch.randn(1, 17, 3, dtype=torch.float64), dim=-1))
    outs = []
    for m in (ref, ckp):
        out = m(pts, nrm, drive, measure_weights=w, **fk)
        out.square().sum().backward()
        outs.append(out)
    assert torch.equal(outs[0], outs[1])
    for (n, p), (_, q) in zip(ref.named_parameters(), ckp.named_parameters()):
        assert p.grad is not None and q.grad is not None, n
        assert torch.allclose(p.grad, q.grad, atol=1e-11, rtol=1e-9), n


def test_query_scalars_contracts():
    """Per-query scalar inputs on the query tokens (e.g. the signed distance
    to the wall): exact rotation/translation covariance is untouched (scalars
    are invariants), the scalar channel changes the output and receives
    gradients, geometric-scale equivariance holds under the similarity gauge
    with the "length" scaling, and misuse raises."""
    torch.manual_seed(0)
    kw = dict(out_scalars=1, out_vectors=1, hidden=32, n_layers=2, n_slices=8,
              query_tokens=True, similarity_gauge=True, n_query_scalars=1)
    m = ISLA(**kw).double()
    pts = torch.randn(1, 40, 3, dtype=torch.float64)
    nrm = torch.nn.functional.normalize(torch.randn(1, 40, 3, dtype=torch.float64), dim=-1)
    drive = torch.tensor([[1.0, 0.3, 0.0]], dtype=torch.float64)
    w = torch.rand(1, 40, dtype=torch.float64) + 0.5
    q = torch.randn(1, 12, 3, dtype=torch.float64)
    qn = torch.nn.functional.normalize(torch.randn(1, 12, 3, dtype=torch.float64), dim=-1)
    sdf = torch.randn(1, 12, dtype=torch.float64) * 0.3
    out = m(pts, nrm, drive, measure_weights=w, query_points=q, query_normals=qn, query_scalars=sdf)
    assert out.shape == (1, 12, 4)
    ### scalar channel is live
    out2 = m(pts, nrm, drive, measure_weights=w, query_points=q, query_normals=qn, query_scalars=sdf * 2)
    assert not torch.allclose(out, out2)
    ### rotation + translation covariance (scalars ride along unchanged)
    R = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))[0]
    if torch.det(R) < 0:
        R[:, 0] = -R[:, 0]
    t = torch.tensor([0.4, -1.1, 2.0], dtype=torch.float64)
    rot = m(pts @ R.T + t, nrm @ R.T, drive @ R.T, measure_weights=w, query_points=q @ R.T + t,
            query_normals=qn @ R.T, query_scalars=sdf)
    assert torch.allclose(rot[..., 0], out[..., 0], atol=1e-10)
    assert torch.allclose(rot[..., 1:], out[..., 1:] @ R.T, atol=1e-10)
    ### geometric-scale equivariance: lengths scale, so must the scalar
    s = 3.7
    sc = m(pts * s, nrm, drive, measure_weights=w * s**2, query_points=q * s, query_normals=qn,
           query_scalars=sdf * s)
    assert torch.allclose(sc, out, atol=1e-10)
    ### gradients reach the scalar embedding
    out.square().sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.qt_scalar_embed.parameters())
    with pytest.raises(ValueError):
        m(pts, nrm, drive, measure_weights=w, query_points=q, query_normals=qn)
    with pytest.raises(ValueError):
        ISLA(out_scalars=1, out_vectors=1, hidden=32, n_layers=1, n_slices=8, n_query_scalars=1)


def test_query_local_features_contracts():
    """Local surface-patch features on the query tokens: exact rotation,
    translation and (with the similarity gauge) scale covariance hold, the
    channel changes the output and receives gradients, and it requires
    query_tokens."""
    torch.manual_seed(0)
    kw = dict(out_scalars=1, out_vectors=1, hidden=32, n_layers=2, n_slices=8,
              query_tokens=True, similarity_gauge=True, n_query_scalars=1,
              query_local_features=True, query_local_radii=(0.1, 0.3))
    m = ISLA(**kw).double()
    m0 = ISLA(**{**kw, "query_local_features": False}).double()
    pts = torch.randn(1, 60, 3, dtype=torch.float64)
    nrm = torch.nn.functional.normalize(torch.randn(1, 60, 3, dtype=torch.float64), dim=-1)
    drive = torch.tensor([[1.0, 0.3, 0.0]], dtype=torch.float64)
    w = torch.rand(1, 60, dtype=torch.float64) + 0.5
    q = torch.randn(1, 12, 3, dtype=torch.float64) * 0.5
    qn = torch.nn.functional.normalize(torch.randn(1, 12, 3, dtype=torch.float64), dim=-1)
    sdf = torch.randn(1, 12, dtype=torch.float64) * 0.3
    args = dict(measure_weights=w, query_points=q, query_normals=qn, query_scalars=sdf)
    out = m(pts, nrm, drive, **args)
    assert out.shape == (1, 12, 4)
    R = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))[0]
    if torch.det(R) < 0:
        R[:, 0] = -R[:, 0]
    t = torch.tensor([0.4, -1.1, 2.0], dtype=torch.float64)
    rot = m(pts @ R.T + t, nrm @ R.T, drive @ R.T, measure_weights=w, query_points=q @ R.T + t,
            query_normals=qn @ R.T, query_scalars=sdf)
    assert torch.allclose(rot[..., 0], out[..., 0], atol=1e-10)
    assert torch.allclose(rot[..., 1:], out[..., 1:] @ R.T, atol=1e-10)
    s = 2.3
    sc = m(pts * s, nrm, drive, measure_weights=w * s**2, query_points=q * s, query_normals=qn,
           query_scalars=sdf * s)
    assert torch.allclose(sc, out, atol=1e-10)
    out.square().sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.qt_local_embed.parameters())
    ### channel is live: zeroing its embedding output recovers the no-channel model's function
    m0.load_state_dict({k: v for k, v in m.state_dict().items() if not k.startswith("qt_local_embed")})
    assert not torch.allclose(m0(pts, nrm, drive, **args), out)
    with pytest.raises(ValueError):
        ISLA(out_scalars=1, out_vectors=1, hidden=32, n_layers=1, n_slices=8, query_local_features=True)


def test_legacy_name_is_an_alias():
    """The previous name and import path keep working (cluster configs, checkpoints)."""
    from physicsnemo.experimental.nn import MeshTransformer2 as legacy_top
    from physicsnemo.experimental.nn.mt2 import MeshTransformer2 as legacy_path

    assert legacy_top is ISLA and legacy_path is ISLA


def test_query_tokens_contracts():
    """Query-token mode (interior queries as interacting tokens): exactly
    rotation/translation-covariant, finite, live (differs from the passive
    interior decode), and -- by design -- NOT query-set independent."""
    torch.manual_seed(0)
    n, nq = 400, 150
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    nrm = torch.nn.functional.normalize(torch.randn(1, n, 3, dtype=torch.float64), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    qpts = torch.randn(1, nq, 3, dtype=torch.float64) * 4.0
    qnrm = torch.nn.functional.normalize(torch.randn(1, nq, 3, dtype=torch.float64), dim=-1)
    m = ISLA(hidden=64, n_layers=2, n_slices=16, query_tokens=True, similarity_gauge=True,
             out_scalars=1, out_vectors=1).double().eval()
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=torch.float64)
    with torch.no_grad():
        base = m(pts, nrm, drv, w, query_points=qpts, query_normals=qnrm)
        rot = m(pts @ q.T + shift, nrm @ q.T, drv @ q.T, w, query_points=qpts @ q.T + shift, query_normals=qnrm @ q.T)
        sub = m(pts, nrm, drv, w, query_points=qpts[:, :50], query_normals=qnrm[:, :50])
    assert base.shape == (1, nq, 4) and torch.isfinite(base).all()
    p0, v0 = _split(base)
    p1, v1 = _split(rot)
    assert torch.allclose(p1, p0, atol=1e-10)
    assert torch.allclose(v1, v0 @ q.T, atol=1e-10)
    # interacting mode: predictions at shared queries depend on the query set
    assert not torch.allclose(sub, base[:, :50], atol=1e-6)
    # every parameter receives a gradient (DDP safety)
    m.train()
    out = m(pts, nrm, drv, w, query_points=qpts, query_normals=qnrm)
    out.square().mean().backward()
    missing = [k for k, p in m.named_parameters() if p.grad is None]
    assert not missing, missing
    with pytest.raises(ValueError):
        ISLA(hidden=32, n_layers=1, n_slices=8, query_tokens=True, query_independent=True)


@pytest.mark.parametrize("extra", [{}, {"n_query_scalars": 1, "query_scalar_scale": "length"}])
def test_query_mass_source_total_refinement_invariance(extra):
    """Audit 2026-09-08: splitting every source token into two half-weight
    copies leaves the discrete source measure unchanged (positions, normals,
    total area, every integral). With query_mass="source_total" the query
    tokens' weight is a fraction of the total source measure, so the output
    is exactly unchanged and the similarity-gauge scale contract still holds;
    the default "geometric_mean" is the trained convention and is asserted to
    keep its (refinement-dependent) formula so checkpoints reproduce."""
    torch.manual_seed(42)
    pts = torch.randn(1, 60, 3, dtype=torch.float64) * torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    nrm = torch.nn.functional.normalize(torch.randn_like(pts), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, 60, dtype=torch.float64) + 0.5
    q, qn = pts[:, :17], nrm[:, :17]
    fk = dict(query_points=q, query_normals=qn)
    if extra:
        fk["query_scalars"] = q.norm(dim=-1)
    kw = dict(hidden=32, n_layers=2, n_slices=8, similarity_gauge=True, query_tokens=True, **extra)
    torch.manual_seed(0)
    total = ISLA(query_mass="source_total", **kw).double().eval()
    torch.manual_seed(0)
    default = ISLA(**kw).double().eval()
    refined = (pts.repeat_interleave(2, 1), nrm.repeat_interleave(2, 1), drv, w.repeat_interleave(2, 1) / 2)
    s = 3.7
    scaled_fk = {**fk, "query_points": q * s}
    if extra:
        scaled_fk["query_scalars"] = fk["query_scalars"] * s
    with torch.no_grad():
        a = total(pts, nrm, drv, w, **fk)
        b = total(*refined, **fk)
        c = total(pts * s, nrm, drv, w * s**2, **scaled_fk)
        a0 = default(pts, nrm, drv, w, **fk)
        b0 = default(*refined, **fk)
    assert torch.allclose(b, a, atol=1e-10, rtol=0.0)
    assert torch.allclose(c, a, atol=1e-10, rtol=0.0)
    ### regression guard: the default convention is unchanged (and therefore
    ### still refinement-dependent); the two conventions are not the same model
    assert not torch.allclose(b0, a0, atol=1e-3)
    assert not torch.allclose(a0, a, atol=1e-6)
    with pytest.raises(ValueError):
        ISLA(hidden=32, n_layers=1, n_slices=8, query_mass="source_total")
    with pytest.raises(ValueError):
        ISLA(hidden=32, n_layers=1, n_slices=8, query_tokens=True, query_mass="mean")


@pytest.mark.parametrize("extra", [{}, {"query_independent": True, "n_decoder_layers": 2}])
def test_similarity_gauge_local_features_scale_equivariance(extra):
    """Audit 2026-09-08: under the similarity gauge the surface patch
    integrals normalize the measure weights to fractions of the total, so a
    geometric rescale (points x k, areas x k^2) leaves log(mass) -- and the
    output -- exactly unchanged, for the encoder seeds and the passive
    decoder's query-side patch integrals alike."""
    torch.manual_seed(0)
    n = 400
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    nrm = torch.nn.functional.normalize(torch.randn(1, n, 3, dtype=torch.float64), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    m = ISLA(hidden=64, n_layers=2, n_slices=16, similarity_gauge=True,
             use_local_features=True, local_radii=(0.5, 1.5), **extra).double().eval()
    k = 2.7
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=torch.float64)
    fk = dict(query_points=pts[:, :50], query_normals=nrm[:, :50]) if extra else {}
    fk_sc = {**fk, "query_points": k * fk["query_points"] + shift} if extra else {}
    with torch.no_grad():
        a = m(pts, nrm, drv, w, **fk)
        b = m(k * pts + shift, nrm, drv, k * k * w, **fk_sc)
    assert torch.allclose(b, a, atol=1e-10, rtol=0.0)


@pytest.mark.parametrize("kw", [{"scale_conditioning": True}, {"seed_mode": "raw"},
                                {"raw_coord_channel": True}, {"odd_head": True},
                                {"odd_head": True, "n_anchors": 30}])
def test_passive_decode_seed_and_head_options(kw):
    """Audit 2026-09-08: passive decoding at a query set whose size differs
    from the source used to fail with a shape error for every seed-widening
    option (the query seed lacked the extra channels) and for the odd head
    (it read the source normal and count after they were overwritten by the
    query-side values). Now: finite outputs of the right shape, and the
    passive contract -- a prediction at one query does not depend on which
    other points are queried -- holds to 1e-12."""
    torch.manual_seed(0)
    pts = torch.randn(1, 60, 3, dtype=torch.float64) * torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    nrm = torch.nn.functional.normalize(torch.randn_like(pts), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, 60, dtype=torch.float64) + 0.5
    q = torch.randn(1, 17, 3, dtype=torch.float64) * 2.0
    qn = torch.nn.functional.normalize(torch.randn_like(q), dim=-1)
    m = ISLA(hidden=32, n_layers=2, n_slices=8, query_independent=True, n_decoder_layers=2, **kw)
    m = m.double().eval()
    if kw.get("odd_head"):
        with torch.no_grad():
            m.odd_gate.weight.normal_(0.0, 0.1)  # make the odd channels live
    with torch.no_grad():
        out = m(pts, nrm, drv, w, query_points=q, query_normals=qn)
        sub = m(pts, nrm, drv, w, query_points=q[:, :8], query_normals=qn[:, :8])
    assert out.shape == (1, 17, 4) and torch.isfinite(out).all()
    assert torch.allclose(sub, out[:, :8], atol=1e-12, rtol=0.0)


def test_passive_decode_boundary_scalars():
    """Audit 2026-09-08: boundary scalars are per-surface-point data. Passive
    decoding of the surface itself (query_points=None) carries them into the
    query seeds and runs; distinct query points have none and must raise
    rather than fail with a shape error."""
    torch.manual_seed(0)
    pts = torch.randn(1, 60, 3, dtype=torch.float64) * torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    nrm = torch.nn.functional.normalize(torch.randn_like(pts), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, 60, dtype=torch.float64) + 0.5
    bs = torch.randn(1, 60, 2, dtype=torch.float64)
    m = ISLA(hidden=32, n_layers=2, n_slices=8, query_independent=True, n_decoder_layers=2,
             n_boundary_scalars=2).double().eval()
    with torch.no_grad():
        out = m(pts, nrm, drv, w, boundary_scalars=bs)
        out2 = m(pts, nrm, drv, w, boundary_scalars=bs * 2)
    assert out.shape == (1, 60, 4) and torch.isfinite(out).all()
    assert not torch.allclose(out, out2, atol=1e-6)  # the channel is live on the query side
    with pytest.raises(ValueError):
        m(pts, nrm, drv, w, boundary_scalars=bs, query_points=pts[:, :17], query_normals=nrm[:, :17])
