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

"""FRAME-FULL: per-case full-surface frame injected from a table and consumed by
ISLA's supplied-frame mode through the ``isla_surface_frame`` template."""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import hydra
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict

from physicsnemo.datapipes.transforms.mesh import TARGET_QUADRATURE_MEASURE_KEY
from physicsnemo.mesh import Mesh

from collate import build_collate_fn
from conftest import make_surface_domain_mesh
from domain_transforms import SetGlobalFieldsFromTable

_RECIPE_ROOT = Path(__file__).resolve().parent.parent
_TABLE = {
    "run_1": {"frame_center": [0.1, -0.2, 0.3], "frame_scale": 0.55, "n_cells": 17},
    "run_10": {"frame_center": [1.0, 2.0, 3.0], "frame_scale": 0.7, "n_cells": 19},
    "_provenance": {"script": "compute_frame_table.py"},
}


def _crc(name: str) -> int:
    return zlib.crc32(name.encode()) & 0x7FFFFFFF


def _mesh(case_key: int | None, dtype=torch.float32) -> Mesh:
    gd = {"U_inf": torch.tensor([30.0, 0.0, 0.0], dtype=dtype)}
    if case_key is not None:
        gd["case_key"] = torch.tensor(case_key, dtype=torch.int64)
    return Mesh(
        points=torch.randn(12, 3, dtype=dtype),
        cells=torch.stack([torch.arange(0, 10), torch.arange(1, 11), torch.arange(2, 12)], dim=1),
        global_data=TensorDict(gd, batch_size=[]),
    )


def test_table_fields_are_injected_by_case_key(tmp_path: Path) -> None:
    table = tmp_path / "frame_table.json"
    table.write_text(json.dumps(_TABLE))
    t = SetGlobalFieldsFromTable(table=str(table), fields=["frame_center", "frame_scale"])
    for name, row in _TABLE.items():
        if name.startswith("_"):
            continue
        out = t(_mesh(_crc(name)))
        assert torch.allclose(out.global_data["frame_center"], torch.tensor(row["frame_center"]))
        assert out.global_data["frame_center"].dtype == torch.float32
        assert out.global_data["frame_scale"].shape == () and float(out.global_data["frame_scale"]) == pytest.approx(row["frame_scale"])
        assert "n_cells" not in out.global_data.keys()  # only the requested fields
        assert "U_inf" in out.global_data.keys()  # existing fields kept
    with pytest.raises(KeyError):
        t(_mesh(_crc("run_999")))
    with pytest.raises(KeyError):
        t(_mesh(None))
    assert "2 cases" in t.extra_repr()


def _compose(model: str, dataset: str, overrides: list[str]):
    ds_cfg = OmegaConf.load(_RECIPE_ROOT / "datasets" / f"{dataset}.yaml")
    targets = OmegaConf.to_container(ds_cfg.targets, resolve=True)
    with initialize_config_dir(config_dir=str(_RECIPE_ROOT / "conf"), version_base=None):
        cfg = compose(config_name="train", overrides=[f"model={model}", f"dataset={dataset}", "+out_dim=4", *overrides])
    return cfg, targets


@pytest.mark.parametrize("overrides", [[], ["+model.scale_mode=global"]])
def test_isla_surface_frame_template_reads_the_frame_from_global_data(overrides: list[str]) -> None:
    """The template's forward_kwargs deliver global_data.frame_center / frame_scale
    to ISLA(center_mode=global); arm B adds scale_mode=global. Shifting the cloud
    and the supplied center together leaves the prediction unchanged."""
    cfg, targets = _compose("isla_surface_frame", "drivaer_ml_surface", overrides)
    assert cfg.model.center_mode == "global"
    domain = make_surface_domain_mesh(targets, n_cells=80)
    domain.interior.point_data[TARGET_QUADRATURE_MEASURE_KEY] = torch.rand(80) + 0.5
    domain.global_data["U_inf_dir"] = torch.tensor([1.0, 0.0, 0.0])
    domain.global_data["frame_center"] = torch.tensor([0.3, -0.1, 0.2])
    domain.global_data["frame_scale"] = torch.tensor(1.7)
    collate = build_collate_fn(
        input_type="tensors",
        forward_kwargs_spec=OmegaConf.to_container(cfg.forward_kwargs, resolve=True),
        target_config=targets,
    )
    batch = collate([(domain, {})])
    ### collated global leaves carry a batch dim (the model reshapes to (B, 3) / (B,))
    assert batch["forward_kwargs"]["frame_center"].numel() == 3
    assert batch["forward_kwargs"]["frame_scale"].numel() == 1
    small = OmegaConf.merge(cfg.model, {"hidden": 32, "n_layers": 2, "n_slices": 8, "geo_checkpoint": False})
    torch.manual_seed(0)
    model = hydra.utils.instantiate(small, _convert_="partial").eval()
    with torch.no_grad():
        out = model(**batch["forward_kwargs"])
        kw = dict(batch["forward_kwargs"])
        shift = torch.tensor([2.0, -3.0, 1.0])
        kw["points"] = kw["points"] + shift
        kw["frame_center"] = kw["frame_center"] + shift
        moved = model(**kw)
    assert out.shape == (1, 80, 4)
    assert torch.allclose(moved, out, atol=1e-5)
