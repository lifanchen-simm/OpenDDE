# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import pytest

from opendde.config.inference import (
    build_inference_config,
    update_gpu_compatible_configs,
    validate_triangle_kernels,
)
from opendde.config.model_base import configs as configs_base

def test_build_inference_config_applies_model_specific_defaults():
    cfg = build_inference_config(fill_required_with_null=True)

    assert cfg.model_name == "opendde_v1"
    assert cfg.c_z == 384
    assert cfg.no_bins == 96
    assert cfg.model.N_cycle == 10
    assert cfg.model.msa_module.c_m == 128
    assert cfg.model.template_embedder.n_blocks == 2
    assert cfg.sample_diffusion.N_step == 200
    assert cfg.confidence.distogram.no_bins == 96

def test_build_inference_config_keeps_cli_overrides_highest_priority():
    cfg = build_inference_config(
        arg_str=(
            "--model_name opendde_v1 "
            "--model.N_cycle 3 "
            "--sample_diffusion.N_step 7 "
            "--triangle_attention torch"
        ),
        fill_required_with_null=True,
    )

    assert cfg.model.N_cycle == 3
    assert cfg.sample_diffusion.N_step == 7
    assert cfg.triangle_attention == "torch"
    assert cfg.c_z == 384

def test_build_inference_config_does_not_mutate_base_defaults():
    build_inference_config(fill_required_with_null=True)

    assert configs_base["c_z"] == 384
    assert configs_base["model"]["N_cycle"] == 10
    assert configs_base["model"]["msa_module"]["c_m"] == 128
    assert configs_base["model"]["template_embedder"]["n_blocks"] == 2
    assert configs_base["confidence"]["distogram"]["min_bin"] == 2.25
    assert configs_base["confidence"]["distogram"]["max_bin"] == 25.75
    assert configs_base["confidence"]["distogram"]["no_bins"] == 96

def test_get_default_runner_config_build_does_not_mutate_base_defaults(monkeypatch):
    from runner import batch_inference

    class DummyRunner:
        def __init__(self, cfg):
            self.configs = cfg

    monkeypatch.setattr(batch_inference, "download_inference_cache", lambda cfg: None)
    monkeypatch.setattr(batch_inference, "InferenceRunner", DummyRunner)

    runner = batch_inference.get_default_runner(
        seeds=[101],
        n_cycle=2,
        n_step=3,
        n_sample=1,
        dtype="fp32",
        use_msa=False,
        trimul_kernel="torch",
        triatt_kernel="torch",
        enable_cache=False,
        enable_fusion=False,
    )

    assert runner.configs.c_z == 384
    assert runner.configs.model.N_cycle == 2
    assert runner.configs.sample_diffusion.N_step == 3
    assert configs_base["c_z"] == 384
    assert configs_base["model"]["N_cycle"] == 10
    assert configs_base["model"]["msa_module"]["c_m"] == 128
    assert configs_base["model"]["template_embedder"]["n_blocks"] == 2
    assert configs_base["confidence"]["distogram"]["no_bins"] == 96

def test_runner_inference_run_config_build_does_not_mutate_base_defaults(monkeypatch):
    from runner import inference

    captured_configs = []

    monkeypatch.setattr(inference, "parse_sys_args", lambda: "")
    monkeypatch.setattr(inference, "download_inference_cache", lambda cfg: None)
    monkeypatch.setattr(inference, "main", lambda cfg: captured_configs.append(cfg))

    inference.run()

    assert captured_configs[0].c_z == 384
    assert configs_base["c_z"] == 384
    assert configs_base["model"]["N_cycle"] == 10
    assert configs_base["model"]["msa_module"]["c_m"] == 128
    assert configs_base["model"]["template_embedder"]["n_blocks"] == 2
    assert configs_base["confidence"]["distogram"]["no_bins"] == 96

def test_validate_triangle_kernels_rejects_unknown_values():
    validate_triangle_kernels("auto", "cuequivariance")
    validate_triangle_kernels("torch", "cuequivariance")
    with pytest.raises(ValueError):
        validate_triangle_kernels("unsupported", "torch")
    with pytest.raises(ValueError):
        validate_triangle_kernels("torch", "unsupported")

def test_auto_triangle_kernels_resolve_to_detected_runtime(monkeypatch):
    import opendde.config.inference as inference_config

    cfg = build_inference_config(
        arg_str="--triangle_attention auto --triangle_multiplicative auto",
        fill_required_with_null=True,
    )

    monkeypatch.setattr(inference_config, "select_triangle_kernel", lambda: "torch")

    cfg = update_gpu_compatible_configs(cfg)

    assert cfg.triangle_attention == "torch"
    assert cfg.triangle_multiplicative == "torch"


def test_get_default_runner_allows_tfg_guidance_with_foldcp(monkeypatch):
    from runner import batch_inference

    class DummyRunner:
        def __init__(self, cfg):
            self.configs = cfg

    monkeypatch.setattr(batch_inference, "download_inference_cache", lambda cfg: None)
    monkeypatch.setattr(batch_inference, "InferenceRunner", DummyRunner)
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    runner = batch_inference.get_default_runner(
        seeds=[101],
        n_cycle=1,
        n_step=1,
        n_sample=1,
        use_tfg_guidance=True,
        foldcp_mode="distributed",
        foldcp_size_dp=1,
        foldcp_size_cp=4,
        foldcp_devices="0,1,2,3",
        foldcp_metrics_jsonl="metrics.jsonl",
    )

    assert runner.configs.sample_diffusion.guidance["enable"] is True
    assert runner.configs.foldcp_mode == "distributed"
    assert runner.configs.foldcp_size_dp == 1
    assert runner.configs.foldcp_size_cp == 4
    assert runner.configs.foldcp_devices == "0,1,2,3"
    assert runner.configs.foldcp_metrics_jsonl == "metrics.jsonl"
    assert runner.configs.foldcp["enabled"] is True
    assert runner.configs.foldcp["cp_mesh_shape"] == (2, 2)
