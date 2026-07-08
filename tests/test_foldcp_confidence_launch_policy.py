from __future__ import annotations

from types import SimpleNamespace

import torch

from opendde.distributed.foldcp import confidence
from opendde.distributed.foldcp import real_pairformer
from opendde.distributed.foldcp.pair_sharding import FoldCPPairShardSpec
from opendde.model.modules import diffusion as diffusion_module
from opendde.model.modules.confidence import ConfidenceHead


def test_confidence_logits_use_shared_row_slab_launch_policy(monkeypatch):
    z_pair_local = torch.zeros(2, 2, 3)
    z_row_slab = torch.arange(4 * 4 * 3, dtype=torch.float32).reshape(4, 4, 3)
    spec = FoldCPPairShardSpec(
        original_shape=(4, 4, 3),
        padded_shape=(4, 4, 3),
        pair_dims=(0, 1),
        row_range=(0, 2),
        col_range=(0, 2),
        mesh_shape=(2, 2),
        mesh_coord=(0, 0),
    )
    mesh = SimpleNamespace(coord=(0, 0))
    linear = torch.nn.Linear(3, 2, bias=False)

    monkeypatch.setattr(
        confidence,
        "_collect_pair_row_slab",
        lambda _z_pair_local, _mesh: z_row_slab,
    )

    def forbidden_source_grid(*_args, **_kwargs):
        raise AssertionError("confidence logits must use the shared launch policy")

    monkeypatch.setattr(
        confidence,
        "_linear_pair_row_slab_with_source_grid_launch",
        forbidden_source_grid,
    )

    calls = []

    def fake_launch_policy(linear_module, x, **kwargs):
        calls.append(kwargs)
        return linear_module(x)

    monkeypatch.setattr(
        confidence,
        "foldcp_pair_row_slab_linear_with_source_launch_policy",
        fake_launch_policy,
        raising=False,
    )

    logits = confidence._confidence_pair_logits_local_rowslab(
        z_pair_local=z_pair_local,
        z_pair_spec=spec,
        mesh=mesh,
        layer_norm=torch.nn.Identity(),
        linear=linear,
    )

    assert logits.shape == (2, 2, 2)
    assert calls == [
        {
            "original_n": 4,
            "row_start": 0,
            "col_start": 0,
            "valid_rows": 2,
            "valid_cols": 4,
        }
    ]


def test_confidence_large_projection_bypasses_source_grid_launch_policy(monkeypatch):
    z_pair_local = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    spec = FoldCPPairShardSpec(
        original_shape=(2049, 2049, 4),
        padded_shape=(2049, 2049, 4),
        pair_dims=(0, 1),
        row_range=(0, 2),
        col_range=(0, 3),
        mesh_shape=(2, 2),
        mesh_coord=(0, 0),
    )
    mesh = SimpleNamespace(coord=(0, 0))
    linear = torch.nn.Linear(4, 2, bias=False)

    monkeypatch.setattr(
        confidence,
        "_collect_pair_row_slab",
        lambda _z_pair_local, _mesh: z_pair_local,
    )

    def forbidden_source_grid(*_args, **_kwargs):
        raise AssertionError("large confidence projection must avoid source-grid launch")

    monkeypatch.setattr(
        confidence,
        "foldcp_pair_row_slab_linear_with_source_launch_policy",
        forbidden_source_grid,
        raising=False,
    )

    monkeypatch.setenv("OPENDDE_FOLDCP_CONFIDENCE_FLAT_CHUNK", "4")
    calls = []

    def fake_source_launch_shape(linear_module, x, *, source_rows):
        calls.append((tuple(x.shape), source_rows))
        return linear_module(x)

    monkeypatch.setattr(
        confidence,
        "foldcp_linear_with_source_launch_shape",
        fake_source_launch_shape,
        raising=False,
    )

    logits = confidence._confidence_pair_logits_local_rowslab(
        z_pair_local=z_pair_local,
        z_pair_spec=spec,
        mesh=mesh,
        layer_norm=torch.nn.Identity(),
        linear=linear,
    )

    assert logits.shape == (2, 3, 2)
    assert calls == [((4, 4), 2049 * 2049), ((2, 4), 2049 * 2049)]


def test_confidence_stream_to_rank0_projects_row_chunks(monkeypatch):
    z_pair_local = torch.zeros(3, 2, 4)
    z_row_slab = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(3, 4, 4)
    spec = FoldCPPairShardSpec(
        original_shape=(4, 4, 4),
        padded_shape=(4, 4, 4),
        pair_dims=(0, 1),
        row_range=(0, 3),
        col_range=(0, 2),
        mesh_shape=(1, 1),
        mesh_coord=(0, 0),
    )
    mesh = SimpleNamespace(coord=(0, 0), group_2d=object())
    linear = torch.nn.Linear(4, 2, bias=False)

    monkeypatch.setenv("OPENDDE_FOLDCP_CONFIDENCE_ROW_CHUNK", "2")
    monkeypatch.setattr(
        confidence,
        "_collect_pair_row_slab",
        lambda _z_pair_local, _mesh: z_row_slab,
    )
    monkeypatch.setattr(confidence.dist, "get_rank", lambda _group: 0)

    def forbidden_full_local_logits(*_args, **_kwargs):
        raise AssertionError("streaming path must not build full local logits first")

    monkeypatch.setattr(
        confidence,
        "_confidence_pair_logits_local_rowslab",
        forbidden_full_local_logits,
    )

    launch_calls = []

    def fake_launch_policy(linear_module, x, **kwargs):
        launch_calls.append((tuple(x.shape), kwargs))
        return linear_module(x)

    monkeypatch.setattr(
        confidence,
        "foldcp_pair_row_slab_linear_with_source_launch_policy",
        fake_launch_policy,
        raising=False,
    )

    gathered_shapes = []

    def fake_gather(**kwargs):
        gathered_shapes.append(tuple(kwargs["local_chunk"].shape))

    monkeypatch.setattr(confidence, "_gather_pair_logit_chunk_to_rank0", fake_gather)

    full_output = confidence._stream_pair_logits_to_rank0(
        z_pair_local=z_pair_local,
        z_pair_spec=spec,
        mesh=mesh,
        layer_norm=torch.nn.Identity(),
        linear=linear,
    )

    assert full_output is not None
    assert full_output.shape == (4, 4, 2)
    assert gathered_shapes == [(2, 2, 2), (1, 2, 2)]
    assert [shape for shape, _kwargs in launch_calls] == [(2, 4, 4), (1, 4, 4)]



def test_foldcp_confidence_non_output_rank_processes_every_sample(monkeypatch):
    head = ConfidenceHead.__new__(ConfidenceHead)
    torch.nn.Module.__init__(head)
    head.input_strunk_ln = torch.nn.Identity()
    head.pairformer_stack = SimpleNamespace(blocks=[SimpleNamespace(c_s=1)])
    mesh = SimpleNamespace()
    spec = FoldCPPairShardSpec(
        original_shape=(4, 4, 1),
        padded_shape=(4, 4, 1),
        pair_dims=(0, 1),
        row_range=(0, 2),
        col_range=(0, 2),
        mesh_shape=(2, 2),
        mesh_coord=(0, 0),
    )

    monkeypatch.setattr(head, "_maybe_create_foldcp_mesh", lambda: mesh)
    monkeypatch.setattr(head, "_foldcp_is_non_output_rank", lambda: True)
    monkeypatch.setattr(
        head,
        "_build_confidence_z_init_local",
        lambda **kwargs: torch.zeros_like(kwargs["reference"]),
    )
    monkeypatch.setattr(
        head,
        "_select_distogram_rep_atom_mask",
        lambda input_feature_dict, n_token: torch.ones(n_token, dtype=torch.bool),
    )

    calls = []

    def fake_foldcp_local(**kwargs):
        calls.append(tuple(kwargs["x_pred_rep_coords"].shape))
        return None, None, None, None

    monkeypatch.setattr(head, "memory_efficient_forward_foldcp_local", fake_foldcp_local)

    result = head.forward(
        input_feature_dict={},
        s_inputs=torch.zeros(4, 1),
        s_trunk=torch.zeros(4, 1),
        z_trunk=torch.zeros(2, 2, 1),
        pair_mask=torch.ones(4, 4),
        x_pred_coords=torch.zeros(3, 4, 3),
        z_trunk_spec=spec,
    )

    assert result == (None, None, None, None)
    assert calls == [(4, 3), (4, 3), (4, 3)]


def test_diffusion_cache_projection_uses_shared_row_slab_launch_policy(monkeypatch):
    module = diffusion_module.DiffusionConditioning(
        c_z=3,
        c_z_pair_diffusion=2,
        c_s=4,
        c_s_inputs=2,
        c_noise_embedding=2,
    )
    z_trunk = torch.arange(4 * 4 * 3, dtype=torch.float32).reshape(4, 4, 3)
    pair_z = torch.arange(4 * 4 * 4, dtype=torch.float32).reshape(4, 4, 4)

    def forbidden_source_grid(*_args, **_kwargs):
        raise AssertionError("diffusion cache must use the shared launch policy")

    monkeypatch.setattr(
        diffusion_module.DiffusionConditioning,
        "_linear_pair_row_slab_source_grid_launch",
        staticmethod(forbidden_source_grid),
    )

    calls = []

    def fake_launch_policy(linear_module, x, **kwargs):
        calls.append(kwargs)
        return linear_module(x)

    monkeypatch.setattr(
        diffusion_module,
        "foldcp_pair_row_slab_linear_with_source_launch_policy",
        fake_launch_policy,
        raising=False,
    )

    projected = module._project_z_trunk_source_launch(
        z_trunk,
        source_rows=16,
        original_n=4,
        row_start=0,
        valid_rows=4,
    )

    pair_projected = module._project_pair_z_source_launch(
        pair_z,
        source_rows=16,
        original_n=4,
        row_start=0,
        valid_rows=4,
    )

    assert projected.shape == (4, 4, 2)
    assert pair_projected.shape == (4, 4, 2)
    assert calls == [
        {
            "original_n": 4,
            "row_start": 0,
            "col_start": 0,
            "valid_rows": 4,
            "valid_cols": 4,
        },
        {
            "original_n": 4,
            "row_start": 0,
            "col_start": 0,
            "valid_rows": 4,
            "valid_cols": 4,
        },
    ]



def test_diffusion_cache_large_projection_bypasses_source_grid_launch_policy(monkeypatch):
    module = diffusion_module.DiffusionConditioning(
        c_z=4,
        c_z_pair_diffusion=2,
        c_s=4,
        c_s_inputs=2,
        c_noise_embedding=2,
    )
    pair_z = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)

    def forbidden_source_grid(*_args, **_kwargs):
        raise AssertionError("large diffusion cache projection must avoid source-grid launch")

    monkeypatch.setattr(
        diffusion_module,
        "foldcp_pair_row_slab_linear_with_source_launch_policy",
        forbidden_source_grid,
        raising=False,
    )

    calls = []

    def fake_source_launch_shape(linear_module, x, *, source_rows):
        calls.append((tuple(x.shape), source_rows))
        return linear_module(x)

    monkeypatch.setattr(
        diffusion_module,
        "foldcp_linear_with_source_launch_shape",
        fake_source_launch_shape,
        raising=False,
    )

    original_n = 2049
    projected = module._project_pair_z_source_launch(
        pair_z,
        source_rows=original_n * original_n,
        original_n=original_n,
        row_start=0,
        valid_rows=2,
    )

    assert projected.shape == (2, 3, 2)
    assert calls == [((2, 3, 4), original_n * original_n)]


def test_diffusion_cache_foldcp_local_uses_local_tile_without_row_slab(monkeypatch):
    module = diffusion_module.DiffusionConditioning(
        c_z=3,
        c_z_pair_diffusion=2,
        c_s=4,
        c_s_inputs=2,
        c_noise_embedding=2,
    )
    spec = FoldCPPairShardSpec(
        original_shape=(5, 5, 3),
        padded_shape=(6, 6, 3),
        pair_dims=(0, 1),
        row_range=(0, 3),
        col_range=(3, 6),
        mesh_shape=(2, 2),
        mesh_coord=(0, 1),
    )
    z_trunk_local = torch.arange(3 * 3 * 3, dtype=torch.float32).reshape(3, 3, 3)
    relp = torch.arange(5 * 5 * 139, dtype=torch.float32).reshape(5, 5, 139) / 1000.0

    def forbidden_row_slab(*_args, **_kwargs):
        raise AssertionError("diffusion cache must not gather a full row slab")

    monkeypatch.setattr(
        diffusion_module.DiffusionConditioning,
        "_collect_pair_row_slab",
        staticmethod(forbidden_row_slab),
    )

    launch_calls = []

    def fake_launch_policy(linear_module, x, **kwargs):
        launch_calls.append((tuple(x.shape), kwargs))
        return linear_module(x)

    monkeypatch.setattr(
        diffusion_module,
        "foldcp_pair_row_slab_linear_with_source_launch_policy",
        fake_launch_policy,
        raising=False,
    )

    pair_z_local, pair_spec = module.prepare_cache_foldcp_local(
        relp_feature=relp,
        z_trunk_local=z_trunk_local,
        z_spec=spec,
        mesh=SimpleNamespace(coord=(0, 1)),
    )

    assert pair_z_local.shape == (3, 3, 2)
    assert pair_spec.original_shape == (5, 5, 2)
    assert [kwargs for _shape, kwargs in launch_calls[:2]] == [
        {
            "original_n": 5,
            "row_start": 0,
            "col_start": 3,
            "valid_rows": 3,
            "valid_cols": 2,
        },
        {
            "original_n": 5,
            "row_start": 0,
            "col_start": 3,
            "valid_rows": 3,
            "valid_cols": 2,
        },
    ]



def _fill_module_deterministic(module):
    with torch.no_grad():
        for index, param in enumerate(module.parameters()):
            values = torch.arange(param.numel(), dtype=torch.float32).reshape(param.shape)
            values = (values.remainder(17 + index) - (8 + index % 3)) / (37.0 + index)
            param.copy_(values.to(dtype=param.dtype))


def test_diffusion_cache_foldcp_local_tiles_equal_serial_cache():
    torch.manual_seed(7)
    module = diffusion_module.DiffusionConditioning(
        c_z=3,
        c_z_pair_diffusion=2,
        c_s=4,
        c_s_inputs=2,
        c_noise_embedding=2,
    )
    _fill_module_deterministic(module)
    n_token = 5
    padded_n = 6
    tile = 3
    z_trunk = torch.arange(n_token * n_token * 3, dtype=torch.float32).reshape(
        n_token, n_token, 3
    ) / 19.0
    relp = torch.arange(n_token * n_token * 139, dtype=torch.float32).reshape(
        n_token, n_token, 139
    ) / 257.0

    serial = module.prepare_cache(relp, z_trunk)
    gathered = torch.zeros_like(serial)
    for row_coord in range(2):
        for col_coord in range(2):
            row_start = row_coord * tile
            col_start = col_coord * tile
            row_end = row_start + tile
            col_end = col_start + tile
            spec = FoldCPPairShardSpec(
                original_shape=(n_token, n_token, 3),
                padded_shape=(padded_n, padded_n, 3),
                pair_dims=(0, 1),
                row_range=(row_start, row_end),
                col_range=(col_start, col_end),
                mesh_shape=(2, 2),
                mesh_coord=(row_coord, col_coord),
            )
            z_local = torch.zeros(tile, tile, 3, dtype=z_trunk.dtype)
            valid_row_end = min(row_end, n_token)
            valid_col_end = min(col_end, n_token)
            valid_rows = max(0, valid_row_end - row_start)
            valid_cols = max(0, valid_col_end - col_start)
            if valid_rows and valid_cols:
                z_local[:valid_rows, :valid_cols] = z_trunk[
                    row_start:valid_row_end,
                    col_start:valid_col_end,
                ]
            local, _local_spec = module.prepare_cache_foldcp_local(
                relp_feature=relp,
                z_trunk_local=z_local,
                z_spec=spec,
                mesh=SimpleNamespace(coord=(row_coord, col_coord)),
            )
            if valid_rows and valid_cols:
                gathered[
                    row_start:valid_row_end,
                    col_start:valid_col_end,
                ] = local[:valid_rows, :valid_cols]

    torch.testing.assert_close(gathered, serial, atol=1e-6, rtol=0)



def test_real_pairformer_tile_source_launch_uses_shared_policy(monkeypatch):
    linear = torch.nn.Linear(3, 2, bias=False)
    x = torch.arange(4 * 5 * 3, dtype=torch.float32).reshape(4, 5, 3)
    calls = []

    def fake_launch_policy(linear_module, launch_x, **kwargs):
        calls.append(kwargs)
        return linear_module(launch_x)

    monkeypatch.setattr(
        real_pairformer,
        "foldcp_pair_row_slab_linear_with_source_launch_policy",
        fake_launch_policy,
        raising=False,
    )

    out = real_pairformer._linear_pair_tile_with_source_grid_launch(
        linear,
        x,
        original_n=8,
        row_start=2,
        col_start=1,
        valid_rows=4,
        valid_cols=5,
    )

    assert out.shape == (4, 5, 2)
    assert calls == [
        {
            "original_n": 8,
            "row_start": 2,
            "col_start": 1,
            "valid_rows": 4,
            "valid_cols": 5,
        }
    ]


def test_starting_triangle_bias_collects_only_target_source_tile():
    local = torch.full((2, 3, 4), 1.0)
    remote = torch.full((2, 3, 4), 7.0)

    class FakeColComm:
        def __init__(self):
            self.calls = 0

        def exchange(self, value):
            self.calls += 1
            assert tuple(value.shape) == tuple(local.shape)
            return remote

    comm = FakeColComm()
    mesh = SimpleNamespace(
        coord=(0, 1),
        layout=SimpleNamespace(shape=(2, 2)),
        ring_comm=lambda: SimpleNamespace(comm_col=comm),
    )

    selected = real_pairformer._collect_column_source_tile(
        local,
        mesh,
        target_source_row=1,
    )

    assert comm.calls == 1
    assert selected.shape == local.shape
    assert torch.equal(selected, remote)


def test_pair_transition_source_flat_chunks_bound_launch_rows():
    calls = []

    class AddOne(torch.nn.Module):
        def forward(self, x):
            calls.append(tuple(x.shape))
            return x + 1

    x = torch.arange(3 * 5 * 2, dtype=torch.float32).reshape(3, 5, 2)

    out = real_pairformer._apply_pair_transition_source_flat_chunks(
        AddOne(),
        x,
        original_n=5,
        row_start=1,
        valid_rows=3,
        flat_chunk_size=4,
    )

    assert torch.equal(out, x + 1)
    assert calls
    assert max(shape[0] for shape in calls) <= 4


def test_pair_transition_source_flat_chunks_preserve_source_offsets():
    class AddLaunchOffset(torch.nn.Module):
        def forward(self, x):
            offset = torch.arange(x.shape[0], dtype=x.dtype).unsqueeze(-1)



def test_pair_transition_update_uses_source_flat_chunks(monkeypatch):
    calls = []

    class AddOne(torch.nn.Module):
        def forward(self, x):
            return x + 1

    def fake_gather_by_row(z_local, mesh, dim, length=None):
        assert dim == -2
        assert length == 8
        return z_local

    def fake_apply(transition, x_row_slab, **kwargs):
        calls.append(kwargs)
        assert x_row_slab.shape == (4, 8, 2)
        return torch.ones(4, 8, 2)

    monkeypatch.setattr(real_pairformer, "_ring_gather_by_row", fake_gather_by_row)
    monkeypatch.setattr(
        real_pairformer,
        "_apply_pair_transition_source_flat_chunks",
        fake_apply,
    )
    z_local = torch.zeros(4, 8, 2)
    z_spec = SimpleNamespace(
        original_shape=(8, 8, 2),
        pair_dims=(0, 1),
        row_range=(4, 8),
        col_range=(2, 6),
    )

    with torch.no_grad():
        out = real_pairformer.distributed_pair_transition_update(
            AddOne(),
            z_local,
            mesh=object(),
            z_spec=z_spec,
        )

    assert out.shape == z_local.shape
    assert torch.equal(out[:4, :4], torch.ones(4, 4, 2))
    assert calls == [
        {
            "original_n": 8,
            "row_start": 4,
            "valid_rows": 4,
            "flat_chunk_size": 262144,
        }
    ]


def test_pair_transition_residual_path_updates_local_tile_in_place(monkeypatch):
    class AddOne(torch.nn.Module):
        def forward(self, x):
            return x + 1

    def fake_gather_by_row(z_local, mesh, dim, length=None):
        assert dim == -2
        assert length == 8
        return z_local

    def fake_apply(transition, x_row_slab, **kwargs):
        assert x_row_slab.shape == (4, 8, 2)
        return torch.ones(4, 8, 2)

    monkeypatch.setattr(real_pairformer, "_ring_gather_by_row", fake_gather_by_row)
    monkeypatch.setattr(
        real_pairformer,
        "_apply_pair_transition_source_flat_chunks",
        fake_apply,
    )
    z_local = torch.zeros(4, 8, 2)
    z_spec = SimpleNamespace(
        original_shape=(8, 8, 2),
        pair_dims=(0, 1),
        row_range=(4, 8),
        col_range=(2, 6),
    )

    with torch.no_grad():
        out = real_pairformer.distributed_pair_transition_update(
            AddOne(),
            z_local,
            mesh=object(),
            residual_local=z_local,
            z_spec=z_spec,
        )

    assert out.data_ptr() == z_local.data_ptr()
    assert torch.equal(out[:4, :4], torch.ones(4, 4, 2))
    assert torch.equal(out[:4, 4:], torch.zeros(4, 4, 2))



def test_triangle_attention_row_chunk_keeps_small_residue_default(monkeypatch):
    monkeypatch.delenv("OPENDDE_FOLDCP_TRIATT_ATTENTION_MAX_ELEMENTS", raising=False)

    row_chunk = real_pairformer._triatt_attention_row_chunk_size(
        valid_rows=512,
        original_n=1000,
        valid_query=1000,
    )

    assert row_chunk == 256


def test_triangle_attention_row_chunk_caps_large_residue_workspace(monkeypatch):
    monkeypatch.delenv("OPENDDE_FOLDCP_TRIATT_ATTENTION_MAX_ELEMENTS", raising=False)

    row_chunk = real_pairformer._triatt_attention_row_chunk_size(
        valid_rows=1000,
        original_n=2000,
        valid_query=1000,
    )

    assert row_chunk == 32


def test_triangle_attention_row_chunk_caps_large_structural_workspace(monkeypatch):
    monkeypatch.delenv("OPENDDE_FOLDCP_TRIATT_ATTENTION_MAX_ELEMENTS", raising=False)

    row_chunk = real_pairformer._triatt_attention_row_chunk_size(
        valid_rows=1900,
        original_n=3799,
        valid_query=1900,
    )

    assert row_chunk == 8


def test_triangle_attention_row_chunk_env_override(monkeypatch):
    monkeypatch.setenv("OPENDDE_FOLDCP_TRIATT_ATTENTION_MAX_ELEMENTS", "1000")

    row_chunk = real_pairformer._triatt_attention_row_chunk_size(
        valid_rows=32,
        original_n=100,
        valid_query=100,
    )

    assert row_chunk == 1


def test_triangle_source_column_chunk_scales_with_global_k(monkeypatch):
    monkeypatch.delenv("OPENDDE_FOLDCP_TRIANGLE_SOURCE_COLUMN_MAX_ELEMENTS", raising=False)
    monkeypatch.delenv("OPENDDE_FOLDCP_TRIANGLE_SOURCE_COLUMN_CHUNK", raising=False)

    assert real_pairformer._triangle_source_column_chunk_size(1000) == 256
    assert real_pairformer._triangle_source_column_chunk_size(2000) == 128
    assert real_pairformer._triangle_source_column_chunk_size(3799) == 64
    assert real_pairformer._triangle_source_column_chunk_size(5500) == 32


def test_triangle_source_column_chunk_env_override(monkeypatch):
    monkeypatch.setenv("OPENDDE_FOLDCP_TRIANGLE_SOURCE_COLUMN_CHUNK", "96")

    assert real_pairformer._triangle_source_column_chunk_size(5500) == 96


def test_triangle_multiplication_auto_channel_chunk_large_tiles(monkeypatch):
    monkeypatch.delenv("OPENDDE_FOLDCP_TRIMUL_PROJECT_CHANNEL_CHUNK", raising=False)
    monkeypatch.delenv("OPENDDE_FOLDCP_TRIMUL_CHANNEL_CHUNK", raising=False)
    module = SimpleNamespace(c_hidden=128)
    z_spec = SimpleNamespace(
        original_shape=(4096, 4096, 128),
        pair_dims=(0, 1),
        row_range=(2048, 4096),
        col_range=(2048, 4096),
    )

    assert real_pairformer._trimul_project_channel_chunk_size(module, z_spec) == 32


def test_triangle_multiplication_auto_channel_chunk_keeps_small_tiles(monkeypatch):
    monkeypatch.delenv("OPENDDE_FOLDCP_TRIMUL_PROJECT_CHANNEL_CHUNK", raising=False)
    monkeypatch.delenv("OPENDDE_FOLDCP_TRIMUL_CHANNEL_CHUNK", raising=False)
    module = SimpleNamespace(c_hidden=128)
    z_spec = SimpleNamespace(
        original_shape=(512, 512, 128),
        pair_dims=(0, 1),
        row_range=(0, 256),
        col_range=(0, 256),
    )

    assert real_pairformer._trimul_project_channel_chunk_size(module, z_spec) == 0


def test_triangle_multiplication_channel_chunk_env_override(monkeypatch):
    monkeypatch.setenv("OPENDDE_FOLDCP_TRIMUL_PROJECT_CHANNEL_CHUNK", "16")
    module = SimpleNamespace(c_hidden=128)
    z_spec = SimpleNamespace(
        original_shape=(4096, 4096, 128),
        pair_dims=(0, 1),
        row_range=(2048, 4096),
        col_range=(2048, 4096),
    )

    assert real_pairformer._trimul_project_channel_chunk_size(module, z_spec) == 16



def test_triangle_layer_norm_source_row_slab_keeps_local_tile_shape():
    calls = []

    class AddOne(torch.nn.Module):
        def forward(self, x):
            calls.append(tuple(x.shape))
            return x + 1

    z = torch.zeros(4, 4, 2)
    z_spec = SimpleNamespace(
        original_shape=(8, 8, 2),
        pair_dims=(0, 1),
        row_range=(0, 4),
        col_range=(2, 6),
    )

    out = real_pairformer._triangle_layer_norm_source_row_slab(
        AddOne(),
        z,
        mesh=object(),
        z_spec=z_spec,
    )

    assert calls == [(4, 4, 2)]
    assert torch.equal(out, torch.ones_like(z))
