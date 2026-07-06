# Supported Models


OpenDDE currently exposes one public model:

| Model name | MSA / Constraint / RNA MSA / Template | Model Parameters (M) | Data cutoff |
| --- | :---: | ---: | :---: |
| `opendde_v1` | ✓ / × / ✓ / ✓ | 656 | 2021-09-30 |

Exact parameter count: `655,791,538`, rounded to `656 M`.
`opendde_v1` uses the defaults in `opendde/config/model_base.py`.

Use it with:

```bash
opendde pred -i examples/input.json -o ./output -n opendde_v1
```

Checkpoint path by default:

```text
$OPENDDE_ROOT_DIR/checkpoint/opendde.pt
```

## Released Checkpoints

| Checkpoint | Use case | Download |
| --- | --- | --- |
| `opendde.pt` | General-purpose OpenDDE checkpoint. | [opendde.pt](https://huggingface.co/aurekaresearch/OpenDDE/resolve/main/opendde.pt) |
| `opendde_abag.pt` | ABAG-optimized checkpoint for antibody-antigen complexes. | [opendde_abag.pt](https://huggingface.co/aurekaresearch/OpenDDE/resolve/main/opendde_abag.pt) |

`opendde.pt` is the default checkpoint for `-n opendde_v1`. Keep the
ABAG-optimized checkpoint as `opendde_abag.pt` and pass it explicitly with
`--load_checkpoint_path`.

Recommended inference defaults:

- `model.N_cycle = 10`
- `sample_diffusion.N_step = 200`
- triangle kernels: `auto`

These are also the current `opendde pred` CLI defaults for `opendde_v1`.

Legacy `constraint` fields are ignored by the inference-only build. Use
`covalent_bonds` for supported covalent links.
