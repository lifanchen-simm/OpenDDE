# Inference Instructions


Concise reference for installing OpenDDE, preparing runtime data, and running
`opendde` commands. For Docker, see
[docker_installation.md](./docker_installation.md).

## Install

OpenDDE requires Python `>=3.11`. Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

GPU install:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install --torch-backend cu126 'opendde[gpu]'
opendde doctor
```

CPU install:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install --torch-backend cpu 'opendde[cpu]'
opendde doctor
```

Source install from a checkout:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install --torch-backend cu126 -e '.[gpu]'
opendde doctor
```

## Runtime data

Set `OPENDDE_ROOT_DIR` to the directory that stores checkpoints and runtime data:

```text
$OPENDDE_ROOT_DIR/
├── checkpoint/opendde.pt
├── common/
└── search_database/        # needed for local template/RNA-MSA search
```

Prepare data from a source checkout:

```bash
export OPENDDE_ROOT_DIR=/path/to/opendde_data
bash scripts/download_opendde_data.sh
```

For a protein-only prediction that disables MSA, template, and RNA-MSA features,
search databases are not needed:

```bash
bash scripts/download_opendde_data.sh --skip-search-database
```

If you already have a checkpoint:

```bash
mkdir -p "$OPENDDE_ROOT_DIR/checkpoint"
cp /path/to/opendde.pt "$OPENDDE_ROOT_DIR/checkpoint/opendde.pt"
```

Alternatively, keep the checkpoint anywhere readable and pass it directly with
`opendde pred --load_checkpoint_path /path/to/opendde_abag.pt`.

Released checkpoints:

| Checkpoint | Use case | Download |
| --- | --- | --- |
| `opendde.pt` | General-purpose OpenDDE checkpoint. | [opendde.pt](https://huggingface.co/aurekaresearch/OpenDDE/resolve/main/opendde.pt) |
| `opendde_abag.pt` | ABAG-optimized checkpoint for antibody-antigen complexes. | [opendde_abag.pt](https://huggingface.co/aurekaresearch/OpenDDE/resolve/main/opendde_abag.pt) |

Use `opendde.pt` with `-n opendde_v1` as the default general-purpose
checkpoint. To use the ABAG-optimized checkpoint, keep it as
`opendde_abag.pt` and pass it with `--load_checkpoint_path`, for example
`opendde pred --load_checkpoint_path "$OPENDDE_ROOT_DIR/checkpoint/opendde_abag.pt"`.

Concrete setup:

```bash
export OPENDDE_ROOT_DIR=/path/to/opendde_data
mkdir -p "$OPENDDE_ROOT_DIR/checkpoint"

# General-purpose checkpoint used by default with -n opendde_v1.
curl -L \
  -o "$OPENDDE_ROOT_DIR/checkpoint/opendde.pt" \
  https://huggingface.co/aurekaresearch/OpenDDE/resolve/main/opendde.pt

# ABAG-optimized checkpoint, selected with --load_checkpoint_path.
curl -L \
  -o "$OPENDDE_ROOT_DIR/checkpoint/opendde_abag.pt" \
  https://huggingface.co/aurekaresearch/OpenDDE/resolve/main/opendde_abag.pt
```

Then run general-purpose inference without an explicit checkpoint path. For ABAG
inference, add:

```bash
--load_checkpoint_path "$OPENDDE_ROOT_DIR/checkpoint/opendde_abag.pt"
```

Useful environment variables:

| Variable | Purpose |
| --- | --- |
| `OPENDDE_ROOT_DIR` | Checkpoints, common files, search databases. Defaults to `~/.cache/opendde`. |
| `OPENDDE_DEPENDENCY_URL` | Override checkpoint download root. |
| `OPENDDE_COMMON_URL` | Override common runtime file download root. Falls back to `OPENDDE_DEPENDENCY_URL` when set. |
| `OPENDDE_SEARCH_DATABASE_URL` | Override template/RNA-MSA database download root. |
| `LAYERNORM_TYPE` | LayerNorm backend; defaults to `torch`. Set to `fast_layernorm` to opt into the fused kernel. |

Template/RNA-MSA preprocessing also needs HMMER. Template inference may need
`kalign`:

```bash
apt-get update && apt-get install -y hmmer kalign
```

## Input JSON

OpenDDE input is a top-level list of jobs:

```json
[
  {
    "name": "tiny",
    "modelSeeds": [101],
    "sequences": [
      {
        "proteinChain": {
          "sequence": "ACDEFGHIK",
          "count": 1
        }
      }
    ]
  }
]
```

`covalent_bonds` is optional and may be omitted from a job; include it only to
declare explicit covalent links between entities.

Full schema: [infer_json_format.md](./infer_json_format.md).

Convert a structure file to JSON:

```bash
opendde json -i examples/7pzb.pdb -o ./output --altloc first
opendde json -i examples/2lwu.cif -o ./output --altloc first --assembly_id 1
```

## Preprocess optional features

```bash
# Protein MSA
opendde msa -i examples/input.json -o ./output

# Protein MSA + template search
opendde mt -i examples/input.json -o ./output

# Protein MSA + template search + RNA MSA when RNA is present
opendde prep -i examples/input.json -o ./output
```

Notes:

- Protein MSA uses the public ColabFold MMseqs2 API unless A3M paths are already
  present in the JSON.
- Template and RNA-MSA search use local databases under
  `$OPENDDE_ROOT_DIR/search_database/`.
- Updated JSON files are written next to the input JSON.

Details: [msa_template_pipeline.md](./msa_template_pipeline.md).

## Run prediction

Standard run:

```bash
opendde pred -i examples/input.json -o ./output -n opendde_v1
```

Compatibility run with the standard step/cycle counts:

```bash
opendde pred \
  -i examples/input.json \
  -o ./output \
  -n opendde_v1 \
  --use_msa false \
  --use_template false \
  --use_rna_msa false \
  --sample 1 \
  --step 200 \
  --cycle 10
```

Inference defaults to `fp32` and `auto` triangle kernels (PyTorch on CPU,
cuEquivariance on a CUDA GPU), so neither needs to be set explicitly.


## 4-GPU Fold-CP inference

> Note: the current Fold-CP path is provided as a distributed-inference demo.
> It verifies that OpenDDE can execute the four-GPU context-parallel path,
> including MSA-enabled inputs, but memory capacity and runtime performance are
> still being actively optimized. We plan to continue improving this path in
> collaboration with NVIDIA, including integration and tuning for acceleration
> libraries such as cuEquivariance where applicable.

Fold-CP distributes token-pair-heavy inference work over four GPUs. Launch it
with `torchrun` and expose exactly the GPUs you want to use:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node 4 \
  -m runner.batch_inference pred \
  -i examples/protein_200.json \
  -o ./output_cp4 \
  -n opendde_v1 \
  --use_msa false \
  --use_template false \
  --use_rna_msa false \
  --sample 1 \
  --step 200 \
  --cycle 10 \
  --foldcp_mode distributed \
  --foldcp_size_dp 1 \
  --foldcp_size_cp 4
```

Runtime notes:

- `--nproc_per_node 4` must match `--foldcp_size_dp 1` times
  `--foldcp_size_cp 4`.
- `--foldcp_size_cp 4` creates a 2 x 2 context-parallel mesh.
- The same input, model, dtype, cycle, step, sample, MSA, template, and kernel
  settings should be used when comparing single-GPU and Fold-CP outputs.
- Outputs are written under the requested `-o/--out_dir` just like normal
  inference.
- Optional `--foldcp_metrics_jsonl path/to/metrics.jsonl` records Fold-CP timing
  and memory metrics.

For single-GPU inference, omit the Fold-CP flags or set
`--foldcp_mode single --foldcp_size_cp 1`.

Use prepared features:

```bash
opendde pred -i examples/examples_with_template/example_9fm7.json \
  -o ./output -n opendde_v1 \
  --use_msa true --use_template true

opendde pred -i examples/examples_with_rna_msa/example_9gmw_2.json \
  -o ./output -n opendde_v1 \
  --use_rna_msa true
```

## Optional TFG Guidance

OpenDDE includes default-off Training-Free Guidance (TFG) for protein-ligand
runs. TFG refines each sampled trajectory with geometry potentials while keeping
the requested `--sample` count unchanged.

```bash
opendde pred -i examples/input.json -o ./output -n opendde_v1 \
  --use_tfg_guidance true
```

Outputs are written to:

```text
<out_dir>/<job_name>/seed_<seed>/predictions/
```

## Common flags

| Flag | Meaning |
| --- | --- |
| `-n`, `--model_name` | Model name. Currently `opendde_v1`. |
| `--load_checkpoint_path` | Explicit checkpoint path. |
| `--seeds` | Comma-separated seeds, e.g. `101,102`. Overrides the job's `modelSeeds`; if unset, `modelSeeds` are used, or a random seed when both are absent. |
| `--use_msa` | Use/generate protein MSA features. |
| `--use_template` | Use/generate template features. |
| `--use_rna_msa` | Use/generate RNA MSA features. |
| `--use_tfg_guidance` | Enable Training-Free Guidance. |
| `--foldcp_mode` | `single` or `distributed`; use `distributed` with `torchrun` for four-GPU Fold-CP inference. |
| `--foldcp_size_cp` | Number of context-parallel ranks. Four-GPU Fold-CP uses `4`. |
| `--foldcp_metrics_jsonl` | Optional JSONL path for Fold-CP timing and memory metrics. |
| `--dtype` | `bf16`, `fp16`, or `fp32`. |
| `--trimul_kernel`, `--triatt_kernel` | `auto`, `cuequivariance`, or `torch`. |

Run `opendde <command> --help` for the full option list.
