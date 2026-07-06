# Docker Installation


Use Docker for GPU inference on a Linux host with an NVIDIA GPU. All examples
below are one-shot `docker run` commands executed from the host. For non-Docker
installation, see [inference_instructions.md](./inference_instructions.md).

## 1. Verify Docker GPU support

Install Docker and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html),
then verify that containers can see the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

## 2. Get the image

Pull the prebuilt image:

```bash
docker pull aurekaresearch/opendde:v1
```

Or build from the repository root:

```bash
docker build -t aurekaresearch/opendde:v1 .
```

## 3. Prepare runtime data

OpenDDE reads checkpoints and runtime data from `OPENDDE_ROOT_DIR`. If you have
local checkpoints, place `opendde.pt` and/or `opendde_abag.pt` under
`checkpoint/` in that directory:

```bash
export OPENDDE_ROOT_DIR="$PWD/opendde_data"

mkdir -p "$OPENDDE_ROOT_DIR/checkpoint"
cp /absolute/path/to/opendde.pt "$OPENDDE_ROOT_DIR/checkpoint/opendde.pt"
```

Released checkpoints:

| Checkpoint | Use case | Download |
| --- | --- | --- |
| `opendde.pt` | General-purpose OpenDDE checkpoint. | [opendde.pt](https://huggingface.co/aurekaresearch/OpenDDE/resolve/main/opendde.pt) |
| `opendde_abag.pt` | ABAG-optimized checkpoint for antibody-antigen complexes. | [opendde_abag.pt](https://huggingface.co/aurekaresearch/OpenDDE/resolve/main/opendde_abag.pt) |

For the default Docker command below, place the general-purpose checkpoint at
`$OPENDDE_ROOT_DIR/checkpoint/opendde.pt`. When using the ABAG-optimized
checkpoint, add this to the `opendde pred` command:

```bash
--load_checkpoint_path /opendde_data/checkpoint/opendde_abag.pt
```

Download one checkpoint directly into the default host path:

```bash
# General-purpose checkpoint:
curl -L \
  -o "$OPENDDE_ROOT_DIR/checkpoint/opendde.pt" \
  https://huggingface.co/aurekaresearch/OpenDDE/resolve/main/opendde.pt

# ABAG-optimized checkpoint:
curl -L \
  -o "$OPENDDE_ROOT_DIR/checkpoint/opendde_abag.pt" \
  https://huggingface.co/aurekaresearch/OpenDDE/resolve/main/opendde_abag.pt
```

Download or verify the remaining runtime files with Docker:

```bash
docker run --rm \
  -v "$OPENDDE_ROOT_DIR":/opendde_data \
  aurekaresearch/opendde:v1 \
  bash scripts/download_opendde_data.sh \
    --root /opendde_data
```

For protein-only smoke tests that disable MSA/template/RNA-MSA preprocessing, you
can skip search databases:

```bash
docker run --rm \
  -v "$OPENDDE_ROOT_DIR":/opendde_data \
  aurekaresearch/opendde:v1 \
  bash scripts/download_opendde_data.sh \
    --root /opendde_data \
    --skip-search-database
```

## 4. Run inference

The command below assumes `tiny.json` exists in the current host directory. See
[../README.md](../README.md) for the minimal input example.

```bash
mkdir -p output

docker run --rm --gpus all --shm-size=4g \
  -e OPENDDE_ROOT_DIR=/opendde_data \
  -v "$OPENDDE_ROOT_DIR":/opendde_data:ro \
  -v "$PWD":/workspace \
  -v "$PWD/output":/output \
  aurekaresearch/opendde:v1 \
  opendde pred \
    -i /workspace/tiny.json \
    -o /output \
    -n opendde_v1 \
    --use_msa false \
    --use_template false \
    --use_rna_msa false \
    --sample 1 \
    --step 200 \
    --cycle 10
```

For production inference options, MSA/template preprocessing, and checkpoint
configuration, see [inference_instructions.md](./inference_instructions.md).
