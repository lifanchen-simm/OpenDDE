# Kernel Configuration


OpenDDE has a safe PyTorch path and optional GPU kernels.

## LayerNorm

Default:

```bash
export LAYERNORM_TYPE=torch
```

Optional CUDA LayerNorm:

```bash
export LAYERNORM_TYPE=fast_layernorm
```

Use `torch` for CPU, debugging, or environments where CUDA extension compilation
is unavailable.

## Triangle kernels

Both triangle attention and triangle multiplication support:

| Value | Meaning |
| --- | --- |
| `auto` | Use cuEquivariance when available, otherwise PyTorch. |
| `cuequivariance` | Force cuEquivariance GPU kernels. |
| `torch` | Force PyTorch fallback. |

CLI flags:

```bash
opendde pred \
  --triatt_kernel auto \
  --trimul_kernel auto
```

## Compatibility run

```bash
LAYERNORM_TYPE=torch opendde pred \
  -i examples/input.json \
  -o ./output \
  -n opendde_v1 \
  --use_msa false \
  --use_template false \
  --use_rna_msa false \
  --triatt_kernel torch \
  --trimul_kernel torch \
  --sample 1 \
  --step 200 \
  --cycle 10 \
  --dtype fp32
```
