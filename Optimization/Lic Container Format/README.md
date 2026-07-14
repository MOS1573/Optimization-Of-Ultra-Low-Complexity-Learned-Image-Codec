# LoC-LIC Container Format (.lic)

Custom binary container for storing LoC-LIC compressed images.
The format is self-describing: a decoder only needs the `.lic` file to know
which model (lambda + quantization type) to load for decompression.

---

## Binary Layout

All fields are big-endian.

```
Offset  Size   Type     Field
──────  ────   ──────   ─────────────────────────────────────────────────────
0       4      bytes    Magic: b'LIC1'
4       1      uint8    Version (current: 1)
5       1      uint8    Lambda ID  → 0=0.0018  1=0.0035  2=0.0067  3=0.025
6       1      uint8    Quant type → 0=fp32    1=int8_gptq  2=qat_best
7       2      uint16   img_H  — original image height before padding
9       2      uint16   img_W  — original image width  before padding
11      2      uint16   z_H    — hyperprior spatial height (from compress())
13      2      uint16   z_W    — hyperprior spatial width  (from compress())
15      4      uint32   CRC32  — over the entire payload (bytes 19 onward)
19      4      uint32   y_len
23      y_len  bytes    y_bytes — raw entropy-coded latent y
...     4      uint32   z_len
...     z_len  bytes    z_bytes — raw entropy-coded latent z
```

**Header size: 19 bytes.**  No pickle, no zlib re-compression.
The `y_bytes` and `z_bytes` come directly from CompressAI's `model.compress()`
and are fed directly to `model.decompress()` on read.

---

## Python API

```python
from lic_format import write_lic, read_lic, lic_info

# ── Compress and save ──────────────────────────────────────────────────────
img_shape = (H_orig, W_orig)           # before any padding
compress_out = model.compress(x)       # {'strings': [[y], [z]], 'shape': (zH, zW)}

write_lic(
    'output.lic',
    compress_out,
    lambda_val = 0.0067,       # one of {0.0018, 0.0035, 0.0067, 0.025}
    quant_type = 'int8_gptq',  # 'fp32' | 'int8_gptq' | 'qat_best'
    img_shape  = img_shape,
)

# ── Load and decompress ────────────────────────────────────────────────────
compress_out, lambda_val, quant_type, img_shape = read_lic('output.lic')

# compress_out is ready to pass directly to model.decompress():
x_hat = model.decompress(**compress_out)['x_hat']

# Crop to original size (removes padding added before compress):
H, W = img_shape
x_hat = x_hat[:, :, :H, :W]

# ── Inspect without loading bitstream ─────────────────────────────────────
info = lic_info('output.lic')
# {'lambda': 0.0067, 'quant_type': 'int8_gptq',
#  'img_shape': (512, 768), 'z_shape': (8, 12),
#  'payload_bytes': 12345, 'bpp': 0.2638}
```

---

## Integration Points

| Component | How it uses .lic |
|-----------|-----------------|
| **Eval harness** | `write_lic` after encoding, `read_lic` to verify bitstream integrity |
| **Windows desktop app** | User presses Save → `write_lic`; Open → `read_lic` → select model → decompress |
| **Benchmark scripts** | `lic_info` to read BPP and metadata without a full decode |

---

## Lambda ID Mapping

| Lambda ID | λ value | Role |
|-----------|---------|------|
| 0 | 0.0018 | Low bitrate |
| 1 | 0.0035 | Low-mid bitrate |
| 2 | 0.0067 | Mid bitrate |
| 3 | 0.025  | High bitrate |

λ=0.013 is permanently excluded from this format (corrupted GPTQ checkpoint).

---

## Running the self-test

```bash
python "Optimization/Lic Container Format/lic_format.py"
```

No model or dataset required — tests round-trip and corruption detection
with synthetic byte strings.
