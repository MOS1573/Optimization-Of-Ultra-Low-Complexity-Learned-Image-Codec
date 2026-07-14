"""
LoC-LIC Container Format (.lic)

Binary layout (all big-endian):

  Bytes       Field
  ──────────  ──────────────────────────────────────────────────────
  0–3         Magic:       b'LIC1'
  4           Version:     uint8  (current: 1)
  5           Lambda ID:   uint8  0=0.0018  1=0.0035  2=0.0067  3=0.025
  6           Quant type:  uint8  0=fp32    1=int8_gptq  2=qat_best
  7–8         img_H:       uint16 (original image height, before any padding)
  9–10        img_W:       uint16 (original image width,  before any padding)
  11–12       z_H:         uint16 (hyperprior spatial height from compress())
  13–14       z_W:         uint16 (hyperprior spatial width  from compress())
  15–18       CRC32:       uint32 (over the entire payload that follows)
  ── payload ──────────────────────────────────────────────────────
  19          n_groups:    uint8  (number of string groups, typically 2: y and z)
  ── per group ────────────────────────────────────────────────────
              n_items:     uint8  (strings per group; y may have 10 for slice coding)
  ── per item ─────────────────────────────────────────────────────
              item_len:    uint32
              item_bytes:  raw entropy-coded bytes

The strings structure mirrors CompressAI's model.compress() output exactly:
  {'strings': [[y_slice1, ..., y_sliceN], [z_bytes]], 'shape': (z_H, z_W)}

No pickle, no zlib re-compression.  Raw entropy-coded bytes stored directly.
Readable from any language (Python, C++, C#).
"""

import os
import struct
import zlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ── Constants ─────────────────────────────────────────────────────────────────

MAGIC   = b'LIC1'
VERSION = 1

LAMBDA_IDS: Dict[float, int] = {0.0018: 0, 0.0035: 1, 0.0067: 2, 0.025: 3}
LAMBDA_IDS_INV: Dict[int, float] = {v: k for k, v in LAMBDA_IDS.items()}

QUANT_IDS: Dict[str, int] = {'fp32': 0, 'int8_gptq': 1, 'qat_best': 2}
QUANT_IDS_INV: Dict[int, str] = {v: k for k, v in QUANT_IDS.items()}

# magic(4) + version(1) + lambda_id(1) + quant_id(1) + img_H(2) + img_W(2) + z_H(2) + z_W(2) + crc32(4)
_HEADER_FMT  = '>4sBBBHHHHI'
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)   # 19 bytes


# ── Strings serialization ─────────────────────────────────────────────────────
# Handles any number of groups and items per group, matching CompressAI's
# variable-slice entropy coding (e.g. 10 y-slices + 1 z string).

def _serialize_strings(strings: List[List[bytes]]) -> bytes:
    """Serialize a nested list of byte strings to raw bytes (no pickle)."""
    parts = [struct.pack('>B', len(strings))]          # n_groups
    for group in strings:
        parts.append(struct.pack('>B', len(group)))    # n_items
        for item in group:
            b = bytes(item)
            parts.append(struct.pack('>I', len(b)))    # item_len
            parts.append(b)                            # item_bytes
    return b''.join(parts)


def _deserialize_strings(data: bytes, offset: int = 0) -> Tuple[List[List[bytes]], int]:
    """Deserialize strings from raw bytes. Returns (strings, new_offset)."""
    n_groups = struct.unpack_from('>B', data, offset)[0]; offset += 1
    strings: List[List[bytes]] = []
    for _ in range(n_groups):
        n_items = struct.unpack_from('>B', data, offset)[0]; offset += 1
        group: List[bytes] = []
        for _ in range(n_items):
            item_len = struct.unpack_from('>I', data, offset)[0]; offset += 4
            item = data[offset: offset + item_len];              offset += item_len
            group.append(item)
        strings.append(group)
    return strings, offset


# ── Public API ────────────────────────────────────────────────────────────────

def write_lic(
    filepath: "str | Path",
    compress_out: Dict[str, Any],
    lambda_val: float,
    quant_type: str,
    img_shape: Tuple[int, int],
) -> None:
    """Write a .lic file from a CompressAI compress() output.

    Args:
        filepath:     Destination path, e.g. 'image.lic'.
        compress_out: Dict returned by model.compress(x):
                        {'strings': [[y_slice1, ...], [z_bytes]], 'shape': (z_H, z_W)}
                        Extra keys (e.g. 'cost_time') are ignored.
        lambda_val:   R-D lambda — one of {0.0018, 0.0035, 0.0067, 0.025}.
        quant_type:   'fp32', 'int8_gptq', or 'qat_best'.
        img_shape:    (H, W) of the original image before padding.
    """
    if lambda_val not in LAMBDA_IDS:
        raise ValueError(f"Unknown lambda {lambda_val}. Valid: {sorted(LAMBDA_IDS)}")
    if quant_type not in QUANT_IDS:
        raise ValueError(f"Unknown quant_type '{quant_type}'. Valid: {list(QUANT_IDS)}")

    z_H, z_W     = compress_out['shape']
    img_H, img_W = img_shape

    payload = _serialize_strings(compress_out['strings'])
    crc     = zlib.crc32(payload) & 0xFFFFFFFF

    header = struct.pack(
        _HEADER_FMT,
        MAGIC,
        VERSION,
        LAMBDA_IDS[lambda_val],
        QUANT_IDS[quant_type],
        img_H, img_W,
        z_H, z_W,
        crc,
    )

    with open(filepath, 'wb') as f:
        f.write(header)
        f.write(payload)


def read_lic(
    filepath: "str | Path",
) -> Tuple[Dict[str, Any], float, str, Tuple[int, int]]:
    """Read and validate a .lic file.

    Returns:
        compress_out: {'strings': [[y_slice1, ...], [z_bytes]], 'shape': (z_H, z_W)}
                      Ready to pass directly to model.decompress().
        lambda_val:   float, one of {0.0018, 0.0035, 0.0067, 0.025}.
        quant_type:   str, 'fp32', 'int8_gptq', or 'qat_best'.
        img_shape:    (H, W) of the original image (crop decoded output to this).
    """
    with open(filepath, 'rb') as f:
        raw_header = f.read(_HEADER_SIZE)
        if len(raw_header) < _HEADER_SIZE:
            raise ValueError("Truncated .lic file: header incomplete")

        magic, version, lambda_id, quant_id, img_H, img_W, z_H, z_W, stored_crc = \
            struct.unpack(_HEADER_FMT, raw_header)

        if magic != MAGIC:
            raise ValueError(f"Not a .lic file (bad magic: {magic!r})")
        if version != VERSION:
            raise ValueError(f"Unsupported .lic version {version} (expected {VERSION})")
        if lambda_id not in LAMBDA_IDS_INV:
            raise ValueError(f"Unknown lambda ID: {lambda_id}")
        if quant_id not in QUANT_IDS_INV:
            raise ValueError(f"Unknown quant ID: {quant_id}")

        payload = f.read()

    actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
    if actual_crc != stored_crc:
        raise ValueError(
            f"CRC32 mismatch — file is corrupt "
            f"(got {actual_crc:#010x}, expected {stored_crc:#010x})"
        )

    strings, _ = _deserialize_strings(payload)

    compress_out = {
        'strings': strings,
        'shape':   (z_H, z_W),
    }
    return compress_out, LAMBDA_IDS_INV[lambda_id], QUANT_IDS_INV[quant_id], (img_H, img_W)


def lic_info(filepath: "str | Path") -> Dict[str, Any]:
    """Return metadata from a .lic file without loading the full bitstream."""
    with open(filepath, 'rb') as f:
        raw_header = f.read(_HEADER_SIZE)
    if len(raw_header) < _HEADER_SIZE:
        raise ValueError("Truncated .lic file")

    magic, version, lambda_id, quant_id, img_H, img_W, z_H, z_W, _ = \
        struct.unpack(_HEADER_FMT, raw_header)

    if magic != MAGIC:
        raise ValueError(f"Not a .lic file (bad magic: {magic!r})")

    file_size    = Path(filepath).stat().st_size
    payload_size = file_size - _HEADER_SIZE

    return {
        'version':       version,
        'lambda':        LAMBDA_IDS_INV.get(lambda_id, f'unknown({lambda_id})'),
        'quant_type':    QUANT_IDS_INV.get(quant_id,  f'unknown({quant_id})'),
        'img_shape':     (img_H, img_W),
        'z_shape':       (z_H, z_W),
        'payload_bytes': payload_size,
        'bpp':           payload_size * 8 / (img_H * img_W) if img_H and img_W else None,
    }


# ── Self-test ─────────────────────────────────────────────────────────────────

def _test() -> None:
    """Verify round-trip and corruption detection without a real model."""
    import tempfile

    # Simulate LoC-LIC's 10-slice y coding + 1 z string
    y_slices = [bytes(range(i, i + 128)) for i in range(10)]   # 10 y slices
    z_bytes  = bytes(range(128)) * 2                            # 1 z string

    compress_out = {
        'strings': [y_slices, [z_bytes]],
        'shape':   (8, 12),
        'cost_time': 0.05,   # extra key — should be silently ignored on write
    }
    img_shape = (512, 768)

    with tempfile.NamedTemporaryFile(suffix='.lic', delete=False) as tmp:
        path = tmp.name

    try:
        # ── round-trip ────────────────────────────────────────────────────────
        write_lic(path, compress_out, lambda_val=0.0067, quant_type='int8_gptq',
                  img_shape=img_shape)

        out, lam, qtype, shape = read_lic(path)
        assert out['strings'][0] == y_slices,  "y_slices mismatch"
        assert out['strings'][1] == [z_bytes], "z_bytes mismatch"
        assert out['shape']      == (8, 12),   "z_shape mismatch"
        assert lam               == 0.0067,    "lambda mismatch"
        assert qtype             == 'int8_gptq', "quant_type mismatch"
        assert shape             == img_shape,  "img_shape mismatch"
        assert 'cost_time' not in out,          "cost_time should not be stored"

        info = lic_info(path)
        print(f"  header size : {_HEADER_SIZE} bytes")
        print(f"  file size   : {Path(path).stat().st_size} bytes")
        print(f"  lambda      : {info['lambda']}")
        print(f"  quant_type  : {info['quant_type']}")
        print(f"  img_shape   : {info['img_shape']}")
        print(f"  z_shape     : {info['z_shape']}")
        print(f"  payload     : {info['payload_bytes']} bytes")
        print(f"  bpp         : {info['bpp']:.6f}")

        # ── single-string case (simple models) ────────────────────────────────
        simple_out = {'strings': [[b'\x00' * 512], [b'\xff' * 64]], 'shape': (4, 6)}
        write_lic(path, simple_out, lambda_val=0.0018, quant_type='fp32',
                  img_shape=(256, 384))
        out2, lam2, _, _ = read_lic(path)
        assert out2['strings'] == simple_out['strings'], "single-string round-trip failed"
        print("  Single-string round-trip OK")

        # ── all 4 lambdas and all quant types ─────────────────────────────────
        for lv in (0.0018, 0.0035, 0.0067, 0.025):
            for qt in ('fp32', 'int8_gptq', 'qat_best'):
                write_lic(path, compress_out, lambda_val=lv, quant_type=qt,
                          img_shape=img_shape)
                _, lam3, qt3, _ = read_lic(path)
                assert lam3 == lv and qt3 == qt, f"Round-trip failed for {lv}/{qt}"
        print("  All lambda/quant combinations OK")

        # ── corruption detection ──────────────────────────────────────────────
        write_lic(path, compress_out, lambda_val=0.0018, quant_type='fp32',
                  img_shape=img_shape)
        with open(path, 'r+b') as f:
            f.seek(-1, 2)
            f.write(b'\x00')
        try:
            read_lic(path)
            raise AssertionError("Should have raised on corrupt file")
        except ValueError as e:
            print(f"  Corruption detected OK: {e}")

        print("\nAll tests passed.")

    finally:
        os.unlink(path)


if __name__ == '__main__':
    _test()
