from __future__ import annotations
import struct
import math
from typing import List, Tuple
from custom_types import Image, Pixel

def _clamp(value: int) -> int:
    return max(0, min(255, value))

def _average_block(block_pixels: List[Pixel]) -> Pixel:
    if not block_pixels:
        return (0, 0, 0)
    r = sum(p[0] for p in block_pixels) // len(block_pixels)
    g = sum(p[1] for p in block_pixels) // len(block_pixels)
    b = sum(p[2] for p in block_pixels) // len(block_pixels)
    return (_clamp(r), _clamp(g), _clamp(b))

def compress_image(img: Image, block_size: int = 8) -> bytes:
    """
    Compress an Image using block-wise average color encoding.
    Header format (big endian):
        4 bytes: width (uint32)
        4 bytes: height (uint32)
        1 byte: block_size (uint8)
    Followed by block data: for each block, 3 bytes (R, G, B).
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    width, height = img.width, img.height
    # Compute number of blocks
    blocks_x = math.ceil(width / block_size)
    blocks_y = math.ceil(height / block_size)
    header = struct.pack(">IIB", width, height, block_size)
    block_bytes = bytearray()
    for by in range(blocks_y):
        for bx in range(blocks_x):
            # Determine block bounds
            x0 = bx * block_size
            y0 = by * block_size
            x1 = min(x0 + block_size, width)
            y1 = min(y0 + block_size, height)
            # Collect pixels in block
            block_pixels: List[Pixel] = []
            for y in range(y0, y1):
                row = img.data[y]
                block_pixels.extend(row[x0:x1])
            avg = _average_block(block_pixels)
            block_bytes.extend(avg)
    return header + block_bytes

def decompress_image(data: bytes) -> Image:
    """
    Decompress data produced by compress_image.
    Returns an Image object.
    """
    if len(data) < 9:
        raise ValueError("Data too short to contain header")
    width, height, block_size = struct.unpack(">IIB", data[:9])
    if block_size <= 0:
        raise ValueError("Invalid block_size in header")
    blocks_x = math.ceil(width / block_size)
    blocks_y = math.ceil(height / block_size)
    expected_len = 9 + blocks_x * blocks_y * 3
    if len(data) != expected_len:
        raise ValueError(f"Data length {len(data)} does not match expected {expected_len}")
    block_data = data[9:]
    # Prepare empty image data
    img_data: List[List[Pixel]] = [[(0, 0, 0) for _ in range(width)] for _ in range(height)]
    idx = 0
    for by in range(blocks_y):
        for bx in range(blocks_x):
            r, g, b = block_data[idx], block_data[idx+1], block_data[idx+2]
            idx += 3
            # Determine block bounds
            x0 = bx * block_size
            y0 = by * block_size
            x1 = min(x0 + block_size, width)
            y1 = min(y0 + block_size, height)
            for y in range(y0, y1):
                row = img_data[y]
                for x in range(x0, x1):
                    row[x] = (r, g, b)
    return Image(width, height, img_data)
