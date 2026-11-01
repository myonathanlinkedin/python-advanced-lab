from __future__ import annotations
import random
import math
from custom_types import Image, Pixel
from engine import compress_image, decompress_image

def _create_random_image(width: int, height: int) -> Image:
    data = [[(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
             for _ in range(width)] for _ in range(height)]
    return Image(width, height, data)

def _assert_images_equal(img1: Image, img2: Image) -> None:
    assert img1.width == img2.width, "Width mismatch"
    assert img1.height == img2.height, "Height mismatch"
    for y in range(img1.height):
        for x in range(img1.width):
            assert img1.data[y][x] == img2.data[y][x], f"Pixel mismatch at ({x},{y})"

def _test_zero_size_image() -> None:
    img = Image(0, 0, [])
    compressed = compress_image(img, block_size=8)
    decompressed = decompress_image(compressed)
    assert decompressed.width == 0 and decompressed.height == 0
    assert decompressed.data == []

def _test_single_pixel() -> None:
    img = Image(1, 1, [[(123, 45, 67)]])
    compressed = compress_image(img, block_size=8)
    decompressed = decompress_image(compressed)
    _assert_images_equal(img, decompressed)

def _test_uniform_image() -> None:
    img = Image(16, 16, [[(200, 200, 200) for _ in range(16)] for _ in range(16)])
    compressed = compress_image(img, block_size=8)
    decompressed = decompress_image(compressed)
    _assert_images_equal(img, decompressed)

def _test_partial_blocks() -> None:
    img = Image(10, 10, [[(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                          for _ in range(10)] for _ in range(10)])
    compressed = compress_image(img, block_size=8)
    decompressed = decompress_image(compressed)
    # Decompressed image should have uniform blocks equal to block averages
    # Verify that each block in decompressed matches the computed average
    block_size = 8
    for by in range(math.ceil(10 / block_size)):
        for bx in range(math.ceil(10 / block_size)):
            x0 = bx * block_size
            y0 = by * block_size
            x1 = min(x0 + block_size, 10)
            y1 = min(y0 + block_size, 10)
            # Compute average of original block
            block_pixels = []
            for y in range(y0, y1):
                block_pixels.extend(img.data[y][x0:x1])
            avg = (sum(p[0] for p in block_pixels) // len(block_pixels),
                   sum(p[1] for p in block_pixels) // len(block_pixels),
                   sum(p[2] for p in block_pixels) // len(block_pixels))
            # Check decompressed block
            for y in range(y0, y1):
                for x in range(x0, x1):
                    assert decompressed.data[y][x] == avg, f"Block average mismatch at ({x},{y})"

def _test_block_size_greater_than_image() -> None:
    img = Image(5, 5, [[(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                          for _ in range(5)] for _ in range(5)])
    compressed = compress_image(img, block_size=10)
    decompressed = decompress_image(compressed)
    # Decompressed image should be uniform with average of entire image
    block_pixels = []
    for row in img.data:
        block_pixels.extend(row)
    avg = (sum(p[0] for p in block_pixels) // len(block_pixels),
           sum(p[1] for p in block_pixels) // len(block_pixels),
           sum(p[2] for p in block_pixels) // len(block_pixels))
    for y in range(5):
        for x in range(5):
            assert decompressed.data[y][x] == avg, f"Pixel mismatch at ({x},{y})"

def _test_block_size_one() -> None:
    img = Image(4, 4, [[(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                          for _ in range(4)] for _ in range(4)])
    compressed = compress_image(img, block_size=1)
    decompressed = decompress_image(compressed)
    _assert_images_equal(img, decompressed)

def _test_random_images() -> None:
    for _ in range(5):
        w = random.randint(1, 32)
        h = random.randint(1, 32)
        img = _create_random_image(w, h)
        compressed = compress_image(img, block_size=8)
        decompressed = decompress_image(compressed)
        # Verify dimensions
        assert decompressed.width == w and decompressed.height == h
        # Verify that each block in decompressed matches the computed average
        block_size = 8
        for by in range(math.ceil(h / block_size)):
            for bx in range(math.ceil(w / block_size)):
                x0 = bx * block_size
                y0 = by * block_size
                x1 = min(x0 + block_size, w)
                y1 = min(y0 + block_size, h)
                block_pixels = []
                for y in range(y0, y1):
                    block_pixels.extend(img.data[y][x0:x1])
                avg = (sum(p[0] for p in block_pixels) // len(block_pixels),
                       sum(p[1] for p in block_pixels) // len(block_pixels),
                       sum(p[2] for p in block_pixels) // len(block_pixels))
                for y in range(y0, y1):
                    for x in range(x0, x1):
                        assert decompressed.data[y][x] == avg, f"Random image block mismatch at ({x},{y})"

def _demo() -> None:
    print("=== Nibnib Demo ===")
    img = _create_random_image(64, 64)
    compressed = compress_image(img, block_size=8)
    print(f"Original size: {len(img.data) * len(img.data[0]) * 3} bytes")
    print(f"Compressed size: {len(compressed)} bytes")
    decompressed = decompress_image(compressed)
    print(f"Decompressed image dimensions: {decompressed.width}x{decompressed.height}")
    # Verify that decompressed image has uniform blocks
    block_size = 8
    for by in range(math.ceil(64 / block_size)):
        for bx in range(math.ceil(64 / block_size)):
            x0 = bx * block_size
            y0 = by * block_size
            x1 = min(x0 + block_size, 64)
            y1 = min(y0 + block_size, 64)
            block_pixels = []
            for y in range(y0, y1):
                block_pixels.extend(img.data[y][x0:x1])
            avg = (sum(p[0] for p in block_pixels) // len(block_pixels),
                   sum(p[1] for p in block_pixels) // len(block_pixels),
                   sum(p[2] for p in block_pixels) // len(block_pixels))
            for y in range(y0, y1):
                for x in range(x0, x1):
                    if decompressed.data[y][x] != avg:
                        print(f"Mismatch at ({x},{y})")
                        return
    print("Demo completed successfully. All blocks match expected averages.")

if __name__ == "__main__":
    # Run unit tests
    _test_zero_size_image()
    _test_single_pixel()
    _test_uniform_image()
    _test_partial_blocks()
    _test_block_size_greater_than_image()
    _test_block_size_one()
    _test_random_images()
    print("All unit tests passed.")
    # Run demo
    _demo()
