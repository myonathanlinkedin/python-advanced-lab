# Union-Find: The Matrix of Disjoint Sets in Python

High-performance **Union-Find: The Matrix of Disjoint Sets** primitive implemented in idiomatic **Python**. Built from scratch using standard library constructs with zero external dependencies.

## Implementation Details

* **Category**: `Computational Mathematics & Transformation`
* **Data Structure Foundation**: `Lookup Tables & Bitwise Bitvectors`
* **Allocation Pattern**: Memory allocations are kept minimal to avoid allocator contention and preserve CPU cache locality.
* **Invariant Integrity**: State transitions adhere to strict ordering guarantees with explicit synchronization fences where necessary.

## Performance Characteristics

* **Time**: `$O(N \log N)$` average, with `$O(N \log N)$` best-case response under ideal conditions.
* **Space**: `$O(N)$` memory usage.

## Test Harness

To compile and execute the test assertions for this module:

```bash
python3 main.py
```

---

*Curated as part of the Polyglot Systems Lab • Maintained by [@myonathanlinkedin](https://github.com/myonathanlinkedin)*