# SUSpMV: A High Frequency Sparse Matrix Vector Multiplier on HBM Enabled FPGA written in

Core **Python** implementation for **SUSpMV: A High Frequency Sparse Matrix Vector Multiplier on HBM Enabled FPGA written in**, structured for computational clarity, explicit data structures, and deterministic unit test coverage.

### Core Highlights
* **Language & Standard**: Modern `Python` standard library conventions.
* **Architecture Pattern**: Designed for `Computational Mathematics & Transformation` using `Lookup Tables & Bitwise Bitvectors`.
* **Runtime Overhead**: Buffer boundaries and collection indices are explicitly validated to prevent out-of-bounds access.
* **Concurrency & Safety**: State transitions follow clear ordering guarantees with explicit validation at each phase.

---

### Complexity Analysis

| Dimension | Bound |
| :--- | :--- |
| **Time (Best Case)** | `O(N log N)` |
| **Time (Worst Case)** | `O(N * M)` |
| **Auxiliary Space** | `O(N)` |

---

### Test Suite Execution

Self-contained verification drivers are embedded directly in `main.py` to validate happy paths, boundary inputs, and invariant preservation.

```bash
python3 main.py
```

---

*Part of the Polyglot Systems Lab • Maintained by [@myonathanlinkedin](https://github.com/myonathanlinkedin)*