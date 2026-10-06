# Beyond Masked Sparsity: SNACK Enables Truly Sparse Neural Networks on GPU

An in-memory reference implementation of **Beyond Masked Sparsity: SNACK Enables Truly Sparse Neural Networks on GPU** in **Python**, adhering to standard library idioms, clean data structures, and assertion test suites.

### Core Highlights
* **Language & Standard**: Modern `Python` standard library conventions.
* **Architecture Pattern**: Designed for `Algorithmic Engineering` using `Standard Memory Primitives`.
* **Runtime Overhead**: Memory allocations are kept minimal to maintain clear data locality and predictable memory bounds.
* **Concurrency & Safety**: Execution behavior is validated against nominal workflows and boundary edge cases.

---

### Complexity Analysis

| Dimension | Bound |
| :--- | :--- |
| **Time (Best Case)** | `O(1)` |
| **Time (Worst Case)** | `O(N log N)` |
| **Auxiliary Space** | `O(N)` |

---

### Test Suite Execution

Self-contained verification drivers are embedded directly in `main.py` to validate happy paths, boundary inputs, and invariant preservation.

```bash
python3 main.py
```

---

*Reference implementation verified by [@myonathanlinkedin](https://github.com/myonathanlinkedin)*
