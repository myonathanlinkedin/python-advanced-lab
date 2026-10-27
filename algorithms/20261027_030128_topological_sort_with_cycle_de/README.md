# Topological Sort with Cycle Detection in Directed Graphs

Self-contained **Topological Sort with Cycle Detection in Directed Graphs** algorithmic primitive written in idiomatic **Python**. Built from scratch using standard library constructs with zero external dependencies.

---

## 🏛️ Architecture & Design Decisions

This module organizes `Topological Sort with Cycle Detection in Directed Graphs` into an isolated, self-contained unit:
* **Domain Focus**: `Graph Topology & Traversal`
* **Primary Primitives**: `Adjacency List & Priority Heap`
* **Memory Strategy**: Zero external heap dependencies; designed as a pure in-memory algorithmic component.
* **Correctness Model**: State transitions follow clear ordering guarantees with explicit validation at each phase.

### Asymptotic Complexity

| Metric | Bound | Characteristics |
| :--- | :---: | :--- |
| **Best Case Time** | `O(V + E)` | Optimized fast-path execution |
| **Average / Worst Time** | `O((V + E) log V)` | Deterministic upper bound for generalized workloads |
| **Space Complexity** | `O(V + E)` | Strict bounds without unconstrained heap growth |

---

## 🧪 Verification Suite

The accompanying `main.py` driver executes self-contained verification tests:
1. **Nominal Flow**: Validates baseline correctness under typical real-world inputs.
2. **Boundary Conditions**: Exercises extreme edge cases (empty inputs, singletons, capacity limits).
3. **Invariant Preservation**: Validates internal state consistency throughout mutation lifecycles.

### Running Locally

```bash
python3 main.py
```

---

> ### 🔬 Module Engineering Origin & Verification Disclaimer
> **Algorithmic Origin & Operational Risk Allocation:**
> - **Autonomous Algorithmic Synthesis:** The implementation of `Topological Sort with Cycle Detection in Directed Graphs` in `Python`, including accompanying unit tests, harnesses, and build scripts, was synthesized via an autonomous AI engineering agent directed by system-level prompt specifications.
> - **Verification Scope:** Correctness assertions are derived solely from deterministic test execution and deterministic test suites. The maintainer governs the testing parameters and algorithmic criteria.
> - **ASSUMPTION OF OPERATIONAL RISK:** THIS MODULE IS DISTRIBUTED AS AN OPEN ALGORITHMIC STUDY. THE MAINTAINER DISCLAIMS ANY WARRANTY OF FITNESS FOR PRODUCTION WORKLOADS. USERS ASSUME COMPLETE LIABILITY FOR VALIDATING MEMORY SAFETY, THREAD SAFETY, CONCURRENCY HAZARDS, AND SECURITY INTEGRITY PRIOR TO PRODUCTION INTEGRATION.

---

*Source code released under the MIT License • [@myonathanlinkedin](https://github.com/myonathanlinkedin)*