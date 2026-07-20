Summary

Re-enabled `_expr`-based `get_unitary` for **`RXGate`, `RYGate`, `RZGate`, `U1Gate`, `RXXGate`, `RYYGate`, `RZZGate`** — these were only ever blocked by the panic bug (which I fixed), not by any precision issue, since they each use a single cos/sin argument pair where `cos²+sin²=1` holds to machine precision regardless of the underlying computation path. Verified:

- Correctness (unitarity) and gradients (finite-difference match) for all 7.
- A 200-gate deep-circuit stress test (RX/RZ/CNOT) against a pure-numpy reference — max diff ~9e-16, no compounding error.
- The extreme-parameter case now raises a catchable `ValueError` instead of crashing, as intended.
- Found and fixed one test (`test_frozenparams.py::test_u3_rx`) whose hypothesis strategy generated unbounded floats up to `f64::MAX` — that's a case where _no_ implementation can produce a numerically meaningful answer, and the test was only passing before because the old numpy path silently returned garbage instead of catching it. Bounded the strategy to `±1e15` (physically the angle range is never remotely close to float extremes anyway) with a comment explaining why.
- Full `tests/ir` suite: 2364/2364 passing.

**`HGate` was deliberately left alone** — its `radix=2` case has a separate, unrelated bug: `1/sqrt(2)` (openqudit) vs `sqrt(2)/2` (bqskit's original) differ by 1 ULP, and I proved earlier this specific difference compounds into ~1e-7 error in deep circuits (the `test_cnot2cz` failure). Neither of today's fixes touches that — re-enabling `HGate` would reintroduce that regression, so it stays on the hand-written implementation.

Want me to run the rest of the test suite (`tests/compiler`, `tests/passes`, `tests/exec`, etc.) to fully close this out, or are you good with what's verified so far?
