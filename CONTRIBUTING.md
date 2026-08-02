# Contributing to CGC

Thank you for helping improve the Conservation-Guided Classification framework!

## Code Style

- **Formatting**: CGC follows [Ruff](https://docs.astral.sh/ruff/) rules (equivalent to `black` + `isort` + `flake8`). Run `ruff check cgc/` before committing.
- **Type annotations**: Use type hints for all public API functions. Run `mypy cgc/engine/ --ignore-missing-imports` to verify.
- **Docstrings**: Every public function should have a docstring explaining what it does, its parameters, and return value. Follow the existing numpy-style pattern.

## Testing

**All new features must include tests.** CGC uses three test layers:

1. **Unit tests** (`cgc/tests/test_*.py`): Use pytest. Verify specific inputs produce expected outputs.
2. **Property-based tests** (`cgc/tests/test_properties.py`): Use Hypothesis. Verify invariants hold for randomized inputs.
3. **Edge case tests** (`cgc/tests/test_edge_cases.py`): Validate boundary conditions, degenerate limits, and degradation behavior.

Run all tests before submitting a PR:

```bash
pytest cgc/tests/ -v
python cgc/tests/test_properties.py
python cgc/tests/test_edge_cases.py
```

## Adding a New Operator Channel

See `docs/extending.md` for the full guide. In brief:

1. Define an `OperatorSpec` in `cgc/channels/my_operator.py`
2. Add a CLI registry entry in `cgc/benchmarks/run.py`
3. Add a unit test in `cgc/tests/`

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes and add/update tests
3. Run the full verification suite: `cgc-benchmark`
4. Run lint and type checks: `ruff check cgc/` and `mypy cgc/engine/ --ignore-missing-imports`
5. Submit a PR with a clear description of what you changed and why
6. CI will automatically run all tests across Python 3.10–3.13

## Verification Layers

CGC's validation network has five layers. When making changes, ensure:

| Layer | What it checks | Minimum requirement |
|:---|:---|:---|
| L0 | Unit tests (pytest) | All pass |
| L1 | Internal self-consistency (4 benchmarks) | Zero delta vs reference |
| L2 | Cross-validation (2 benchmarks) | Consistent across methods |
| L4 | Known-model benchmarks (5 benchmarks) | Match physical expectations |
| STABILITY | Numerical stability (5 dimensions) | Degradation < 50% at coarsest grid |

## Questions?

Open an issue or discussion on the repository. Tag with `question`, `enhancement`, or `bug` as appropriate.
