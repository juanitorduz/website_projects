# 09 — Dev Stack

## Type Hints

### `jaxtyping` + `beartype`

All public functions use `jaxtyping` shape annotations for array arguments:

```python
from jaxtyping import Float, Array

def crps_empirical(
    truth: Float[Array, "*batch"],
    pred: Float[Array, "n_samples *batch"],
) -> Float[Array, ""]:
```

Runtime shape checking is enabled via `beartype` during development and testing:

```python
# In conftest.py or via pytest plugin
from jaxtyping import install_import_hook
install_import_hook("probcast", "beartype.beartype")
```

**Source:** `var_numpyro.ipynb` uses this pattern:

```python
%load_ext jaxtyping
%jaxtyping.typechecker beartype.beartype
```

### Standard type hints

Non-array arguments use standard Python type hints (`int`, `float`, `str`, `Callable`, etc.). Pydantic models (`MCMCParams`, `SVIParams`) provide runtime validation for configuration.

## Package Management

### `uv`

- `pyproject.toml` as single source of truth for metadata, dependencies, and tool config.
- `uv` for dependency resolution, virtual environments, and publishing.
- Lock file (`uv.lock`) committed for reproducible CI builds.

### `pyproject.toml` structure

```toml
[build-system]
requires = ["hatchling>=1.25"]
build-backend = "hatchling.build"

[project]
name = "probcast"
version = "0.1.0"
description = "Bayesian forecasting with NumPyro and JAX."
readme = "README.md"
requires-python = ">=3.11"
license = { text = "Apache-2.0" }
authors = [
    { name = "Juan Orduz", email = "juanitorduz@gmail.com" },
]
keywords = [
    "bayesian",
    "forecasting",
    "jax",
    "numpyro",
    "probabilistic-programming",
    "time-series",
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering",
]
dependencies = [
    "jax>=0.4",
    "numpyro>=0.15",
    "pydantic>=2.0",
]

[project.optional-dependencies]
viz = [
    "arviz>1.0.0",
    "matplotlib>=3.8",
]
cv = [
    "xarray>=2024.1",
    "arviz>1.0.0",
]
nn = [
    "flax>=0.10",
]  # For DeepAR / attention models (uses flax.nnx API)
dev = [
    "pytest>=8.0",
    "pytest-xdist",
    "jaxtyping>=0.2",
    "beartype>=0.18",
    "ruff>=0.4",
    "pre-commit>=3.7",
    "mypy>=1.10",
]
docs = [
    "sphinx>=7",
    "myst-nb>=1.0",
    "sphinx-book-theme",
]
all = ["probcast[viz,cv,nn,dev,docs]"]

[project.urls]
Homepage = "https://juanitorduz.github.io"
Repository = "https://github.com/juanitorduz/probcast"
Documentation = "https://probcast.readthedocs.io"
Issues = "https://github.com/juanitorduz/probcast/issues"

[tool.uv]
default-groups = []

[tool.ruff]
line-length = 99
target-version = "py311"
# Canonical Ruff policy is defined in the dedicated "Linting and Formatting"
# section below. Keep only one Ruff configuration in the real pyproject.

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --tb=short"

[tool.mypy]
python_version = "3.11"
warn_unused_configs = true
disallow_untyped_defs = false
ignore_missing_imports = true
```

The structure should be explicit in the design review:

- `[build-system]` defines how the wheel/sdist is built. Use `hatchling` unless there is a strong reason to choose another backend.
- `[project]` contains the package identity and only the runtime dependencies required for the core library.
- `[project.optional-dependencies]` is the contract for extras. Keep visualization, CV, docs, neural components, and development tooling in separate tiers.
- `[project.urls]` should exist from day one for PyPI and documentation discoverability.
- `[tool.uv]` owns environment-resolution defaults, while linting, tests, and typing live under their respective `[tool.*]` sections.

Recommended dependency policy:

- Keep `jax` in core dependencies, but do not hard-code `jaxlib` in the initial package spec. Installation of accelerator-specific wheels should be documented separately because CPU/GPU/TPU installs vary by platform.
- Keep `matplotlib` and `ArviZ > 1.0.0` in the `viz` extra.
- Do not add `seaborn`.
- If `xarray` appears in public APIs like `CVResult`, either promote it to required status for those APIs or make those APIs clearly optional and lazily imported.
- The first implementation pass should create a real `pyproject.toml` matching this structure so packaging decisions are not deferred to the end.

## Testing

### `pytest`

Structure mirrors the module layout:

```
tests/
├── conftest.py              # Shared fixtures
├── test_core/
│   ├── test_types.py        # Protocol compliance checks
│   └── test_params.py       # Pydantic validation
├── test_components/
│   ├── test_level.py        # Deterministic transition checks
│   └── ...
├── test_models/
│   ├── test_exponential_smoothing.py
│   └── ...
├── test_metrics/
│   ├── test_crps.py         # Known-value tests, edge cases
│   └── test_point.py
├── test_cv/
│   └── test_time_series.py
├── test_utils/
│   ├── test_features.py     # Fourier basis properties
│   └── test_data.py
└── integration/
    ├── test_es_pipeline.py  # model → inference → forecast → metrics
    ├── test_intermittent.py
    └── test_var.py
```

### Testing strategy

- **Unit tests:** Deterministic checks on components (known input → known output), metric edge cases, data prep helpers.
- **Integration tests:** End-to-end pipelines with short MCMC runs (`num_warmup=50, num_samples=50, num_chains=1`). Verify shapes, no NaN, reasonable posterior ranges.
- **Property tests:** CRPS ≥ 0, CRPS of perfect forecast = 0, metrics monotonicity properties.

### Fixtures (`conftest.py`)

```python
import pytest
from jax import random

@pytest.fixture
def rng_key():
    return random.PRNGKey(42)

@pytest.fixture
def sample_univariate():
    """Short synthetic time series for fast tests."""
    return jnp.sin(jnp.linspace(0, 4 * jnp.pi, 50)) + 0.1 * random.normal(random.PRNGKey(0), (50,))

@pytest.fixture
def fast_mcmc_params():
    return MCMCParams(num_warmup=50, num_samples=50, num_chains=1)
```

## Linting and Formatting

### `ruff`

```toml
[tool.ruff]
extend-include = ["*.ipynb"]
line-length = 99
target-version = "py311"

[tool.ruff.format]
docstring-code-format = true

[tool.ruff.lint]
select = ["B", "D", "DOC", "E", "F", "I", "RUF", "S", "UP", "W"]
ignore = [
    "B008",   # Allow pydantic Field/default factories in argument defaults
    "B904",   # Do not force raise-from in every internal exception path
    "RUF001", # Allow scientific symbols when justified
    "RUF002", # Allow scientific symbols in docstrings when justified
    "RUF012", # Mutable class attributes sometimes appear in config patterns
]

[tool.ruff.lint.pydocstyle]
convention = "numpy"

[tool.ruff.lint.per-file-ignores]
"tests/*" = [
    "D",
    "S101",
]
"docs/tutorials/*" = [
    "B018",
    "D103",
]
"docs/examples/*" = [
    "B018",
    "D103",
]

[tool.ruff.lint.pycodestyle]
max-line-length = 120
```

This is the canonical Ruff policy for the project and should be the only Ruff configuration copied into the real `pyproject.toml`. The important design choice is to stay strict on bugbear, docstrings, Ruff-native rules, security-oriented checks, and modernization rules, while allowing narrow exceptions for tests and notebook-style documentation.

### `pre-commit`

```yaml
ci:
  autofix_prs: false

repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.15.6
    hooks:
      - id: ruff-check
        types_or: [python, pyi, jupyter]
        args: ["--fix", "--output-format=full"]
        exclude: ^docs/examples/dev/
      - id: ruff-format
        types_or: [python, pyi, jupyter]
        exclude: ^docs/examples/dev/
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.19.1
    hooks:
      - id: mypy
        files: ^probcast/
        additional_dependencies: [jaxtyping, numpyro, pydantic]
        args: [--ignore-missing-imports]
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v6.0.0
    hooks:
      - id: no-commit-to-branch
        args: [--branch, main]
        stages: [pre-commit, pre-merge-commit, pre-push, manual]
      - id: debug-statements
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-toml
      - id: check-yaml
      - id: check-added-large-files
        exclude: ^(docs/tutorials/|docs/examples/|docs/_static/)
  - repo: local
    hooks:
      - id: validate-api-docs
        name: Validate API documentation completeness
        entry: python3 scripts/validate_api_docs.py
        language: system
        pass_filenames: false
        files: ^(docs/api/|probcast/.*\.py)$
```

This is intentionally similar to the PyMC Marketing pre-commit setup in their [`.pre-commit-config.yaml`](https://github.com/pymc-labs/pymc-marketing/blob/main/.pre-commit-config.yaml), but adjusted for this repository:

- keep Ruff as the primary formatter and linter, including notebooks
- run `mypy` only against the package source tree
- add standard repository hygiene hooks
- prevent accidental commits to `main`
- reserve a local hook slot for package-specific validation such as API doc coverage

## Plotting Policy

- Plotting utilities use `matplotlib` directly.
- Do not introduce `seaborn` as a dependency.
- ArviZ-backed visualization or diagnostics helpers must be compatible with `ArviZ > 1.0.0`.
- Keep plotting optional so the core package remains lightweight.

## Docstrings

NumPy-style docstrings with `jaxtyping` shapes referenced in parameter descriptions:

```python
def run_mcmc(
    rng_key: Array,
    model: Callable,
    params: MCMCParams,
    *model_args,
    **nuts_kwargs,
) -> MCMC:
    """Run NUTS/MCMC inference on a model function.

    Parameters
    ----------
    rng_key : Array
        JAX PRNG key.
    model : Callable
        NumPyro model function following the ModelFn protocol.
    params : MCMCParams
        MCMC configuration.

    Returns
    -------
    MCMC
        Fitted MCMC object.
    """
```

## License

**Apache License 2.0** — permissive, compatible with JAX/NumPyro ecosystem (both Apache 2.0).

## AI-Friendly Files

### `AGENTS.md`

Instructions for AI coding assistants working on the project. Should include:
- Repository structure overview and module responsibilities.
- Coding conventions (functional style, jaxtyping annotations, NumPy docstrings).
- Testing patterns (deterministic unit tests for components, short MCMC integration tests).
- Common pitfalls (scan+condition interaction, forward_mode_differentiation).
- How to add a new model (component → model → tests → docs).

### `SKILLS.md`

Package capabilities reference for AI tools. Should include:
- Available model types and their signatures.
- Inference API (MCMC vs SVI, when to use which).
- Metrics and CV utilities.
- Example workflows (fit → forecast → evaluate).

## Contributing Guide (`CONTRIBUTING.md`)

Should cover:
- Dev environment setup with `uv`.
- Running tests (`uv run pytest`), linting (`uv run ruff check .`).
- PR guidelines: one feature per PR, tests required, NumPy docstrings.
- How to add a new model: create component functions → model function → unit + integration tests → example notebook.
- Code of conduct reference.

## Code of Conduct (`CODE_OF_CONDUCT.md`)

Adopt the [Contributor Covenant](https://www.contributor-covenant.org/) v2.1 — standard in the Python open source ecosystem.

## CI/CD

GitHub Actions workflow:

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --all-extras
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run mypy probcast/
      - run: uv run python3 scripts/validate_api_docs.py
      - run: uv run sphinx-build -W docs docs/_build/html
      - run: uv run pytest tests/ -x --tb=short
```

### CI parity and release policy

- CI must enforce the same quality gates as local pre-commit for source files: Ruff, mypy, and API docs validation.
- Documentation is a required gate: `sphinx-build -W` must pass before merge.
- Notebook docs policy must be explicit (execution mode, timeout budget, and failure behavior).
- Release policy:
  - use semantic versioning for public API changes,
  - document deprecations for at least one minor release before removal,
  - require changelog entries for user-visible API, statistical, or CI behavior changes.
