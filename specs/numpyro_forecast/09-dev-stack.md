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
[project]
name = "probcast"
requires-python = ">=3.11"
license = "Apache-2.0"
dependencies = [
    "jax>=0.4",
    "jaxlib>=0.4",
    "numpyro>=0.15",
    "pydantic>=2.0",
]

[project.optional-dependencies]
viz = ["arviz>=0.18", "matplotlib>=3.8"]
cv = ["xarray>=2024.1", "arviz>=0.18"]
nn = ["flax>=0.10"]  # For DeepAR / attention models (uses flax.nnx API)
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
```

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
line-length = 99
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "TCH"]
```

### `pre-commit`

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    hooks:
      - id: mypy
        additional_dependencies: [jaxtyping, numpyro, pydantic]
        args: [--ignore-missing-imports]
```

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
      - run: uv run pytest tests/ -x --tb=short
```
