# 03 — Core Abstractions

## Design Philosophy

**Protocols over base classes.** JAX's functional model (pure functions, JIT compilation, `vmap`) is incompatible with traditional OOP patterns. NumPyro models are plain functions — the package should respect that. We use Python `Protocol` classes for type checking and `NamedTuple`s for structured return values.

## `ModelFn` Protocol

Every forecasting model follows this signature:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class ModelFn(Protocol):
    def __call__(self, y: Array, *args, future: int = 0, **kwargs) -> None: ...
```

**Contract:**

- `y` — observed time series as a JAX array. Shape `(t_max,)` for univariate or `(t_max, *batch)` for panel/multi-series. All models must handle both shapes seamlessly — components broadcast over trailing batch dimensions.
- `*args` — model-specific positional arguments (e.g., `n_seasons`, `n_lags`, covariates).
- `future` — number of future time steps to forecast. When `future=0`, the model only conditions on observations. When `future > 0`, it produces `numpyro.deterministic` forecast sites.
- `**kwargs` — injectable priors and configuration overrides.
- Returns `None` (side effects via NumPyro's effect system).

**Rationale:** This matches the existing pattern across most source notebooks:

```python
# From exponential_smoothing_numpyro.ipynb
def level_model(y: ArrayImpl, future: int = 0) -> None: ...
def holt_winters_model(y: ArrayImpl, n_seasons: int, future: int = 0) -> None: ...

# From var_numpyro.ipynb
def model(y: Float[Array, "time vars"], n_lags: int, future: int = 0) -> None: ...

# From numpyro_forecasting_univariate.ipynb (covariate-first variant, see note below)
def model(covariates: Float[Array, "t_max feature_dim"], y=None) -> None: ...
```

The protocol is intentionally loose — it type-checks the common `(y, ..., future=0)` pattern without restricting model-specific arguments.

**Exception: covariate-first models.** The `local_level_fourier_model` (from `numpyro_forecasting_univariate.ipynb`) takes `(covariates, y=None)` instead of `(y, *args, future=0)`. This model does not conform to `ModelFn` because it uses covariates as the primary input and `y` as an optional observation argument. For such models:
- They cannot be used directly with `time_slice_cv()` — users wrap them with a `functools.partial` or lambda to match the expected signature.
- The generic `forecast()` and `run_mcmc()` helpers still work since they forward `*model_args` and `**model_kwargs` to the model positionally.
- This is a deliberate tradeoff: forcing covariates into `*args` would obscure the model's API. Instead, we document the exception and provide usage examples in the model's docstring.

## Inference Parameter Configs

Pydantic models for validated inference configuration. These replace the `InferenceParams` class found across notebooks.

### `MCMCParams`

```python
from pydantic import BaseModel, Field

class MCMCParams(BaseModel):
    """Configuration for MCMC inference via NUTS."""

    num_warmup: int = Field(2_000, ge=1)
    num_samples: int = Field(2_000, ge=1)
    num_chains: int = Field(4, ge=1)
```

**Source:** `InferenceParams` in `exponential_smoothing_numpyro.ipynb`, `tsb_numpyro.ipynb`, `arma_numpyro.ipynb` (all identical structure, varying defaults).

### `SVIParams`

```python
class SVIParams(BaseModel):
    """Configuration for SVI inference."""

    num_steps: int = Field(10_000, ge=1)
    num_samples: int = Field(5_000, ge=1, description="Posterior samples to draw after optimization")
    learning_rate: float = Field(0.005, gt=0)
    optimizer: str = Field("Adam", description="Optax or numpyro.optim optimizer name")
    guide: str = Field("AutoNormal", description="AutoGuide class name")
    stable_update: bool = Field(True, description="Use stable ELBO update")
```

**Source:** SVI setup in `hierarchical_exponential_smoothing.ipynb` (Adam, lr=0.03, 15k steps, AutoDiagonalNormal) and `numpyro_forecasting_univariate.ipynb` (Adam, lr=0.005, 50k steps, AutoNormal).

## Result Containers

### `ForecastResult`

```python
from typing import NamedTuple
from jaxtyping import Float, Array

class ForecastResult(NamedTuple):
    """Container for forecast output."""

    samples: dict[str, Float[Array, "..."]]
    """Raw posterior predictive samples keyed by site name."""

    idata: "az.InferenceData | None" = None
    """Optional ArviZ InferenceData with posterior_predictive group."""
```

**Why NamedTuple:** Immutable, unpacking-friendly, works with JAX tree utilities. The `samples` dict mirrors what `Predictive` returns in all source notebooks:

```python
# From croston_numpyro.ipynb
predictive = Predictive(model=model, posterior_samples=samples, return_sites=["z_forecast", "p_inv_forecast", "forecast"])
return predictive(rng_key, *model_args)
```

### `CVResult`

```python
class CVResult(NamedTuple):
    """Container for cross-validation output."""

    forecasts: "xr.Dataset"
    """Concatenated posterior predictive forecasts across folds, indexed by time."""

    metrics: dict[str, Float[Array, "..."]]
    """Per-fold and aggregate metric values."""

    n_folds: int
    """Number of CV folds executed."""
```

**Source:** The CV functions in `croston_numpyro.ipynb`, `tsb_numpyro.ipynb`, and `zi_tsb_numpyro.ipynb` all return `xr.Dataset` by concatenating per-fold `arviz.InferenceData` objects:

```python
return xr.concat(
    [x["posterior_predictive"] for x in forecast_list],
    dim=("t"),
)
```

`CVResult` wraps this pattern with accompanying metrics.

## Type Aliases

```python
from jaxtyping import Float, Array

# Common array shapes used in type hints
TimeSeries = Float[Array, "t_max"]
BatchTimeSeries = Float[Array, "t_max *batch"]
PanelTimeSeries = Float[Array, "t_max n_series"]
Samples = dict[str, Float[Array, "..."]]
RNGKey = Array
```

These provide readable type hints throughout the codebase without introducing runtime overhead. Combined with `jaxtyping` + `beartype` (see [09-dev-stack.md](09-dev-stack.md)), they enable shape-checked function signatures as seen in `var_numpyro.ipynb`:

```python
def model(y: Float[Array, "time vars"], n_lags: int, future: int = 0) -> None: ...
```

## Batch Dimension Design

A core design principle: **every model works on both a single time series and a batch of time series without code changes.**

### How it works

1. **Components** use `Float[Array, "..."]` for state and parameters — the ellipsis absorbs batch dimensions. A level transition that works on a scalar also works on a `(n_series,)` vector via JAX broadcasting.

2. **Models** accept `y: Float[Array, "t_max *batch"]`. When `*batch` is empty, it's univariate. When it's `(n_series,)`, it's panel data. The `scan` loop runs over `t_max` and all batch dimensions are handled by broadcasting within each step.

3. **Hierarchical models** use `numpyro.plate("series", n_series)` to sample per-series parameters. Non-hierarchical models can use `jax.vmap` to fit independent models across series.

4. **Inference helpers** are batch-agnostic — they forward arrays to the model without reshaping.

### Example: univariate to panel

```python
# Univariate — works as-is
ucm_model(y_single, future=12)  # y_single.shape == (100,)

# Panel — same function, same call pattern
ucm_model(y_panel, future=12)   # y_panel.shape == (100, 50)
```

No `vmap` wrapper needed for the common case. The scan loop and all components broadcast naturally.
