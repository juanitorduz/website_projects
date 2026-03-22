# 03 — Core Abstractions

## Design Philosophy

**Protocols over base classes.** JAX's functional model (pure functions, JIT compilation, `vmap`) is incompatible with traditional OOP patterns. NumPyro models are plain functions — the package should respect that. We use Python `Protocol` classes for type checking and `NamedTuple`s for structured return values.

## `ModelFn` Protocol

Every forecasting model follows this signature:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class ModelFn(Protocol):
    def __call__(self, y: Array, *args, *, future: int = 0, **kwargs) -> None: ...
```

**Contract:**

- `y` — observed time series as a JAX array. Shape `(time,)` for univariate or `(time, *batch)` for panel/multi-series. All models must handle both shapes seamlessly — components broadcast over trailing batch dimensions.
- `*args` — model-specific positional arguments (e.g., `n_seasons`, `n_lags`, covariates).
- `future` — number of future time steps to forecast. This is keyword-only (`future=...`) in public APIs and examples to avoid positional ambiguity across model families. When `future=0`, the model only conditions on observations. When `future > 0`, it produces `numpyro.deterministic` forecast sites.
- `priors` — optional `dict[str, Prior]` for prior overrides. Models merge user-provided priors with their `DEFAULT_PRIORS` via shallow dict merge so users only override what they need.
- `**kwargs` — additional model-specific configuration overrides.
- Returns `None` (side effects via NumPyro's effect system).

**Rationale:** This matches the existing pattern across most source notebooks:

```python
# From exponential_smoothing_numpyro.ipynb
def level_model(y: ArrayImpl, *, future: int = 0) -> None: ...
def holt_winters_model(
    y: ArrayImpl, n_seasons: int, *, future: int = 0, damped: bool = False
) -> None: ...

# From var_numpyro.ipynb
def model(y: Float[Array, "time vars"], n_lags: int, *, future: int = 0) -> None: ...

# From numpyro_forecasting_univariate.ipynb (covariate-first variant, see note below)
def model(covariates: Float[Array, "time feature_dim"], y=None) -> None: ...
```

The protocol is intentionally loose — it type-checks the common `(y, ..., future=0)` pattern without restricting model-specific arguments. The shape contract is seamless: a single model interface covers univariate, panel, and multivariate use cases by treating all non-time axes as trailing structure handled by JAX broadcasting or model-specific linear algebra.

**Covariate convention.** Models that accept exogenous covariates use a consistent naming convention inspired by [Pyro's `ForecastingModel.model(zero_data, covariates)`](https://docs.pyro.ai/en/dev/_modules/pyro/contrib/forecast/forecaster.html#ForecastingModel.model):

- `covariates: Float[Array, "time n_features *batch"] | None = None` — observed covariate matrix covering the training period.
- `future_covariates: Float[Array, "future n_features *batch"] | None = None` — covariate values for the forecast horizon (must be provided when `future > 0` and `covariates` is not None).

Both are keyword-only arguments to avoid positional ambiguity. All models that accept covariates (UCM with regression, SARIMAX, DeepAR, HSGP-based) follow this convention. DeepAR additionally takes a pre-built `flax.nnx.Module` (`rnn`) as its second positional argument — architecture choices belong to NN construction, not the model signature — giving a `(y, rnn, ..., future=0)` contract. The original `local_level_fourier_model` (from `numpyro_forecasting_univariate.ipynb`) used a covariate-first `(covariates, y=None)` signature — this pattern is now subsumed by the UCM with trigonometric seasonality and the `covariates` argument.

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
    optimizer: str = Field("Adam", description="Optax optimizer name")
    guide: str = Field("AutoNormal", description="AutoGuide class name")
    stable_update: bool = Field(True, description="Use stable ELBO update")

    def build_optimizer(self) -> "optax.GradientTransformation":
        """Resolve optimizer name to an optax optimizer instance.

        Uses ``self.learning_rate`` as the learning rate.
        """
        import optax
        optimizer_cls = getattr(optax, self.optimizer.lower(), None)
        if optimizer_cls is None:
            raise ValueError(f"Unknown optax optimizer: {self.optimizer!r}")
        return optimizer_cls(self.learning_rate)

    def build_guide(self, model: "Callable") -> "AutoGuide":
        """Resolve guide name to a NumPyro AutoGuide instance.

        Parameters
        ----------
        model
            The NumPyro model function to wrap.
        """
        from numpyro.infer.autoguide import AutoNormal, AutoDiagonalNormal
        guides = {"AutoNormal": AutoNormal, "AutoDiagonalNormal": AutoDiagonalNormal}
        guide_cls = guides.get(self.guide)
        if guide_cls is None:
            raise ValueError(f"Unknown guide: {self.guide!r}")
        return guide_cls(model)
```

**Source:** SVI setup in `hierarchical_exponential_smoothing.ipynb` (Adam, lr=0.03, 15k steps, AutoDiagonalNormal) and `numpyro_forecasting_univariate.ipynb` (Adam, lr=0.005, 50k steps, AutoNormal).

## Prior Specification (`core/prior.py`)

### `Prior`

A Pydantic `BaseModel` that wraps NumPyro distributions for ergonomic prior injection. Priors can be nested to express hyperprior trees. The class is intentionally simple — plate contexts and reparameterization are model-level concerns handled by NumPyro's native `numpyro.plate` and `numpyro.handlers.reparam` / `LocScaleReparam`.

```python
from __future__ import annotations

from pydantic import BaseModel, Field
import numpyro
import numpyro.distributions as dist
from jaxtyping import Array

class Prior(BaseModel):
    """Lightweight wrapper around a NumPyro distribution for prior injection.

    Design inspired by the ``Prior`` class in
    `PyMC-Extras <https://github.com/pymc-devs/pymc-extras/blob/main/pymc_extras/prior.py>`_
    (originally developed in PyMC-Marketing), adapted here for NumPyro/JAX.

    Parameters
    ----------
    distribution
        Name of a ``numpyro.distributions`` class (e.g. ``"Beta"``, ``"Normal"``).
    params
        Distribution parameters. Values can be floats, ints, or nested ``Prior``
        instances for hierarchical (hyperprior) trees.
    """

    distribution: str
    params: dict[str, float | int | Prior] = Field(default_factory=dict)

    def sample(self, name: str) -> Array:
        """Resolve nested priors and call ``numpyro.sample``.

        Child ``Prior`` params are sampled first (with scoped names),
        then the parent distribution is instantiated with the resolved
        values.
        """
        resolved = {}
        for k, v in self.params.items():
            if isinstance(v, Prior):
                resolved[k] = v.sample(f"{name}_{k}")
            else:
                resolved[k] = v
        dist_cls = getattr(dist, self.distribution, None)
        if dist_cls is None:
            raise ValueError(
                f"Unknown distribution: {self.distribution!r}. "
                f"Must be a class in numpyro.distributions."
            )
        return numpyro.sample(name, dist_cls(**resolved))
```

### Limitations

`Prior.sample()` resolves the full tree in the current NumPyro scope. For multi-level hierarchies requiring intermediate plates (e.g., 3-level hierarchies with `group_mapping`), the model function must walk the tree manually — see the hierarchical example in [04-models-module.md](04-models-module.md).

### `sample_hierarchical`

A standalone helper in `core/prior.py` that encapsulates the 3-level (global -> group -> series) hierarchical sampling pattern. Every panel-capable model that supports `group_mapping` needs this logic, so centralising it avoids fragile, repetitive tree-walking code in each model function.

```python
def sample_hierarchical(
    prior: Prior,
    name: str,
    group_mapping: Array,
    n_groups: int,
    n_series: int,
) -> Array:
    """Sample a 3-level hierarchical prior: global -> group -> series.

    1. Hyperpriors (leaf ``Prior`` children) are sampled at global scope.
    2. Group-level params are sampled inside ``plate("groups", n_groups)``.
    3. Series-level values are produced by indexing group params via
       ``group_mapping``.

    Parameters
    ----------
    prior
        A ``Prior`` whose ``params`` contain nested ``Prior`` children
        (the hyperprior tree).
    name
        Base site name for ``numpyro.sample`` calls.
    group_mapping
        Integer array of shape ``(n_series,)`` mapping each series to
        its group index.
    n_groups
        Number of groups.
    n_series
        Number of series.

    Returns
    -------
    Array
        Sampled values of shape ``(n_series,)``.
    """
    # 1. Sample hyperpriors at global scope
    hyperparams = {}
    for k, v in prior.params.items():
        if isinstance(v, Prior):
            hyperparams[k] = v.sample(f"{name}_{k}")
        else:
            hyperparams[k] = v

    # 2. Sample group-level params
    with numpyro.plate("groups", n_groups):
        group_params = numpyro.sample(
            f"{name}_group", getattr(dist, prior.distribution)(**hyperparams)
        )

    # 3. Index into series
    return group_params[group_mapping]
```

When `group_mapping` is `None`, models fall back to `prior.sample(name)` directly inside a `plate("series", n_series)`.

### Usage

```python
# Flat prior (equivalent to dist.Beta(1, 1) in the old kwargs style)
Prior("Beta", params={"concentration1": 1.0, "concentration0": 1.0})

# Hyperprior tree — parent samples after children resolve
Prior(
    "Beta",
    params={
        "concentration1": Prior("HalfNormal", params={"scale": 8.0}),
        "concentration0": Prior("HalfNormal", params={"scale": 8.0}),
    },
)
```

### `DEFAULT_PRIORS` pattern

Each model defines a module-level `*_DEFAULT_PRIORS: dict[str, Prior]` constant. Model functions accept `priors: dict[str, Prior] | None = None` and merge user overrides via `merge_priors`, which validates that all user-supplied keys are known:

```python
def merge_priors(
    defaults: dict[str, Prior],
    overrides: dict[str, Prior] | None,
) -> dict[str, Prior]:
    """Merge user prior overrides with defaults, validating keys.

    Raises
    ------
    ValueError
        If ``overrides`` contains keys not present in ``defaults``.
    """
    if overrides:
        unknown = set(overrides) - set(defaults)
        if unknown:
            raise ValueError(
                f"Unknown prior keys: {unknown}. Valid keys: {sorted(defaults)}"
            )
    return {**defaults, **(overrides or {})}
```

This helper lives in `core/prior.py` alongside the `Prior` class and is used by every model:

```python
from probcast.core.prior import Prior, merge_priors

UC_DEFAULT_PRIORS: dict[str, Prior] = {
    "level_smoothing": Prior("Beta", params={"concentration1": 1.0, "concentration0": 1.0}),
    "level_init": Prior("Normal", params={"loc": 0.0, "scale": 1.0}),
    "sigma": Prior("HalfNormal", params={"scale": 1.0}),
    # ... one entry per model parameter ...
}

def uc_model(y, *, future=0, priors=None, **config):
    resolved = merge_priors(UC_DEFAULT_PRIORS, priors)
    level_smoothing = resolved["level_smoothing"].sample("level_smoothing")
    ...
```

Users only override what they need. Typos in prior key names raise `ValueError` immediately instead of silently falling back to defaults. Each model documents its valid prior keys (the keys in its `DEFAULT_PRIORS`).

### Hierarchical priors

Hierarchical structure is expressed by nesting `Prior` params. The model function wraps the `prior.sample(...)` call inside the appropriate `numpyro.plate` and applies `LocScaleReparam` where needed — these are model-level concerns, not prior-level:

```python
# User passes hierarchical priors
holt_winters_model(
    y_panel, n_seasons=12, future=12,
    priors={
        "level_smoothing": Prior(
            "Beta",
            params={
                "concentration1": Prior("Gamma", params={"concentration": 8.0, "rate": 4.0}),
                "concentration0": Prior("Gamma", params={"concentration": 8.0, "rate": 4.0}),
            },
        ),
    },
)

# Inside the model function — plate and reparam are model concerns
with numpyro.plate("series", n_series):
    level_smoothing = resolved["level_smoothing"].sample("level_smoothing")
```

### Serialization

`Prior` supports `to_dict()` / `from_dict()` for reproducibility and configuration persistence:

```python
prior = Prior("Normal", params={"loc": 0.0, "scale": 1.0})
d = prior.model_dump()
# {"distribution": "Normal", "params": {"loc": 0.0, "scale": 1.0}}
Prior.model_validate(d)  # round-trips via Pydantic
```

## Encoding and Mapping (`core/encoding.py`)

Hierarchical models consume integer `group_mapping` arrays, but users may start from different dataframe backends. To keep the API backend-agnostic while preserving a Polars-like expression style, encoding helpers use **Narwhals-first** contracts (`narwhals.from_native(...)`) and return backend-neutral/JAX-ready artifacts.

**Optional dependency:** Narwhals (and scikit-learn for `LabelEncoder`) are optional dependencies, installed via `pip install probcast[dataframes]`. The encoding functions raise `ImportError` with a clear install instruction if Narwhals or scikit-learn is missing. Users who only use univariate models or pass JAX arrays directly do not need these dependencies.

### `LabelEncoderData`

```python
from pydantic import BaseModel, ConfigDict, Field
from sklearn.preprocessing import LabelEncoder
from jaxtyping import Array

class LabelEncoderData(BaseModel):
    """Container for one encoded categorical variable."""

    name: str = Field(description="Encoded column name.")
    label_encoder: LabelEncoder = Field(
        description="Fitted sklearn LabelEncoder instance."
    )
    labels: Array = Field(
        description="Encoded labels aligned with the dataframe row order."
    )
    classes: list[str] = Field(
        description="Sorted classes as learned by LabelEncoder."
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
    )
```

### Helper contracts

```python
from typing import Any
import narwhals as nw

def label_encode_column(df: Any, column: str) -> LabelEncoderData:
    """Encode one categorical column from any Narwhals-supported dataframe backend."""

def build_group_mapping(
    df: Any,
    *,
    series_col: str,
    group_col: str,
    sort_by: str | None = None,
) -> Array:
    """Build shape-(n_series,) group_mapping for panel-capable models."""

def build_levels_mapping(
    df: Any,
    *,
    higher_level_col: str,
    lower_level_col: str,
    sort_by: str | None = None,
) -> Array:
    """Build shape-(n_lower,) parent-index mapping.

    Output contract:
    - shape: ``(n_lower,)`` where ``n_lower`` is number of unique lower-level ids;
    - dtype: ``int32`` (JAX-compatible integer array);
    - index basis: position ``i`` corresponds to the i-th lower-level class in
      deterministic lower-level class order;
    - value semantics: ``mapping[i]`` is the integer class index of the parent
      (higher-level) category for that lower-level class.
    """
```

### Validation and determinism rules

- Inputs are accepted from any Narwhals-supported backend (`pandas`, `polars`, `pyarrow`, etc.) via `nw.from_native`.
- Required columns must exist; missing columns raise `ValueError` with explicit column names.
- Each lower-level id must map to exactly one higher-level id (one-to-one parent relationship). Violations raise `ValueError`.
- Mapping outputs are deterministic: classes and mapping indices are derived with stable ordering (`LabelEncoder` ordering plus explicit row-order policy via `sort_by` when provided).
- Returned mapping arrays are `int32` and directly consumable as `group_mapping` in model calls.
- `build_group_mapping(...)` output contract:
  - shape is `(n_series,)` aligned with deterministic `series_col` class order;
  - values are integer group ids in deterministic `group_col` class order.
- `build_levels_mapping(...)` output contract:
  - shape is `(n_lower,)` aligned with deterministic `lower_level_col` class order;
  - values are parent indices in deterministic `higher_level_col` class order.
- `sort_by` policy:
  - when provided, ordering-sensitive preprocessing must be computed on data sorted by `sort_by`;
  - when omitted, class ordering is still deterministic via encoder class ordering.
- callback/evaluation note:
  - callback-style mapping preparation (for example CV fold callbacks) expects eager dataframe inputs; lazy query plans must be materialized before encoding/mapping helpers are called.
- required error cases:
  - lower-level class mapped to multiple higher-level classes (`ValueError`);
  - nulls in required mapping columns unless explicitly pre-cleaned (`ValueError`);
  - empty post-filter input after required column selection (`ValueError`).

### Unit-test contract (`tests/test_core/test_encoding.py`)

`core/encoding.py` must be covered by deterministic unit tests with backend parity:

- `test_label_encode_column_backend_parity`:
  - run on equivalent pandas and Polars dataframes;
  - assert identical `classes` and `labels` from `LabelEncoderData`.
- `test_label_encode_column_missing_column_raises`:
  - missing `column` raises `ValueError` with the missing name in the message.
- `test_build_group_mapping_one_parent_per_series`:
  - valid one-to-one series->group relationships produce integer mapping of shape `(n_series,)`.
- `test_build_group_mapping_many_parents_raises`:
  - one series mapped to multiple groups raises `ValueError`.
- `test_build_levels_mapping_deterministic_with_sort_by`:
  - repeated calls with the same input and `sort_by` produce identical mappings.
- `test_build_levels_mapping_parent_child_integrity`:
  - every lower-level id maps to exactly one higher-level id; violations raise `ValueError`.

## Result Containers

### `ForecastResult`

```python
from typing import NamedTuple
from jaxtyping import Float, Array
import xarray as xr

class ForecastResult(NamedTuple):
    """Container for forecast output."""

    samples: dict[str, Float[Array, "..."]]
    """Raw posterior predictive samples keyed by site name."""

    datatree: "xr.DataTree"
    """xarray DataTree (ArviZ >= 1.0.0) with posterior_predictive group.
    Created via ``arviz.from_numpyro``."""

    coords: dict | None = None
    """Coordinate metadata (e.g. time index, series ids) attached to the DataTree."""

    dims: dict | None = None
    """Dimension names for each site (e.g. ``{"pred": ["time", "series"]}``)."""
```

**Why NamedTuple:** Immutable, unpacking-friendly, works with JAX tree utilities. The `samples` dict mirrors what `Predictive` returns in all source notebooks:

```python
# From croston_numpyro.ipynb
predictive = Predictive(model=model, posterior_samples=samples, return_sites=["z_forecast", "p_inv_forecast", "forecast"])
return predictive(rng_key, *model_args)
```

**ArviZ >= 1.0.0 note:** The `datatree` field uses `xarray.DataTree` (the standard container for ArviZ >= 1.0.0, replacing the legacy `arviz.InferenceData`). It is created via `arviz.from_numpyro` and can be consumed by all ArviZ plotting and diagnostic functions.

### `CVResult`

```python
class CVResult(NamedTuple):
    """Container for cross-validation output.
    """

    forecasts: "xr.DataTree"
    """Concatenated posterior predictive forecasts across folds as a DataTree,
    indexed by time. Created via ``arviz.from_numpyro``."""

    metrics: dict[str, Float[Array, "..."]]
    """Per-fold and aggregate metric values."""

    n_folds: int
    """Number of CV folds executed."""

    fold_info: list[dict[str, int]]
    """Per-fold metadata dictionaries.

    Minimum keys per fold:
    - ``fold_idx``
    - ``horizon``
    - ``train_end_idx``
    - ``train_size``
    - ``forecast_origin_idx``
    """
```

**Source:** The CV functions in `croston_numpyro.ipynb`, `tsb_numpyro.ipynb`, and `zi_tsb_numpyro.ipynb` all return concatenated per-fold results. In ArviZ >= 1.0.0, `xarray.DataTree` replaces `arviz.InferenceData` as the standard container:

```python
return xr.concat(
    [x["posterior_predictive"] for x in forecast_list],
    dim=("t"),
)
```

`CVResult` wraps this pattern with accompanying metrics and fold metadata.

## Type Aliases

```python
from jaxtyping import Float, Array

# Common array shapes used in type hints
TimeSeries = Float[Array, "time"]
BatchTimeSeries = Float[Array, "time *batch"]
PanelTimeSeries = Float[Array, "time n_series"]
Samples = dict[str, Float[Array, "..."]]
RNGKey = Array
GroupMapping = Array
```

These provide readable type hints throughout the codebase without introducing runtime overhead. Combined with `jaxtyping` + `beartype` (see [09-dev-stack.md](09-dev-stack.md)), they enable shape-checked function signatures as seen in `var_numpyro.ipynb`:

```python
def model(y: Float[Array, "time vars"], n_lags: int, *, future: int = 0) -> None: ...
```

## Batch Dimension Design

A core design principle: **every model works on both a single time series and a batch of time series without code changes.**

### How it works

1. **Components** use `Float[Array, "..."]` for state and parameters — the ellipsis absorbs batch dimensions. A level transition that works on a scalar also works on a `(n_series,)` vector via JAX broadcasting.

2. **Models** accept `y: Float[Array, "time *batch"]`. When `*batch` is empty, it's univariate. When it's `(n_series,)`, it's panel data. The `scan` loop runs over the time dimension and all batch dimensions are handled by broadcasting within each step.

3. **Hierarchical models** use `numpyro.plate("series", n_series)` to sample per-series parameters. Non-hierarchical models can use `jax.vmap` to fit independent models across series.

4. **Inference helpers** are batch-agnostic — they forward arrays to the model without reshaping.

### Example: univariate to panel

```python
# Univariate — works as-is
uc_model(y_single, future=12)  # y_single.shape == (100,)

# Panel — same function, same call pattern
uc_model(y_panel, future=12)   # y_panel.shape == (100, 50)
```

No `vmap` wrapper needed for the common case. The scan loop and all components broadcast naturally.
