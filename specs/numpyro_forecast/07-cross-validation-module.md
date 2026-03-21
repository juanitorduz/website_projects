# 07 — Cross-Validation Module

## Overview

Time-series cross-validation with rolling or expanding windows. Designed to work with any `ModelFn` and handle data preparation for specialized models (Croston/TSB) via a callback.

## Validation Policy (No Leakage)

All CV routines must enforce leakage-safe semantics:
- training folds may only use observations and covariates available up to the forecast origin;
- `prepare_data_fn` must consume training-slice inputs only (never full-series arrays);
- when exogenous regressors are used, forecast horizons must use aligned `future_covariates` for that fold only;
- metrics are computed strictly on held-out horizon windows.

## Time-Slice CV (`cv/time_series.py`)

### `time_slice_cv`

Rolling-origin cross-validation with fixed training window and configurable forecast horizon.

```python
def time_slice_cv(
    rng_key: Array,
    model: Callable,
    y: Float[Array, "..."],
    n_splits: int,
    inference_params: MCMCParams | SVIParams,
    *model_args,
    horizon: int = 1,
    prepare_data_fn: Callable[..., tuple[tuple, dict]] | None = None,
    return_sites: list[str] | None = None,
    metrics_fn: Callable | None = None,
    **model_kwargs,
) -> CVResult:
    """Rolling-origin time-slice cross-validation.

    For each fold i in range(n_splits):
        1. Split: y_train = y[:-(n_splits - i)], y_test = y[len(y_train):len(y_train)+horizon]
        2. Prepare: if prepare_data_fn is provided, transform y_train into model args
        3. Infer: run MCMC or SVI on training data
        4. Forecast: generate horizon-step-ahead predictions
        5. Score: compute metrics on y_test

    Parameters
    ----------
    rng_key
        JAX PRNG key (split internally per fold).
    model
        Model function following the ModelFn protocol.
    y
        Full time series (train + test combined).
    n_splits
        Number of CV folds (rolling origins).
    inference_params
        MCMCParams for MCMC or SVIParams for SVI.
    *model_args
        Additional positional arguments for the model (e.g., n_seasons).
    horizon
        Number of steps to forecast at each fold.
    prepare_data_fn
        Optional callback to transform training data into model arguments.
        Signature: ``(y_train, **fold_info) -> tuple[tuple, dict]``
        where the return value is ``(model_args_tuple, model_kwargs_dict)``,
        and ``fold_info`` includes ``fold_idx``, ``train_end_idx``, ``horizon``.
        For exogenous models, the callback should slice covariates from
        the full arrays passed via ``model_kwargs``.
        Used for Croston/TSB where raw y must be decomposed into z, p_inv.
    return_sites
        Sites to return from Predictive.
    metrics_fn
        Optional function ``(y_true, y_pred_samples) -> dict[str, float]``
        to compute per-fold metrics. Defaults to ``crps_empirical``.
    **model_kwargs
        Keyword arguments forwarded to the model.

    Returns
    -------
    CVResult
        Contains forecasts (xr.DataTree with a ``posterior_predictive``
        Dataset concatenated across folds), per-fold metrics, and n_folds.
    """
```

**Source pattern** (from `croston_numpyro.ipynb`, `tsb_numpyro.ipynb`, `zi_tsb_numpyro.ipynb`):

```python
# Current: model-specific CV functions duplicated across notebooks
def croston_time_slice_cross_validation(rng_key, y, n_splits, inference_params):
    forecast_list = []
    for i in tqdm(range(n_splits)):
        y_train = y[:-(n_splits - i)]
        z = y_train[y_train != 0]
        p_idx = jnp.flatnonzero(y_train).astype(jnp.float32)
        p = jnp.diff(p_idx, prepend=-1)
        p_inv = 1 / p
        # ... inference + forecast ...

# Package: generic CV with prepare_data_fn callback
def prepare_intermittent_data(y_train, **fold_info):
    z = y_train[y_train != 0]
    p_idx = jnp.flatnonzero(y_train).astype(jnp.float32)
    p = jnp.diff(p_idx, prepend=-1)
    p_inv = 1 / p
    return (z, p_inv), {}

cv_result = time_slice_cv(
    rng_key, croston_model, y, n_splits=20,
    inference_params=params, prepare_data_fn=prepare_intermittent_data,
)
```

**SARIMAX CV example** — slicing exogenous regressors per fold:

```python
def prepare_sarimax_data(y_train, *, covariates_full, future_covariates_full, fold_idx, train_end_idx, horizon, **_):
    covariates_train = covariates_full[:train_end_idx]
    future_covariates = future_covariates_full[train_end_idx:train_end_idx + horizon]
    return (y_train,), {"covariates": covariates_train, "future_covariates": future_covariates}
```

### `expanding_window_cv`

Expanding window variant where training size grows at each fold.

```python
def expanding_window_cv(
    rng_key: Array,
    model: Callable,
    y: Float[Array, "..."],
    min_train_size: int,
    step: int,
    inference_params: MCMCParams | SVIParams,
    *model_args,
    horizon: int = 1,
    prepare_data_fn: Callable[..., tuple[tuple, dict]] | None = None,
    return_sites: list[str] | None = None,
    metrics_fn: Callable | None = None,
    **model_kwargs,
) -> CVResult:
    """Expanding-window cross-validation.

    Parameters
    ----------
    min_train_size
        Minimum number of observations in the first training window.
    step
        Number of observations to advance between folds.
    """
```

**Relationship to `time_slice_cv`:** `expanding_window_cv` is parameterized by `min_train_size + step` rather than `n_splits`. Both share the same internal loop structure but differ in how `y_train` bounds are computed.

## `CVResult` Output Format

```python
class CVResult(NamedTuple):
    forecasts: xr.DataTree
    metrics: dict[str, Float[Array, "..."]]
    n_folds: int
```

`CVResult` always includes `xr.DataTree` forecasts as part of the core package contract.

### `forecasts` structure

An `xr.DataTree` (ArviZ >= 1.0.0 container, replacing `arviz.InferenceData`) with dimensions `(chain, draw, t)` where `t` indexes the forecast origins. This matches the pattern from all three intermittent demand notebooks:

```python
xr.concat(
    [x["posterior_predictive"] for x in forecast_list],
    dim="t",
)
```

### `metrics` structure

```python
{
    "crps": jnp.array([...]),       # Per-fold CRPS values, shape (n_folds,)
    "crps_mean": jnp.float32(...),  # Aggregate mean CRPS
    "mae": jnp.array([...]),        # Per-fold MAE (optional)
    "crps_by_horizon": jnp.array([...]),      # Shape (n_folds, horizon)
    "coverage80_by_horizon": jnp.array([...]) # Shape (n_folds, horizon), optional
}
```

Fold metadata should also include forecast origin indices and effective train sizes to make leakage audits reproducible.

## Data Preparation (`cv/prepare.py`)

### `train_test_split`

```python
def train_test_split(
    y: Float[Array, "time *rest"],
    n_test: int,
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """Split a time series into train and test by slicing the last n_test steps."""
    return y[:-n_test], y[-n_test:]
```

### `prepare_hierarchical_mapping`

```python
def prepare_hierarchical_mapping(
    group_labels: Sequence[str],
) -> tuple[Float[Array, " n_series"], int]:
    """Encode group labels as integer indices.

    Parameters
    ----------
    group_labels
        Group membership for each series (e.g., state names).

    Returns
    -------
    mapping_idx
        Integer array mapping each series to its group index.
    n_groups
        Number of unique groups.
    """
```

**Source:** `LabelEncoder` usage in `hierarchical_exponential_smoothing.ipynb`.

## `prepare_data_fn` Callbacks

Pre-built callbacks for common data transformations (live in `cv/prepare.py`):

```python
def prepare_intermittent_data(y_train, **fold_info):
    """Decompose intermittent series into demand sizes and period inverses."""
    z = y_train[y_train != 0]
    p_idx = jnp.flatnonzero(y_train).astype(jnp.float32)
    p = jnp.diff(p_idx, prepend=-1)
    p_inv = 1 / p
    return (z, p_inv), {}

def prepare_tsb_data(y_train, **fold_info):
    """Trim leading zeros and compute initial z0, p0."""
    y_train_trim = jnp.trim_zeros(y_train, trim="f")
    p_idx = jnp.flatnonzero(y_train)
    p_diff = jnp.diff(p_idx, prepend=-1)
    z0 = y_train[p_idx[0]]
    p0 = 1 / p_diff.mean()
    return (y_train_trim, z0, p0), {}
```

**Source:** `get_model_args` in `tsb_numpyro.ipynb` and `zi_tsb_numpyro.ipynb`. All data preparation callbacks and helpers live in `cv/prepare.py`.

`prepare_data_fn` callbacks must validate minimum train-length requirements for enabled model structure (for example, seasonal period, AR lag order, and HSGP basis assumptions) and fail fast with clear errors if a fold is too short.

## Usage Example

```python
from probcast.cv import time_slice_cv
from probcast.cv.prepare import prepare_tsb_data
from probcast.models import tsb_model
from probcast.core import MCMCParams

params = MCMCParams(num_warmup=1_000, num_samples=1_000, num_chains=2)

cv_result = time_slice_cv(
    rng_key,
    tsb_model,
    y,
    n_splits=20,
    inference_params=params,
    prepare_data_fn=prepare_tsb_data,
    return_sites=["ts_forecast"],
)

print(f"Mean CRPS: {cv_result.metrics['crps_mean']:.3f}")
```
