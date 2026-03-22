# 12 — Quickstart Contracts (Phase 1-4)

## Purpose

This spec defines canonical end-to-end quickstart flows for the core release model families:

- UCM and local aliases
- Exponential smoothing wrappers
- SARIMAX
- Intermittent demand (Croston, TSB, ZI-TSB)
- ARMA
- VAR (+ IRF)

The goal is contract clarity: every quickstart uses the same step order and explicit data/API rules.

## Canonical End-to-End Flow

For all covered model families, the expected usage flow is:

1. Prepare arrays with contract-compliant shapes.
2. Define the model call and model-specific kwargs.
3. Run inference (`run_mcmc` or `run_svi` where applicable).
4. Generate forecasts with explicit `return_sites`.
5. Run diagnostics and metrics.
6. Plot from `ForecastResult.datatree` / `CVResult.forecasts`.

## Global Data and API Contracts

### Array shape contracts

- **Univariate target:** `y.shape == (time,)`
- **Panel target:** `y.shape == (time, n_series)`
- **Multivariate VAR target:** `y.shape == (time, n_vars)`
- **Exogenous regressors:** `covariates.shape == (time, n_features, *batch)`
- **Future regressors:** `future_covariates.shape == (future, n_features, *batch)`
- **Hierarchical mapping:** `group_mapping.shape == (n_series,)` with integer group ids

### Forecast output contracts

- `ForecastResult.samples` is the raw site dictionary from `Predictive`.
- `ForecastResult.datatree` is always present (`xr.DataTree`).
- `CVResult.forecasts` is always present (`xr.DataTree`).
- Core naming:
  - `pred`: full observed+future sampled site when model uses a direct observation site.
  - `y_forecast`: **standardized primary forecast site across all models** — forecast slice (`future` only) exposed as deterministic. `forecast()` defaults to `return_sites=["y_forecast"]`.

### Inference contracts

- `future` is always keyword-only.
- `run_mcmc(..., model_kwargs=...)` and `run_svi(..., model_kwargs=...)` are the canonical forwarding pattern.
- Use `check_diagnostics()` on MCMC runs before treating forecasts as baseline-ready.
- Fit-stage kwargs should not include forecast horizon. Pass horizon via `forecast(..., future=horizon, ...)`.

### Narwhals encoding contracts (hierarchical workflows)

Use Narwhals-first helpers so mapping creation is backend-agnostic:

```python
from probcast.core import label_encode_column, build_group_mapping, build_levels_mapping

# `data_native` may be pandas/polars/pyarrow/etc.
sku_encoder = label_encode_column(data_native, "sku_id")
state_encoder = label_encode_column(data_native, "state")
group_mapping = build_group_mapping(
    data_native,
    series_col="sku_id",
    group_col="state",
    sort_by="date",
)
```

#### Pandas input example

```python
import pandas as pd
from probcast.core import build_group_mapping
from probcast.models import holt_winters_model

df_pd = pd.DataFrame(
    {
        "date": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
        "sku_id": ["sku_a", "sku_a", "sku_b", "sku_b"],
        "state": ["CA", "CA", "NY", "NY"],
    }
)

group_mapping = build_group_mapping(
    df_pd,
    series_col="sku_id",
    group_col="state",
    sort_by="date",
)

fit_model_kwargs = {"group_mapping": group_mapping}
# run_mcmc(..., holt_winters_model, ..., model_kwargs=fit_model_kwargs)
# forecast(..., future=12, model_kwargs=fit_model_kwargs)
```

#### Polars input example

```python
import polars as pl
from probcast.core import build_group_mapping
from probcast.models import arma_model

df_pl = pl.DataFrame(
    {
        "date": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
        "sku_id": ["sku_a", "sku_a", "sku_b", "sku_b"],
        "state": ["CA", "CA", "NY", "NY"],
    }
)

group_mapping = build_group_mapping(
    df_pl,
    series_col="sku_id",
    group_col="state",
    sort_by="date",
)

fit_model_kwargs = {"order": (1, 1), "group_mapping": group_mapping}
# run_mcmc(..., arma_model, ..., model_kwargs=fit_model_kwargs)
# forecast(..., future=12, model_kwargs=fit_model_kwargs)
```

Both examples must produce equivalent integer mappings for equivalent tabular content.

Multi-level parent-child mapping (for hierarchical priors with intermediate plates):

```python
# level3 -> sku mapping and level2 -> level3 mapping
level3_to_sku = build_levels_mapping(
    data_native,
    higher_level_col="level3_category",
    lower_level_col="sku_id",
    sort_by="date",
)
level2_to_level3 = build_levels_mapping(
    data_native,
    higher_level_col="level2_category",
    lower_level_col="level3_category",
    sort_by="date",
)
```

Mapping outputs are integer arrays and plug directly into model kwargs such as `group_mapping`.

## Shared E2E Skeleton

```python
import jax
from probcast.core import MCMCParams
from probcast.inference import run_mcmc, forecast, check_diagnostics
from probcast.metrics import crps_empirical
from probcast.plotting import plot_forecast

rng_key = jax.random.PRNGKey(42)
params = MCMCParams(num_warmup=1_000, num_samples=1_000, num_chains=2)

fit_model_kwargs = {}
# Default: forecast uses the same kwargs as fit.
forecast_model_kwargs = fit_model_kwargs

# 1) fit
mcmc = run_mcmc(
    rng_key,
    model_fn,
    params,
    *model_args,
    model_kwargs=fit_model_kwargs,  # optional
)

# 2) diagnose
diag = check_diagnostics(mcmc)

# 3) forecast
result = forecast(
    rng_key,
    model_fn,
    mcmc.get_samples(),
    *model_args,
    future=horizon,
    model_kwargs=forecast_model_kwargs,
    return_sites=return_sites,
    coords=coords,
    dims=dims,
)

# 4) score
pred_mean = result.samples[return_sites[0]].mean(axis=0)
score = crps_empirical(result.samples[return_sites[0]], y_test)

# 5) plot
ax = plot_forecast(result.datatree, y_train=y_train, y_test=y_test, var_name=return_sites[0])
```

---

## UCM + Local Alias Quickstart

### Step 1 — data contract

```python
# Univariate
y = y_train  # shape (time,)

# Panel
# y = y_panel  # shape (time, n_series)
# group_mapping = group_ids  # shape (n_series,)
```

### Step 2 — model call contract

```python
from probcast.models import uc_model, local_level_model, local_linear_trend_model

# Canonical direct UCM call
model_fn = uc_model
model_args = (y,)
fit_model_kwargs = {
    "level": True,
    "trend": "local linear",
    "seasonal": 12,              # additive HW seasonality
    # "freq_seasonal": [{"period": 365.25, "harmonics": 6}],  # trigonometric alternative
    # "group_mapping": group_mapping,  # panel only
}

# Local alias equivalent
# model_fn = local_linear_trend_model
# fit_model_kwargs = {}
# forecast_model_kwargs = fit_model_kwargs
```

### Step 3-6 — inference, forecast, diagnostics, metrics, plotting

```python
forecast_model_kwargs = fit_model_kwargs
return_sites = ["y_forecast"]
coords = {"time": time_index_future}
dims = {"y_forecast": ["time"]}
```

---

## Exponential Smoothing Quickstart

### Step 1 — data contract

```python
y = y_train  # shape (time,) or (time, n_series)
```

### Step 2 — model call contract

```python
from probcast.models import holt_winters_model

model_fn = holt_winters_model
model_args = (y, 12)  # n_seasons
fit_model_kwargs = {
    "damped": False,  # set True for damped Holt-Winters
    # "group_mapping": group_mapping,  # panel hierarchy
}
```

### Step 3-6

```python
forecast_model_kwargs = fit_model_kwargs
return_sites = ["y_forecast"]
coords = {"time": time_index_future}
dims = {"y_forecast": ["time"]}
```

---

## SARIMAX Quickstart

### Step 1 — data contract

```python
y = y_train  # (time,) or (time, n_series)
covariates = x_train  # (time, n_features, *batch)
future_covariates = x_future  # (future, n_features, *batch)
```

### Step 2 — model call contract

```python
from probcast.models import sarimax_model

model_fn = sarimax_model
model_args = (y,)
fit_model_kwargs = {
    "order": (1, 1, 1),
    "seasonal_order": (1, 0, 1, 12),
    "covariates": covariates,
}
forecast_model_kwargs = {
    **fit_model_kwargs,
    "future_covariates": future_covariates,
}
```

### Step 3-6

```python
return_sites = ["y_forecast"]
forecast_model_kwargs = {
    **fit_model_kwargs,
    "future_covariates": future_covariates,
}
coords = {"time": time_index_future}
dims = {"y_forecast": ["time"]}
```

---

## Intermittent Demand Quickstart

### Step 1 — data contract

```python
from probcast.cv.prepare import prepare_intermittent_data, prepare_tsb_data

# Croston
(z, p_inv), _ = prepare_intermittent_data(y_train)

# TSB / ZI-TSB
(ts_trim, z0, p0), _ = prepare_tsb_data(y_train)
```

### Step 2 — model call contracts

```python
from probcast.models import croston_model, tsb_model, zi_tsb_model

# Croston
model_fn = croston_model
model_args = (z, p_inv)
fit_model_kwargs = {}
forecast_model_kwargs = fit_model_kwargs
return_sites = ["y_forecast"]  # standardized; use ["y_forecast", "z_forecast", "p_inv_forecast"] for diagnostics

# TSB
# model_fn = tsb_model
# model_args = (ts_trim, z0, p0)
# return_sites = ["y_forecast"]

# ZI-TSB
# model_fn = zi_tsb_model
# model_args = (ts_trim, z0, p0)
# return_sites = ["y_forecast"]
```

Note: all models expose `"y_forecast"` as the primary forecast site. For Croston, sub-component diagnostics are available via `return_sites=["y_forecast", "z_forecast", "p_inv_forecast"]`.

### Step 3-6

```python
forecast_model_kwargs = fit_model_kwargs
coords = {"time": time_index_future}
dims = {site: ["time"] for site in return_sites}
```

---

## ARMA Quickstart

### Step 1 — data contract

```python
y = y_train  # (time,) or (time, n_series)
```

### Step 2 — model call contract

```python
from probcast.models import arma_model

model_fn = arma_model
model_args = (y,)
fit_model_kwargs = {
    "order": (1, 1),  # (p, q)
    # "group_mapping": group_mapping,  # panel hierarchy
}
```

### Step 3-6

```python
forecast_model_kwargs = fit_model_kwargs
return_sites = ["y_forecast"]  # add "errors" for diagnostics: ["y_forecast", "errors"]
coords = {"time": time_index_future}
dims = {"y_forecast": ["time"]}
```

---

## VAR + IRF Quickstart

### Step 1 — data contract

```python
y = y_train  # shape (time, n_vars)
```

### Step 2 — model call contract

```python
from probcast.models import var_model

model_fn = var_model
model_args = (y,)
fit_model_kwargs = {"n_lags": 2}
forecast_model_kwargs = fit_model_kwargs
```

### Step 3-6

```python
forecast_model_kwargs = fit_model_kwargs
return_sites = ["y_forecast"]  # add "irf" for impulse response: ["y_forecast", "irf"]
coords = {"time": time_index_future, "var": var_names}
dims = {"y_forecast": ["time", "var"]}
```

---

## Cross-Validation Quickstart (Canonical)

```python
from probcast.cv import time_slice_cv
from probcast.cv.prepare import prepare_intermittent_data, prepare_tsb_data
from probcast.core import MCMCParams

params = MCMCParams(num_warmup=1_000, num_samples=1_000, num_chains=2)

# Example: intermittent CV
cv_result = time_slice_cv(
    rng_key,
    croston_model,
    y,
    n_splits=20,
    inference_params=params,
    prepare_data_fn=prepare_intermittent_data,
    return_sites=["y_forecast"],
)

# Contract: always present DataTree
dt = cv_result.forecasts
metrics = cv_result.metrics
```

Expected CV outputs:

- `cv_result.forecasts`: `xr.DataTree` with concatenated fold forecasts
- `cv_result.metrics`: per-fold and aggregate metric dictionary
- `cv_result.n_folds`: executed fold count

---

## Contract Violation Appendix

These are the expected validation failures and guidance:

- **Shape mismatch (`y`, `covariates`, `future_covariates`)**
  - Raise `ValueError` with expected and received shapes.
- **Missing `future_covariates` when `future > 0` and covariates are enabled**
  - Raise `ValueError` with explicit requirement message.
- **Invalid `group_mapping`**
  - Raise `ValueError` when mapping length does not equal `n_series` or contains invalid ids.
- **Unsupported `return_sites`**
  - Raise `KeyError`/`ValueError` listing valid sites for the selected model family.

## Out of Scope (Phase 5)

This quickstart spec intentionally excludes DeepAR, attention DeepAR, and HSGP custom-model flows. Those belong to advanced/optional quickstarts after the core release quickstarts are stable.
