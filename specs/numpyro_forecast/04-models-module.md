# 04 — Models Module

## Design Principle

Pre-built models are **plain functions** that follow the `ModelFn` protocol. Priors are injectable via a `priors: dict[str, Prior] | None = None` parameter with sensible defaults defined in a module-level `*_DEFAULT_PRIORS` constant. Each model:

1. Merges user-provided priors with `DEFAULT_PRIORS`: `resolved = {**DEFAULT_PRIORS, **(priors or {})}`.
2. Samples each resolved prior via `resolved["key"].sample("key")`.
3. Defines a transition function using components from `components/`.
4. Runs `scan` + `condition` for inference.
5. Optionally produces forecast deterministics when `future > 0`.

See [03-core-abstractions.md](03-core-abstractions.md) for the `Prior` class specification.

## Input Validation

Models validate inputs at entry and raise `ValueError` for:
- Unknown `trend` strings (must be one of: `None`, `"local linear"`, `"smooth"`, `"deterministic"`, `"damped"`)
- `n_seasons < 2` or `autoregressive < 0`
- `y.shape[0] < n_lags` (series shorter than required lag order)
- `future > 0` with `covariates` provided but `future_covariates` missing
- `future_covariates.shape[0] != future` shape mismatch

NaN policy: Models do not handle NaN internally. Users must impute or mask before passing to model functions. Document this in model docstrings.

**Batch dimension:** All models accept `y: Float[Array, "time *batch"]`. When `*batch` is empty it's univariate; when `(n_series,)` it's panel data. Components broadcast over batch dims automatically.

## Unobserved Components Model (`models/ucm.py`) — **Core Model**

The UCM is the central composable model. It generalizes local level, local linear trend, Holt-Winters, and more into a single function where users enable/disable structural components. This is a Bayesian, JAX-native counterpart to [statsmodels `UnobservedComponents`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.structural.UnobservedComponents.html).

### `UCM_DEFAULT_PRIORS`

```python
from probcast.core.prior import Prior

UCM_DEFAULT_PRIORS: dict[str, Prior] = {
    # Level
    "level_smoothing": Prior("Beta", params={"concentration1": 1.0, "concentration0": 1.0}),
    "level_init": Prior("Normal", params={"loc": 0.0, "scale": 1.0}),
    # Trend
    "trend_smoothing": Prior("Beta", params={"concentration1": 1.0, "concentration0": 1.0}),
    "trend_init": Prior("Normal", params={"loc": 0.0, "scale": 1.0}),
    "damping": Prior("Beta", params={"concentration1": 8.0, "concentration0": 2.0}),
    # Seasonality
    "seasonality_smoothing": Prior("Beta", params={"concentration1": 1.0, "concentration0": 1.0}),
    "seasonality_init": Prior("Normal", params={"loc": 0.0, "scale": 1.0}),
    # Cycle
    "cycle_frequency": Prior("Uniform", params={"low": 0.01, "high": 3.14159}),
    "cycle_damping": Prior("Beta", params={"concentration1": 8.0, "concentration0": 2.0}),
    "cycle_innovation": Prior("Normal", params={"loc": 0.0, "scale": 0.1}),
    # AR
    "ar": Prior("Normal", params={"loc": 0.0, "scale": 0.5}),
    # Regression
    "beta": Prior("Normal", params={"loc": 0.0, "scale": 1.0}),
    # Observation noise
    "sigma": Prior("HalfNormal", params={"scale": 1.0}),
    # Note: enabling level + local linear trend + cycle simultaneously can cause
    # identifiability issues. Consider fixing one component or using informative priors.
}
```

### `ucm_model`

```python
def ucm_model(
    y: Float[Array, "time *batch"],
    *,
    future: int = 0,
    # --- Structural component toggles ---
    level: bool = True,
    trend: str | None = None,           # "local linear", "smooth", "deterministic", "damped", or None
    seasonal: int | dict | None = None, # int = additive HW with n_seasons; dict = trigonometric config
    cycle: bool = False,
    autoregressive: int = 0,            # AR order on the irregular component
    covariates: Float[Array, "time n_features *batch"] | None = None,
    future_covariates: Float[Array, "future n_features *batch"] | None = None,
    group_mapping: Float[Array, "n_series"] | None = None,
    # --- Prior overrides ---
    priors: dict[str, Prior] | None = None,
    likelihood: str = "normal",  # "normal", "studentt"
) -> None:
    resolved = {**UCM_DEFAULT_PRIORS, **(priors or {})}
    # Only sample priors for enabled components ...
```

**Key design:**
- Only samples priors and creates state for **enabled** components — no wasted computation.
- Transition function composes enabled components additively: `mu = level + trend + seasonal + cycle + ar + regression`.
- `seasonal` accepts either `int` (additive HW with `n_seasons`) or `dict` for trigonometric: `{"type": "trigonometric", "period": 12, "harmonics": 4}` or a list of dicts for multiple seasonal periods.
- `trend` accepts string matching statsmodels conventions: `"local linear"`, `"smooth"`, `"deterministic"`, `"damped"`.
- Works seamlessly on `(time,)` or `(time, n_series)` via broadcasting.
- Valid prior keys: all keys in `UCM_DEFAULT_PRIORS`. The model only samples priors for components that are enabled.

**Seasonal config grammar:**
- `None` — no seasonality
- `int` — additive HW seasonality with that many seasons
- `dict` — trigonometric: `{"type": "trigonometric", "period": float, "harmonics": int}`
- `list[dict]` — multiple trigonometric periods: `[{"type": "trigonometric", "period": 365.25, "harmonics": 6}, {"type": "trigonometric", "period": 7, "harmonics": 3}]`

**Identifiability:** Not all component combinations are well-identified. The model does not enforce identifiability constraints — users should check R-hat and ESS via `check_diagnostics()` and consult the UCM tutorial for recommended configurations.

### UCM Configuration Recipes

| Recipe | level | trend | seasonal | cycle | AR | Equivalent |
|--------|-------|-------|----------|-------|----|------------|
| Local level | True | None | None | False | 0 | `local_level_model` |
| Local linear trend | True | "local linear" | None | False | 0 | `local_linear_trend_model` |
| Smooth trend | True | "smooth" | None | False | 0 | `smooth_trend_model` |
| Holt-Winters | True | "local linear" | n_seasons | False | 0 | `holt_winters_model` |
| Damped HW | True | "damped" | n_seasons | False | 0 | `damped_holt_winters_model` |
| BSM (basic structural) | True | "local linear" | {"type": "trigonometric", ...} | True | 0 | — |
| UCM + AR | True | "smooth" | n_seasons | False | 2 | — |

### Convenience aliases

```python
# Thin wrappers calling ucm_model with specific configurations.
# All forward `priors` so users can override defaults via the same dict pattern.
def local_level_model(y, *, future=0, priors=None, **kwargs):
    return ucm_model(y, future=future, level=True, priors=priors, **kwargs)

def local_linear_trend_model(y, *, future=0, priors=None, **kwargs):
    return ucm_model(y, future=future, level=True, trend="local linear", priors=priors, **kwargs)

def smooth_trend_model(y, *, future=0, priors=None, **kwargs):
    return ucm_model(y, future=future, level=True, trend="smooth", priors=priors, **kwargs)
```

### Validation: UCM vs statsmodels

The UCM implementation must be validated against [`statsmodels.tsa.statespace.structural.UnobservedComponents`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.structural.UnobservedComponents.html). A dedicated test suite (`tests/test_models/test_ucm_statsmodels.py`) should:

1. Fit both `probcast.ucm_model` (with weakly informative priors and MCMC) and `statsmodels.UnobservedComponents` on the same dataset for each UCM configuration recipe (local level, local linear trend, smooth trend, Holt-Winters, BSM).
2. Compare the **posterior mean of predictions** against the statsmodels MLE point forecasts (within reasonable tolerance, accounting for Bayesian vs frequentist differences).
3. Compare **inferred parameter means** (e.g., smoothing parameters, noise variances) against MLE estimates.
4. Use weakly informative priors centred near the MLE values to ensure the Bayesian posterior concentrates around the frequentist solution.

`statsmodels>=0.14` is a dev dependency for this purpose. See [09-dev-stack.md](09-dev-stack.md).

## Exponential Smoothing (`models/exponential_smoothing.py`)

These are **convenience wrappers** around `ucm_model` with specific component configurations and the ES-style smoothing parameterisation (alpha, beta, gamma instead of innovation variances). They accept `y: Float[Array, "time *batch"]` and broadcast over batch dimensions. All forward `priors` to `ucm_model`.

### `level_model`

Simple exponential smoothing (level only). Equivalent to `ucm_model(y, level=True)`.

```python
def level_model(
    y: Float[Array, "time *batch"],
    *,
    future: int = 0,
    priors: dict[str, Prior] | None = None,
) -> None:
    return ucm_model(y, future=future, level=True, priors=priors)
```

**Source:** `exponential_smoothing_numpyro.ipynb` — `level_model(y, future=0)`

Valid prior keys: `"level_smoothing"`, `"level_init"`, `"sigma"`.

### `level_trend_model`

Exponential smoothing with additive trend. Equivalent to `ucm_model(y, level=True, trend="local linear")`.

```python
def level_trend_model(
    y: Float[Array, "time *batch"],
    *,
    future: int = 0,
    priors: dict[str, Prior] | None = None,
) -> None:
    return ucm_model(y, future=future, level=True, trend="local linear", priors=priors)
```

Valid prior keys: `"level_smoothing"`, `"trend_smoothing"`, `"level_init"`, `"trend_init"`, `"sigma"`.

### `holt_winters_model`

Additive Holt-Winters. Equivalent to `ucm_model(y, level=True, trend="local linear", seasonal=n_seasons)`.

```python
def holt_winters_model(
    y: Float[Array, "time *batch"],
    n_seasons: int,
    *,
    future: int = 0,
    group_mapping: Float[Array, "n_series"] | None = None,
    priors: dict[str, Prior] | None = None,
) -> None:
    return ucm_model(
        y, future=future, level=True, trend="local linear",
        seasonal=n_seasons, group_mapping=group_mapping, priors=priors,
    )
```

Valid prior keys: `"level_smoothing"`, `"trend_smoothing"`, `"seasonality_smoothing"`, `"level_init"`, `"trend_init"`, `"seasonality_init"`, `"sigma"`.

### `damped_holt_winters_model`

Damped trend variant. Equivalent to `ucm_model(y, level=True, trend="damped", seasonal=n_seasons)`.

```python
def damped_holt_winters_model(
    y: Float[Array, "time *batch"],
    n_seasons: int,
    *,
    future: int = 0,
    group_mapping: Float[Array, "n_series"] | None = None,
    priors: dict[str, Prior] | None = None,
) -> None:
    return ucm_model(
        y, future=future, level=True, trend="damped",
        seasonal=n_seasons, group_mapping=group_mapping, priors=priors,
    )
```

Valid prior keys: `"level_smoothing"`, `"trend_smoothing"`, `"seasonality_smoothing"`, `"damping"`, `"level_init"`, `"trend_init"`, `"seasonality_init"`, `"sigma"`.

**Source:** `exponential_smoothing_numpyro.ipynb`

## Intermittent Demand (`models/intermittent.py`)

> **Note:** Intermittent models do not currently accept `group_mapping` because their input contract (`z`, `p_inv`) differs from the standard `y` convention and they are typically applied to individual SKUs. Hierarchical intermittent demand (e.g., pooling smoothing parameters across related SKUs) is a future extension.

### `CROSTON_DEFAULT_PRIORS`

```python
CROSTON_DEFAULT_PRIORS: dict[str, Prior] = {
    "level_smoothing": Prior("Beta", params={"concentration1": 2.0, "concentration0": 20.0}),
    "level_init": Prior("Normal", params={"loc": 0.0, "scale": 1.0}),
    "sigma": Prior("HalfNormal", params={"scale": 1.0}),
}
```

### `croston_model`

Croston's method via scoped sub-models for demand sizes and inter-arrival periods.

```python
def croston_model(
    z: Float[Array, "n_demands *batch"],
    p_inv: Float[Array, "n_demands *batch"],
    *,
    future: int = 0,
    priors: dict[str, Prior] | None = None,
) -> None:
    resolved = {**CROSTON_DEFAULT_PRIORS, **(priors or {})}
    ...
```

**Key pattern:** Uses `numpyro.handlers.scope` to namespace the same `level_model` for demand and period components:

```python
z_forecast = scope(level_model, "demand")(z, future=future, priors=resolved)
p_inv_forecast = scope(level_model, "period_inv")(p_inv, future=future, priors=resolved)
```

**Source:** `croston_numpyro.ipynb`

### `TSB_DEFAULT_PRIORS`

```python
TSB_DEFAULT_PRIORS: dict[str, Prior] = {
    "z_smoothing": Prior("Beta", params={"concentration1": 10.0, "concentration0": 40.0}),
    "p_smoothing": Prior("Beta", params={"concentration1": 10.0, "concentration0": 40.0}),
    "sigma": Prior("HalfNormal", params={"scale": 1.0}),
}
```

### `tsb_model`

Teunter-Syntetos-Babai method with dual-state (demand level + demand probability) transition.

```python
def tsb_model(
    ts_trim: Float[Array, "time *batch"],
    z0: float,
    p0: float,
    *,
    future: int = 0,
    priors: dict[str, Prior] | None = None,
) -> None:
    resolved = {**TSB_DEFAULT_PRIORS, **(priors or {})}
    ...
```

**Source:** `tsb_numpyro.ipynb`. Transition updates `z_next` and `p_next` in a tuple carry; forecast `mu = z_next * p_next`.

### `ZI_TSB_DEFAULT_PRIORS`

```python
ZI_TSB_DEFAULT_PRIORS: dict[str, Prior] = {
    "z_smoothing": Prior("Beta", params={"concentration1": 10.0, "concentration0": 60.0}),
    "p_smoothing": Prior("Beta", params={"concentration1": 10.0, "concentration0": 60.0}),
    "concentration": Prior("HalfNormal", params={"scale": 1.0}),
}
```

### `zi_tsb_model`

Zero-inflated TSB with count likelihood.

```python
def zi_tsb_model(
    ts_trim: Float[Array, "time *batch"],
    z0: float,
    p0: float,
    *,
    future: int = 0,
    priors: dict[str, Prior] | None = None,
) -> None:
    resolved = {**ZI_TSB_DEFAULT_PRIORS, **(priors or {})}
    ...
```

**Key difference:** Uses `dist.ZeroInflatedNegativeBinomial2(mean=mu, concentration=concentration, gate=1-p_next)` instead of Gaussian likelihood.

**Source:** `zi_tsb_numpyro.ipynb`

## ARMA (`models/arma.py`)

### `ARMA_DEFAULT_PRIORS`

```python
ARMA_DEFAULT_PRIORS: dict[str, Prior] = {
    "mu": Prior("Normal", params={"loc": 0.0, "scale": 1.0}),
    "phi": Prior("Uniform", params={"low": -1.0, "high": 1.0}),
    "theta": Prior("Uniform", params={"low": -1.0, "high": 1.0}),
    "sigma": Prior("HalfNormal", params={"scale": 1.0}),
    # Note: ARMA(p,q) with p,q > 2 may have identifiability issues. Start with small orders.
}
```

### `arma_model`

ARMA(p,q) model with error conditioning pattern.

```python
def arma_model(
    y: Float[Array, "time *batch"],
    p: int = 1,
    q: int = 1,
    *,
    future: int = 0,
    group_mapping: Float[Array, "n_series"] | None = None,
    priors: dict[str, Prior] | None = None,
) -> None:
    resolved = {**ARMA_DEFAULT_PRIORS, **(priors or {})}
    ...
```

**Key pattern:** Conditions on errors, not observations. The transition function computes `error = y[t] - pred`, then all errors are observed jointly:

```python
numpyro.sample("errors", dist.Normal(loc=0, scale=sigma), obs=errors)
```

This is necessary because direct observation conditioning would make `theta` (MA coefficient) unidentifiable.

Valid prior keys: `"mu"`, `"phi"`, `"theta"`, `"sigma"`.

**Source:** `arma_numpyro.ipynb` — `arma_1_1(y, future=0)`

## VAR (`models/var.py`)

### `VAR_DEFAULT_PRIORS`

```python
VAR_DEFAULT_PRIORS: dict[str, Prior] = {
    "constant": Prior("Normal", params={"loc": 0.0, "scale": 1.0}),
    "phi": Prior("Normal", params={"loc": 0.0, "scale": 10.0}),
    "sigma": Prior("HalfNormal", params={"scale": 1.0}),
    "lkj": Prior("LKJCholesky", params={"concentration": 1.0}),
    # Note: VAR with many lags and variables grows parameters as O(n_vars^2 * n_lags).
    # Keep n_lags small or use regularizing priors.
}
```

### `var_model`

Vector autoregressive model of order p with LKJ correlation prior.

```python
def var_model(
    y: Float[Array, "time vars"],
    n_lags: int,
    *,
    future: int = 0,
    group_mapping: Float[Array, "n_vars"] | None = None,
    priors: dict[str, Prior] | None = None,
) -> None:
    resolved = {**VAR_DEFAULT_PRIORS, **(priors or {})}
    ...
```

**Key patterns:**
- Einsum for lag contributions: `jnp.einsum("lij,lj->i", phi[::-1], y_lags)`
- LKJ Cholesky prior for cross-series correlation: `dist.LKJCholesky(n_vars, concentration)`
- Covariance scaling: `jnp.einsum("...i,...ij->...ij", sigma, l_omega)`

Valid prior keys: `"constant"`, `"phi"`, `"sigma"`, `"lkj"`.

**Source:** `var_numpyro.ipynb`

### `compute_irf`

Impulse response function (standalone utility, lives in `models/var.py`).

```python
def compute_irf(
    phi: Float[Array, "*sample n_lags n_vars n_vars"],
    n_steps: int,
    shock_size: float = 1.0,
) -> Float[Array, "*sample n_steps n_vars n_vars"]:
```

Computes MA(∞) representation via `lax.scan`. JIT-compilable with `static_argnames=["n_steps", "shock_size"]`. Vectorizable over posterior samples via `vmap`.

**Source:** `var_numpyro.ipynb`

### Validation: VAR vs statsmodels

The VAR implementation must be validated against [`statsmodels.tsa.vector_ar.var_model.VAR`](https://www.statsmodels.org/stable/vector_ar.html). A dedicated test suite (`tests/test_models/test_var_statsmodels.py`) should:

1. Fit both `probcast.var_model` (with weakly informative priors and MCMC) and `statsmodels.VAR` on the same publicly available dataset (from the [var_numpyro blog post](https://juanitorduz.github.io/var_numpyro/)).
2. Compare **posterior mean predictions** against the OLS point forecasts.
3. Compare **inferred AR coefficient matrices** (posterior means) against OLS estimates.
4. Compare **impulse response functions** from `compute_irf` against `statsmodels.tsa.vector_ar.irf.IRAnalysis` — these should match closely since IRFs are a deterministic function of the AR coefficients.

`statsmodels>=0.14` is a dev dependency for this purpose. See [09-dev-stack.md](09-dev-stack.md).

## Local Level + Fourier (subsumed by UCM)

The previous `local_level_fourier_model` from `numpyro_forecasting_univariate.ipynb` is now expressed as a UCM configuration:

```python
# Local level with trigonometric Fourier seasonality and Student-T likelihood
ucm_model(
    y, future=12,
    level=True, trend=None,
    seasonal={"type": "trigonometric", "period": 365.25, "harmonics": 6},
    likelihood="studentt",
)
```

For the specific pattern using `LocScaleReparam` and Fourier regression covariates (as in the original notebook), users can compose `level_transition` + `fourier_regression` in a custom model function. The UCM provides the common case; the components enable the advanced case.

## SARIMAX (`models/sarimax.py`)

### `SARIMAX_DEFAULT_PRIORS`

```python
SARIMAX_DEFAULT_PRIORS: dict[str, Prior] = {
    "phi": Prior("Normal", params={"loc": 0.0, "scale": 0.5}),
    "theta": Prior("Normal", params={"loc": 0.0, "scale": 0.5}),
    "seasonal_phi": Prior("Normal", params={"loc": 0.0, "scale": 0.5}),
    "seasonal_theta": Prior("Normal", params={"loc": 0.0, "scale": 0.5}),
    "beta": Prior("Normal", params={"loc": 0.0, "scale": 1.0}),
    "sigma": Prior("HalfNormal", params={"scale": 1.0}),
    # Note: high-order seasonal AR/MA with short seasonal cycles can be poorly identified.
    # Use `check_diagnostics()` and watch R-hat.
}
```

### `sarimax_model`

Seasonal ARIMA with exogenous regressors. Builds on the ARMA components with differencing and seasonal AR/MA terms.

```python
def sarimax_model(
    y: Float[Array, "time *batch"],
    order: tuple[int, int, int] = (1, 0, 0),
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 1),
    covariates: Float[Array, "time n_features *batch"] | None = None,
    *,
    future: int = 0,
    future_covariates: Float[Array, "future n_features *batch"] | None = None,
    group_mapping: Float[Array, "n_series"] | None = None,
    priors: dict[str, Prior] | None = None,
) -> None:
    resolved = {**SARIMAX_DEFAULT_PRIORS, **(priors or {})}
    ...
```

**Key patterns:**
- `order = (p, d, q)` — non-seasonal AR order, differencing, MA order.
- `seasonal_order = (P, D, Q, s)` — seasonal AR, differencing, MA, and period.
- Differencing applied as a data transform before the scan loop.
- Exogenous regressors added as linear regression term to the mean.
- `future_covariates` must be provided when `future > 0` and `covariates` is not None.

Valid prior keys: `"phi"`, `"theta"`, `"seasonal_phi"`, `"seasonal_theta"`, `"beta"`, `"sigma"`.

## Hierarchical Models (Cross-Cutting Pattern)

Hierarchical structure is a **cross-cutting capability** supported by all panel-capable models, not a separate model. Any model that accepts `y: (time, *batch)` and `priors: dict[str, Prior]` also accepts `group_mapping: Array | None` to enable multi-level hierarchical priors. Hierarchy is expressed entirely through the `Prior` dict: nested `Prior` objects create hyperprior trees, and the model function wraps `prior.sample(...)` inside `numpyro.plate` and applies `LocScaleReparam` where needed.

### How it works

1. **2-level hierarchy (global → series):** Use nested `Prior` objects. `Prior.sample()` resolves children (hyperpriors) at global scope, then samples the parent inside `plate("series")`.
2. **3-level hierarchy (global → group → series):** Pass `group_mapping` (an integer array mapping each series to its group index). The model creates an intermediate `plate("groups")` between global and series-level sampling.
3. **No hierarchy:** Use flat `Prior` objects (no nesting). All parameters are sampled independently per series (or globally for univariate data).

### Models supporting `group_mapping`

| Model | `group_mapping` type | Notes |
|-------|---------------------|-------|
| `ucm_model` | `Float[Array, "n_series"]` | Core model — all ES wrappers delegate here |
| `holt_winters_model` | `Float[Array, "n_series"]` | Forwards to `ucm_model` |
| `damped_holt_winters_model` | `Float[Array, "n_series"]` | Forwards to `ucm_model` |
| `local_level_model` etc. | via `**kwargs` | UCM convenience aliases forward `group_mapping` |
| `sarimax_model` | `Float[Array, "n_series"]` | Standalone — own plate logic |
| `arma_model` | `Float[Array, "n_series"]` | Standalone — own plate logic |
| `var_model` | `Float[Array, "n_vars"]` | Groups variables, not series |
| `deepar_model` | `Float[Array, "n_series"]` | SVI-only — own plate logic |
| `attention_deepar_model` | `Float[Array, "n_series"]` | SVI-only — own plate logic |

Intermittent models (`croston_model`, `tsb_model`, `zi_tsb_model`) do not currently accept `group_mapping` — see the note in the Intermittent Demand section.

### Example 1: 3-level hierarchical Holt-Winters (from `hierarchical_exponential_smoothing.ipynb`)

This example directly corresponds to the [hierarchical exponential smoothing notebook](Python/hierarchical_exponential_smoothing.ipynb), which fits a Holt-Winters model to 308 Australian tourism series grouped by state/territory. The hierarchy has three levels:

- **Global:** hyperpriors for trend and seasonality smoothing concentrations, noise scale.
- **Group (state):** trend smoothing concentrations sampled inside `plate("groups")`.
- **Series:** all smoothing parameters sampled inside `plate("series")`, with trend parameters indexed into group-level params via `group_mapping`.

```python
# ts_state_mapping_idx: shape (n_series,) — maps each series to its state/territory group
holt_winters_model(
    y_panel,  # shape (time, n_series)
    n_seasons=4,
    future=8,
    group_mapping=ts_state_mapping_idx,
    priors={
        # Level: per-series, flat (no hierarchy)
        "level_smoothing": Prior("Beta", params={"concentration1": 1.0, "concentration0": 1.0}),
        "level_init": Prior("Normal", params={"loc": 0.0, "scale": 1.0}),

        # Trend: 3-level hierarchy (global -> state -> series)
        # Nesting depth 3: leaf Gamma hyperpriors are sampled at global scope,
        # mid-level Gamma priors are sampled inside plate("groups"),
        # and the Beta parent is sampled inside plate("series") with group_mapping indexing.
        "trend_smoothing": Prior(
            "Beta",
            params={
                "concentration1": Prior(
                    "Gamma",
                    params={
                        "concentration": Prior("Gamma", params={"concentration": 8.0, "rate": 4.0}),
                        "rate": Prior("Gamma", params={"concentration": 8.0, "rate": 4.0}),
                    },
                ),
                "concentration0": Prior(
                    "Gamma",
                    params={
                        "concentration": Prior("Gamma", params={"concentration": 8.0, "rate": 4.0}),
                        "rate": Prior("Gamma", params={"concentration": 8.0, "rate": 4.0}),
                    },
                ),
            },
        ),
        "trend_init": Prior("Normal", params={"loc": 0.0, "scale": 1.0}),

        # Seasonality: 2-level hierarchy (global -> series)
        # Children (Gamma) are sampled at global scope, parent (Beta) inside plate("series").
        "seasonality_smoothing": Prior(
            "Beta",
            params={
                "concentration1": Prior("Gamma", params={"concentration": 4.0, "rate": 2.0}),
                "concentration0": Prior("Gamma", params={"concentration": 4.0, "rate": 2.0}),
            },
        ),
        "seasonality_init": Prior("Normal", params={"loc": 0.0, "scale": 1.0}),

        # Noise: 2-level hierarchy (global -> series)
        # The global Gamma scale is sampled first, then HalfNormal inside plate("series").
        "sigma": Prior(
            "HalfNormal",
            params={"scale": Prior("Gamma", params={"concentration": 80.0, "rate": 3.0})},
        ),
    },
)
```

**Source:** `hierarchical_exponential_smoothing.ipynb`

### Example 2: Hierarchical ARMA (same pattern, different model)

The identical `group_mapping` + nested `Prior` pattern works with any panel-capable model. Here, an ARMA(1,1) pools AR and MA coefficients across groups of related series:

```python
arma_model(
    y_panel,  # shape (time, n_series)
    p=1, q=1,
    future=12,
    group_mapping=region_mapping_idx,  # shape (n_series,) — maps series to region
    priors={
        # AR coefficient: 2-level hierarchy (global -> series)
        "phi": Prior(
            "Normal",
            params={
                "loc": Prior("Normal", params={"loc": 0.0, "scale": 0.5}),
                "scale": Prior("HalfNormal", params={"scale": 0.3}),
            },
        ),
        # MA coefficient: flat (no pooling)
        "theta": Prior("Uniform", params={"low": -1.0, "high": 1.0}),
        # Noise: 2-level hierarchy
        "sigma": Prior(
            "HalfNormal",
            params={"scale": Prior("Gamma", params={"concentration": 4.0, "rate": 2.0})},
        ),
    },
)
```

### Example 3: Hierarchical SARIMAX

```python
sarimax_model(
    y_panel,  # shape (time, n_series)
    order=(1, 0, 0),
    seasonal_order=(1, 0, 0, 12),
    covariates=covariates_panel,
    future=12,
    future_covariates=future_covariates_panel,
    group_mapping=store_cluster_idx,  # shape (n_series,) — maps stores to clusters
    priors={
        # Seasonal AR: 3-level hierarchy (global -> cluster -> store)
        "seasonal_phi": Prior(
            "Normal",
            params={
                "loc": Prior(
                    "Normal",
                    params={
                        "loc": Prior("Normal", params={"loc": 0.0, "scale": 0.5}),
                        "scale": Prior("HalfNormal", params={"scale": 0.3}),
                    },
                ),
                "scale": Prior("HalfNormal", params={"scale": 0.2}),
            },
        ),
    },
)
```

### Inside the model function (general pattern)

> **Note:** `Prior.sample()` handles 1-level and 2-level hierarchies automatically. For 3-level hierarchies with `group_mapping`, the model manually decomposes the prior tree to insert `plate('groups')` between global and series scopes. This is the **one case** where manual tree-walking is required (because an intermediate plate must be inserted between levels).

Every model that accepts `group_mapping` follows the same internal logic. Here it is illustrated with `holt_winters_model`, but the pattern applies identically to `arma_model`, `sarimax_model`, `deepar_model`, etc.:

```python
def holt_winters_model(y, n_seasons, *, future=0, group_mapping=None, priors=None):
    resolved = {**HOLT_WINTERS_DEFAULT_PRIORS, **(priors or {})}
    n_time, n_series = y.shape[0], y.shape[1] if y.ndim > 1 else None

    if group_mapping is not None:
        n_groups = jnp.unique(group_mapping).shape[0]

    # --- 3-level priors (trend): global -> group -> series ---
    # Leaf hyperpriors (depth 0) are sampled at global scope by Prior.sample()
    # Mid-level priors (depth 1) are sampled inside plate("groups")
    if group_mapping is not None:
        with numpyro.plate("groups", n_groups):
            trend_c1 = resolved["trend_smoothing"].params["concentration1"].sample(
                "trend_smoothing_concentration1"
            )
            trend_c0 = resolved["trend_smoothing"].params["concentration0"].sample(
                "trend_smoothing_concentration0"
            )

    with numpyro.plate("series", n_series):
        # Flat priors: sample directly in series plate
        level_smoothing = resolved["level_smoothing"].sample("level_smoothing")
        level_init = resolved["level_init"].sample("level_init")

        # 3-level prior: index group params into series via group_mapping
        if group_mapping is not None:
            trend_smoothing = numpyro.sample(
                "trend_smoothing",
                dist.Beta(
                    concentration1=trend_c1[group_mapping],
                    concentration0=trend_c0[group_mapping],
                ),
            )
        else:
            trend_smoothing = resolved["trend_smoothing"].sample("trend_smoothing")
        trend_init = resolved["trend_init"].sample("trend_init")

        # 2-level priors: children sampled globally, parent in series plate
        seasonality_smoothing = resolved["seasonality_smoothing"].sample("seasonality_smoothing")
        sigma = resolved["sigma"].sample("sigma")

        with numpyro.plate("n_seasons", n_seasons, dim=-2):
            seasonality_init = resolved["seasonality_init"].sample("seasonality_init")

    # ... transition_fn, scan+condition as in non-hierarchical case ...
```

When a `Prior` has nested children, `prior.sample(name)` recursively samples the hyperpriors first (at the global level), then the parent distribution is instantiated with the resolved values and sampled — giving the hierarchical pooling structure. For 3-level hierarchies with `group_mapping`, the model function manually walks the prior tree to insert an intermediate group plate between the global and series levels.

The model can also apply `numpyro.handlers.reparam` with `LocScaleReparam` for non-centered parameterization when needed.

**Key patterns:**
- **Universal:** The `group_mapping` + nested `Prior` pattern works identically across all panel-capable models.
- `group_mapping: Array | None` — optional mapping from series index to group index, enabling the intermediate plate.
- `numpyro.plate("groups", n_groups)` for group-level parameters.
- `numpyro.plate("series", n_series)` for series-level parameters indexed into group params via `group_mapping`.
- Transition operates on `(time, n_series)` arrays with vectorized updates.
- Recommend SVI for large hierarchical models (hundreds of series). MCMC is viable for smaller panels.
- The pattern was validated with both NUTS and SVI on the tourism dataset (308 series, `hierarchical_exponential_smoothing.ipynb`).

## DeepAR (`models/deepar.py`)

### Neural network integration pattern

DeepAR uses `flax.nnx` modules registered with NumPyro via [`numpyro.contrib.module`](https://num.pyro.ai/en/stable/contrib.html#module). Following the pattern from [hierarchical forecasting Part III](https://juanitorduz.github.io/numpyro_hierarchical_forecasting_3/):

1. The neural network is **built externally** as a `flax.nnx.Module`.
2. It is **passed into the model function** as a positional argument.
3. Inside the model, it is **registered with NumPyro** via `nnx_module` (deterministic, default) or `random_nnx_module` (Bayesian weights).

This keeps architecture decisions (hidden size, number of layers, cell type) outside the model function, making the NN fully composable and swappable.

### `DEEPAR_DEFAULT_PRIORS`

```python
DEEPAR_DEFAULT_PRIORS: dict[str, Prior] = {
    "sigma": Prior("HalfNormal", params={"scale": 1.0}),
}
```

The `priors` dict covers emission distribution parameters only. NN weights are handled by `nnx_module` / `random_nnx_module`, not by `Prior`.

### `deepar_model`

A simple DeepAR-style probabilistic forecaster using an autoregressive RNN. This is intentionally not a fully general implementation — it provides a usable baseline that can be extended.

```python
from numpyro.contrib.module import nnx_module, random_nnx_module

def deepar_model(
    y: Float[Array, "time *batch"],
    rnn: nnx.Module,
    covariates: Float[Array, "time n_features"] | None = None,
    *,
    future: int = 0,
    future_covariates: Float[Array, "future n_features"] | None = None,
    likelihood: str = "normal",
    bayesian_nn: bool = False,
    group_mapping: Float[Array, "n_series"] | None = None,
    priors: dict[str, Prior] | None = None,
) -> None:
    resolved = {**DEEPAR_DEFAULT_PRIORS, **(priors or {})}

    # Register NN with NumPyro
    if bayesian_nn:
        rnn = random_nnx_module("rnn", rnn, prior=dist.Normal(0, 0.1))
    else:
        rnn = nnx_module("rnn", rnn)

    sigma = resolved["sigma"].sample("sigma")
    # ... autoregressive scan driven by rnn ...
```

**Note:** DeepAR requires `y` to have at least one batch dimension (`n_series >= 1`) because the RNN operates per-series. For a single series, reshape to `(time, 1)`.

**Parameters:**
- `rnn` — a pre-built `flax.nnx.Module` (e.g. `DeepARCell` from `nn/rnn.py`). Architecture choices (hidden size, number of layers, cell type) are made at construction time, not in the model signature.
- `bayesian_nn` — if `False` (default), uses `nnx_module`: weights are deterministic variational parameters trained via SVI. If `True`, uses `random_nnx_module`: weights get a prior and are sampled (still SVI-recommended for practical performance).
- `likelihood` — controls the output distribution: `"normal"`, `"studentt"`, `"negative_binomial"`.

### Usage example

```python
from flax import nnx
from jax import random
from probcast.nn.rnn import DeepARCell
from probcast.models.deepar import deepar_model
from probcast.inference.svi import run_svi, forecast_svi
from probcast.core.params import SVIParams

# 1. Build the NN externally
rnn = DeepARCell(
    input_size=1,        # y_{t-1} features
    hidden_size=40,
    n_layers=2,
    cell_type="gru",
    rngs=nnx.Rngs(random.PRNGKey(0)),
)

# 2. Run SVI (deterministic weights by default)
svi_result = run_svi(rng_key, deepar_model, SVIParams(), y_train, rnn)

# 3. Forecast
forecast = forecast_svi(rng_key, deepar_model, guide, svi_result.params,
                        y_train, rnn, future=24)

# 4. Bayesian NN weights (optional — puts a prior on all weights)
svi_result_bayes = run_svi(
    rng_key, deepar_model, SVIParams(), y_train, rnn, bayesian_nn=True,
)
```

### Expected NN interface (`nn/rnn.py`)

The `rnn` module passed to `deepar_model` must implement a `__call__` compatible with autoregressive `lax.scan`:

```python
class DeepARCell(nnx.Module):
    """RNN cell for DeepAR autoregressive forecasting.

    At each time step, takes [y_{t-1}, covariates_t] as input and outputs
    distribution parameters (loc, scale for Normal; rate for Poisson, etc.).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 40,
        n_layers: int = 2,
        cell_type: str = "gru",  # "gru" or "lstm"
        rngs: nnx.Rngs = ...,
    ) -> None: ...

    def __call__(
        self,
        carry: Float[Array, "n_layers hidden_size"],
        x_t: Float[Array, "input_dim"],
    ) -> tuple[Float[Array, "n_layers hidden_size"], Float[Array, "output_dim"]]:
        """Single-step RNN forward pass.

        output_dim depends on likelihood:
        - "normal": 2 (loc, scale)
        - "studentt": 3 (df, loc, scale)
        - "negative_binomial": 2 (mean, concentration)

        The model function maps RNN output to distribution parameters.
        Alternatively, output_dim is always hidden_size and a final
        linear layer inside the model maps to the required params.

        Returns
        -------
        carry
            Updated hidden state.
        output
            Distribution parameters (e.g. loc and scale for Normal likelihood).
        """
        ...
```

**Key patterns:**
- The `rnn` is built externally and passed as a positional argument — architecture is not part of the model signature.
- `nnx_module("rnn", rnn)` registers the NN so NumPyro's SVI optimizer can find and update its parameters.
- `random_nnx_module("rnn", rnn, prior=...)` places a prior on all NN weights, enabling Bayesian neural network inference (still SVI-recommended).
- At each time step, the RNN takes `[y_{t-1}, covariates_t]` as input and outputs distribution parameters.
- Training via SVI (MCMC is impractical for neural network parameters with `nnx_module`; Bayesian weights via `random_nnx_module` are also SVI-recommended).
- `likelihood` controls the output distribution: `"normal"`, `"studentt"`, `"negative_binomial"`.

**Design note:** DeepAR is fundamentally different from the scan+condition pattern used by other models. The NN is a positional argument (not a keyword argument) because it is required — there is no sensible default. Covariates are passed as keyword arguments, keeping the `(y, rnn, ..., future=0)` contract.

### Optional: `attention_deepar_model`

A variant that adds a simple temporal self-attention layer on top of the RNN hidden states before producing distribution parameters. This is a lightweight extension, not a full Transformer. It follows the same `nnx_module` pattern:

```python
def attention_deepar_model(
    y: Float[Array, "time *batch"],
    rnn: nnx.Module,
    covariates: Float[Array, "time n_features"] | None = None,
    *,
    future: int = 0,
    n_heads: int = 2,
    likelihood: str = "normal",
    bayesian_nn: bool = False,
    group_mapping: Float[Array, "n_series"] | None = None,
    priors: dict[str, Prior] | None = None,
) -> None:
    # Register NN
    if bayesian_nn:
        rnn = random_nnx_module("rnn", rnn, prior=dist.Normal(0, 0.1))
    else:
        rnn = nnx_module("rnn", rnn)
    # ... run RNN, apply self-attention over hidden states, produce distribution params ...
```

## HSGP Time-Varying Covariates

Time-varying covariate effects can be added to any model via the HSGP component in `components/hsgp.py`. This wraps [`numpyro.contrib.hsgp`](https://github.com/pyro-ppl/numpyro/tree/master/numpyro/contrib/hsgp) to provide smooth, non-parametric covariate effects that evolve over time.

### Usage pattern

```python
from probcast.components.hsgp import hsgp_covariate_effect

def custom_model_with_hsgp(y, covariates, *, future=0, priors=None, **kwargs):
    # ... standard model components ...

    # Add time-varying covariate effect via HSGP
    covariate_effect = hsgp_covariate_effect(
        x=covariates,
        m=20,            # number of basis functions
        c=1.5,           # boundary factor
        priors=priors,   # forwarded — HSGP reads "length_scale", "amplitude" keys
    )

    mu = level + trend + covariate_effect
    # ... likelihood ...
```

This keeps the HSGP as a composable component rather than baking it into specific model functions.
