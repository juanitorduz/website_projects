# 04 — Models Module

## Design Principle

Pre-built models are **plain functions** that follow the `ModelFn` protocol. Priors are injectable via keyword arguments with sensible defaults. Each model:

1. Samples priors (using defaults or user-provided distributions).
2. Defines a transition function using components from `components/`.
3. Runs `scan` + `condition` for inference.
4. Optionally produces forecast deterministics when `future > 0`.

## Exponential Smoothing (`models/exponential_smoothing.py`)

### `level_model`

Simple exponential smoothing (level only).

```python
def level_model(
    y: Float[Array, " t_max"],
    future: int = 0,
    *,
    level_smoothing_prior: dist.Distribution = dist.Beta(1, 1),
    level_init_prior: dist.Distribution = dist.Normal(0, 1),
    noise_prior: dist.Distribution = dist.HalfNormal(1),
) -> None:
```

**Source:** `exponential_smoothing_numpyro.ipynb` — `level_model(y, future=0)`

### `level_trend_model`

Exponential smoothing with additive trend.

```python
def level_trend_model(
    y: Float[Array, " t_max"],
    future: int = 0,
    *,
    level_smoothing_prior: dist.Distribution = dist.Beta(1, 1),
    trend_smoothing_prior: dist.Distribution = dist.Beta(1, 1),
    level_init_prior: dist.Distribution = dist.Normal(0, 1),
    trend_init_prior: dist.Distribution = dist.Normal(0, 1),
    noise_prior: dist.Distribution = dist.HalfNormal(1),
) -> None:
```

**Source:** `exponential_smoothing_numpyro.ipynb` — `level_trend_model(y, future=0)`

### `holt_winters_model`

Additive Holt-Winters with seasonal component.

```python
def holt_winters_model(
    y: Float[Array, " t_max"],
    n_seasons: int,
    future: int = 0,
    *,
    level_smoothing_prior: dist.Distribution = dist.Beta(1, 1),
    trend_smoothing_prior: dist.Distribution = dist.Beta(1, 1),
    seasonality_smoothing_prior: dist.Distribution = dist.Beta(1, 1),
    level_init_prior: dist.Distribution = dist.Normal(0, 1),
    trend_init_prior: dist.Distribution = dist.Normal(0, 1),
    seasonality_init_prior: dist.Distribution = dist.Normal(0, 1),
    noise_prior: dist.Distribution = dist.HalfNormal(1),
) -> None:
```

**Source:** `exponential_smoothing_numpyro.ipynb` — `holt_winters_model(y, n_seasons, future=0)`

### `damped_holt_winters_model`

Damped trend variant using `fori_loop` for multi-step-ahead damping.

```python
def damped_holt_winters_model(
    y: Float[Array, " t_max"],
    n_seasons: int,
    future: int = 0,
    *,
    level_smoothing_prior: dist.Distribution = dist.Beta(1, 1),
    trend_smoothing_prior: dist.Distribution = dist.Beta(1, 1),
    seasonality_smoothing_prior: dist.Distribution = dist.Beta(1, 1),
    damping_prior: dist.Distribution = dist.Beta(8, 2),
    level_init_prior: dist.Distribution = dist.Normal(0, 1),
    trend_init_prior: dist.Distribution = dist.Normal(0, 1),
    seasonality_init_prior: dist.Distribution = dist.Normal(0, 1),
    noise_prior: dist.Distribution = dist.HalfNormal(1),
) -> None:
```

**Source:** `exponential_smoothing_numpyro.ipynb` — `damped_holt_winters_model(y, n_seasons, future=0)`. Uses `fori_loop` for the damped cumulative sum: `phi_step = fori_loop(1, step+1, lambda i, val: val + phi**i, 0)`.

## Intermittent Demand (`models/intermittent.py`)

### `croston_model`

Croston's method via scoped sub-models for demand sizes and inter-arrival periods.

```python
def croston_model(
    z: Float[Array, " n_demands"],
    p_inv: Float[Array, " n_demands"],
    future: int = 0,
    *,
    level_smoothing_prior: dist.Distribution = dist.Beta(2, 20),
    level_init_prior: dist.Distribution = dist.Normal(0, 1),
    noise_prior: dist.Distribution = dist.HalfNormal(1),
) -> None:
```

**Key pattern:** Uses `numpyro.handlers.scope` to namespace the same `level_model` for demand and period components:

```python
z_forecast = scope(level_model, "demand")(z, future)
p_inv_forecast = scope(level_model, "period_inv")(p_inv, future)
```

**Source:** `croston_numpyro.ipynb`

### `tsb_model`

Teunter-Syntetos-Babai method with dual-state (demand level + demand probability) transition.

```python
def tsb_model(
    ts_trim: Float[Array, " t_max"],
    z0: float,
    p0: float,
    future: int = 0,
    *,
    z_smoothing_prior: dist.Distribution = dist.Beta(10, 40),
    p_smoothing_prior: dist.Distribution = dist.Beta(10, 40),
    noise_prior: dist.Distribution = dist.HalfNormal(1),
) -> None:
```

**Source:** `tsb_numpyro.ipynb`. Transition updates `z_next` and `p_next` in a tuple carry; forecast `mu = z_next * p_next`.

### `zi_tsb_model`

Zero-inflated TSB with count likelihood.

```python
def zi_tsb_model(
    ts_trim: Float[Array, " t_max"],
    z0: float,
    p0: float,
    future: int = 0,
    *,
    z_smoothing_prior: dist.Distribution = dist.Beta(10, 60),
    p_smoothing_prior: dist.Distribution = dist.Beta(10, 60),
    concentration_prior: dist.Distribution = dist.HalfNormal(1),
) -> None:
```

**Key difference:** Uses `dist.ZeroInflatedNegativeBinomial2(mean=mu, concentration=concentration, gate=1-p_next)` instead of Gaussian likelihood.

**Source:** `zi_tsb_numpyro.ipynb`

## ARMA (`models/arma.py`)

### `arma_model`

ARMA(p,q) model with error conditioning pattern.

```python
def arma_model(
    y: Float[Array, " t_max"],
    p: int = 1,
    q: int = 1,
    future: int = 0,
    *,
    mu_prior: dist.Distribution = dist.Normal(0, 1),
    phi_prior: dist.Distribution = dist.Uniform(-1, 1),
    theta_prior: dist.Distribution = dist.Uniform(-1, 1),
    sigma_prior: dist.Distribution = dist.HalfNormal(1),
) -> None:
```

**Key pattern:** Conditions on errors, not observations. The transition function computes `error = y[t] - pred`, then all errors are observed jointly:

```python
numpyro.sample("errors", dist.Normal(loc=0, scale=sigma), obs=errors)
```

This is necessary because direct observation conditioning would make `theta` (MA coefficient) unidentifiable.

**Source:** `arma_numpyro.ipynb` — `arma_1_1(y, future=0)`

## VAR (`models/var.py`)

### `var_model`

Vector autoregressive model of order p with LKJ correlation prior.

```python
def var_model(
    y: Float[Array, "time vars"],
    n_lags: int,
    future: int = 0,
    *,
    constant_prior: dist.Distribution = dist.Normal(0, 1),
    phi_prior: dist.Distribution = dist.Normal(0, 10),
    sigma_prior: dist.Distribution = dist.HalfNormal(1),
    lkj_concentration: float = 1.0,
) -> None:
```

**Key patterns:**
- Einsum for lag contributions: `jnp.einsum("lij,lj->i", phi[::-1], y_lags)`
- LKJ Cholesky prior for cross-series correlation: `dist.LKJCholesky(n_vars, concentration)`
- Covariance scaling: `jnp.einsum("...i,...ij->...ij", sigma, l_omega)`

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

## Local Level + Fourier (`models/local_level.py`)

### `local_level_fourier_model`

Local level model with Fourier seasonal features and Student-T likelihood.

**Note:** This model uses a covariate-first signature and does not conform to the `ModelFn` protocol (see [03-core-abstractions.md](03-core-abstractions.md) for details). For use with `time_slice_cv`, wrap with `functools.partial` to bind covariates.

```python
def local_level_fourier_model(
    covariates: Float[Array, "t_max feature_dim"],
    y: Float[Array, "t_max n_series"] | None = None,
    *,
    bias_prior: dist.Distribution = dist.Normal(0, 10),
    weight_prior: dist.Distribution = dist.Normal(0, 0.1),
    drift_scale_prior: dist.Distribution = dist.LogNormal(-20, 5),
    nu_prior: dist.Distribution = dist.Gamma(10, 2),
    sigma_prior: dist.Distribution = dist.LogNormal(-5, 5),
) -> None:
```

**Key patterns:**
- `LocScaleReparam` with learned `centered` parameter for the drift.
- `numpyro.plate("time", t_max)` for the drift block.
- `scan` for computing latent levels from drift.
- Student-T likelihood for heavy tails.

**Source:** `numpyro_forecasting_univariate.ipynb`

## SARIMAX (`models/sarimax.py`)

### `sarimax_model`

Seasonal ARIMA with exogenous regressors. Builds on the ARMA components with differencing and seasonal AR/MA terms.

```python
def sarimax_model(
    y: Float[Array, " t_max"],
    order: tuple[int, int, int] = (1, 0, 0),
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 1),
    exog: Float[Array, "t_max n_exog"] | None = None,
    future: int = 0,
    future_exog: Float[Array, "future n_exog"] | None = None,
    *,
    phi_prior: dist.Distribution = dist.Normal(0, 0.5),
    theta_prior: dist.Distribution = dist.Normal(0, 0.5),
    seasonal_phi_prior: dist.Distribution = dist.Normal(0, 0.5),
    seasonal_theta_prior: dist.Distribution = dist.Normal(0, 0.5),
    beta_prior: dist.Distribution = dist.Normal(0, 1),
    sigma_prior: dist.Distribution = dist.HalfNormal(1),
) -> None:
```

**Key patterns:**
- `order = (p, d, q)` — non-seasonal AR order, differencing, MA order.
- `seasonal_order = (P, D, Q, s)` — seasonal AR, differencing, MA, and period.
- Differencing applied as a data transform before the scan loop.
- Exogenous regressors added as linear regression term to the mean.
- `future_exog` must be provided when `future > 0` and `exog` is not None.

## Hierarchical Exponential Smoothing (`models/hierarchical.py`)

### `hierarchical_holt_winters_model`

Multi-series Holt-Winters with hierarchical pooling of smoothing parameters.

```python
def hierarchical_holt_winters_model(
    y: Float[Array, "t_max n_series"],
    ts_group_mapping_idx: Float[Array, " n_series"],
    n_seasons: int,
    future: int = 0,
    *,
    trend_smoothing_concentration_prior: dist.Distribution = dist.Gamma(8, 4),
    seasonality_smoothing_concentration_prior: dist.Distribution = dist.Gamma(4, 2),
    noise_scale_prior: dist.Distribution = dist.Gamma(80, 3),
) -> None:
```

**Key patterns:**
- Three-level hierarchy: Global → Group (state) → Series.
- `numpyro.plate("groups", n_groups)` for group-level parameters.
- `numpyro.plate("series", n_series)` for series-level parameters indexed into group params via `mapping_idx`.
- `numpyro.plate("n_seasons", n_seasons, dim=-2)` for seasonal init.
- Transition operates on `(t_max, n_series)` arrays with vectorized updates.

**Source:** `hierarchical_exponential_smoothing.ipynb`

## DeepAR (`models/deepar.py`)

### `deepar_model`

A simple DeepAR-style probabilistic forecaster using an autoregressive RNN. This is intentionally not a fully general implementation — it provides a usable baseline that can be extended.

```python
def deepar_model(
    y: Float[Array, "t_max n_series"],
    covariates: Float[Array, "t_max n_features"] | None = None,
    future: int = 0,
    future_covariates: Float[Array, "future n_features"] | None = None,
    *,
    hidden_size: int = 40,
    n_layers: int = 2,
    likelihood: str = "normal",
    rnn_cell: str = "gru",
) -> None:
```

**Key patterns:**
- Uses `nn/rnn.py` GRU (or LSTM) cells with `lax.scan` for sequential processing.
- At each time step, the RNN takes `[y_{t-1}, covariates_t]` as input and outputs distribution parameters (loc, scale for Normal; or rate for Poisson, etc.).
- Training via SVI (MCMC is impractical for neural network parameters).
- The RNN weights are treated as variational parameters, not sampled — the model samples only the emission distribution parameters.
- `likelihood` controls the output distribution: `"normal"`, `"studentt"`, `"negative_binomial"`.

**Design note:** DeepAR is fundamentally different from the scan+condition pattern used by other models. It uses SVI exclusively and the `ModelFn` protocol is adapted via the `covariates` argument pattern (similar to `local_level_fourier_model`).

### Optional: `attention_deepar_model`

A variant that adds a simple temporal self-attention layer on top of the RNN hidden states before producing distribution parameters. This is a lightweight extension, not a full Transformer.

```python
def attention_deepar_model(
    y: Float[Array, "t_max n_series"],
    covariates: Float[Array, "t_max n_features"] | None = None,
    future: int = 0,
    *,
    hidden_size: int = 40,
    n_heads: int = 2,
    likelihood: str = "normal",
) -> None:
```

## HSGP Time-Varying Covariates

Time-varying covariate effects can be added to any model via the HSGP component in `components/hsgp.py`. This wraps [`numpyro.contrib.hsgp`](https://github.com/pyro-ppl/numpyro/tree/master/numpyro/contrib/hsgp) to provide smooth, non-parametric covariate effects that evolve over time.

### Usage pattern

```python
from probcast.components.hsgp import hsgp_covariate_effect

def custom_model_with_hsgp(y, covariates, future=0, **kwargs):
    # ... standard model components ...

    # Add time-varying covariate effect via HSGP
    covariate_effect = hsgp_covariate_effect(
        x=covariates,
        m=20,            # number of basis functions
        c=1.5,           # boundary factor
        length_scale_prior=dist.InverseGamma(5, 5),
        amplitude_prior=dist.HalfNormal(1),
    )

    mu = level + trend + covariate_effect
    # ... likelihood ...
```

This keeps the HSGP as a composable component rather than baking it into specific model functions.
