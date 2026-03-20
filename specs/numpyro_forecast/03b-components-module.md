# 03b — Components Module

## Design Principle

Components are **pure transition functions** — they implement a single forecasting mechanism (level update, trend update, seasonal rotation, cycle, etc.) without sampling priors. The calling model function is responsible for sampling priors and passing them as arguments.

This separation enables:
- **Reuse:** The same level component appears in simple ES, Holt-Winters, Croston, UCM, and hierarchical models.
- **Unit testing:** Components can be tested deterministically with known inputs — no inference needed.
- **Composition:** Users can mix components to build novel models — the UCM model is the canonical example.

## Batch Dimension Convention

All components use `Float[Array, "..."]` for state and parameters. The `...` absorbs trailing batch dimensions, so the same component code works for:
- **Univariate:** `carry` is a scalar, `y` has shape `(t_max,)`
- **Panel:** `carry` has shape `(n_series,)`, `y` has shape `(t_max, n_series)`

JAX broadcasting handles the rest. No separate univariate/panel implementations needed.

## UCM Component Catalogue

The following components map 1:1 to [statsmodels `UnobservedComponents`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.structural.UnobservedComponents.html) but in a Bayesian, composable form:

| Component | statsmodels equivalent | File |
|-----------|----------------------|------|
| Local level (random walk) | `level='local level'` | `level.py` |
| Local linear trend | `trend='local linear trend'` | `trend.py` |
| Smooth trend | `trend='smooth trend'` (fixed level, stochastic slope) | `trend.py` |
| Random walk with drift | `trend='random walk with drift'` | `trend.py` |
| Damped trend | `trend='damped'` | `trend.py` |
| Deterministic trend | `trend='deterministic trend'` | `trend.py` |
| Additive seasonality (HW) | `seasonal=n` (state-space rotation) | `seasonality.py` |
| Trigonometric seasonality | `freq_seasonal` (Fourier harmonics in state space) | `seasonality.py` |
| Stochastic cycle | `cycle=True` | `cycle.py` |
| Autoregressive | `autoregressive=p` | `ar.py` |
| Regression | `exog` | `regression.py` |
| Irregular (noise) | Always present | (handled in model, not a component) |

## Level (`components/level.py`)

### `level_transition`

Core exponential smoothing update.

```python
def level_transition(
    carry: Float[Array, "..."],
    t: int,
    y: Float[Array, " t_max"],
    t_max: int,
    level_smoothing: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Update the level state via exponential smoothing.

    level_t = alpha * y_t + (1 - alpha) * level_{t-1}   (if t < t_max)
    level_t = level_{t-1}                                (if t >= t_max, forecast mode)

    Parameters
    ----------
    carry
        Previous level value(s). Scalar for univariate, array for panel.
    t
        Current time index.
    y
        Observed time series.
    t_max
        Length of observed data (beyond this, level is frozen for forecasting).
    level_smoothing
        Smoothing parameter alpha in [0, 1].

    Returns
    -------
    Updated level value(s), same shape as carry.
    """
    previous_level = carry
    level = jnp.where(
        t < t_max,
        level_smoothing * y[t] + (1 - level_smoothing) * previous_level,
        previous_level,
    )
    return level
```

**Source:** Extracted from `transition_fn` in `exponential_smoothing_numpyro.ipynb` (level_model).

## Trend (`components/trend.py`)

Provides multiple trend specifications matching the statsmodels UCM trend options.

### `local_linear_trend_transition`

Full stochastic trend: both level and slope have innovation terms.

```python
def local_linear_trend_transition(
    level: Float[Array, "..."],
    previous_level: Float[Array, "..."],
    previous_trend: Float[Array, "..."],
    t: int,
    t_max: int,
    trend_smoothing: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Update the trend state for local linear trend model.

    trend_t = beta * (level_t - level_{t-1}) + (1 - beta) * trend_{t-1}

    In state-space form:
        level_t = level_{t-1} + trend_{t-1} + eta_level_t
        trend_t = trend_{t-1} + eta_trend_t
    """
```

### `smooth_trend_transition`

Smooth trend: level is deterministic (no innovation), slope is stochastic.

```python
def smooth_trend_transition(
    previous_level: Float[Array, "..."],
    previous_trend: Float[Array, "..."],
    trend_innovation: Float[Array, "..."],
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """Smooth trend: level follows slope deterministically, slope has innovations.

    level_t = level_{t-1} + trend_{t-1}
    trend_t = trend_{t-1} + eta_trend_t
    """
```

### `deterministic_trend`

No stochastic component — useful as a baseline or for short series.

```python
def deterministic_trend(
    t: int,
    intercept: Float[Array, "..."],
    slope: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Deterministic linear trend: mu_t = intercept + slope * t"""
```

### `damped_trend_prediction`

```python
def damped_trend_prediction(
    level: Float[Array, "..."],
    trend: Float[Array, "..."],
    phi: Float[Array, "..."],
    step: int,
) -> Float[Array, "..."]:
    """Compute damped trend prediction h steps ahead.

    mu = level + sum_{i=1}^{step} phi^i * trend

    Uses ``jax.lax.fori_loop`` for JIT-compatible summation.
    """
```

**Source:** `damped_holt_winters_model` in `exponential_smoothing_numpyro.ipynb`, which uses `fori_loop(1, step+1, lambda i, val: val + phi**i, 0)`.

## Seasonality (`components/seasonality.py`)

### `additive_seasonality_transition`

```python
def additive_seasonality_transition(
    y_t: Float[Array, "..."],
    level: Float[Array, "..."],
    trend: Float[Array, "..."],
    previous_seasonality: Float[Array, "n_seasons ..."],
    t: int,
    t_max: int,
    seasonality_smoothing: Float[Array, "..."],
) -> tuple[Float[Array, "..."], Float[Array, "n_seasons ..."]]:
    """Update the seasonal state via additive Holt-Winters recursion.

    s_new = gamma * (y_t - level - trend) + (1 - gamma) * s_old
    Rotates the seasonal array: drops current season, appends updated.

    Returns
    -------
    current_season
        Current seasonal component (s_old before rotation).
    seasonality
        Updated seasonal array after rotation.
    """
```

**Source:** `holt_winters_model` transition function in `exponential_smoothing_numpyro.ipynb` and `hierarchical_exponential_smoothing.ipynb`.

### `trigonometric_seasonal_transition`

State-space trigonometric seasonality (stochastic Fourier harmonics). Each harmonic `j` has a 2D state `[gamma_j, gamma_j*]` that rotates at frequency `2*pi*j/s`.

```python
def trigonometric_seasonal_transition(
    previous_state: Float[Array, "2 n_harmonics ..."],
    frequency: Float[Array, " n_harmonics"],
    innovation: Float[Array, "2 n_harmonics ..."] | None = None,
) -> tuple[Float[Array, "..."], Float[Array, "2 n_harmonics ..."]]:
    """Trigonometric seasonal transition (freq_seasonal in statsmodels).

    For each harmonic j:
        gamma_j(t)  = cos(lambda_j) * gamma_j(t-1) + sin(lambda_j) * gamma_j*(t-1) + omega_j(t)
        gamma_j*(t) = -sin(lambda_j) * gamma_j(t-1) + cos(lambda_j) * gamma_j*(t-1) + omega_j*(t)

    where lambda_j = 2*pi*j/s.

    Returns
    -------
    seasonal_effect
        Sum of gamma_j across harmonics — the seasonal contribution at time t.
    updated_state
        New state array for carrying forward.
    """
```

**Source:** Corresponds to `freq_seasonal` in statsmodels UCM. More flexible than additive HW seasonality because it supports multiple seasonal periods simultaneously (e.g., weekly + yearly).

### `fourier_regression`

```python
def fourier_regression(
    covariates: Float[Array, "t_max feature_dim"],
    weight: Float[Array, " feature_dim"],
    bias: float,
) -> Float[Array, " t_max"]:
    """Compute seasonal component as weighted sum of Fourier features.

    mu = bias + covariates @ weight

    Note: This is a *deterministic* (regression-based) Fourier approach.
    For stochastic Fourier seasonality, use ``trigonometric_seasonal_transition``.
    """
```

**Source:** `numpyro_forecasting_univariate.ipynb` — `(weight * covariates).sum(axis=-1, keepdims=True)`.

## Autoregressive (`components/ar.py`)

### `ar_transition`

```python
def ar_transition(
    y_lags: Float[Array, "n_lags ..."],
    phi: Float[Array, "n_lags ... ..."],
    constant: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Compute AR(p) prediction from lagged values.

    For univariate: mu = constant + sum_j phi_j * y_{t-j}
    For multivariate (VAR): mu = constant + einsum("lij,lj->i", phi, y_lags)
    """
```

**Source:** VAR transition in `var_numpyro.ipynb` — `jnp.einsum("lij,lj->i", phi[::-1], y_lags)`.

### `update_lags`

```python
def update_lags(
    y_lags: Float[Array, "n_lags ..."],
    y_new: Float[Array, "..."],
) -> Float[Array, "n_lags ..."]:
    """Shift lag window: drop oldest observation, append newest.

    new_lags = concat([y_lags[1:], y_new[None]], axis=0)
    """
```

## Moving Average (`components/ma.py`)

### `ma_error_step`

```python
def ma_error_step(
    y_t: float,
    predicted: float,
    previous_errors: Float[Array, " q"],
    theta: Float[Array, " q"],
) -> tuple[float, Float[Array, " q"]]:
    """Compute MA error and update error history.

    error = y_t - predicted
    ma_contribution = theta @ previous_errors

    Returns
    -------
    error
        Current prediction error.
    updated_errors
        Error history with newest error appended, oldest dropped.
    """
```

**Source:** `arma_1_1` in `arma_numpyro.ipynb` — `error = y[t] - pred; ma_part = theta * error_prev`.

## Cycle (`components/cycle.py`)

### `stochastic_cycle_transition`

Damped stochastic cycle — captures medium-term cyclical behaviour (business cycles, etc.) that is distinct from seasonality.

```python
def stochastic_cycle_transition(
    previous_state: Float[Array, "2 ..."],
    frequency: Float[Array, "..."],
    damping: Float[Array, "..."],
    innovation: Float[Array, "2 ..."] | None = None,
) -> tuple[Float[Array, "..."], Float[Array, "2 ..."]]:
    """Stochastic damped cycle transition.

    c_t     = rho * cos(lambda) * c_{t-1}   + rho * sin(lambda) * c*_{t-1} + kappa_t
    c*_t    = -rho * sin(lambda) * c_{t-1}  + rho * cos(lambda) * c*_{t-1} + kappa*_t

    Parameters
    ----------
    previous_state
        2D state ``[c, c*]`` with optional batch dimensions.
    frequency
        Cycle frequency lambda in (0, pi).
    damping
        Damping factor rho in (0, 1). Values close to 1 give persistent cycles.
    innovation
        Optional stochastic innovations ``[kappa, kappa*]``.

    Returns
    -------
    cycle_effect
        The cycle contribution at time t (first element of state).
    updated_state
        New ``[c, c*]`` state.
    """
```

**Source:** Corresponds to `cycle=True, damped_cycle=True` in statsmodels UCM. Uses the same 2D rotation structure as trigonometric seasonality but with a damping factor and a single (typically estimated) frequency.

## Regression (`components/regression.py`)

### `regression_effect`

Static (time-invariant) regression of exogenous covariates. For time-varying coefficients, use the HSGP component instead.

```python
def regression_effect(
    exog: Float[Array, "t_max n_exog *batch"],
    beta: Float[Array, "n_exog *batch"],
) -> Float[Array, "t_max *batch"]:
    """Compute linear regression effect: mu_t = exog_t @ beta.

    Parameters
    ----------
    exog
        Exogenous covariate matrix.
    beta
        Regression coefficients.

    Returns
    -------
    Regression contribution at each time step.
    """
```

**Source:** Standard exogenous regression as in SARIMAX and statsmodels UCM (`exog` argument). This is a pure function — the model samples `beta` from a prior and passes it in.

## Intermittent (`components/intermittent.py`)

### `tsb_transition`

```python
def tsb_transition(
    carry: tuple[Float[Array, "..."], Float[Array, "..."]],
    t: int,
    ts: Float[Array, " t_max"],
    t_max: int,
    z_smoothing: Float[Array, "..."],
    p_smoothing: Float[Array, "..."],
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """TSB dual-state transition: update demand level (z) and demand probability (p).

    z_next = alpha_z * y_t + (1 - alpha_z) * z_prev   (if y_t > 0)
    p_next = alpha_p + (1 - alpha_p) * p_prev          (if y_t > 0)
    p_next = (1 - alpha_p) * p_prev                    (if y_t = 0)

    Returns
    -------
    (z_next, p_next)
    """
```

**Source:** `tsb_model` in `tsb_numpyro.ipynb` and `zi_tsb_model` in `zi_tsb_numpyro.ipynb` (identical transition logic, different likelihoods).

## HSGP (`components/hsgp.py`)

### `hsgp_covariate_effect`

Wraps `numpyro.contrib.hsgp` to provide smooth, non-parametric time-varying covariate effects. This enables any model to incorporate covariates whose influence changes over time without specifying a parametric form.

```python
def hsgp_covariate_effect(
    x: Float[Array, "t_max n_features"],
    m: int = 20,
    c: float = 1.5,
    *,
    length_scale_prior: dist.Distribution = dist.InverseGamma(5, 5),
    amplitude_prior: dist.Distribution = dist.HalfNormal(1),
    name: str = "hsgp",
) -> Float[Array, " t_max"]:
    """Compute a time-varying covariate effect using Hilbert Space GP approximation.

    Parameters
    ----------
    x
        Covariate matrix (time x features). For a single time index,
        pass ``jnp.arange(t_max).reshape(-1, 1)``.
    m
        Number of basis functions for the HSGP approximation.
    c
        Boundary extension factor (controls how far the GP extends
        beyond the data range).
    length_scale_prior
        Prior on the GP length scale.
    amplitude_prior
        Prior on the GP amplitude (output scale).
    name
        NumPyro scope name for the GP parameters.

    Returns
    -------
    Time-varying effect, shape ``(t_max,)``.
    """
```

**Source:** [`numpyro.contrib.hsgp`](https://github.com/pyro-ppl/numpyro/tree/master/numpyro/contrib/hsgp). The component handles basis function computation and parameter sampling, returning a ready-to-use effect array that can be added to any model's mean function.

**Key design choice:** The HSGP component *does* include `numpyro.sample` calls (unlike other components which are pure transition functions). This is because the GP kernel hyperparameters (length scale, amplitude) are inherently part of the GP specification and separating them would make the API awkward. The component is scoped via the `name` parameter to avoid site name collisions.

## How Components Compose into Models

### Example 1: UCM model — the general case

The UCM model accepts configuration flags to enable/disable components. Internally it assembles the enabled components into a single scan loop:

```python
def ucm_model(
    y, future=0, *,
    level=True,           # local level (random walk)
    trend="local linear", # None, "local linear", "smooth", "deterministic", "damped"
    seasonal=None,        # None, int (additive HW), or {"type": "trigonometric", "period": 12, "harmonics": 4}
    cycle=False,          # stochastic damped cycle
    autoregressive=0,     # AR order
    exog=None,            # exogenous regressors (t_max, n_exog)
    # ... injectable priors for each enabled component ...
):
    # 1. Sample priors only for enabled components
    # 2. Build composite transition_fn
    def transition_fn(carry, t):
        state = {}
        mu = 0.0

        if level:
            state["level"] = level_transition(...)
            mu += state["level"]
        if trend:
            state["trend"] = local_linear_trend_transition(...)  # or smooth/damped/...
            mu += state["trend"]
        if seasonal:
            effect, state["seasonal"] = additive_seasonality_transition(...)  # or trigonometric
            mu += effect
        if cycle:
            effect, state["cycle"] = stochastic_cycle_transition(...)
            mu += effect
        if autoregressive > 0:
            mu += ar_transition(...)
        if exog is not None:
            mu += regression_effect(exog[t], beta)

        pred = numpyro.sample("pred", dist.Normal(loc=mu, scale=noise))
        return state, pred

    # 3. Run scan+condition
    with numpyro.handlers.condition(data={"pred": y}):
        _, preds = scan(transition_fn, init_carry, jnp.arange(t_max + future))
```

### Example 2: Holt-Winters as a UCM convenience wrapper

```python
def holt_winters_model(y, n_seasons, future=0, **prior_kwargs):
    """Additive Holt-Winters — a specific UCM configuration."""
    return ucm_model(
        y, future=future,
        level=True, trend="local linear",
        seasonal=n_seasons,
        **prior_kwargs,
    )
```

### Example 3: Batch dimensions — same model, single vs panel

```python
# Univariate — y.shape == (100,)
ucm_model(y_single, future=12, level=True, trend="smooth", seasonal=12)

# Panel — y.shape == (100, 50) — same function, zero changes
# Components broadcast over the 50-series batch dimension
ucm_model(y_panel, future=12, level=True, trend="smooth", seasonal=12)
```

Components handle the state update logic; models handle prior sampling, likelihood, and the scan+condition wrapper.
