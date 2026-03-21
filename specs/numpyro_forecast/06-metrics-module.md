# 06 — Metrics Module

## Design Principle

All metric functions are **pure JAX** — no PyMC, no NumPy at runtime. This keeps the dependency footprint minimal and allows metrics to be JIT-compiled, vmapped over series, and used inside JAX-based cross-validation loops.

## Probabilistic Metrics (`metrics/crps.py`)

### `crps_empirical`

Computes the Continuous Ranked Probability Score from posterior predictive samples.

```python
def crps_empirical(
    truth: Float[Array, "*batch"],
    pred: Float[Array, "n_samples *batch"],
    sample_weight: Float[Array, "*batch"] | None = None,
) -> Float[Array, ""]:
    """Compute CRPS from empirical samples.

    CRPS(F, y) = E|X - y| - 0.5 * E|X - X'|

    where X, X' are independent draws from the forecast distribution F.

    Uses the efficient sorted-pairs estimator (Hersbach, 2000):
        CRPS = mean|X - y| - sum(diff * weight) / n^2

    Parameters
    ----------
    truth
        Observed values. Shape ``(*batch,)``.
    pred
        Posterior predictive samples. Shape ``(n_samples, *batch)``.
        First axis is the sample dimension.
    sample_weight
        Optional weights for averaging over batch dimensions.

    Returns
    -------
    Scalar CRPS value (lower is better).
    """
```

**Implementation (from `hierarchical_exponential_smoothing.ipynb` and `numpyro_forecasting_univariate.ipynb`):**

```python
absolute_error = jnp.mean(jnp.abs(pred - truth), axis=0)

num_samples = pred.shape[0]
if num_samples == 1:
    return jnp.average(absolute_error, weights=sample_weight)

pred = jnp.sort(pred, axis=0)
diff = pred[1:] - pred[:-1]
weight = jnp.arange(1, num_samples) * jnp.arange(num_samples - 1, 0, -1)
weight = weight.reshape(weight.shape + (1,) * (diff.ndim - 1))

per_obs_crps = absolute_error - jnp.sum(diff * weight, axis=0) / num_samples**2
return jnp.average(per_obs_crps, weights=sample_weight)
```

**Why standalone?** The `crps.ipynb` notebook currently imports from `pymc_marketing.metrics`. A standalone JAX implementation removes the PyMC dependency and enables JIT compilation.

### `per_obs_crps`

Returns per-observation CRPS values (useful for diagnostics — identifying which time points have poor calibration).

```python
def per_obs_crps(
    truth: Float[Array, "*batch"],
    pred: Float[Array, "n_samples *batch"],
) -> Float[Array, "*batch"]:
    """Compute per-observation CRPS (not averaged over batch dimensions)."""
```

Same algorithm as `crps_empirical` but returns before the final `jnp.average`.

### `energy_score`

Multivariate generalization of CRPS for models like VAR.

```python
def energy_score(
    truth: Float[Array, "time n_vars"],
    pred: Float[Array, "n_samples time n_vars"],
) -> Float[Array, ""]:
    """Compute multivariate energy score.

    ES(F, y) = E||X - y|| - 0.5 * E||X - X'||

    where ||.|| is the Euclidean norm over the variable dimension.
    """
```

**Rationale:** VAR models (`var_numpyro.ipynb`) produce multivariate forecasts where cross-variable calibration matters. CRPS applied independently per variable misses correlation structure.

## Point Metrics (`metrics/point.py`)

Standard metrics computed from the posterior predictive mean (or median).

```python
def mae(
    truth: Float[Array, "*batch"],
    pred: Float[Array, "*batch"],
) -> Float[Array, ""]:
    """Mean Absolute Error."""
    return jnp.mean(jnp.abs(truth - pred))


def rmse(
    truth: Float[Array, "*batch"],
    pred: Float[Array, "*batch"],
) -> Float[Array, ""]:
    """Root Mean Squared Error."""
    return jnp.sqrt(jnp.mean((truth - pred) ** 2))


def mape(
    truth: Float[Array, "*batch"],
    pred: Float[Array, "*batch"],
) -> Float[Array, ""]:
    """Mean Absolute Percentage Error.

    Warning: undefined when truth contains zeros.
    """
    return jnp.mean(jnp.abs((truth - pred) / truth))


def wape(
    truth: Float[Array, "*batch"],
    pred: Float[Array, "*batch"],
) -> Float[Array, ""]:
    """Weighted Absolute Percentage Error (scale-independent)."""
    return jnp.sum(jnp.abs(truth - pred)) / jnp.sum(jnp.abs(truth))


def log_score(
    truth: Float[Array, "*batch"],
    pred: Float[Array, "n_samples *batch"],
) -> Float[Array, ""]:
    """Deferred for v0.1 unless estimator is fully specified.

    If implemented, the estimator, bandwidth/binning policy, and
    multivariate behavior must be documented explicitly.
    """
```

### `interval_coverage`

Calibration metric for posterior predictive intervals at target nominal levels.

```python
def interval_coverage(
    truth: Float[Array, "horizon *batch"],
    pred: Float[Array, "n_samples horizon *batch"],
    levels: tuple[float, ...] = (0.5, 0.8, 0.95),
) -> dict[str, Float[Array, "horizon"]]:
    """Compute empirical coverage by forecast horizon for each nominal level."""
```

### `pit_values`

Probability integral transform values for calibration diagnostics.

```python
def pit_values(
    truth: Float[Array, "horizon *batch"],
    pred: Float[Array, "n_samples horizon *batch"],
) -> Float[Array, "horizon *batch"]:
    """Compute PIT values from empirical posterior predictive samples."""
```

### Horizon-conditioned outputs

For forecasting evaluation, probabilistic metrics should support per-horizon outputs:
- CRPS by horizon for univariate/panel forecasts,
- energy score by horizon for multivariate forecasts,
- coverage by horizon for selected interval levels,
- PIT values for calibration visualization and tests.

Baseline reporting for tutorials and integration tests should include:
- CRPS (or energy score) by horizon,
- interval coverage at 50/80/95% by horizon,
- PIT diagnostic artifact (histogram or equivalent summary).

## Usage Example

```python
from probcast.metrics import crps_empirical, mae, per_obs_crps

# After generating posterior predictive samples
crps_val = crps_empirical(y_test, forecast_samples)
mae_val = mae(y_test, forecast_samples.mean(axis=0))
obs_crps = per_obs_crps(y_test, forecast_samples)  # Per-timestep CRPS
```

## Comparison with Existing Approaches

| Metric | Current Source | Package |
|--------|---------------|---------|
| CRPS | `pymc_marketing.metrics.crps` (crps.ipynb) or inline (hierarchical notebook) | `probcast.metrics.crps_empirical` — pure JAX, no PyMC dep |
| MAE | `sklearn.metrics.mean_absolute_error` (crps.ipynb) | `probcast.metrics.mae` — pure JAX |
| Energy Score | Not implemented | `probcast.metrics.energy_score` — new |
