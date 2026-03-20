# 05 — Inference Module

## Overview

The inference module wraps NumPyro's MCMC and SVI machinery into ergonomic helpers that reduce boilerplate while preserving full access to underlying options via `**kwargs` passthrough. All functions work with any `ModelFn`-compatible function.

**Key design goal:** Support both MCMC and SVI with **custom optimizers and samplers**. Users should be able to swap NUTS for HMC or SA, use any `optax` optimizer with SVI, and draw arbitrary numbers of posterior samples.

## MCMC Inference (`inference/mcmc.py`)

### `run_mcmc`

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
    rng_key
        JAX PRNG key.
    model
        NumPyro model function following the ModelFn protocol.
    params
        MCMC configuration (num_warmup, num_samples, num_chains).
    *model_args
        Positional arguments forwarded to the model function.
    **nuts_kwargs
        Additional keyword arguments passed to ``numpyro.infer.NUTS``
        (e.g., ``target_accept_prob``, ``max_tree_depth``,
        ``forward_mode_differentiation``).

    Returns
    -------
    MCMC
        Fitted MCMC object. Access samples via ``mcmc.get_samples()``.
    """
    sampler = NUTS(model, **nuts_kwargs)
    mcmc = MCMC(
        sampler=sampler,
        num_warmup=params.num_warmup,
        num_samples=params.num_samples,
        num_chains=params.num_chains,
    )
    mcmc.run(rng_key, *model_args)
    return mcmc


def run_mcmc_custom(
    rng_key: Array,
    model: Callable,
    sampler: MCMCSampler,
    params: MCMCParams,
    *model_args,
) -> MCMC:
    """Run MCMC with a user-provided sampler (HMC, SA, BarkerMH, etc.).

    Parameters
    ----------
    sampler
        Any NumPyro MCMC kernel (``NUTS``, ``HMC``, ``SA``, ``BarkerMH``).
    """
    mcmc = MCMC(
        sampler=sampler,
        num_warmup=params.num_warmup,
        num_samples=params.num_samples,
        num_chains=params.num_chains,
    )
    mcmc.run(rng_key, *model_args)
    return mcmc
```

**Source:** `run_inference` in `exponential_smoothing_numpyro.ipynb`, `tsb_numpyro.ipynb`, `arma_numpyro.ipynb` (identical pattern).

**Design note:** `nuts_kwargs` passthrough is critical for models that need `forward_mode_differentiation=True` (required when `scan` interacts with `condition` in some configurations).

### `forecast`

```python
def forecast(
    rng_key: Array,
    model: Callable,
    samples: dict[str, Array],
    *model_args,
    future: int = 0,
    model_kwargs: dict[str, Any] | None = None,
    return_sites: list[str] | None = None,
) -> ForecastResult:
    """Generate posterior predictive forecasts from MCMC samples.

    Parameters
    ----------
    rng_key
        JAX PRNG key.
    model
        The same model function used for inference.
    samples
        Posterior samples from ``mcmc.get_samples()``.
    *model_args
        Positional arguments forwarded to the model.
    future
        Forecast horizon. Must be passed as keyword argument to keep calls
        consistent across model families.
    model_kwargs
        Additional keyword arguments forwarded to the model (for example
        ``exog`` or ``future_exog``).
    return_sites
        Sites to return. If None, returns all deterministic sites.

    Returns
    -------
    ForecastResult
        Named tuple with ``samples`` dict and optional ``idata``.
    """
    predictive = Predictive(
        model=model,
        posterior_samples=samples,
        return_sites=return_sites,
    )
    pred_samples = predictive(
        rng_key,
        *model_args,
        future=future,
        **(model_kwargs or {}),
    )
    return ForecastResult(samples=pred_samples)
```

**Source:** `forecast` helper in all notebooks. The `return_sites` parameter varies per model:
- Exponential smoothing: `["pred"]` (full series) or implicit
- Croston: `["z_forecast", "p_inv_forecast", "forecast"]`
- TSB/ZI-TSB: `["ts_forecast"]`
- ARMA: `["y_forecast", "errors"]`

## SVI Inference (`inference/svi.py`)

### `run_svi`

```python
def run_svi(
    rng_key: Array,
    model: Callable,
    params: SVIParams,
    *model_args,
    guide: AutoGuide | None = None,
    optimizer: optax.GradientTransformation | None = None,
    **model_kwargs,
) -> SVIRunResult:
    """Run SVI optimization on a model function.

    Parameters
    ----------
    rng_key
        JAX PRNG key.
    model
        NumPyro model function.
    params
        SVI configuration (num_steps, learning_rate, etc.).
    *model_args
        Positional arguments forwarded to the model.
    guide
        AutoGuide instance. If None, uses ``AutoNormal(model)``.
    optimizer
        Optax optimizer. If None, uses ``Adam(params.learning_rate)``.
    **model_kwargs
        Keyword arguments forwarded to the model.

    Returns
    -------
    SVIRunResult
        Result object with ``.params`` and ``.losses``.
    """
```

**Source:** SVI setup in `hierarchical_exponential_smoothing.ipynb` (`AutoDiagonalNormal`, Adam lr=0.03, 15k steps) and `numpyro_forecasting_univariate.ipynb` (`AutoNormal`, Adam lr=0.005, 50k steps).

### `forecast_svi`

```python
def forecast_svi(
    rng_key: Array,
    model: Callable,
    guide: AutoGuide,
    svi_params: dict,
    *model_args,
    future: int = 0,
    model_kwargs: dict[str, Any] | None = None,
    num_samples: int = 5_000,
    return_sites: list[str] | None = None,
) -> ForecastResult:
    """Generate posterior predictive forecasts from SVI parameters.

    Uses ``Predictive`` with guide and optimized params.
    """
    predictive = Predictive(
        model=model,
        guide=guide,
        params=svi_params,
        num_samples=num_samples,
        return_sites=return_sites,
    )
    pred_samples = predictive(
        rng_key,
        *model_args,
        future=future,
        **(model_kwargs or {}),
    )
    return ForecastResult(samples=pred_samples)
```

**Source:** `numpyro_forecasting_univariate.ipynb`:

```python
posterior = Predictive(
    model=model, guide=guide, params=svi_result.params,
    num_samples=5_000, return_sites=["obs"],
)
```

## Diagnostics (`inference/diagnostics.py`)

### `check_diagnostics`

```python
def check_diagnostics(
    mcmc: MCMC,
    *,
    rhat_threshold: float = 1.01,
    min_ess: int = 100,
    warn: bool = True,
) -> dict[str, Any]:
    """Check MCMC convergence diagnostics.

    Parameters
    ----------
    mcmc
        Fitted MCMC object.
    rhat_threshold
        Maximum acceptable R-hat value.
    min_ess
        Minimum acceptable effective sample size.
    warn
        If True, emit warnings for failed checks.

    Returns
    -------
    dict
        Keys: ``rhat_ok``, ``ess_ok``, ``divergences``, ``max_rhat``,
        ``min_ess``, ``num_divergences``.
    """
```

This consolidates the ad-hoc diagnostic checks scattered across notebooks into a single reusable function. Uses ArviZ summary statistics under the hood.

### Diagnostic policy (mandatory for baseline examples)

For release examples and integration tests, diagnostics are considered passing when:
- rank-normalized ``R-hat <= 1.01`` for key latent and forecast sites,
- ``ESS_bulk >= 400`` and ``ESS_tail >= 200`` for key sites,
- ``num_divergences == 0`` (or explicit documented rationale),
- no persistent max treedepth saturation,
- BFMI has no warning-level failures.

If these checks fail, either:
- reparameterize/tune and rerun, or
- mark the example/model as experimental with an explicit warning.

## ArviZ Integration

All inference functions support optional conversion to `arviz.InferenceData`:

```python
def to_arviz(
    mcmc: MCMC | None = None,
    posterior_predictive: dict[str, Array] | None = None,
    *,
    coords: dict | None = None,
    dims: dict | None = None,
    prior_config: dict[str, "Prior"] | None = None,
) -> "az.InferenceData":
    """Convert inference results to ArviZ InferenceData.

    Parameters
    ----------
    prior_config
        Optional dict of ``Prior`` objects used in the model run.
        When provided, serialized prior metadata is attached to
        ``idata.attrs["prior_config"]`` for reproducibility.
    """
```

This wraps `az.from_numpyro()` with sensible defaults and handles both MCMC and SVI outputs. When `prior_config` is provided, the `Prior` objects are serialized via `model_dump()` and stored as InferenceData attributes so that the exact prior configuration is recoverable from saved results.
