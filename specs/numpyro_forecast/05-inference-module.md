# 05 — Inference Module

## Overview

The inference module wraps NumPyro's MCMC and SVI machinery into ergonomic helpers that reduce boilerplate while preserving full access to underlying options via `**kwargs` passthrough. All functions work with any `ModelFn`-compatible function.

**Key design goal:** Support both MCMC and SVI with **custom optimizers and samplers**. Users should be able to swap NUTS for HMC or SA, use any `optax` optimizer with SVI, and draw arbitrary numbers of posterior samples.

## MCMC Inference (`inference/mcmc.py`)

### `run_mcmc`

A single unified function for all MCMC inference. Defaults to NUTS when no sampler is provided; accepts a pre-built sampler for HMC, SA, BarkerMH, etc.

```python
def run_mcmc(
    rng_key: Array,
    model: Callable,
    params: MCMCParams,
    *model_args,
    sampler: MCMCSampler | None = None,
    model_kwargs: dict[str, Any] | None = None,
    **nuts_kwargs,
) -> MCMC:
    """Run MCMC inference on a model function.

    When ``sampler`` is None (default), creates a NUTS sampler with
    ``**nuts_kwargs``. When ``sampler`` is provided, uses it directly
    and ``nuts_kwargs`` must be empty (raises ``ValueError`` if both
    are given).

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
    sampler
        Optional pre-built MCMC kernel (``NUTS``, ``HMC``, ``SA``,
        ``BarkerMH``). When None, a NUTS sampler is created with
        ``**nuts_kwargs``.
    model_kwargs
        Keyword arguments forwarded to the model (e.g., ``future``,
        ``priors``, ``covariates``). Passed to ``mcmc.run()``.
    **nuts_kwargs
        Additional keyword arguments passed to ``numpyro.infer.NUTS``
        when ``sampler`` is None (e.g., ``target_accept_prob``,
        ``max_tree_depth``, ``forward_mode_differentiation``).
        Ignored (with warning) if ``sampler`` is provided.

    Returns
    -------
    MCMC
        Fitted MCMC object. Access samples via ``mcmc.get_samples()``.

    Raises
    ------
    ValueError
        If both ``sampler`` and ``nuts_kwargs`` are provided.
    """
    if sampler is not None and nuts_kwargs:
        raise ValueError(
            "Cannot pass both `sampler` and `**nuts_kwargs`. "
            "When using a custom sampler, configure it before passing."
        )
    if sampler is None:
        sampler = NUTS(model, **nuts_kwargs)
    mcmc = MCMC(
        sampler=sampler,
        num_warmup=params.num_warmup,
        num_samples=params.num_samples,
        num_chains=params.num_chains,
    )
    mcmc.run(rng_key, *model_args, **(model_kwargs or {}))
    return mcmc
```

**Source:** `run_inference` in `exponential_smoothing_numpyro.ipynb`, `tsb_numpyro.ipynb`, `arma_numpyro.ipynb` (identical pattern).

**Design note:** The optional `sampler` parameter replaces the previous `run_mcmc_custom` function — one entry point instead of two. `nuts_kwargs` passthrough is critical for models that need `forward_mode_differentiation=True` (required when `scan` interacts with `condition` in some configurations).

**Usage examples:**
```python
# Default: NUTS with custom kwargs
mcmc = run_mcmc(rng_key, model, params, y, forward_mode_differentiation=True)

# Custom sampler: HMC
from numpyro.infer import HMC
mcmc = run_mcmc(rng_key, model, params, y, sampler=HMC(model, step_size=0.01))
```

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
    coords: dict | None = None,
    dims: dict | None = None,
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
        ``covariates`` or ``future_covariates``).
    return_sites
        Sites to return. Defaults to ``["y_forecast"]`` — the standardized
        primary forecast site across all models. Pass an explicit list to
        include model-specific diagnostic sites (e.g., ``["y_forecast", "errors"]``
        for ARMA). Pass ``None`` for NumPyro ``Predictive`` default behavior
        (all sample and deterministic sites).
    coords
        Coordinate metadata (e.g. time index, series ids) passed to
        ``to_datatree()``.
    dims
        Dimension names for each site (e.g. ``{"pred": ["time"]}``),
        passed to ``to_datatree()``.

    Returns
    -------
    ForecastResult
        Named tuple with ``samples`` dict and ``datatree``
        (``xarray.DataTree`` via ArviZ >= 1.0.0).
    """
    # Default to standardized primary forecast site
    if return_sites is None:
        return_sites = ["y_forecast"]
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
    dt = to_datatree(posterior_predictive=pred_samples, coords=coords, dims=dims)
    return ForecastResult(samples=pred_samples, datatree=dt, coords=coords, dims=dims)
```

**Source:** `forecast` helper in all notebooks. All models expose `"y_forecast"` as the primary forecast site. Model-specific diagnostic sites are available by passing an explicit `return_sites` list:
- Croston: `["y_forecast", "z_forecast", "p_inv_forecast"]`
- ARMA: `["y_forecast", "errors"]`

## SVI Inference (`inference/svi.py`)

### `run_svi`

```python
def run_svi(
    rng_key: Array,
    model: Callable,
    params: SVIParams,
    *model_args,
    model_kwargs: dict[str, Any] | None = None,
    guide: AutoGuide | None = None,
    optimizer: optax.GradientTransformation | None = None,
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
    model_kwargs
        Keyword arguments forwarded to the model (e.g., ``future``,
        ``priors``, ``covariates``). Passed to the SVI model call.
    guide
        AutoGuide instance. If None, resolved from ``params.build_guide(model)``.
    optimizer
        Optax optimizer. If None, resolved from ``params.build_optimizer()``.

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
    coords: dict | None = None,
    dims: dict | None = None,
) -> ForecastResult:
    """Generate posterior predictive forecasts from SVI parameters.

    Uses ``Predictive`` with guide and optimized params.

    Parameters
    ----------
    coords
        Coordinate metadata (e.g. time index, series ids) passed to
        ``to_datatree()``.
    dims
        Dimension names for each site, passed to ``to_datatree()``.

    Returns
    -------
    ForecastResult
        Named tuple with ``samples`` dict and ``datatree``
        (``xarray.DataTree`` via ArviZ >= 1.0.0).
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
    dt = to_datatree(posterior_predictive=pred_samples, coords=coords, dims=dims)
    return ForecastResult(samples=pred_samples, datatree=dt, coords=coords, dims=dims)
```

**Source:** `numpyro_forecasting_univariate.ipynb`:

```python
posterior = Predictive(
    model=model, guide=guide, params=svi_result.params,
    num_samples=5_000, return_sites=["y_forecast"],
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

This consolidates the ad-hoc diagnostic checks scattered across notebooks into a single reusable function. Uses ArviZ >= 1.0.0 summary statistics under the hood.

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

## ArviZ Integration (ArviZ >= 1.0.0 / `xarray.DataTree`)

All inference functions convert results to `xarray.DataTree` via `arviz.from_numpyro`:

```python
def to_datatree(
    mcmc: MCMC | None = None,
    posterior_predictive: dict[str, Array] | None = None,
    *,
    coords: dict | None = None,
    dims: dict | None = None,
    prior_config: dict[str, "Prior"] | None = None,
) -> "xr.DataTree":
    """Convert inference results to an xarray DataTree (ArviZ >= 1.0.0).

    Parameters
    ----------
    mcmc
        Fitted MCMC object.
    posterior_predictive
        Posterior predictive samples dict.
    coords
        Coordinate metadata (e.g. time index, series ids).
    dims
        Dimension names for each site.
    prior_config
        Optional dict of ``Prior`` objects used in the model run.
        When provided, serialized prior metadata is attached to
        ``datatree.attrs["prior_config"]`` for reproducibility.

    Returns
    -------
    xr.DataTree
        DataTree with posterior, posterior_predictive, and other groups.

    See Also
    --------
    arviz.from_numpyro : https://python.arviz.org/en/stable/api/generated/arviz.from_numpyro.html
    """
```

This wraps `arviz.from_numpyro()` with sensible defaults and handles both MCMC and SVI outputs. When `prior_config` is provided, the `Prior` objects are serialized via `model_dump()` and stored as DataTree attributes so that the exact prior configuration is recoverable from saved results.

## Model Serialization (Future — Phase 4-5)

**Gap:** Users need to save fitted models to disk and reload them for reproducibility, sharing, and deployment without re-running inference.

**Design direction:** Leverage existing serialization capabilities:
- `ForecastResult.datatree` is an `xr.DataTree` with native NetCDF/Zarr serialization.
- `Prior.model_dump()` gives JSON-serializable prior configs.
- MCMC samples are plain `dict[str, Array]` (saveable via `jnp.save` or pickle).

A future `save_result(path, result, model_config)` / `load_result(path)` pair should:
1. Save the DataTree (with posterior predictive samples and prior config in attrs) to NetCDF.
2. Save model configuration (prior keys, component toggles, model function name) as JSON metadata.
3. Optionally save raw MCMC samples as a separate artifact for re-forecasting.

This is deferred to Phase 4-5 but should be specced before the core release to ensure `to_datatree` stores sufficient metadata for round-tripping.
