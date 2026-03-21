# 01 — Package Overview

## Name

`probcast`

## Vision

A flexible and powerful forecasting library built on NumPyro and JAX. At its heart is a **modular Unobserved Components Model (UCM)** — a Bayesian extension of [statsmodels `UnobservedComponents`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.structural.UnobservedComponents.html) — where users compose level, trend, seasonality, cycle, autoregressive, and regression components in any combination. The package also provides SARIMAX, ARMA, VAR, intermittent demand (Croston, TSB), hierarchical models, and advanced probabilistic models including DeepAR and time-varying covariates via Hilbert Space Gaussian Processes (HSGP).

Inspired by [Pyro's `contrib.forecast`](https://github.com/pyro-ppl/pyro/tree/dev/pyro/contrib/forecast) module, but designed around JAX's functional paradigm rather than PyTorch's object-oriented style.

The core strength of the package is **customizability** — reducing the boilerplate of writing custom NumPyro forecasting models (like those in the [NumPyro time series tutorial](https://num.pyro.ai/en/stable/tutorials/time_series_forecasting.html) and [availability-aware TSB](https://juanitorduz.github.io/availability_tsb/)) while offering simpler wrappers for the most common models.

## Core Principles

1. **Functional-first.** Models are plain Python functions following NumPyro's `def model(y, ..., future=0)` convention. No base classes to inherit from — just functions you can compose, scope, and pass around.

2. **Composable components.** Transition functions (level, trend, seasonality, cycle, AR, MA, regression) are standalone building blocks. Assemble them into full models by composition, not inheritance. The **UCM model** is the canonical example: pick which components to include, and probcast assembles the scan loop.

3. **Batch-native.** All components and models vectorize seamlessly over batch dimensions. A model that works on a single series `(time,)` works identically on a panel `(time, n_series)` — no code changes needed. Components use `...` trailing dimensions for broadcasting; `numpyro.plate` or `jax.vmap` handles the series axis.

4. **Two-layer API.**
   - **Toolkit layer** — inference runners, forecast helpers, metrics, CV routines. Use these with *any* NumPyro model function.
   - **Convenience layer** — pre-built model functions (UCM, exponential smoothing, Croston, ARMA, VAR, etc.) with injectable priors via keyword arguments.

5. **JAX-native.** All numerical code uses `jax.numpy` and `jax.lax.scan`. Models are JIT-compilable and `vmap`-friendly. No NumPy/SciPy at runtime.

6. **ArviZ integration.** Inference results convert to `xarray.DataTree` via `arviz.from_numpyro` (ArviZ >= 1.0.0) for diagnostics, plotting, and comparison. Forecast outputs align with ArviZ's `posterior_predictive` conventions.

7. **Typed-by-default core dependencies.** Core: `numpyro`, `jax`, `jaxlib`, `pydantic`, `jaxtyping`, `beartype`, `arviz`, `matplotlib`. Optional extras are reserved for non-core capabilities (for example neural-network backends and docs tooling).

8. **AI-friendly.** The repository includes `AGENTS.md` and `SKILLS.md` to make it easy for AI coding assistants to understand and contribute to the project.

## Target Users

- Practitioners who already write NumPyro models and want less boilerplate for common forecasting patterns.
- Teams that need probabilistic forecasts with proper uncertainty quantification (CRPS, calibration) and time-series cross-validation.
- Researchers exploring custom models who want a composable toolkit rather than a rigid framework.

## What This Package Is *Not*

- Not a black-box AutoML forecasting tool (see StatsForecast, Prophet).
- Not a replacement for NumPyro — it builds on top of it.
- Not tied to a specific data format — models operate on JAX arrays; data wrangling is the user's responsibility.

## Source Notebooks

The package design is grounded in these reference implementations:

| Notebook | Key Patterns |
|----------|-------------|
| `exponential_smoothing_numpyro.ipynb` | scan+condition, InferenceParams, run_inference, forecast helpers |
| `hierarchical_exponential_smoothing.ipynb` | Hierarchical plates, CRPS, SVI at scale |
| `numpyro_forecasting_univariate.ipynb` | Local level + Fourier, SVI, LocScaleReparam, periodic_features |
| `var_numpyro.ipynb` | VAR(p), einsum lags, IRF with lax.scan+vmap+jit, jaxtyping |
| `croston_numpyro.ipynb` | Croston with scope, time-slice CV |
| `tsb_numpyro.ipynb` | TSB with data prep helpers |
| `zi_tsb_numpyro.ipynb` | Zero-inflated likelihood, custom CV |
| `arma_numpyro.ipynb` | ARMA with error conditioning |
| `crps.ipynb` | CRPS metric analysis and comparison with MAE |
| `availability_tsb` ([blog post](https://juanitorduz.github.io/availability_tsb/)) | Custom model with availability covariates, highlights extensibility |

## External References

| Resource | Relevance |
|----------|-----------|
| [Pyro contrib.forecast](https://github.com/pyro-ppl/pyro/tree/dev/pyro/contrib/forecast) | API inspiration — functional forecast module on Pyro/PyTorch |
| [Pyro Forecasting III](https://pyro.ai/examples/forecasting_iii.html) | Advanced patterns (hierarchical, covariates) |
| [Pyro Simple Forecasting](https://pyro.ai/examples/forecast_simple.html) | Minimal forecasting API example |
| [Pyro DLM](https://pyro.ai/examples/forecasting_dlm.html) | Dynamic linear model in Pyro |
| [NumPyro HSGP](https://github.com/pyro-ppl/numpyro/tree/master/numpyro/contrib/hsgp) | Hilbert Space GP for time-varying covariates |
| [NumPyro Time Series Tutorial](https://num.pyro.ai/en/stable/tutorials/time_series_forecasting.html) | Base patterns this package should simplify |
| [Pyro-M5-Starter-Kit](https://github.com/pyro-ppl/Pyro-M5-Starter-Kit) | M5 competition models — validation target for probcast (all three models must be reproducible) |
| [statsmodels UnobservedComponents](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.structural.UnobservedComponents.html) | Reference for UCM component catalogue (level, trend, seasonal, cycle, AR, regression) |

