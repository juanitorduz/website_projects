# 02 — Module Structure

## Directory Tree

```
probcast/
├── __init__.py                  # Public API re-exports
├── py.typed                     # PEP 561 marker
│
├── core/                        # Abstractions and type definitions
│   ├── __init__.py
│   ├── types.py                 # ModelFn protocol, ForecastResult, CVResult
│   ├── params.py                # MCMCParams, SVIParams (Pydantic)
│   └── prior.py                 # Prior class (Pydantic) for prior injection and hierarchical composition
│
├── components/                  # Composable building blocks (transition functions)
│   ├── __init__.py
│   ├── level.py                 # Level transitions (local level / random walk)
│   ├── trend.py                 # Trend transitions (local linear, smooth, damped, deterministic)
│   ├── seasonality.py           # Seasonal transitions (additive HW, trigonometric/Fourier, dummy) + feature helpers (periodic_features, fourier_modes, periodic_repeat)
│   ├── cycle.py                 # Stochastic cycle with damping and frequency
│   ├── ar.py                    # Autoregressive component
│   ├── ma.py                    # Moving average component
│   ├── regression.py            # Exogenous regression component (static coefficients)
│   ├── intermittent.py          # Demand/probability transitions (Croston, TSB)
│   └── hsgp.py                  # HSGP time-varying covariate component (wraps numpyro.contrib.hsgp)
│
├── models/                      # Pre-built model functions
│   ├── __init__.py
│   ├── ucm.py                   # Unobserved Components Model (composable: level, trend, season, cycle, AR, regression)
│   ├── exponential_smoothing.py # Level, level+trend, Holt-Winters, damped HW (UCM convenience wrappers)
│   ├── sarimax.py               # SARIMAX(p,d,q)(P,D,Q,s)
│   ├── intermittent.py          # Croston, TSB, ZI-TSB
│   ├── arma.py                  # ARMA(p,q)
│   ├── var.py                   # VAR(p) with IRF
│   └── deepar.py                # Simple DeepAR (RNN-based) probabilistic forecaster
│
├── nn/                          # Neural network building blocks (flax.nnx)
│   ├── __init__.py
│   ├── rnn.py                   # GRU/LSTM cells for DeepAR
│   └── attention.py             # Simple temporal attention layer
│
├── inference/                   # Inference runners and forecast helpers
│   ├── __init__.py
│   ├── mcmc.py                  # run_mcmc(), forecast()
│   ├── svi.py                   # run_svi(), forecast_svi()
│   └── diagnostics.py           # check_diagnostics()
│
├── metrics/                     # Scoring functions
│   ├── __init__.py
│   ├── crps.py                  # crps_empirical(), per_obs_crps(), energy_score()
│   └── point.py                 # mae, rmse, mape, wape, log_score
│
├── cv/                          # Cross-validation routines
│   ├── __init__.py
│   ├── time_series.py           # time_slice_cv(), expanding_window_cv(), train_test_split()
│   └── prepare.py               # prepare_intermittent_data(), prepare_tsb_data(), prepare_hierarchical_mapping()
│
└── plotting/                    # Visualization helpers (optional: requires matplotlib)
    ├── __init__.py
    ├── forecast.py              # plot_forecast()
    ├── cv.py                    # plot_cv_results()
    └── irf.py                   # plot_irf()
```

```
tests/
├── conftest.py                  # Shared fixtures (RNG keys, sample data)
├── test_core/
├── test_components/
├── test_models/
├── test_nn/
├── test_inference/
├── test_metrics/
├── test_cv/
├── test_plotting/
└── integration/                 # End-to-end tests (model → inference → forecast → metrics)
    ├── test_ucm.py
    ├── test_exponential_smoothing.py
    ├── test_sarimax.py
    ├── test_intermittent.py
    ├── test_var.py
    └── test_deepar.py
```

```
docs/
├── conf.py
├── index.rst
├── api/                         # Auto-generated API reference
├── tutorials/                   # Narrative tutorials (myst-nb notebooks)
└── _static/

# Root-level AI-friendly files
AGENTS.md                        # Agent instructions for AI coding assistants
SKILLS.md                        # Package skills/capabilities reference for AI tools
CONTRIBUTING.md                  # Developer setup, testing, PR guidelines
CODE_OF_CONDUCT.md               # Community code of conduct
LICENSE                          # Apache License 2.0
```

## Design Decision: Components vs Models

### Components (`components/`)

Low-level **transition functions** and **prior blocks** that implement a single forecasting mechanism. These are the building blocks:

```python
# Example: a level transition function
def level_transition(carry, t, y, time, level_smoothing):
    previous_level = carry
    level = jnp.where(
        t < time,
        level_smoothing * y[t] + (1 - level_smoothing) * previous_level,
        previous_level,
    )
    return level
```

Components are **pure functions** — no `numpyro.sample` calls for priors. The calling model function samples priors and passes them in.

**Explicit exception:** `components/hsgp.py` uses `numpyro.sample` internally via `Prior.sample()`, following the same `DEFAULT_PRIORS` + override pattern as model functions. This is the only component that samples — all others are pure transition functions. The justification is that GP kernel hyperparameters are intrinsic to the component's definition and cannot be meaningfully separated. This must be documented in the component docstring.

### Models (`models/`)

Complete **model functions** that assemble components, sample priors, and define the observation model via scan+condition. These follow the `ModelFn` protocol. Priors are injected via a `priors: dict[str, Prior] | None = None` parameter, merged with each model's `DEFAULT_PRIORS` constant:

```python
def level_model(y, *, future=0, priors=None):
    resolved = {**LEVEL_DEFAULT_PRIORS, **(priors or {})}
    level_smoothing = resolved["level_smoothing"].sample("level_smoothing")
    sigma = resolved["sigma"].sample("sigma")
    # Build transition_fn from components
    # Run scan+condition
    # Return forecast deterministic if future > 0
```

Models are what users pass to `run_mcmc()` or `run_svi()`. See [03-core-abstractions.md](03-core-abstractions.md) for the `Prior` class specification.

### Why This Split?

- **Reuse.** The same level component appears in simple exponential smoothing, Holt-Winters, Croston, and UCM. All panel-capable models support hierarchical priors via `group_mapping` and nested `Prior` objects — hierarchy is a cross-cutting capability, not a separate model.
- **Testability.** Components can be unit-tested with known inputs. Models require inference to test.
- **Customization.** Users can mix components to build novel models without touching inference code.
- **UCM as the canonical example.** The UCM model composes level + trend + seasonality + cycle + AR + regression components. The exponential smoothing models are thin convenience wrappers around specific UCM configurations.

### Batch Dimension Convention

All components and models follow a consistent batch dimension convention:

- **Univariate:** `y` has shape `(time,)`. Components carry scalar state.
- **Panel / multi-series:** `y` has shape `(time, *batch)` where `*batch` is typically `(n_series,)`. Components carry state with matching batch shape.

Components use `...` (ellipsis) in `jaxtyping` annotations for trailing batch dimensions. This means the same component code works for both univariate and panel data — no separate implementations. Models use `numpyro.plate` for the series dimension to enable hierarchical priors, or `jax.vmap` for independent fits.

## Public API Surface

The following canonical imports define the public API. These are re-exported from `probcast/__init__.py` and subpackage `__init__.py` files, and constitute the stable interface users should rely on.

```python
# Core abstractions
from probcast.core import MCMCParams, SVIParams, Prior, ForecastResult, CVResult

# Models
from probcast.models import (
    uc_model,
    level_model,
    holt_winters_model,
    damped_holt_winters_model,
    sarimax_model,
    croston_model,
    tsb_model,
    zi_tsb_model,
    arma_model,
    var_model,
    deepar_model,
)

# Inference
from probcast.inference import run_mcmc, run_svi, forecast, forecast_svi, to_datatree
from probcast.inference import check_diagnostics

# Metrics
from probcast.metrics import crps_empirical, per_obs_crps, energy_score
from probcast.metrics import mae, rmse, mape, wape, log_score

# Cross-validation
from probcast.cv import time_slice_cv, expanding_window_cv
from probcast.cv.prepare import train_test_split, prepare_croston_data, prepare_tsb_data

# Plotting (optional — requires matplotlib)
from probcast.plotting import plot_forecast, plot_cv_results, plot_irf
```

Each subpackage `__init__.py` should define `__all__` listing exactly the symbols above. Internal helpers, component functions, and NN building blocks are not part of the public API — users may import them directly from their submodules (e.g., `from probcast.components.level import level_transition`) but these are not guaranteed stable across minor versions.

## Dependency Flow

```
core/  ←  components/  ←  models/
  ↑            ↑              ↓
  │         nn/ ←─── models/deepar
  └──── inference/ ←─────────┘
            ↓
        metrics/  ←  cv/
            ↓
        plotting/ (optional, matplotlib)
```

No circular dependencies. `core/` depends on nothing internal (except `numpyro` and `pydantic` for the `Prior` class). `plotting/` is optional — only required if `matplotlib` is installed. `nn/` is optional — only required for DeepAR/attention models (uses `flax.nnx`). There is no catch-all `utils/` module — each function lives in its natural domain: Fourier/periodic helpers in `components/seasonality.py`, data preparation callbacks in `cv/prepare.py`, and plotting in `plotting/`.
