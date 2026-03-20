# 11 — Implementation Sequence

## Phase 1: Foundation (Core + Metrics + Utils)

**Goal:** Remove duplication from existing notebooks immediately. After this phase, notebooks can `import` instead of copy-pasting helpers.

### Deliverables

- `core/types.py` — `ModelFn` protocol, `ForecastResult`, `CVResult`
- `core/params.py` — `MCMCParams`, `SVIParams`
- `inference/mcmc.py` — `run_mcmc()`, `forecast()`
- `inference/svi.py` — `run_svi()`, `forecast_svi()`
- `inference/diagnostics.py` — `check_diagnostics()`
- `metrics/crps.py` — `crps_empirical()`, `per_obs_crps()`
- `metrics/point.py` — `mae`, `rmse`, `mape`, `wape`
- `utils/features.py` — `periodic_features()`, `fourier_modes()`
- `utils/data.py` — `train_test_split()`, `prepare_intermittent_data()`, `prepare_tsb_data()`
- `pyproject.toml` — project metadata, dependencies, ruff config
- `tests/conftest.py` + unit tests for metrics and utils

### Why first?

These are the most duplicated pieces across notebooks. Every notebook has its own `run_inference`, `forecast`, and `InferenceParams`. Consolidating them provides immediate value and establishes the API patterns that models and CV build on.

## Phase 2: Components (Building Blocks)

**Goal:** Extract reusable transition functions from existing models.

### Deliverables

- `components/level.py` — level transition (exponential smoothing core)
- `components/trend.py` — additive trend, damped trend
- `components/seasonality.py` — additive seasonality rotation, Fourier seasonality
- `components/ar.py` — autoregressive transition
- `components/ma.py` — moving average with error state
- `components/intermittent.py` — TSB demand/probability updates
- Unit tests for each component (deterministic, known-value)

### Why second?

Components are extracted *from* the models. Having the core and inference layer stable means we can test components by plugging them into real inference pipelines.

## Phase 3: Pre-built Models (Classical)

**Goal:** Provide ready-to-use model functions with injectable priors for classical time series models.

### Deliverables

- `models/exponential_smoothing.py` — `level_model`, `level_trend_model`, `holt_winters_model`, `damped_holt_winters_model`
- `models/sarimax.py` — `sarimax_model`
- `models/intermittent.py` — `croston_model`, `tsb_model`, `zi_tsb_model`
- `models/arma.py` — `arma_model`
- `models/var.py` — `var_model`, `compute_irf`
- `models/local_level.py` — `local_level_fourier_model`
- `models/hierarchical.py` — `hierarchical_holt_winters_model`
- Integration tests for each model (short MCMC runs, shape checks)

### Why third?

Models compose components + core. They are the user-facing API and need both layers to be stable before building.

## Phase 4: Cross-Validation

**Goal:** Generic time-series CV routines.

### Deliverables

- `cv/time_series.py` — `time_slice_cv()`, `expanding_window_cv()`
- `utils/plotting.py` — `plot_forecast()`, `plot_cv_results()`, `plot_irf()`
- Integration tests with real models + CV

### Why fourth?

CV depends on inference + models + metrics. It's the capstone that ties everything together. Plotting is deferred here because it's only needed for visual validation.

## Phase 5: Advanced Models (DeepAR + HSGP)

**Goal:** Add neural-network-based and GP-based models for users who need more flexibility.

### Deliverables

- `nn/rnn.py` — GRU/LSTM cells for DeepAR
- `nn/attention.py` — simple temporal attention layer
- `models/deepar.py` — `deepar_model`, `attention_deepar_model`
- `components/hsgp.py` — `hsgp_covariate_effect` (wrapping `numpyro.contrib.hsgp`)
- Integration tests for DeepAR (SVI-only, shape checks) and HSGP component
- Optional dependency on `flax` for neural network layers

### Why fifth?

These are more advanced models that depend on all previous layers being stable. DeepAR requires SVI infrastructure. HSGP is a component that can be composed with any existing model. Both can be shipped as optional features without blocking the core release.

## Phase 6: Documentation + Packaging + CI/CD

**Goal:** Make the package installable, documented, and tested in CI.

### Deliverables

- `docs/` — Sphinx setup, API reference, tutorials (including HSGP and DeepAR), examples
- GitHub Actions CI (lint + test matrix)
- PyPI packaging with `uv`
- README with quickstart example
- `CONTRIBUTING.md` — dev setup, testing, PR guidelines
- `CODE_OF_CONDUCT.md` — Contributor Covenant v2.1
- `AGENTS.md` — AI coding assistant instructions
- `SKILLS.md` — package capabilities reference for AI tools
- `LICENSE` — Apache License 2.0

### Why last?

Docs are best written against a stable API. Premature docs create maintenance burden when signatures change.

## Known Challenges

### 0. DeepAR + SVI-only constraint

DeepAR models use neural network weights that are impractical to sample via MCMC. They are SVI-only, which means:
- `run_mcmc` will not work with DeepAR — this should be documented clearly and raise an informative error.
- The `ModelFn` protocol still applies, but the inference path is restricted.
- The `flax` dependency is optional — DeepAR imports should fail gracefully with a clear message if flax is not installed.

### 1. `scan` + `condition` interaction

The `scan + condition` pattern used in all exponential smoothing models can cause gradient issues. Some configurations require `forward_mode_differentiation=True` in NUTS:

```python
mcmc = run_mcmc(rng_key, model, params, y, forward_mode_differentiation=True)
```

**Mitigation:** Document this in model docstrings. The `**nuts_kwargs` passthrough in `run_mcmc` makes it easy to enable.

### 2. `forward_mode_differentiation` performance

Forward-mode AD is slower than reverse-mode for models with many parameters. Hierarchical models with hundreds of series may hit performance walls with MCMC.

**Mitigation:** Recommend SVI for large hierarchical models (as done in `hierarchical_exponential_smoothing.ipynb`). Document the MCMC vs SVI tradeoff.

### 3. Hierarchical tensor shapes

The hierarchical model uses plates with `dim=-1` and `dim=-2` for series and seasons. Getting broadcasting right across transition functions is error-prone.

**Mitigation:** Thorough shape annotations with `jaxtyping`. Integration tests that verify output shapes match expected `(t_max, n_series)`.

### 4. Intermittent data preparation

Croston/TSB models require non-trivial data preprocessing (extracting non-zero demands, computing inter-arrival periods). The `prepare_data_fn` callback pattern in CV must handle edge cases (all-zero series, single demand).

**Mitigation:** Validate inputs in `prepare_intermittent_data` and `prepare_tsb_data`. Add edge-case tests.

### 5. ARMA error conditioning

The error conditioning pattern in ARMA is conceptually different from the `scan + condition` pattern. Users building custom ARMA variants need to understand why direct observation conditioning fails for MA terms.

**Mitigation:** Document the pattern clearly in `models/arma.py` docstrings and the custom model tutorial.

## Dependency Graph

```
Phase 1: core/ + inference/ + metrics/ + utils/ (features, data)
    ↓
Phase 2: components/ (including hsgp)
    ↓
Phase 3: models/ (classical: ES, SARIMAX, ARMA, VAR, intermittent, hierarchical)
    ↓
Phase 4: cv/ + utils/plotting
    ↓
Phase 5: nn/ + models/deepar + components/hsgp integration
    ↓
Phase 6: docs/ + CI/CD + packaging + AGENTS.md + SKILLS.md
```

Each phase is independently releasable. Phase 1 alone provides value to existing notebook workflows. Phases 1–4 constitute the core release; Phase 5 is an optional advanced feature set.
