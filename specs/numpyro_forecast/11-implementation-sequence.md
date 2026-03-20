# 11 — Implementation Sequence

## Phase 1: Foundation (Core + Metrics + Utils)

**Goal:** Remove duplication from existing notebooks immediately. After this phase, notebooks can `import` instead of copy-pasting helpers.

### Deliverables

- `core/types.py` — `ModelFn` protocol, `ForecastResult`, `CVResult`
- `core/params.py` — `MCMCParams`, `SVIParams`
- `core/prior.py` — `Prior` Pydantic class for prior injection and hierarchical composition
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

### Exit criteria

- Core types/configs compile and are importable from package root.
- `Prior` class supports flat and nested (hierarchical) prior trees with `sample()`.
- `future` usage is keyword-only in documented public call patterns.
- Metrics/utils unit tests pass with deterministic fixtures.

## Phase 2: Components (Building Blocks)

**Goal:** Extract reusable transition functions from existing models. All components must broadcast over batch dimensions from day one.

### Deliverables

- `components/level.py` — level transition (local level / random walk)
- `components/trend.py` — local linear trend, smooth trend, deterministic trend, damped trend
- `components/seasonality.py` — additive seasonality rotation, trigonometric (Fourier state-space), Fourier regression
- `components/cycle.py` — stochastic damped cycle
- `components/ar.py` — autoregressive transition
- `components/ma.py` — moving average with error state
- `components/regression.py` — exogenous regression (static coefficients)
- `components/intermittent.py` — TSB demand/probability updates
- Unit tests for each component (deterministic, known-value, **both univariate and batch shapes**)

### Why second?

Components are extracted *from* the models. Having the core and inference layer stable means we can test components by plugging them into real inference pipelines.

### Exit criteria

- Every component has deterministic unit tests for univariate and batch shapes.
- HSGP exception is explicitly documented as the only component-level sampling exception.
- No unresolved shape-contract issues remain for component composition.

## Phase 3: Pre-built Models (Classical)

**Goal:** Provide ready-to-use model functions with the `priors: dict[str, Prior]` injection pattern. The UCM is the centrepiece; other models are either UCM wrappers or distinct model families. All models must work on both `(t_max,)` and `(t_max, n_series)` shapes. All panel-capable models accept `group_mapping: Array | None` for hierarchical priors — hierarchy is a cross-cutting capability expressed through nested `Prior` objects, not a separate model.

### Deliverables

- `models/ucm.py` — **`ucm_model`** (composable: level, trend, seasonal, cycle, AR, regression) + convenience aliases (`local_level_model`, `local_linear_trend_model`, `smooth_trend_model`). Accepts `group_mapping`.
- `models/exponential_smoothing.py` — `level_model`, `level_trend_model`, `holt_winters_model`, `damped_holt_winters_model` (thin UCM wrappers). All forward `group_mapping` to `ucm_model`.
- `models/sarimax.py` — `sarimax_model`. Accepts `group_mapping`.
- `models/intermittent.py` — `croston_model`, `tsb_model`, `zi_tsb_model` (hierarchical extension deferred — different input contract).
- `models/arma.py` — `arma_model`. Accepts `group_mapping`.
- `models/var.py` — `var_model`, `compute_irf`. Accepts `group_mapping`.
- Integration tests for each model (short MCMC runs, **shape checks for both univariate and panel**, hierarchical prior tests with `group_mapping`)

### Why third?

Models compose components + core. They are the user-facing API and need both layers to be stable before building. The UCM comes first in this phase because the ES wrappers depend on it.

### Exit criteria

- Each model has at least one passing short-run integration test.
- Every model defines a `*_DEFAULT_PRIORS` constant and accepts `priors: dict[str, Prior] | None = None`.
- All panel-capable models accept `group_mapping` and correctly create `plate("groups")` + `plate("series")` when it is provided.
- Hierarchical behavior is validated via nested `Prior` objects with `numpyro.plate` on at least two model families (e.g., ES and ARMA).
- Baseline model docs include identifiability/stability notes where applicable (ARMA/SARIMAX/VAR/HSGP/UCM).
- Forecast outputs expose stable and documented `return_sites`.

## Phase 4: Cross-Validation

**Goal:** Generic time-series CV routines.

### Deliverables

- `cv/time_series.py` — `time_slice_cv()`, `expanding_window_cv()`
- `utils/plotting.py` — `plot_forecast()`, `plot_cv_results()`, `plot_irf()`
- Integration tests with real models + CV

### Why fourth?

CV depends on inference + models + metrics. It's the capstone that ties everything together. Plotting is deferred here because it's only needed for visual validation.

### Exit criteria

- CV routines enforce no-leakage semantics and fold-local data preparation.
- Per-horizon metrics are emitted and documented.
- Integration tests cover at least one exogenous and one intermittent workflow.

## Phase 5: Advanced Models (DeepAR + HSGP)

**Goal:** Add neural-network-based and GP-based models for users who need more flexibility.

### Deliverables

- `nn/rnn.py` — `DeepARCell` (`flax.nnx.Module`) implementing GRU/LSTM cells for DeepAR. Architecture choices (hidden size, layers, cell type) are made at construction time.
- `nn/attention.py` — simple temporal attention layer (`flax.nnx.Module`)
- `models/deepar.py` — `deepar_model(y, rnn, ...)`, `attention_deepar_model(y, rnn, ...)`. The NN is a pre-built `flax.nnx.Module` passed as a positional argument and registered inside the model via `numpyro.contrib.module.nnx_module` (deterministic, default) or `random_nnx_module` (Bayesian weights, opt-in via `bayesian_nn=True`).
- `components/hsgp.py` — `hsgp_covariate_effect` (wrapping `numpyro.contrib.hsgp`)
- Integration tests for DeepAR (SVI-only, shape checks, both `nnx_module` and `random_nnx_module` paths) and HSGP component
- Optional dependency on `flax` (nnx API) for neural network layers

### Why fifth?

These are more advanced models that depend on all previous layers being stable. DeepAR requires SVI infrastructure. HSGP is a component that can be composed with any existing model. Both can be shipped as optional features without blocking the core release.

### Exit criteria

- DeepAR SVI-only behavior is documented and validated by tests.
- Both `nnx_module` (deterministic) and `random_nnx_module` (Bayesian) paths are tested.
- HSGP integration examples include prior sensitivity guidance.
- Optional dependency failures produce clear user-facing messages.

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

### Exit criteria

- CI enforces lint, typing, tests, docs strict build, and API docs validation.
- Release workflow and deprecation/versioning policy are documented.
- Principal sign-off checklist items are traceable to spec sections.

## Statistical quality gate (cross-phase)

Before a model family is considered complete, baseline examples/integration tests must satisfy:
- MCMC diagnostics: `R-hat <= 1.01`, adequate ESS bulk/tail, no unexplained divergences.
- Calibration artifacts: CRPS (or energy score) by horizon, and interval coverage/PIT diagnostics.
- If checks fail, model is either reparameterized or explicitly marked experimental.

## Known Challenges

### 0. DeepAR + SVI-only constraint

DeepAR models use `flax.nnx` modules registered with NumPyro via `nnx_module` / `random_nnx_module`. The NN is built externally and passed into the model function as a positional argument. Key constraints:
- `run_mcmc` will not work with DeepAR — this should be documented clearly and raise an informative error.
- The `ModelFn` protocol still applies (the `rnn` is just the second positional arg), but the inference path is restricted to SVI.
- With `bayesian_nn=True`, `random_nnx_module` places a prior on all NN weights — this is still SVI-recommended, not MCMC.
- The `flax` (nnx) dependency is optional — DeepAR imports should fail gracefully with a clear message if flax is not installed.

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

All panel-capable models use plates with `dim=-1` and `dim=-2` for series and seasons when `group_mapping` is provided. Getting broadcasting right across transition functions is error-prone.

**Mitigation:** Thorough shape annotations with `jaxtyping`. Integration tests that verify output shapes match expected `(t_max, n_series)`.

### 4. Intermittent data preparation

Croston/TSB models require non-trivial data preprocessing (extracting non-zero demands, computing inter-arrival periods). The `prepare_data_fn` callback pattern in CV must handle edge cases (all-zero series, single demand).

**Mitigation:** Validate inputs in `prepare_intermittent_data` and `prepare_tsb_data`. Add edge-case tests.

### 5. Batch dimension broadcasting

Components must broadcast correctly over `...` trailing batch dimensions. This is straightforward for element-wise operations (level, trend) but requires care for:
- **Seasonal rotation:** `jnp.roll` and array indexing with batch dims.
- **Trigonometric seasonality:** State is `(2, n_harmonics, *batch)` — matrix operations must use correct axes.
- **AR lags:** `update_lags` shifts along axis 0 with batch dims trailing.
- **Plates vs vmap:** Models with `group_mapping` (or nested `Prior` objects) use `numpyro.plate` for shared/pooled priors across series; non-hierarchical models can use `jax.vmap` for independent fits. The choice affects shape conventions.

**Mitigation:** Comprehensive shape tests for every component with `(t_max,)`, `(t_max, 1)`, and `(t_max, n_series)` inputs. Use `jaxtyping` + `beartype` to catch shape mismatches early.

### 6. ARMA error conditioning

The error conditioning pattern in ARMA is conceptually different from the `scan + condition` pattern. Users building custom ARMA variants need to understand why direct observation conditioning fails for MA terms.

**Mitigation:** Document the pattern clearly in `models/arma.py` docstrings and the custom model tutorial.

## Dependency Graph

```
Phase 1: core/ + inference/ + metrics/ + utils/ (features, data)
    ↓
Phase 2: components/ (core deterministic components)
    ↓
Phase 3: models/ (UCM core + ES wrappers, SARIMAX, ARMA, VAR, intermittent — all with group_mapping)
    ↓
Phase 4: cv/ + utils/plotting
    ↓
Phase 5: nn/ + models/deepar + components/hsgp
    ↓
Phase 6: docs/ + CI/CD + packaging + AGENTS.md + SKILLS.md
```

Each phase is independently releasable. Phase 1 alone provides value to existing notebook workflows. Phases 1–4 constitute the core release; Phase 5 is an optional advanced feature set.
