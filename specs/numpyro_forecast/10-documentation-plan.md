# 10 — Documentation Plan

## Stack

- **Sphinx** with **`myst-nb`** for rendering Jupyter notebooks as documentation pages.
- **`sphinx-book-theme`** for clean, readable layout.
- **ReadTheDocs** for hosting and automated builds on push.
- Plotting examples should use `matplotlib` directly; do not introduce `seaborn`.
- Any ArviZ examples or diagnostics pages should assume `ArviZ >= 1.0.0` and use `xarray.DataTree` (via `arviz_base.from_numpyro`) instead of `arviz.InferenceData`.

**Source:** Follows NumPyro's own docs structure (`numpyro/docs/`), as referenced in the project requirements.

## Structure

```
docs/
├── conf.py
├── index.rst               # Landing page with overview and quick links
├── installation.rst        # pip/uv install instructions, optional deps
│
├── api/                    # Auto-generated API reference (one page per module)
│   ├── index.rst
│   ├── core.rst            # ModelFn, Prior, MCMCParams, SVIParams, ForecastResult, CVResult
│   ├── models.rst          # All pre-built model functions
│   ├── inference.rst       # run_mcmc, run_svi, forecast, check_diagnostics
│   ├── metrics.rst         # crps_empirical, per_obs_crps, energy_score, mae, etc.
│   ├── cv.rst              # time_slice_cv, expanding_window_cv, data preparation helpers
│   └── plotting.rst        # plot_forecast, plot_cv_results, plot_irf
│
├── tutorials/              # Narrative tutorials as myst-nb notebooks
│   ├── quickstart.ipynb
│   ├── ucm_guide.ipynb
│   ├── prior_config.ipynb
│   ├── custom_model.ipynb
│   ├── batch_panel.ipynb
│   ├── hierarchical.ipynb
│   ├── cv_workflow.ipynb
│   ├── intermittent.ipynb
│   ├── hsgp_covariates.ipynb
│   └── deepar.ipynb
│
├── examples/               # Adapted from existing notebooks (lighter, focused)
│   ├── ucm_components.ipynb
│   ├── exponential_smoothing.ipynb
│   ├── hierarchical_es.ipynb
│   ├── sarimax.ipynb
│   ├── arma.ipynb
│   ├── var_irf.ipynb
│   ├── croston_tsb.ipynb
│   └── deepar.ipynb
│
└── _static/
    └── logo.png
```

## API Reference

Auto-generated via `sphinx.ext.autodoc` + `sphinx.ext.napoleon` (NumPy-style docstrings):

```rst
.. automodule:: probcast.models.exponential_smoothing
   :members:
   :undoc-members:
```

Each API page groups functions by module with cross-references to related tutorials.

## Tutorials

### 1. Quickstart (`tutorials/quickstart.ipynb`)

End-to-end example: generate data → fit UCM (local level + trend) → forecast → evaluate CRPS → plot. Shows both univariate and panel data.

**Covers:** `uc_model`, `run_mcmc`, `forecast`, `crps_empirical`, `plot_forecast`, batch dimensions.

### 2. UCM Guide (`tutorials/ucm_guide.ipynb`)

Deep dive into the Unobserved Components Model. Shows how to compose level, trend (all variants), seasonality (additive, trigonometric), cycle, AR, and regression components. Compares different configurations on the same dataset.

**Covers:** `uc_model` with all component combinations, interpreting component decomposition, comparison with statsmodels UCM.

### 3. Prior Configuration (`tutorials/prior_config.ipynb`)

How to configure, override, and compose priors using the `Prior` class. Covers flat priors, hyperprior trees for hierarchical models, serialization for reproducibility, and attaching prior metadata to ArviZ results.

**Covers:** `Prior`, `DEFAULT_PRIORS` pattern, nested hierarchical priors, `model_dump()` / `model_validate()` round-trip, `to_arviz(prior_config=...)`.

### 4. Custom Model (`tutorials/custom_model.ipynb`)

Build a model from scratch using components. Shows the `scan + condition` pattern, how to add custom priors, and how to use the inference toolkit with a non-standard model.

**Covers:** `components/`, `ModelFn` protocol, `run_mcmc`, `forecast`.

### 5. Batch & Panel Forecasting (`tutorials/batch_panel.ipynb`)

Shows how the same model works on a single series and a panel of series. Demonstrates `numpyro.plate` for shared/hierarchical priors vs `jax.vmap` for independent fits.

**Covers:** Batch dimension convention, `uc_model` on panel data, `holt_winters_model` on panel data, performance tips.

### 6. Hierarchical Forecasting (`tutorials/hierarchical.ipynb`)

Demonstrates hierarchical priors as a cross-cutting capability applicable to any panel-capable model. Uses `group_mapping` + nested `Prior` objects for multi-level pooling. The primary worked example is hierarchical Holt-Winters (3-level: global → state → series), with a brief second example showing the same pattern applied to ARMA to prove universality. Demonstrates SVI for scalability.

**Covers:** `group_mapping` parameter, nested `Prior` hierarchies, `numpyro.plate`, `LocScaleReparam`, `run_svi`, `forecast_svi`, `prepare_hierarchical_mapping`. Shows that the same pattern works across model families.

**Adapted from:** `hierarchical_exponential_smoothing.ipynb`.

### 7. Cross-Validation Workflow (`tutorials/cv_workflow.ipynb`)

Time-slice CV on intermittent demand data. Shows `prepare_data_fn` pattern and per-fold metrics.

**Covers:** `time_slice_cv`, `prepare_tsb_data`, `plot_cv_results`.

**Adapted from:** `tsb_numpyro.ipynb`, `zi_tsb_numpyro.ipynb`.

### 8. Intermittent Demand (`tutorials/intermittent.ipynb`)

Comparison of Croston, TSB, and ZI-TSB on the same dataset.

**Covers:** `croston_model`, `tsb_model`, `zi_tsb_model`, data prep utilities.

**Adapted from:** `croston_numpyro.ipynb`, `tsb_numpyro.ipynb`, `zi_tsb_numpyro.ipynb`.

### 9. HSGP Time-Varying Covariates — Bikes GP (`tutorials/hsgp_covariates.ipynb`)

Reproduces and extends the [bikes GP blog post](https://juanitorduz.github.io/bikes_gp/) (originally written in PyMC) using probcast's HSGP component. The key additions over the original are: (1) a train-test split to showcase genuine forecasting (the PyMC version does not forecast), and (2) future covariate values (wind speed, temperature) are assumed known and passed via `future_covariates`.

**Covers:** `hsgp_covariate_effect` component, composing with UCM or custom models, train-test split with known future covariates, interpreting the GP effect, comparison with parametric seasonality.

**Adapted from:** [bikes_gp](https://juanitorduz.github.io/bikes_gp/) (PyMC implementation).

### 10. M5 Competition Models (`tutorials/m5_competition.ipynb`)

Reproduces all three models from the [Pyro-M5-Starter-Kit](https://github.com/pyro-ppl/Pyro-M5-Starter-Kit) using probcast. Demonstrates that the package can support the same model complexity as the original Pyro forecasting module. Includes covariates, hierarchical structure, and probabilistic evaluation.

**Covers:** `uc_model` with covariates, hierarchical priors, `run_svi`, `forecast_svi`, `crps_empirical`, custom model composition from components. Ports [model1.py](https://github.com/pyro-ppl/Pyro-M5-Starter-Kit/blob/master/model1.py), [model2.py](https://github.com/pyro-ppl/Pyro-M5-Starter-Kit/blob/master/model2.py), and [model3.py](https://github.com/pyro-ppl/Pyro-M5-Starter-Kit/blob/master/model3.py) to NumPyro/probcast.

**Adapted from:** [Pyro-M5-Starter-Kit](https://github.com/pyro-ppl/Pyro-M5-Starter-Kit).

### 11. DeepAR Forecasting (`tutorials/deepar.ipynb`)

Probabilistic forecasting with a simple RNN-based model. Demonstrates building a `DeepARCell` with `flax.nnx`, passing it into `deepar_model`, and SVI training with both deterministic (`nnx_module`) and Bayesian (`random_nnx_module`) NN weights.

**Covers:** `DeepARCell` construction, `deepar_model(y, rnn, ...)`, `bayesian_nn` flag, `run_svi`, `forecast_svi`, comparing with classical models via CRPS.

## Example Notebooks

Lighter versions of the user's existing notebooks, adapted to use the package API instead of inline code. Each example:

- Uses `probcast` imports instead of inline helpers.
- Focuses on the model and results, not infrastructure.
- Includes ArviZ diagnostics (via `xarray.DataTree`) and CRPS evaluation.

| Example | Source Notebook |
|---------|----------------|
| `ucm_components.ipynb` | New — UCM with various component combinations (local level, trend, cycle, trigonometric seasonal) |
| `exponential_smoothing.ipynb` | `exponential_smoothing_numpyro.ipynb` |
| `hierarchical_es.ipynb` | `hierarchical_exponential_smoothing.ipynb` — 3-level hierarchical Holt-Winters (the same `group_mapping` + nested `Prior` pattern applies to ARMA, SARIMAX, etc.) |
| `arma.ipynb` | `arma_numpyro.ipynb` |
| `var_irf.ipynb` | `var_numpyro.ipynb` |
| `croston_tsb.ipynb` | `croston_numpyro.ipynb`, `tsb_numpyro.ipynb` |
| `sarimax.ipynb` | New — SARIMAX with exogenous regressors |
| `deepar.ipynb` | New — simple DeepAR probabilistic forecasting |

## `conf.py` Key Settings

```python
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_nb",
]

napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "numpyro": ("https://num.pyro.ai/en/stable/", None),
    "arviz": ("https://python.arviz.org/en/stable/", None),
}

nb_execution_mode = "cache"
```

## Documentation Acceptance Gates

The docs pipeline is release-blocking and must satisfy all of the following:
- strict build: `sphinx-build -W docs docs/_build/html`;
- notebook policy: explicit execution mode, per-notebook timeout budget, and fail-on-execution-error behavior;
- API reference completeness check (aligned with `scripts/validate_api_docs.py`);
- link hygiene checks on a scheduled cadence or release workflow.

Minimum notebook execution policy for CI/docs:
- tutorial notebooks must run in a deterministic environment with fixed seeds;
- runtime-heavy notebooks may be marked as pre-executed artifacts, but this must be explicit;
- failures in required notebooks block merge/release.
