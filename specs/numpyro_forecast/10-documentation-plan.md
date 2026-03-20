# 10 — Documentation Plan

## Stack

- **Sphinx** with **`myst-nb`** for rendering Jupyter notebooks as documentation pages.
- **`sphinx-book-theme`** for clean, readable layout.
- **ReadTheDocs** for hosting and automated builds on push.

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
│   ├── core.rst            # ModelFn, MCMCParams, SVIParams, ForecastResult, CVResult
│   ├── models.rst          # All pre-built model functions
│   ├── inference.rst       # run_mcmc, run_svi, forecast, check_diagnostics
│   ├── metrics.rst         # crps_empirical, per_obs_crps, energy_score, mae, etc.
│   ├── cv.rst              # time_slice_cv, expanding_window_cv
│   └── utils.rst           # periodic_features, data helpers, plotting
│
├── tutorials/              # Narrative tutorials as myst-nb notebooks
│   ├── quickstart.ipynb
│   ├── custom_model.ipynb
│   ├── hierarchical.ipynb
│   ├── cv_workflow.ipynb
│   ├── intermittent.ipynb
│   ├── hsgp_covariates.ipynb
│   └── deepar.ipynb
│
├── examples/               # Adapted from existing notebooks (lighter, focused)
│   ├── exponential_smoothing.ipynb
│   ├── sarimax.ipynb
│   ├── arma.ipynb
│   ├── var_irf.ipynb
│   ├── croston_tsb.ipynb
│   ├── local_level_fourier.ipynb
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

End-to-end example: generate data → fit level model → forecast → evaluate CRPS → plot.

**Covers:** `level_model`, `run_mcmc`, `forecast`, `crps_empirical`, `plot_forecast`.

### 2. Custom Model (`tutorials/custom_model.ipynb`)

Build a model from scratch using components. Shows the `scan + condition` pattern, how to add custom priors, and how to use the inference toolkit with a non-standard model.

**Covers:** `components/`, `ModelFn` protocol, `run_mcmc`, `forecast`.

### 3. Hierarchical Forecasting (`tutorials/hierarchical.ipynb`)

Multi-series Holt-Winters with group-level pooling. Demonstrates SVI for scalability.

**Covers:** `hierarchical_holt_winters_model`, `run_svi`, `forecast_svi`, `prepare_hierarchical_mapping`.

**Adapted from:** `hierarchical_exponential_smoothing.ipynb`.

### 4. Cross-Validation Workflow (`tutorials/cv_workflow.ipynb`)

Time-slice CV on intermittent demand data. Shows `prepare_data_fn` pattern and per-fold metrics.

**Covers:** `time_slice_cv`, `prepare_tsb_data`, `plot_cv_results`.

**Adapted from:** `tsb_numpyro.ipynb`, `zi_tsb_numpyro.ipynb`.

### 5. Intermittent Demand (`tutorials/intermittent.ipynb`)

Comparison of Croston, TSB, and ZI-TSB on the same dataset.

**Covers:** `croston_model`, `tsb_model`, `zi_tsb_model`, data prep utilities.

**Adapted from:** `croston_numpyro.ipynb`, `tsb_numpyro.ipynb`, `zi_tsb_numpyro.ipynb`.

### 6. HSGP Time-Varying Covariates (`tutorials/hsgp_covariates.ipynb`)

Adding smooth, non-parametric covariate effects to a forecasting model using Hilbert Space GPs.

**Covers:** `hsgp_covariate_effect` component, composing with existing models, interpreting the GP effect.

### 7. DeepAR Forecasting (`tutorials/deepar.ipynb`)

Probabilistic forecasting with a simple RNN-based model. Demonstrates SVI training and multi-series forecasting.

**Covers:** `deepar_model`, `run_svi`, `forecast_svi`, comparing with classical models via CRPS.

## Example Notebooks

Lighter versions of the user's existing notebooks, adapted to use the package API instead of inline code. Each example:

- Uses `probcast` imports instead of inline helpers.
- Focuses on the model and results, not infrastructure.
- Includes ArviZ diagnostics and CRPS evaluation.

| Example | Source Notebook |
|---------|----------------|
| `exponential_smoothing.ipynb` | `exponential_smoothing_numpyro.ipynb` |
| `arma.ipynb` | `arma_numpyro.ipynb` |
| `var_irf.ipynb` | `var_numpyro.ipynb` |
| `croston_tsb.ipynb` | `croston_numpyro.ipynb`, `tsb_numpyro.ipynb` |
| `local_level_fourier.ipynb` | `numpyro_forecasting_univariate.ipynb` |
| `sarimax.ipynb` | New — demonstrates SARIMAX with exogenous regressors |
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
