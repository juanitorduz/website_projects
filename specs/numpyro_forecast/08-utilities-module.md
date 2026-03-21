# 08 — Utilities Redistribution & Plotting Module

## Design Decision: No Catch-All `utils/`

Functions that would traditionally live in a `utils/` directory are instead placed in their natural domain modules. This avoids an accumulation of loosely related helpers and keeps each module self-contained.

### Where each function now lives

| Function | New Location | Rationale |
|----------|-------------|-----------|
| `periodic_features()` | `components/seasonality.py` | Fourier basis generation is a seasonal building block |
| `periodic_repeat()` | `components/seasonality.py` | Seasonal pattern tiling is a seasonal building block |
| `fourier_modes()` | `components/seasonality.py` | Fourier harmonics at a given period |
| `train_test_split()` | `cv/prepare.py` | Used in CV and evaluation contexts |
| `prepare_intermittent_data()` | `cv/prepare.py` | Croston data preparation for CV callbacks |
| `prepare_tsb_data()` | `cv/prepare.py` | TSB data preparation for CV callbacks |
| `prepare_hierarchical_mapping()` | `cv/prepare.py` | Group mapping encoding used in hierarchical CV |
| `plot_forecast()` | `plotting/forecast.py` | Forecast visualization |
| `plot_cv_results()` | `plotting/cv.py` | CV results visualization |
| `plot_irf()` | `plotting/irf.py` | VAR impulse response function plotting |

See [03b-components-module.md](03b-components-module.md) for the Fourier/seasonal function specs, [07-cross-validation-module.md](07-cross-validation-module.md) for the data preparation specs, and below for the plotting specs.

## Plotting Module (`plotting/`)

Thin plotting helpers built on `matplotlib` directly, with optional ArviZ interoperability targeting `ArviZ >= 1.0.0` (which uses `xarray.DataTree` instead of `arviz.InferenceData`). Do not add a `seaborn` dependency. Optional — only imported if `matplotlib` is available.

### `plot_forecast` (`plotting/forecast.py`)

```python
def plot_forecast(
    y_train: Float[Array, " t_train"],
    y_test: Float[Array, " t_test"] | None = None,
    forecast_samples: Float[Array, "n_samples t_test"] | None = None,
    *,
    hdi_prob: float = 0.94,
    ax: "matplotlib.axes.Axes | None" = None,
    **plot_kwargs,
) -> "matplotlib.axes.Axes":
    """Plot observed data with forecast HDI bands.

    Uses ``arviz_plots`` for credible interval shading when available.
    """
```

### `plot_cv_results` (`plotting/cv.py`)

```python
def plot_cv_results(
    cv_result: "CVResult",
    y: Float[Array, " time"],
    *,
    hdi_prob: float = 0.94,
    max_folds: int | None = None,
    ax: "matplotlib.axes.Axes | None" = None,
) -> "matplotlib.axes.Axes":
    """Plot cross-validation forecasts overlaid on the original series."""
```

### `plot_irf` (`plotting/irf.py`)

```python
def plot_irf(
    irf_samples: Float[Array, "n_samples n_steps n_vars n_vars"],
    var_names: list[str],
    *,
    hdi_prob: float = 0.94,
    axes: "npt.NDArray[matplotlib.axes.Axes] | None" = None,
    figsize: tuple[float, float] | None = None,
) -> "npt.NDArray[matplotlib.axes.Axes]":
    """Plot impulse response functions with HDI bands.

    Creates a (n_vars x n_vars) grid of subplots. If ``axes`` is None,
    a new figure and axes array are created internally. The caller can
    reach the figure via ``axes.flat[0].get_figure()``.
    """
```

**Source:** IRF plotting in `var_numpyro.ipynb`.
