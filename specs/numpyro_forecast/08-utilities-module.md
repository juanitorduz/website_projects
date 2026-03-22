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
| `label_encode_column()` | `core/encoding.py` | Backend-agnostic categorical encoding via Narwhals + LabelEncoder |
| `build_group_mapping()` | `core/encoding.py` | Build `(n_series,)` integer mapping for hierarchical priors |
| `build_levels_mapping()` | `core/encoding.py` | Build multi-level parent-child index mappings |
| `plot_forecast()` | `plotting/forecast.py` | Forecast visualization |
| `plot_cv_results()` | `plotting/cv.py` | CV results visualization |
| `plot_irf()` | `plotting/irf.py` | VAR impulse response function plotting |

See [03b-components-module.md](03b-components-module.md) for the Fourier/seasonal function specs, [07-cross-validation-module.md](07-cross-validation-module.md) for the data preparation specs, and below for the plotting specs.

## Plotting Module (`plotting/`)

Thin plotting helpers built on `matplotlib` directly, with ArviZ interoperability targeting `ArviZ >= 1.0.0` (which uses `xarray.DataTree` instead of `arviz.InferenceData`). Do not add a `seaborn` dependency.

### `plot_forecast` (`plotting/forecast.py`)

```python
def plot_forecast(
    dt: "xr.DataTree",
    y_train: Float[Array, " t_train"] | None = None,
    y_test: Float[Array, " t_test"] | None = None,
    *,
    group: str = "posterior_predictive",
    var_name: str = "y_forecast",
    hdi_prob: float = 0.94,
    ax: "matplotlib.axes.Axes | None" = None,
    **plot_kwargs,
) -> "matplotlib.axes.Axes":
    """Plot observed data with forecast HDI bands.

    Parameters
    ----------
    dt
        ``xr.DataTree`` (ArviZ >= 1.0.0) containing forecast samples.
        Typically obtained from ``ForecastResult.datatree``.
    y_train
        Optional observed training series for overlay.
    y_test
        Optional held-out test series for overlay.
    group
        DataTree group to extract samples from.
    var_name
        Variable name within the group. Use ``"y_forecast"`` for forecast-only
        trajectories (recommended) or ``"pred"`` for full observed+future paths.
    hdi_prob
        Probability mass for the HDI band.

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
    """Plot cross-validation forecasts overlaid on the original series.

    Reads forecast samples from ``cv_result.forecasts`` (an ``xr.DataTree``).
    """
```

### `plot_irf` (`plotting/irf.py`)

```python
def plot_irf(
    dt: "xr.DataTree",
    var_names: list[str],
    *,
    group: str = "posterior_predictive",
    var_name: str = "irf",
    hdi_prob: float = 0.94,
    axes: "npt.NDArray[matplotlib.axes.Axes] | None" = None,
    figsize: tuple[float, float] | None = None,
) -> "npt.NDArray[matplotlib.axes.Axes]":
    """Plot impulse response functions with HDI bands.

    Parameters
    ----------
    dt
        ``xr.DataTree`` (ArviZ >= 1.0.0) containing IRF samples.
    var_names
        Names of the VAR variables.
    group
        DataTree group to extract samples from.
    var_name
        Variable name within the group.

    Creates a (n_vars x n_vars) grid of subplots. If ``axes`` is None,
    a new figure and axes array are created internally. The caller can
    reach the figure via ``axes.flat[0].get_figure()``.
    """
```

**Source:** IRF plotting in `var_numpyro.ipynb`.
