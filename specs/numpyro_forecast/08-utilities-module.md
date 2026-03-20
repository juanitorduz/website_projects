# 08 — Utilities Module

## Features (`utils/features.py`)

### `periodic_features`

JAX translation of Pyro's `periodic_features` for generating Fourier basis functions.

```python
def periodic_features(
    duration: int,
    max_period: float | None = None,
    min_period: float | None = None,
) -> Float[Array, "duration feature_dim"]:
    """Generate periodic (Fourier) features for time series regression.

    Creates a matrix of sine/cosine pairs at multiple frequencies,
    suitable for capturing seasonal patterns.

    Parameters
    ----------
    duration
        Number of time steps.
    max_period
        Maximum period (default: ``duration``).
    min_period
        Minimum period (default: 2, Nyquist cutoff).

    Returns
    -------
    Array of shape ``(duration, 2 * n_frequencies)`` with cosine and
    sine columns at each frequency.
    """
    assert isinstance(duration, int) and duration >= 0
    if max_period is None:
        max_period = duration
    if min_period is None:
        min_period = 2
    assert min_period >= 2, "min_period is below Nyquist cutoff"
    assert min_period <= max_period

    t = jnp.arange(float(duration)).reshape(-1, 1, 1)
    phase = jnp.array([0, jnp.pi / 2]).reshape(1, -1, 1)
    freq = jnp.arange(1, max_period / min_period).reshape(1, 1, -1) * (
        2 * jnp.pi / max_period
    )
    return jnp.cos(freq * t + phase).reshape(duration, -1)
```

**Source:** `periodic_features_jax` in `numpyro_forecasting_univariate.ipynb`. Direct JAX port of `pyro.ops.tensor_utils.periodic_features`.

### `periodic_repeat`

Repeat a seasonal pattern to fill a longer time span.

```python
def periodic_repeat(
    seasonal_init: Float[Array, "n_seasons *batch"],
    t_max: int,
) -> Float[Array, "t_max *batch"]:
    """Tile a seasonal pattern to cover t_max time steps."""
```

### `fourier_modes`

Generate Fourier modes at specific periods (e.g., weekly, yearly).

```python
def fourier_modes(
    t: Float[Array, " t_max"],
    period: float,
    n_modes: int,
) -> Float[Array, "t_max 2*n_modes"]:
    """Generate n sine/cosine pairs at harmonics of the given period.

    Parameters
    ----------
    t
        Time index array (e.g., ``jnp.arange(t_max)``).
    period
        Fundamental period (e.g., 365.25 for yearly, 7 for weekly).
    n_modes
        Number of Fourier harmonics.

    Returns
    -------
    Array of shape ``(t_max, 2 * n_modes)`` — sin and cos columns.
    """
```

## Data (`utils/data.py`)

### `train_test_split`

```python
def train_test_split(
    y: Float[Array, "t_max *rest"],
    n_test: int,
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """Split a time series into train and test by slicing the last n_test steps."""
    return y[:-n_test], y[-n_test:]
```

### `prepare_intermittent_data`

Prepare Croston-style data from a raw intermittent series.

```python
def prepare_intermittent_data(
    y: Float[Array, " t_max"],
) -> tuple[Float[Array, " n_demands"], Float[Array, " n_demands"]]:
    """Decompose an intermittent series into demand sizes and period inverses.

    Parameters
    ----------
    y
        Raw time series with zeros for no-demand periods.

    Returns
    -------
    z
        Non-zero demand values.
    p_inv
        Inverse of inter-demand periods.
    """
    z = y[y != 0]
    p_idx = jnp.flatnonzero(y).astype(jnp.float32)
    p = jnp.diff(p_idx, prepend=-1.0)
    p_inv = 1.0 / p
    return z, p_inv
```

**Source:** Inline in `croston_numpyro.ipynb`.

### `prepare_tsb_data`

Prepare TSB-style data (trim leading zeros, compute initial states).

```python
def prepare_tsb_data(
    y: Float[Array, " t_max"],
) -> tuple[Float[Array, "..."], float, float]:
    """Prepare data for TSB/ZI-TSB models.

    Returns
    -------
    y_trim
        Series with leading zeros removed.
    z0
        Initial demand level (first non-zero value).
    p0
        Initial demand probability (1 / mean inter-demand period).
    """
```

**Source:** `get_model_args` in `tsb_numpyro.ipynb`.

### `prepare_hierarchical_mapping`

Encode group membership for hierarchical models.

```python
def prepare_hierarchical_mapping(
    group_labels: Sequence[str],
) -> tuple[Float[Array, " n_series"], int]:
    """Encode group labels as integer indices.

    Parameters
    ----------
    group_labels
        Group membership for each series (e.g., state names).

    Returns
    -------
    mapping_idx
        Integer array mapping each series to its group index.
    n_groups
        Number of unique groups.
    """
```

**Source:** `LabelEncoder` usage in `hierarchical_exponential_smoothing.ipynb`.

## Plotting (`utils/plotting.py`)

Thin plotting helpers built on `matplotlib` directly, with optional ArviZ interoperability where `ArviZ > 1.0.0` APIs are stable. Do not add a `seaborn` dependency. Optional — only imported if `matplotlib` is available.

### `plot_forecast`

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

    Uses ``arviz.plot_hdi`` for credible interval shading.
    """
```

### `plot_cv_results`

```python
def plot_cv_results(
    cv_result: "CVResult",
    y: Float[Array, " t_max"],
    *,
    hdi_prob: float = 0.94,
    max_folds: int | None = None,
    ax: "matplotlib.axes.Axes | None" = None,
) -> "matplotlib.axes.Axes":
    """Plot cross-validation forecasts overlaid on the original series."""
```

### `plot_irf`

```python
def plot_irf(
    irf_samples: Float[Array, "n_samples n_steps n_vars n_vars"],
    var_names: list[str],
    *,
    hdi_prob: float = 0.94,
    figsize: tuple[float, float] | None = None,
) -> "matplotlib.figure.Figure":
    """Plot impulse response functions with HDI bands.

    Creates a (n_vars x n_vars) grid of subplots.
    """
```

**Source:** IRF plotting in `var_numpyro.ipynb`.
