# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: default
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Interpreting and Communicating Statistical Models
#
# An ad platform charges its stores per click and reports back ROAS (return on ad spend).
# Stores keep spending while ROAS makes the campaigns worth it; when it doesn't, they pause
# for a month or two. We have a panel of monthly booked budgets and want to predict next
# month's budget from this month's signals: ROAS, where the store is in its life-cycle, and
# the time of year.
#
# We'll fit three models of increasing flexibility on the same panel — a Gaussian linear
# baseline, a Hurdle-Gamma GLM with a linear ROAS coefficient, and a Hurdle-Gamma GLM with
# a Gaussian process on ROAS — and use each to sharpen what the next one buys us. The
# headline target is the **varying coefficient** $\beta(\mathrm{roas})$: the non-linear
# return that ad spend gives a store. Past raw coefficients, the whole point of the
# notebook is the move to **quantities of interest** — predictions, comparisons, slopes —
# using the [`marginaleffects`](https://marginaleffects.com/) framework.

# %% [markdown]
# ## Setup

# %%
from typing import NamedTuple

import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

az.style.use("arviz-darkgrid")
plt.rcParams["figure.figsize"] = [10, 6]
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.facecolor"] = "white"

# %load_ext autoreload
# %autoreload 2
# %config InlineBackend.figure_format = "retina"

# %%
seed: int = sum(map(ord, "marginaleffects"))
rng: np.random.Generator = np.random.default_rng(seed=seed)


# %% [markdown]
# ## A data-generating process we can reason about
#
# We need synthetic data where we *know* the truth, so later we can check whether the
# model recovers it. On the log scale, next month's expected budget is
#
# $$
# \log \mu_{t+1} \;=\; \beta_0 \;+\; \text{season}(\text{month}_t) \;+\; \gamma \cdot \text{cohort\_age}_t \;+\; \beta(\text{roas}_t) \cdot \text{roas}_t
# $$
#
# The interesting piece is $\beta(\text{roas})$: a coefficient that varies with ROAS itself.
# Below ROAS=1 stores are losing money, so the marginal effect of ROAS on next month's
# budget is negative; in the sweet spot between 1 and 4 each extra unit of ROAS pulls more
# budget in; past 4 stores hit inventory or capacity ceilings and the effect saturates.

# %% [markdown]
# ### A smooth $\beta(\text{roas})$ by construction
#
# We build $\beta$ as a product of two analytic pieces — a smooth rise and a smooth
# saturation window — so the function is $C^{\infty}$ everywhere with no joining knots.


# %%
def beta_roas(roas: np.ndarray) -> np.ndarray:
    rise = -0.5 + 1.0 / (1.0 + np.exp(-3.0 * (roas - 1.0)))
    saturation = 1.0 / (1.0 + np.exp(0.6 * (roas - 4.0)))
    return rise * saturation


def f_roas(roas: np.ndarray) -> np.ndarray:
    return beta_roas(roas) * roas


# %%
roas_grid = np.linspace(0.0, 10.0, 500)

beta_roas_grid = beta_roas(roas_grid)
f_roas_grid = f_roas(roas_grid)

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(12, 7),
    sharex=True,
    sharey=True,
    layout="constrained",
)
axes[0].plot(roas_grid, beta_roas_grid, color="C0")
axes[0].axhline(0.0, color="black", linewidth=0.8)
axes[0].set_title(r"$\beta(\mathrm{roas})$ — the varying coefficient")
axes[0].set_xlabel("roas")
axes[0].set_ylabel(r"$\beta$")

axes[1].plot(roas_grid, f_roas_grid, color="C1")
axes[1].axhline(0.0, color="black", linewidth=0.8)
axes[1].set_title(
    r"$\beta(\mathrm{roas}) \cdot \mathrm{roas}$ — contribution to $\log \mu$"
)
axes[1].set_xlabel("roas")
(fig.suptitle("Ground truth ROAS effect", fontsize=18, fontweight="bold"))


# %% [markdown]
# Negative for low ROAS, rising through zero around break-even, peaking in the sweet spot,
# then pulled back toward zero as the platform saturates. This is the curve we'll later try
# to recover from a Gaussian process.

# %% [markdown]
# ### Generating the panel
#
# 100 stores observed for 24 months. Each store has its own cohort start (so cohort age
# varies across the panel) and its own ROAS process — an AR(1) on the log scale to get
# realistic month-to-month persistence.
#
# The lag matters: each row pairs **this month's signals** with **next month's budget** —
# the leading indicators a store could act on. When a store is inactive in a given month
# (no spend), it has no ROAS to report — that's encoded as `NaN`, and rows whose lagged
# ROAS is `NaN` are dropped at modelling time.


# %%
class DGPParams(NamedTuple):
    n_stores: int = 100
    n_months: int = 24
    intercept: float = 0.5
    cohort_slope: float = -0.02
    gamma_shape: float = 8.0
    inactive_base_prob: float = 0.05
    inactive_summer_bonus: float = 0.10


def season(month_of_year: np.ndarray) -> np.ndarray:
    return 0.4 * np.sin(2 * np.pi * month_of_year / 12) + 0.2 * np.cos(
        4 * np.pi * month_of_year / 12
    )


class DGP:
    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def run(self, params: DGPParams) -> pl.DataFrame:
        roas, month_of_year, cohort_age, store_ids, inactive = self._simulate_features(
            params
        )
        budget_next, inactive_next = self._draw_response(
            roas, month_of_year, cohort_age, inactive, params
        )
        return self._build_panel(
            store_ids,
            roas,
            month_of_year,
            cohort_age,
            inactive,
            budget_next,
            inactive_next,
            params,
        )

    def _simulate_features(
        self, params: DGPParams
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """AR(1) ROAS, store-start offsets, derived month/cohort arrays, and per-cell
        inactivity flags. Shapes (n_stores, n_months) except store_ids (n_stores,)."""
        store_ids = np.arange(params.n_stores)
        store_starts = self.rng.integers(low=-12, high=1, size=params.n_stores)
        store_log_roas_mean = self.rng.normal(
            loc=np.log(2.5), scale=0.4, size=params.n_stores
        )

        log_roas = np.empty(shape=(params.n_stores, params.n_months))
        log_roas[:, 0] = store_log_roas_mean + self.rng.normal(
            scale=0.3, size=params.n_stores
        )
        for t in range(1, params.n_months):
            log_roas[:, t] = (
                0.6 * log_roas[:, t - 1]
                + 0.4 * store_log_roas_mean
                + self.rng.normal(scale=0.3, size=params.n_stores)
            )
        roas = np.clip(np.exp(log_roas), 0.0, 8.0)

        month_idx = np.broadcast_to(
            np.arange(params.n_months), (params.n_stores, params.n_months)
        )
        month_of_year = (month_idx % 12) + 1
        cohort_age = month_idx - store_starts[:, None]

        inactive_prob = params.inactive_base_prob + params.inactive_summer_bonus * (
            np.isin(month_of_year, [7, 8])
        ).astype(float)
        inactive = (
            self.rng.uniform(size=(params.n_stores, params.n_months)) < inactive_prob
        )

        return roas, month_of_year, cohort_age, store_ids, inactive

    def _draw_response(
        self,
        roas: np.ndarray,
        month_of_year: np.ndarray,
        cohort_age: np.ndarray,
        inactive: np.ndarray,
        params: DGPParams,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Lagged log_mu, gamma draw, target-month inactive mask. Both returned arrays
        have shape (n_stores, n_months - 1)."""
        log_mu_next = (
            params.intercept
            + season(month_of_year[:, :-1])
            + params.cohort_slope * cohort_age[:, :-1]
            + f_roas(roas[:, :-1])
        )
        budget_pos = self.rng.gamma(
            shape=params.gamma_shape,
            scale=np.exp(log_mu_next) / params.gamma_shape,
        )
        inactive_next = inactive[:, 1:]
        budget_next = np.where(inactive_next, 0.0, budget_pos)
        return budget_next, inactive_next

    def _build_panel(
        self,
        store_ids: np.ndarray,
        roas: np.ndarray,
        month_of_year: np.ndarray,
        cohort_age: np.ndarray,
        inactive: np.ndarray,
        budget_next: np.ndarray,
        inactive_next: np.ndarray,
        params: DGPParams,
    ) -> pl.DataFrame:
        """Long polars frame, one row per (store, predictor month t) for
        t in [0, n_months - 2]. ROAS is masked to NaN when the predictor month was
        inactive (no spend → no observable ROAS)."""
        roas_observed = np.where(inactive, np.nan, roas)
        n_pred = params.n_months - 1
        return pl.DataFrame(
            {
                "store_id": np.repeat(store_ids, n_pred),
                "predictor_month_idx": np.tile(np.arange(n_pred), params.n_stores),
                "month_of_year": month_of_year[:, :-1].ravel(),
                "cohort_age": cohort_age[:, :-1].ravel(),
                "roas": roas_observed[:, :-1].ravel(),
                "budget_next": budget_next.ravel(),
                "inactive_next": inactive_next.ravel(),
            }
        )


params = DGPParams()
panel = DGP(rng=rng).run(params)
panel.head()

# %% [markdown]
# ## Exploring the synthetic data
#
# Before we hand anything to a sampler, let's look at the panel and check the structure we
# put in is actually visible.

# %%
panel.describe()

# %%
active_share = (
    panel.group_by("month_of_year")
    .agg((1 - pl.col("inactive_next").cast(pl.Float64)).mean().alias("active_rate"))
    .sort("month_of_year")
)
active_share

# %% [markdown]
# Twelve random stores over time — each store's monthly budget; the model regresses each
# point on the *prior* month's predictors. Watch for seasonal humps and the occasional
# zero month.

# %%
n_random_stores = 12

sample_ids = rng.choice(
    panel["store_id"].unique().to_numpy(), size=n_random_stores, replace=False
)

fig, axes = plt.subplots(
    nrows=3,
    ncols=4,
    figsize=(12, 7),
    sharex=True,
    sharey=True,
    layout="constrained",
)

for ax, sid in zip(axes.flat, sample_ids, strict=True):
    sub = panel.filter(pl.col("store_id").eq(pl.lit(sid))).sort("predictor_month_idx")
    ax.plot(
        sub["predictor_month_idx"].to_numpy(),
        sub["budget_next"].to_numpy(),
        color="black",
    )
    ax.set_title(f"store {sid}")
    ax.set_xlabel("predictor month index")
fig.suptitle(
    "Next-month booked budget for twelve random stores", fontsize=18, fontweight="bold"
)

# %% [markdown]
# Marginal distributions of the three drivers and the response.

# %%
panel

# %%
active = panel.filter(~pl.col("inactive_next"))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(active["budget_next"].to_numpy(), bins=40, color="C0")
axes[0].set_title("budget_next (active months)")
axes[1].hist(
    panel.filter(pl.col("roas").is_not_nan())["roas"].to_numpy(), bins=40, color="C1"
)
axes[1].set_title("roas (predictor month)")
axes[2].hist(panel["cohort_age"].to_numpy(), bins=40, color="C2")
axes[2].set_title("cohort age")
fig.tight_layout()

# %% [markdown]
# Yearly seasonality should be visible by month.

# %%
fig, ax = plt.subplots(figsize=(12, 5))
month_groups = [
    active.filter(pl.col("month_of_year") == m)["budget_next"].to_numpy()
    for m in range(1, 13)
]
ax.boxplot(month_groups, tick_labels=list(range(1, 13)), showfliers=False)
ax.set_xlabel("predictor month of year")
ax.set_ylabel("budget_next")
ax.set_title("Next-month budget by this-month's month-of-year (active rows)")
fig.tight_layout()

# %% [markdown]
# Cohort age vs budget — a mild downward drift as stores get older.

# %%
cohort_summary = (
    active.group_by("cohort_age")
    .agg(pl.col("budget_next").mean().alias("mean_budget"))
    .sort("cohort_age")
)

fig, ax = plt.subplots(figsize=(12, 5))
ax.scatter(
    active["cohort_age"].to_numpy(),
    active["budget_next"].to_numpy(),
    alpha=0.15,
    s=10,
)
ax.plot(
    cohort_summary["cohort_age"].to_numpy(),
    cohort_summary["mean_budget"].to_numpy(),
    color="C3",
    linewidth=2,
    label="binned mean",
)
ax.set_xlabel("cohort age (months)")
ax.set_ylabel("budget_next")
ax.set_title("Cohort-age trend in next-month budget")
ax.legend()
fig.tight_layout()

# %% [markdown]
# And the headline one: next-month budget against this-month's ROAS.

# %%
roas_bins = np.linspace(0, 8, 21)
bin_centers = 0.5 * (roas_bins[:-1] + roas_bins[1:])
scatter_df = active.filter(pl.col("roas").is_not_nan())
roas_arr = scatter_df["roas"].to_numpy()
budget_arr = scatter_df["budget_next"].to_numpy()
bin_idx = np.digitize(roas_arr, roas_bins) - 1
bin_idx = np.clip(bin_idx, 0, len(bin_centers) - 1)
medians = np.array(
    [
        np.median(budget_arr[bin_idx == i]) if np.any(bin_idx == i) else np.nan
        for i in range(len(bin_centers))
    ]
)

fig, ax = plt.subplots(figsize=(12, 5))
ax.scatter(roas_arr, budget_arr, alpha=0.15, s=10)
ax.plot(bin_centers, medians, color="C3", linewidth=2, label="binned median")
ax.set_xlabel("roas (predictor month)")
ax.set_ylabel("budget_next")
ax.set_title("Next-month budget vs this-month's ROAS")
ax.legend()
fig.tight_layout()

# %% [markdown]
# The non-linear shape is visible to the eye: budget rises with ROAS, levels off past
# ROAS≈4. That's the signal we want the model to pick up.

# %% [markdown]
# ## Shared setup for three models
#
# Every model below sees the same fit dataframe, the same posterior-summary helpers, and
# the same ROAS evaluation grid. Lift the shared pieces out once so each model section
# reduces to *spec → prior predictive → fit → ROAS plot*.
#
# We keep every row with an observed lagged ROAS — including rows where `budget_next = 0`.
# The Gaussian baseline will see those zeros as ordinary data; the hurdle-Gamma models
# treat them as draws from the inactive component.

# %%
df_fit = panel.filter(pl.col("roas").is_not_nan()).to_pandas()
print(
    f"fit rows: {len(df_fit)}, of which {(df_fit['budget_next'] == 0).sum()} are zero"
)


# %% [markdown]
# ### Posterior over $\mathbb{E}[Y \mid \text{grid}]$
#
# We'll evaluate every model the same way: posterior over the response mean on a polars
# grid, summarised by mean + 94% CI. `predict_marginal` returns $\psi \cdot \mu$ for hurdle
# families (averaging over the chance the store skips) and plain $\mu$ otherwise, so it's
# safe to call across all three models.


# %%
def predict_mu(model: bmb.Model, idata, grid_pl: pl.DataFrame) -> np.ndarray:
    """Posterior samples of the *conditional* response mean over a grid.
    For hurdle Gamma this is $E[Y \\mid \\text{active}]$; for Gaussian it is the
    response mean itself. Shape (n_samples, n_grid)."""
    new_idata = model.predict(
        idata, data=grid_pl.to_pandas(), kind="response_params", inplace=False
    )
    mu = new_idata.posterior["mu"]
    obs_dim = next(d for d in mu.dims if d not in ("chain", "draw"))
    return mu.stack(sample=("chain", "draw")).transpose("sample", obs_dim).to_numpy()


def predict_marginal(model: bmb.Model, idata, grid_pl: pl.DataFrame) -> np.ndarray:
    """Posterior samples of the *marginal* response mean over a grid:
    $\\psi \\cdot \\mu$ for hurdle families (averaging over the chance the store skips),
    plain $\\mu$ for non-hurdle families. Shape (n_samples, n_grid)."""
    new_idata = model.predict(
        idata, data=grid_pl.to_pandas(), kind="response_params", inplace=False
    )
    mu = new_idata.posterior["mu"]
    obs_dim = next(d for d in mu.dims if d not in ("chain", "draw"))
    mu_arr = mu.stack(sample=("chain", "draw")).transpose("sample", obs_dim).to_numpy()
    if "psi" not in new_idata.posterior:
        return mu_arr
    psi = new_idata.posterior["psi"]
    psi_arr = psi.stack(sample=("chain", "draw")).to_numpy()
    return psi_arr[:, None] * mu_arr


def summarise(samples: np.ndarray, prob: float = 0.94) -> dict[str, np.ndarray]:
    lo, hi = np.quantile(samples, [(1 - prob) / 2, 1 - (1 - prob) / 2], axis=0)
    return {"mean": samples.mean(axis=0), "lo": lo, "hi": hi}


# %% [markdown]
# ### A shared ROAS evaluation grid
#
# A single sweep used by every model's ROAS effect plot — cohort age fixed at 12 months,
# month-of-year fixed at June — plus the Gamma-conditional ground-truth curve to overlay.

# %%
roas_eval = np.linspace(0.2, 6.0, 60)
grid_roas = pl.DataFrame(
    {
        "roas": roas_eval,
        "cohort_age": np.full_like(roas_eval, 12, dtype=np.int64),
        "month_of_year": np.full_like(roas_eval, 6, dtype=np.int64),
    }
)

truth_log_mu = (
    params.intercept
    + season(np.full_like(roas_eval, 6))
    + params.cohort_slope * 12
    + f_roas(roas_eval)
)
truth_budget = np.exp(truth_log_mu)

# %% [markdown]
# ## Baseline 1 — linear Gaussian (identity link)
#
# The simplest thing that could work: a plain linear regression on all four predictors,
# Gaussian noise, identity link. No special handling of the zeros, no log scale, no smooth
# on ROAS. This is the model a stakeholder would draw on a napkin.
#
# Two reasons to expect trouble. First, the response is non-negative with a point mass at
# zero — a Gaussian likelihood has no business there. Second, the DGP's ROAS effect is
# non-linear (a peak in the sweet spot, saturation past ROAS≈4), and a single slope can't
# bend.

# %%
formula_lm = bmb.Formula("budget_next ~ 1 + cohort_age + C(month_of_year) + roas")

priors_lm = {
    "Intercept": bmb.Prior("Normal", mu=0.0, sigma=10.0),
    "cohort_age": bmb.Prior("Normal", mu=0.0, sigma=1.0),
    "C(month_of_year)": bmb.Prior("Normal", mu=0.0, sigma=5.0),
    "roas": bmb.Prior("Normal", mu=0.0, sigma=5.0),
    "sigma": bmb.Prior("HalfNormal", sigma=10.0),
}

model_lm = bmb.Model(
    formula=formula_lm,
    data=df_fit,
    family="gaussian",
    link="identity",
    priors=priors_lm,
)
model_lm.build()
model_lm

# %% [markdown]
# ### Prior predictive
#
# Gaussian noise puts mass below zero — call that out as the first concrete reason to
# prefer Gamma later on.

# %%
idata_prior_lm = model_lm.prior_predictive(draws=500, random_seed=seed)

# %%
fig, ax = plt.subplots(figsize=(10, 5))
az.plot_ppc(idata_prior_lm, group="prior", ax=ax)
ax.set_title("Baseline 1 — prior predictive")
ax.set_xlim(
    left=-np.quantile(df_fit["budget_next"], 0.99) * 2,
    right=np.quantile(df_fit["budget_next"], 0.99) * 3,
)
fig.tight_layout()

# %% [markdown]
# ### Fit

# %%
idata_lm = model_lm.fit(
    draws=1000,
    tune=1000,
    chains=4,
    target_accept=0.9,
    random_seed=seed,
    idata_kwargs={"log_likelihood": True},
)

# %% [markdown]
# ### Diagnostics

# %%
az.summary(
    idata_lm,
    var_names=["Intercept", "cohort_age", "C(month_of_year)", "roas", "sigma"],
    filter_vars="like",
)

# %%
az.plot_trace(
    idata_lm,
    var_names=["Intercept", "cohort_age", "roas", "sigma"],
    compact=True,
    backend_kwargs={"layout": "constrained"},
)

# %%
model_lm.predict(idata_lm, kind="response", inplace=True)

fig, ax = plt.subplots(figsize=(10, 5))
az.plot_ppc(idata_lm, ax=ax)
ax.set_title("Baseline 1 — posterior predictive")
ax.set_xlim(
    left=-np.quantile(df_fit["budget_next"], 0.99),
    right=np.quantile(df_fit["budget_next"], 0.99) * 2,
)
fig.tight_layout()

# %% [markdown]
# ### Adjusted predictions across ROAS
#
# A single slope on ROAS bends to nothing. The posterior mean is a straight line — by
# construction. Compare to the true Gamma-conditional curve in dashed black.

# %%
mu_roas_lm = predict_mu(model_lm, idata_lm, grid_roas)
sum_roas_lm = summarise(mu_roas_lm)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(roas_eval, sum_roas_lm["mean"], color="C0", label="posterior mean")
ax.fill_between(
    roas_eval,
    sum_roas_lm["lo"],
    sum_roas_lm["hi"],
    alpha=0.25,
    color="C0",
    label="94% CI",
)
ax.plot(roas_eval, truth_budget, color="black", linestyle="--", label="ground truth")
ax.set_xlabel("roas")
ax.set_ylabel("expected budget next month")
ax.set_title("Baseline 1 — adjusted predictions across ROAS")
ax.legend()
fig.tight_layout()

# %% [markdown]
# ## Baseline 2 — Hurdle Gamma with a linear ROAS coefficient
#
# Same likelihood family as the model we ultimately want, but ROAS still enters linearly
# on the log scale. The log link turns the linear coefficient into a *multiplicative* effect
# — `exp(linear)` is monotone, so we get a curve rather than a line, but no peak and no
# saturation. The shape is still wrong; the comparison versus the GP model will quantify
# how wrong.

# %%
formula_hgl = bmb.Formula("budget_next ~ 1 + cohort_age + C(month_of_year) + roas")

priors_hgl = {
    "Intercept": bmb.Prior("Normal", mu=0.0, sigma=1.0),
    "cohort_age": bmb.Prior("Normal", mu=0.0, sigma=1.0),
    "C(month_of_year)": bmb.Prior("Normal", mu=0.0, sigma=1.5),
    "roas": bmb.Prior("Normal", mu=0.0, sigma=1.0),
    "alpha": bmb.Prior("HalfNormal", sigma=1.0),
}

model_hgl = bmb.Model(
    formula=formula_hgl,
    data=df_fit,
    family="hurdle_gamma",
    link="log",
    priors=priors_hgl,
)
model_hgl.build()
model_hgl

# %% [markdown]
# ### Prior predictive

# %%
idata_prior_hgl = model_hgl.prior_predictive(draws=500, random_seed=seed)

# %%
fig, ax = plt.subplots(figsize=(10, 5))
az.plot_ppc(idata_prior_hgl, group="prior", ax=ax)
ax.set_title("Baseline 2 — prior predictive")
ax.set_xlim(left=0, right=np.quantile(df_fit["budget_next"], 0.99) * 3)
fig.tight_layout()

# %% [markdown]
# ### Fit

# %%
idata_hgl = model_hgl.fit(
    draws=1000,
    tune=1000,
    chains=4,
    target_accept=0.95,
    random_seed=seed,
    idata_kwargs={"log_likelihood": True},
)

# %% [markdown]
# ### Diagnostics

# %%
az.summary(
    idata_hgl,
    var_names=["Intercept", "cohort_age", "C(month_of_year)", "roas", "alpha", "psi"],
    filter_vars="like",
)

# %%
az.plot_trace(
    idata_hgl,
    var_names=["Intercept", "cohort_age", "roas", "alpha", "psi"],
    compact=True,
    backend_kwargs={"layout": "constrained"},
)

# %%
model_hgl.predict(idata_hgl, kind="response", inplace=True)

fig, ax = plt.subplots(figsize=(10, 5))
az.plot_ppc(idata_hgl, ax=ax)
ax.set_title("Baseline 2 — posterior predictive")
ax.set_xlim(left=0, right=np.quantile(df_fit["budget_next"], 0.99) * 2)
fig.tight_layout()

# %% [markdown]
# ### Adjusted predictions across ROAS
#
# Monotone exponential of a linear slope. Better than baseline 1 (the right family and
# scale) but still wrong shape — no peak, no saturation.

# %%
mu_roas_hgl = predict_mu(model_hgl, idata_hgl, grid_roas)
sum_roas_hgl = summarise(mu_roas_hgl)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(roas_eval, sum_roas_hgl["mean"], color="C0", label="posterior mean")
ax.fill_between(
    roas_eval,
    sum_roas_hgl["lo"],
    sum_roas_hgl["hi"],
    alpha=0.25,
    color="C0",
    label="94% CI",
)
ax.plot(roas_eval, truth_budget, color="black", linestyle="--", label="ground truth")
ax.set_xlabel("roas")
ax.set_ylabel("expected budget next month, given active")
ax.set_title("Baseline 2 — adjusted predictions across ROAS")
ax.legend()
fig.tight_layout()

# %% [markdown]
# ## GLM with varying coefficient — Hurdle Gamma + HSGP(roas)
#
# Booked budget is non-negative with a real point-mass at zero (inactive months) and a
# positive continuous tail otherwise. **Hurdle Gamma** is the natural fit: a Bernoulli for
# "will the store spend at all next month", and a Gamma for "how much, given they spend".
# Bambi exposes it as `family="hurdle_gamma"`. By default the formula drives the Gamma
# mean and the zero-inflation probability $\psi$ is a single scalar — that's what we use
# here; Bambi lets you pass a multi-formula if a stakeholder wants different drivers per
# component.
#
# The new piece is `hsgp(roas, ...)` — a Hilbert-space Gaussian-process basis on ROAS — in
# place of the linear `roas` term. We avoid assuming linearity, polynomial form, or knot
# locations; the GP lets the data shape the curve.

# %% [markdown]
# ### Reading the formula
#
# - `cohort_age` enters linearly. The DGP made it linear on `log_mu`, so a single
#   coefficient is the right parameterization.
# - `month_of_year` is a categorical (`C(...)`), 12 levels dummy-coded. Stakeholders think
#   of "December lift" or "August dip", not sin/cos, so categorical contrasts read better.
# - `hsgp(roas, ...)` is a Hilbert-space Gaussian-process basis on ROAS. We avoid assuming
#   linearity, polynomial form, or knot locations — the GP lets the data shape the curve.
#
# A subtlety: the DGP encodes ROAS as $\beta(\mathrm{roas}) \cdot \mathrm{roas}$, a
# *varying coefficient*. Bambi's formula API doesn't expose that directly — `hsgp(x, by=g)`
# supports group-specific GPs over a categorical `g`, but not a continuous-by-continuous
# product. The varying-coefficient form is straightforward in raw PyMC (see
# [bikes_gp](https://juanitorduz.github.io/bikes_gp/)). For this talk we keep things in
# pure Bambi: a single `hsgp(roas)` is flexible enough to absorb $\beta(\mathrm{roas})
# \cdot \mathrm{roas}$ as one smooth $g(\mathrm{roas})$, and we'll recover
# $\hat{\beta}(\mathrm{roas})$ post-hoc.

# %%
HSGP_M = 20
HSGP_C = 1.5

formula = bmb.Formula(
    f"budget_next ~ 1 + cohort_age + C(month_of_year) + hsgp(roas, m={HSGP_M}, c={HSGP_C})"
)

priors = {
    "Intercept": bmb.Prior("Normal", mu=0.0, sigma=1.0),
    "cohort_age": bmb.Prior("Normal", mu=0.0, sigma=1.0),
    "C(month_of_year)": bmb.Prior("Normal", mu=0.0, sigma=1.5),
    "alpha": bmb.Prior("HalfNormal", sigma=1.0),
}

model = bmb.Model(
    formula=formula, data=df_fit, family="hurdle_gamma", link="log", priors=priors
)
model.build()
model

# %% [markdown]
# ## Prior predictive check
#
# Are the implied budgets in a sensible order of magnitude before the data is touched?

# %%
idata_prior = model.prior_predictive(draws=500, random_seed=seed)

# %%
fig, ax = plt.subplots(figsize=(10, 5))
az.plot_ppc(idata_prior, group="prior", ax=ax)
ax.set_title("Prior predictive")
ax.set_xlim(left=0, right=np.quantile(df_fit["budget_next"], 0.99) * 3)
fig.tight_layout()

# %% [markdown]
# ## Fit

# %%
# Note: nutpie (`inference_method="nutpie"`) is faster but currently trips over commas in
# Bambi's HSGP variable names when arviz parses dimensions. Sticking with the default PyMC
# sampler here.
idata = model.fit(
    draws=1000,
    tune=1000,
    chains=4,
    target_accept=0.95,
    random_seed=seed,
    idata_kwargs={"log_likelihood": True},
)

# %% [markdown]
# ## Diagnostics

# %%
az.summary(
    idata,
    var_names=["Intercept", "cohort_age", "C(month_of_year)", "alpha", "psi"],
    filter_vars="like",
)

# %%
az.plot_trace(
    idata,
    var_names=["Intercept", "cohort_age", "alpha", "psi"],
    compact=True,
    backend_kwargs={"layout": "constrained"},
)

# %%
model.predict(idata, kind="response", inplace=True)

fig, ax = plt.subplots(figsize=(10, 5))
az.plot_ppc(idata, ax=ax)
ax.set_title("Posterior predictive")
ax.set_xlim(left=0, right=np.quantile(df_fit["budget_next"], 0.99) * 2)
fig.tight_layout()

# %% [markdown]
# ## Why raw coefficients don't answer the question
#
# `az.summary` gives us posterior means of the model parameters: an intercept, a
# `cohort_age` slope, eleven contrasts for the months, and a basket of HSGP basis weights.
# None of these answer the question a stakeholder actually asks:
#
# > *"If a store's ROAS goes from 2 to 3 next month, how much more budget will they book?"*
#
# The log link makes the intercept a multiplicative offset, the categorical contrasts are
# differences against a baseline month, and the HSGP weights have no individual meaning.
# We need to ask the model directly, in the units of the data.

# %% [markdown]
# ## Quantities of interest, by hand
#
# The [`marginaleffects`](https://marginaleffects.com/) framework names three primitives:
#
# - **`predictions`** — what does the model say at this scenario?
# - **`comparisons`** — what changes when we move from A to B?
# - **`slopes`** — what is $\partial \hat{Y} / \partial X$ here? (we'll skip this one in the
#   walkthrough; it's the same recipe with a finite-difference twist on top.)
#
# We'll build the first two ourselves so the mechanics are visible. The recipe is always
# the same: pick a reference grid, push it through the posterior, summarise. The Python
# `marginaleffects` package would automate this for `statsmodels` / `sklearn` /
# `linearmodels` / `pyfixest` models, but Bambi/PyMC isn't on its supported list yet — so
# rolling it ourselves is also the only path.

# %% [markdown]
# ### Adjusted predictions across ROAS
#
# Reuse the shared `grid_roas` (ROAS sweep at cohort age 12, month-of-year June) and the
# Gamma-conditional `truth_budget` defined earlier.

# %%
mu_roas = predict_mu(model, idata, grid_roas)
sum_roas = summarise(mu_roas)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(roas_eval, sum_roas["mean"], color="C0", label="posterior mean")
ax.fill_between(
    roas_eval, sum_roas["lo"], sum_roas["hi"], alpha=0.25, color="C0", label="94% CI"
)
ax.plot(roas_eval, truth_budget, color="black", linestyle="--", label="ground truth")
ax.set_xlabel("roas")
ax.set_ylabel("expected budget next month, given active")
ax.set_title("Adjusted predictions across ROAS (cohort_age=12, month=6)")
ax.legend()
fig.tight_layout()

# %% [markdown]
# The recovered curve tracks the ground truth: shallow for low ROAS, climbing through
# break-even, levelling off past 4. We're plotting the **Gamma-conditional** mean here
# (expected budget *given* the store spends); the hurdle's $\psi$ scales it down to the
# unconditional expectation later on.

# %% [markdown]
# ### Recovering $\hat{\beta}(\text{roas})$ post-hoc
#
# We modelled ROAS as a single smooth $g(\text{roas})$ on the log scale, but the DGP wrote
# it as $\beta(\text{roas}) \cdot \text{roas}$. We can recover $\hat{\beta}$ by isolating
# the GP contribution to $\log \mu$ (everything else cancels when we hold cohort age and
# month fixed) and dividing by ROAS.

# %%
roas_ref = 1.0
grid_ref = pl.DataFrame(
    {
        "roas": [roas_ref],
        "cohort_age": [12],
        "month_of_year": [6],
    }
)
mu_ref = predict_mu(model, idata, grid_ref)[:, 0]

# log(mu(r)) - log(mu(r_ref)) is g(r) - g(r_ref) on the log scale.
# Anchor at r_ref=1 using the truth f_roas(1) ≈ 0.
g_anchor = float(f_roas(np.array([roas_ref]))[0])
log_ratio = np.log(mu_roas) - np.log(mu_ref)[:, None]
g_samples = log_ratio + g_anchor
beta_samples = g_samples / roas_eval[None, :]
sum_beta = summarise(beta_samples)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(roas_eval, sum_beta["mean"], color="C0", label=r"$\hat{\beta}(\mathrm{roas})$")
ax.fill_between(
    roas_eval, sum_beta["lo"], sum_beta["hi"], alpha=0.25, color="C0", label="94% CI"
)
ax.plot(
    roas_eval, beta_roas(roas_eval), color="black", linestyle="--", label="ground truth"
)
ax.axhline(0.0, color="gray", linewidth=0.8)
ax.set_xlabel("roas")
ax.set_ylabel(r"$\beta$")
ax.set_title("Recovered varying coefficient from a single-smooth model")
ax.legend()
fig.tight_layout()

# %% [markdown]
# The GP recovered the varying coefficient even though the formula didn't ask for one.

# %% [markdown]
# ### Comparisons — going from break-even to a strong campaign
#
# What's the expected lift in next-month budget if we move a store from ROAS=1 to ROAS=4?
# This is the question the platform team actually asks, and it's the place where the
# hurdle matters: the answer should be the *marginal* expectation $\psi \cdot \mu$, i.e.
# averaged over the chance the store skips. We anchor on a non-summer predictor month
# (February → target March) so the inactive baseline is clean.

# %%
grid_compare = pl.DataFrame(
    {
        "roas": [1.0, 4.0],
        "cohort_age": [12, 12],
        "month_of_year": [2, 2],
    }
)
marg_compare = predict_marginal(model, idata, grid_compare)
diff_samples = marg_compare[:, 1] - marg_compare[:, 0]
diff_mean = float(diff_samples.mean())
diff_lo, diff_hi = np.quantile(diff_samples, [0.03, 0.97])

print(
    f"E[budget_next | roas=4] - E[budget_next | roas=1] = "
    f"{diff_mean:.2f}  (94% CI: {diff_lo:.2f}, {diff_hi:.2f})"
)

# %% [markdown]
# One number, on the response scale, with uncertainty attached — exactly the language the
# platform team would use.

# %% [markdown]
# ### Cohort-age effect

# %%
ages = np.arange(0, params.n_months)
grid_age = pl.DataFrame(
    {
        "roas": np.full_like(ages, 2.5, dtype=float),
        "cohort_age": ages,
        "month_of_year": np.full_like(ages, 6),
    }
)
mu_age = predict_mu(model, idata, grid_age)
sum_age = summarise(mu_age)

truth_age = np.exp(
    params.intercept
    + season(np.full_like(ages, 6, dtype=float))
    + params.cohort_slope * ages
    + f_roas(np.full_like(ages, 2.5, dtype=float))
)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(ages, sum_age["mean"], color="C0", label="posterior mean")
ax.fill_between(
    ages, sum_age["lo"], sum_age["hi"], alpha=0.25, color="C0", label="94% CI"
)
ax.plot(ages, truth_age, color="black", linestyle="--", label="ground truth")
ax.set_xlabel("cohort age (months)")
ax.set_ylabel("expected budget next month")
ax.set_title("Cohort-age effect (roas=2.5, month=6)")
ax.legend()
fig.tight_layout()

# %% [markdown]
# ### Yearly seasonality

# %%
months = np.arange(1, 13)
grid_month = pl.DataFrame(
    {
        "roas": np.full_like(months, 2.5, dtype=float),
        "cohort_age": np.full_like(months, 12),
        "month_of_year": months,
    }
)
mu_month = predict_mu(model, idata, grid_month)
sum_month = summarise(mu_month)

truth_month = np.exp(
    params.intercept
    + season(months.astype(float))
    + params.cohort_slope * 12
    + f_roas(np.full_like(months, 2.5, dtype=float))
)

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(
    months,
    sum_month["mean"],
    yerr=[sum_month["mean"] - sum_month["lo"], sum_month["hi"] - sum_month["mean"]],
    color="C0",
    alpha=0.7,
    label="posterior mean ± 94% CI",
)
ax.plot(
    months, truth_month, color="black", linestyle="--", marker="o", label="ground truth"
)
ax.set_xticks(months)
ax.set_xlabel("month of year")
ax.set_ylabel("expected budget next month")
ax.set_title("Seasonality (roas=2.5, cohort_age=12)")
ax.legend()
fig.tight_layout()

# %% [markdown]
# ## The same answers via Bambi's `interpret` module
#
# Now that we've seen the mechanics, here's what you'd actually write day to day. Bambi
# ships [interpretation tools](https://bambinos.github.io/bambi/notebooks/#tools-to-interpret-model-outputs)
# with the same three primitives, driven by a `conditional` dictionary in place of the
# polars grids we built by hand.

# %%
fig, ax = plt.subplots(figsize=(12, 6))
bmb.interpret.plot_predictions(
    model,
    idata,
    conditional={"roas": roas_eval, "cohort_age": 12, "month_of_year": 6},
    ax=ax,
)
ax.set_title("bmb.interpret.plot_predictions — across ROAS")
fig.tight_layout()

# %%
bmb.interpret.comparisons(
    model,
    idata,
    contrast={"roas": [1.0, 4.0]},
    conditional={"cohort_age": 12, "month_of_year": 6},
)

# %% [markdown]
# Same shapes, same intervals, three lines instead of thirty. Worth knowing both: the by-hand
# version is the one to reach for when the question doesn't fit a built-in primitive.

# %% [markdown]
# ## Model comparison and parameter recovery
#
# Three models, same fit dataframe, same evaluation grid. Two questions:
#
# 1. Which one generalises best out-of-sample? `az.compare` with leave-one-out (LOO).
# 2. Which one actually recovers the truth? We put the ROAS curve, the cohort slope, the
#    seasonality, and $\beta(\mathrm{roas})$ next to their ground-truth values side by side.

# %% [markdown]
# ### Leave-one-out cross-validation
#
# Lower `elpd_loo` is worse. The hurdle-Gamma + HSGP model should sit at the top with
# `elpd_diff = 0`; the linear hurdle-Gamma beats the linear Gaussian thanks to the right
# family and the log link.

# %%
compare_df = az.compare(
    {
        "linear_gaussian": idata_lm,
        "linear_hgamma": idata_hgl,
        "vc_hgamma": idata,
    },
    ic="loo",
)
compare_df

# %%
az.plot_compare(compare_df, insample_dev=False)

# %% [markdown]
# ### ROAS curve recovery
#
# Posterior mean ± 94% CI on the shared `roas_eval` grid for all three models, with the
# Gamma-conditional truth in dashed black. Baseline 1 is a straight line, baseline 2 is a
# monotone exponential of a line, only the varying-coefficient model bends through the
# peak and saturates past ROAS≈4.

# %%
mu_roas_vc = predict_mu(model, idata, grid_roas)
sum_roas_vc = summarise(mu_roas_vc)

model_curves = {
    "linear_gaussian": (sum_roas_lm, "C0"),
    "linear_hgamma": (sum_roas_hgl, "C1"),
    "vc_hgamma": (sum_roas_vc, "C2"),
}

fig, ax = plt.subplots(figsize=(12, 6))
for name, (summary, color) in model_curves.items():
    ax.plot(roas_eval, summary["mean"], color=color, label=name)
    ax.fill_between(
        roas_eval, summary["lo"], summary["hi"], alpha=0.15, color=color
    )
ax.plot(roas_eval, truth_budget, color="black", linestyle="--", label="ground truth")
ax.set_xlabel("roas")
ax.set_ylabel("expected budget next month, given active")
ax.set_title("ROAS curve recovery — three models vs ground truth")
ax.legend()
fig.tight_layout()

# %% [markdown]
# ### Cohort-age slope
#
# Both log-scale models (`linear_hgamma` and `vc_hgamma`) parameterise `cohort_age` with a
# single slope on $\log \mu$ — directly comparable to the DGP's `cohort_slope = -0.02`.
# Baseline 1's slope is on the data scale (not log), so we leave it off the forest plot.

# %%
az.plot_forest(
    [idata_hgl, idata],
    model_names=["linear_hgamma", "vc_hgamma"],
    var_names=["cohort_age"],
    combined=True,
    figsize=(10, 3),
)
plt.axvline(params.cohort_slope, color="black", linestyle="--", label="truth")
plt.legend()
plt.title("Cohort-age slope — log-scale models vs truth")
plt.tight_layout()

# %% [markdown]
# ### Seasonality recovery
#
# Sweep month-of-year, hold cohort age at 12 and ROAS at 2.5. Plot all three models'
# posterior means against the Gamma-conditional truth `exp(season(month) + ...)`.

# %%
months = np.arange(1, 13)
grid_month = pl.DataFrame(
    {
        "roas": np.full_like(months, 2.5, dtype=float),
        "cohort_age": np.full_like(months, 12),
        "month_of_year": months,
    }
)
truth_month = np.exp(
    params.intercept
    + season(months.astype(float))
    + params.cohort_slope * 12
    + f_roas(np.full_like(months, 2.5, dtype=float))
)

month_curves = {
    "linear_gaussian": (model_lm, idata_lm, "C0"),
    "linear_hgamma": (model_hgl, idata_hgl, "C1"),
    "vc_hgamma": (model, idata, "C2"),
}

fig, ax = plt.subplots(figsize=(12, 5))
for name, (mdl, idt, color) in month_curves.items():
    mu_m = predict_mu(mdl, idt, grid_month)
    sum_m = summarise(mu_m)
    ax.plot(months, sum_m["mean"], color=color, marker="o", label=name)
    ax.fill_between(months, sum_m["lo"], sum_m["hi"], alpha=0.15, color=color)
ax.plot(
    months, truth_month, color="black", linestyle="--", marker="s", label="ground truth"
)
ax.set_xticks(months)
ax.set_xlabel("month of year")
ax.set_ylabel("expected budget next month, given active")
ax.set_title("Seasonality recovery — three models vs ground truth")
ax.legend()
fig.tight_layout()

# %% [markdown]
# ### Varying coefficient $\beta(\mathrm{roas})$
#
# The real payoff. For each model we isolate the implied ROAS contribution to $\log \mu$
# at the reference scenario (`cohort_age=12`, `month=6`) and divide by ROAS to get an
# implied $\hat{\beta}(\mathrm{roas})$. Baselines collapse $\beta$ to a single value
# (constant for baseline 2 by construction); only the GP model traces the true curve.

# %%
roas_ref = 1.0
grid_ref = pl.DataFrame(
    {"roas": [roas_ref], "cohort_age": [12], "month_of_year": [6]}
)
g_anchor = float(f_roas(np.array([roas_ref]))[0])


def recover_beta(model_obj: bmb.Model, idata_obj) -> dict[str, np.ndarray]:
    mu_grid = predict_mu(model_obj, idata_obj, grid_roas)
    mu_anchor = predict_mu(model_obj, idata_obj, grid_ref)[:, 0]
    log_ratio = np.log(mu_grid) - np.log(mu_anchor)[:, None]
    return summarise((log_ratio + g_anchor) / roas_eval[None, :])


fig, ax = plt.subplots(figsize=(12, 6))
for name, (mdl, idt, color) in month_curves.items():
    sum_b = recover_beta(mdl, idt)
    ax.plot(roas_eval, sum_b["mean"], color=color, label=name)
    ax.fill_between(
        roas_eval, sum_b["lo"], sum_b["hi"], alpha=0.15, color=color
    )
ax.plot(
    roas_eval, beta_roas(roas_eval), color="black", linestyle="--", label="ground truth"
)
ax.axhline(0.0, color="gray", linewidth=0.8)
ax.set_xlabel("roas")
ax.set_ylabel(r"$\hat{\beta}(\mathrm{roas})$")
ax.set_title("Varying coefficient recovery — only the GP model bends")
ax.legend()
fig.tight_layout()

# %% [markdown]
# The GP model is the only one that recovers $\beta(\mathrm{roas})$ — and it's also the
# one that wins on LOO. The story closes: the right shape is also the right predictor.

# %% [markdown]
# ## Wrapping up
#
# Three models, three layers of flexibility, one true curve. The Gaussian linear model
# missed the response scale, missed the zero point-mass, and flattened the ROAS effect to
# a slope. Switching to Hurdle Gamma fixed the likelihood but left ROAS monotone. Only the
# Hurdle Gamma + HSGP model recovered both the shape of $\beta(\mathrm{roas})$ *and* won
# on LOO — predictive performance and structural recovery aligned.
#
# Once a model has a link function, a smooth term, or contrasts in it, raw posterior
# coefficients stop being a useful unit of communication. **Predictions** answer the
# scenario question and **comparisons** answer the counterfactual question — both in the
# data's own units, with uncertainty attached. The same recipe extends to **slopes** (the
# derivative question, $\partial \hat{Y} / \partial X$) by predicting on a grid nudged by
# $\pm \varepsilon$ and finite-differencing through the posterior. The
# [marginaleffects book](https://marginaleffects.com/) is the reference if you want to go
# further.
