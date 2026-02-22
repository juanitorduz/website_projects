# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # In-Sample $R^2$ is Not a Good Metric for Decision Making
#
# ## A Collider Bias Example
#
# In this notebook we demonstrate, through a simple simulation,
# that in-sample $R^2$ is not a reliable metric for evaluating
# models used in causal decision making. We construct a data
# generating process (DGP) based on a known structural causal
# model and show that including a **collider** variable in a
# regression improves in-sample fit ($R^2$) while simultaneously
# **biasing** the estimated causal effect of marketing spend on
# sales. The model that "fits better" gives worse answers to the
# question that matters: *"How much will sales increase if I
# raise marketing spend?"*

# %% [markdown]
# ## Prepare Notebook

# %%
import graphviz as gr
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import statsmodels.api as sm
from pydantic import BaseModel, ConfigDict, Field

plt.style.use("bmh")
plt.rcParams["figure.figsize"] = [12, 7]
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.facecolor"] = "white"

# %%
seed: int = sum(map(ord, "collider_bias"))
rng: np.random.Generator = np.random.default_rng(seed=seed)

# %% [markdown]
# ## The Causal DAG
#
# The true causal structure is as follows:
#
# - **Season** drives both **Spend** (companies increase
#   marketing during high season) and **Sales** (demand is
#   higher during peak season).
# - **Spend** has a direct causal effect on **Sales** (the
#   effect we want to estimate).
# - **Economy** (e.g. consumer confidence, macroeconomic
#   conditions) affects **Sales** but is **unobserved** — it
#   is not included in either regression. This adds realistic
#   unexplained variance that lowers $R^2$.
# - **Inquiries** (e.g. customer inquiries / website visits)
#   is a **collider**: it is caused by both **Spend** (ads
#   drive traffic) and **Sales** (word-of-mouth, organic
#   interest).
#
# Conditioning on a collider opens a spurious path between
# its parents, biasing the estimated relationship between
# Spend and Sales.

# %%
g = gr.Digraph()

g.node(name="season", label="Season", color="lightblue", style="filled")
g.node(name="spend", label="Spend", color="lightyellow", style="filled")
g.node(name="sales", label="Sales", color="lightgreen", style="filled")
g.node(name="economy", label="Economy\n(Unobserved)", color="lightgrey", style="filled")
g.node(
    name="inquiries", label="Inquiries\n(Collider)", color="lightsalmon", style="filled"
)

g.edge(tail_name="season", head_name="spend")
g.edge(tail_name="season", head_name="sales")
g.edge(tail_name="spend", head_name="sales", label=" β₁ = 3")
g.edge(tail_name="economy", head_name="sales", style="dashed")
g.edge(tail_name="spend", head_name="inquiries")
g.edge(tail_name="sales", head_name="inquiries")

g  # noqa: B018

# %% [markdown]
# ## Data Generating Process
#
# We generate $n = 200$ weeks of data according to:
#
# \begin{align}
# \text{season}_t &= \sin\!\left(\frac{2\pi\, t}{52}\right) \\[6pt]
# \text{spend}_t &= 5 + 2\,\text{season}_t
#     + \varepsilon^{\text{spend}}_t,
#     \quad \varepsilon^{\text{spend}}_t
#     \sim \mathcal{N}(0, 0.5^2) \\[6pt]
# \text{economy}_t &\sim \mathcal{N}(0, 1)
#     \quad \text{(unobserved)} \\[6pt]
# \text{sales}_t &= 10 + 3\,\text{spend}_t
#     + 5\,\text{season}_t
#     + 4\,\text{economy}_t
#     + \varepsilon^{\text{sales}}_t,
#     \quad \varepsilon^{\text{sales}}_t
#     \sim \mathcal{N}(0, 1) \\[6pt]
# \text{inquiries}_t &= \text{spend}_t
#     + 2\,\text{sales}_t
#     + \varepsilon^{\text{inq}}_t,
#     \quad \varepsilon^{\text{inq}}_t
#     \sim \mathcal{N}(0, 2^2)
# \end{align}
#
# The **true causal effect** of spend on sales is $\beta_1 = 3$.
# The variable `economy` affects sales but is **not included** in either regression,
# introducing realistic unexplained variance.
# The variable `inquiries` is a collider (common effect of spend and sales).


# %%
class Simulator(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    rng: np.random.Generator = Field(..., description="NumPy random number generator.")
    n: int = Field(default=200, description="Number of time periods (weeks).", gt=0)
    true_intercept: float = Field(
        default=10.0, description="Intercept in the sales equation."
    )
    true_beta_spend: float = Field(
        default=3.0, description="True causal effect of spend on sales."
    )
    true_beta_season: float = Field(
        default=5.0, description="Effect of seasonality on sales."
    )
    true_beta_economy: float = Field(
        default=4.0, description="Effect of the unobserved economy variable on sales."
    )
    spend_offset: float = Field(
        default=5.0, description="Baseline level of marketing spend."
    )
    spend_season_coeff: float = Field(
        default=2.0, description="How much seasonality drives spend."
    )
    spend_noise_scale: float = Field(
        default=0.5, description="Noise std for marketing spend.", gt=0
    )
    economy_noise_scale: float = Field(
        default=1.0, description="Noise std for the unobserved economy variable.", gt=0
    )
    sales_noise_scale: float = Field(
        default=1.0, description="Noise std for sales.", gt=0
    )
    inquiries_spend_coeff: float = Field(
        default=1.0, description="How much spend drives inquiries (collider)."
    )
    inquiries_sales_coeff: float = Field(
        default=2.0, description="How much sales drives inquiries (collider)."
    )
    inquiries_noise_scale: float = Field(
        default=2.0, description="Noise std for inquiries.", gt=0
    )

    def run(self) -> pl.DataFrame:
        t = np.arange(self.n)
        season = np.sin(2 * np.pi * t / 52)

        spend = (
            self.spend_offset
            + self.spend_season_coeff * season
            + self.rng.normal(loc=0, scale=self.spend_noise_scale, size=self.n)
        )

        economy = self.rng.normal(loc=0, scale=self.economy_noise_scale, size=self.n)

        sales = (
            self.true_intercept
            + self.true_beta_spend * spend
            + self.true_beta_season * season
            + self.true_beta_economy * economy
            + self.rng.normal(loc=0, scale=self.sales_noise_scale, size=self.n)
        )

        inquiries = (
            self.inquiries_spend_coeff * spend
            + self.inquiries_sales_coeff * sales
            + self.rng.normal(loc=0, scale=self.inquiries_noise_scale, size=self.n)
        )

        return pl.DataFrame(
            {
                "t": t,
                "season": season,
                "spend": spend,
                "economy": economy,
                "sales": sales,
                "inquiries": inquiries,
            }
        )


# %%
simulator = Simulator(rng=rng)

df = simulator.run()

df.head(10)  # noqa: B018

# %% [markdown]
# ## Visualize the Data

# %%
fig, axes = plt.subplots(
    nrows=5, ncols=1, figsize=(14, 14), sharex=True, layout="constrained"
)

axes[0].plot(df["t"], df["season"], color="C0")
axes[0].set(ylabel="Season")
axes[0].set_title("Seasonality Component", fontsize=14)

axes[1].plot(df["t"], df["spend"], color="C1")
axes[1].set(ylabel="Spend")
axes[1].set_title("Marketing Spend", fontsize=14)

axes[2].plot(df["t"], df["economy"], color="C4")
axes[2].set(ylabel="Economy")
axes[2].set_title("Economy (Unobserved — not included in regressions)", fontsize=14)

axes[3].plot(df["t"], df["sales"], color="C2")
axes[3].set(ylabel="Sales")
axes[3].set_title("Sales", fontsize=14)

axes[4].plot(df["t"], df["inquiries"], color="C3")
axes[4].set(ylabel="Inquiries")
axes[4].set_title("Inquiries (Collider)", fontsize=14)

axes[4].set_xlabel("Week")

fig.suptitle("Simulated Time Series Data", fontsize=16, fontweight="bold", y=1.01)

# %%
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 5), layout="constrained")

axes[0].scatter(df["spend"], df["sales"], alpha=0.5, edgecolors="k", linewidths=0.3)
axes[0].set(xlabel="Spend", ylabel="Sales", title="Spend vs Sales")

axes[1].scatter(df["spend"], df["inquiries"], alpha=0.5, edgecolors="k", linewidths=0.3)
axes[1].set(xlabel="Spend", ylabel="Inquiries", title="Spend vs Inquiries")

axes[2].scatter(df["sales"], df["inquiries"], alpha=0.5, edgecolors="k", linewidths=0.3)
axes[2].set(xlabel="Sales", ylabel="Inquiries", title="Sales vs Inquiries")

fig.suptitle("Pairwise Scatter Plots", fontsize=16, fontweight="bold")

# %% [markdown]
# ## Fit Both Models
#
# We fit two OLS regressions:
#
# - **Model A (Correct):** `sales ~ spend + season` — the correct structural model.
# - **Model B (Collider):** `sales ~ spend + season + inquiries` — includes the collider.

# %% [markdown]
# ### Model A: Correct Structural Model

# %%
df_pd = df.to_pandas()

X_a = sm.add_constant(df_pd[["spend", "season"]])
model_a = sm.OLS(endog=df_pd["sales"], exog=X_a).fit()

print(model_a.summary())

# %% [markdown]
# ### Model B: Collider Model (includes `inquiries`)

# %%
X_b = sm.add_constant(df_pd[["spend", "season", "inquiries"]])
model_b = sm.OLS(endog=df_pd["sales"], exog=X_b).fit()

print(model_b.summary())

# %% [markdown]
# ## Compare Results
#
# Let's compare the two models side by side on two axes:
# 1. **In-sample $R^2$**: how well does the model fit the data?
# 2. **Estimated causal effect of Spend**: how close is the coefficient to the true $\beta_1 = 3$?

# %%
results = pl.DataFrame(
    {
        "model": ["A (Correct)", "B (Collider)"],
        "r_squared": [model_a.rsquared, model_b.rsquared],
        "beta_spend": [model_a.params["spend"], model_b.params["spend"]],
        "beta_spend_se": [model_a.bse["spend"], model_b.bse["spend"]],
        "true_beta_spend": [simulator.true_beta_spend, simulator.true_beta_spend],
    }
).with_columns(
    beta_spend_error=(pl.col("beta_spend") - pl.col("true_beta_spend")).abs()
)

results  # noqa: B018

# %%
fig, axes = plt.subplots(
    nrows=1, ncols=2, figsize=(14, 5), layout="constrained"
)

colors = ["C0", "C3"]
model_labels = results["model"].to_list()

ax = axes[0]
bars = ax.bar(model_labels, results["r_squared"].to_list(), color=colors, edgecolor="k")
ax.set_ylabel("$R^2$")
ax.set_title("In-Sample $R^2$", fontsize=14)
ax.set_ylim(0, 1.0)
for bar, val in zip(bars, results["r_squared"].to_list()):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.001,
        f"{val:.4f}",
        ha="center",
        fontsize=12,
    )

ax = axes[1]
betas = results["beta_spend"].to_list()
ses = results["beta_spend_se"].to_list()
x_pos = np.arange(len(model_labels))
ax.bar(
    x_pos, betas, yerr=[1.96 * s for s in ses], color=colors, edgecolor="k", capsize=5
)
ax.axhline(
    y=simulator.true_beta_spend,
    color="k",
    linestyle="--",
    linewidth=1.5,
    label=f"True β₁ = {simulator.true_beta_spend}",
)
ax.set_xticks(x_pos)
ax.set_xticklabels(model_labels)
ax.set_ylabel("Estimated β (Spend)")
ax.set_title("Estimated Causal Effect of Spend on Sales", fontsize=14)
ax.legend()

fig.suptitle("Model A (Correct) vs Model B (Collider)", fontsize=16, fontweight="bold")

# %% [markdown]
# **Key observation:** Model B (with the collider) has a **higher** $R^2$ but a **biased**
# estimate of the causal effect of Spend. The collider variable `inquiries` absorbs variation
# in Sales (because it is partly caused by Sales), inflating $R^2$. But conditioning on it
# opens a spurious path, distorting the coefficient on Spend.

# %% [markdown]
# ## Decision-Making Simulation
#
# Suppose a decision maker asks: *"If I increase marketing spend by $\Delta = 5$ units,
# how much additional sales should I expect?"*
#
# The true answer is $\Delta \times \beta_1 = 5 \times 3 = 15$.
# Let's compare what each model predicts.

# %%
delta_spend: float = 5.0
true_lift: float = delta_spend * simulator.true_beta_spend

lift_model_a: float = delta_spend * model_a.params["spend"]
lift_model_b: float = delta_spend * model_b.params["spend"]

lift_results = pl.DataFrame(
    {
        "source": ["True DGP", "Model A (Correct)", "Model B (Collider)"],
        "predicted_sales_lift": [true_lift, lift_model_a, lift_model_b],
    }
).with_columns(error=(pl.col("predicted_sales_lift") - true_lift).abs())

lift_results

# %%
fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")

bar_colors = ["k", "C0", "C3"]
labels = lift_results["source"].to_list()
lifts = lift_results["predicted_sales_lift"].to_list()

bars = ax.bar(labels, lifts, color=bar_colors, edgecolor="k")
ax.axhline(
    y=true_lift,
    color="k",
    linestyle="--",
    linewidth=1.5,
    alpha=0.5,
    label=f"True lift = {true_lift:.1f}",
)

for bar, val in zip(bars, lifts):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f"{val:.2f}",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )

ax.set_ylabel("Predicted Sales Lift")
ax.set_title(
    f"Decision: Increase Spend by Δ = {delta_spend:.0f} units\n"
    f"How much additional sales do we expect?",
    fontsize=14,
)
ax.legend()

# %% [markdown]
# ## Conclusion
#
# This example illustrates a fundamental point: **in-sample $R^2$ measures predictive fit,
# not causal accuracy.** When a variable is a collider (a common effect of both the treatment
# and the outcome), including it in a regression:
#
# 1. **Improves** in-sample $R^2$ — because the collider carries information about the outcome.
# 2. **Biases** the estimated causal effect — because conditioning on a collider opens a
#    spurious (non-causal) path between treatment and outcome.
#
# For decision making (e.g., "how much should I spend on marketing?"), we need a model that
# correctly estimates the causal effect, not one that maximizes $R^2$. The structural causal
# model (Model A) answers the decision question correctly despite having a lower $R^2$.
