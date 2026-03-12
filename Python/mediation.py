# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # Mediation Analysis and (In)Direct Effects with PyMC
#
# Mediation analysis goes beyond asking *"does the treatment work?"* to ask *"how does the treatment work?"* Understanding the mechanisms by which an intervention achieves its effect can have serious consequences for what treatments or policy changes are preferable. For instance, a family intervention program during adolescence might reduce substance use disorder in young adulthood — but through which pathways? Should the intervention focus on reducing peer influence, or on curbing direct experimentation?
#
# This notebook demonstrates how to perform **causal mediation analysis** using [PyMC](https://docs.pymc.io/en/stable/) and the `do` operator. We decompose the total causal effect of a treatment into direct and indirect components, quantifying each pathway's contribution with full Bayesian uncertainty.
#
# **What you will learn:**
#
# - Build a joint generative model encoding a causal DAG with multiple mediators
# - Decompose total effects into direct, indirect, interaction, and dependence components
# - Compute interventional effects analytically and via the `do` operator
# - Cross-validate both approaches
#
# ## Approach
#
# This notebook ports the [ChiRho mediation analysis example](https://basisresearch.github.io/chirho/mediation.html) to PyMC, following the same style as our [backdoor adjustment tutorial](https://juanitorduz.github.io/intro_causal_inference_ppl_pymc/). For the mediation decomposition, we follow the [StatsNotebook causal mediation analysis](https://statsnotebook.io/blog/analysis/mediation/) blogpost, which implements the **interventional effects** framework (Vansteelandt and Daniel, 2017; Chan and Leung, 2020).
#
# ## References
#
# - Vansteelandt, S., & Daniel, R. M. (2017). Interventional effects for mediation analysis with
#   multiple mediators. *Epidemiology*, 28(2), 258.
# - Chan, G., & Leung, J. (2020). Causal mediation analysis using the interventional effect
#   approach.
# - Pearl, J. (2001). Direct and indirect effects. *Proceedings of the 17th Conference on
#   Uncertainty in Artificial Intelligence*.

# %% [markdown]
# ## Prepare Notebook

# %%
from itertools import product

import arviz as az
import graphviz as gr
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pymc.model.transform.conditioning import do, observe
from scipy.special import expit

az.style.use("arviz-darkgrid")
plt.rcParams["figure.figsize"] = [10, 6]
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.facecolor"] = "white"

# %load_ext autoreload
# %autoreload 2
# %config InlineBackend.figure_format = "retina"

# %%
seed: int = 42
rng: np.random.Generator = np.random.default_rng(seed=seed)

# %% [markdown]
# ## Read and Preprocess Data
#
# We use a synthetic dataset with 553 simulated individuals studying the effect of family intervention during adolescence on future substance use disorder. The dataset was discussed in a [StatsNotebook blogpost](https://statsnotebook.io/blog/analysis/mediation/) and the data can be found [here](https://statsnotebook.io/blog/data_management/example_data/substance.csv).
#
# **Variables:**
#
# - `gender`: binary (Female / Male)
# - `conflict`: level of family conflict (continuous, ~1-5)
# - `fam_int`: participation in family intervention during adolescence (binary, treatment)
# - `dev_peer`: engagement with deviant peer groups (binary, mediator 1)
# - `sub_exp`: experimentation with drugs (binary, mediator 2)
# - `sub_disorder`: diagnosis of substance use disorder in young adulthood (binary, outcome)
#
# **Remark (missing data):** The original blogpost handles missing data using multiple imputation (20 imputations via MICE). For simplicity, we drop rows with any missing values. This reduces the sample from 553 to ~410 observations (~25% drop). Since missingness may be related to covariates or outcomes, this could introduce selection bias. Our results will be qualitatively similar to the blogpost but not numerically identical — this is a plausible source of discrepancy.

# %%
data_url = "https://statsnotebook.io/blog/data_management/example_data/substance.csv"
raw_df = pl.read_csv(data_url, null_values="NA")

print(f"Number of individuals: {len(raw_df)}")

data_df = raw_df.drop_nulls().with_columns(
    pl.col("gender").eq(pl.lit("Male")).cast(pl.Int64)
)

n_obs = len(data_df)
print(f"Number of individuals without missing values: {n_obs}")

data_df.head()

# %% [markdown]
# ## Exploratory Data Analysis

# %%
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 8), layout="constrained")

for ax, col in zip(axes.flatten(), data_df.columns):
    if data_df[col].n_unique() <= 2:
        vc = data_df[col].value_counts().sort(col)
        ax.bar(vc[col].to_list(), vc["count"].to_list(), color=["C0", "C1"])
        ax.set_xlabel("No / Yes" if col != "gender" else "Female / Male")
    else:
        ax.hist(data_df[col].to_numpy(), bins=20, edgecolor="white")
        ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.set_title(col)

fig.suptitle("Marginal Distributions", fontsize=16, fontweight="bold")

# %%
binary_vars = ["fam_int", "dev_peer", "sub_exp", "sub_disorder"]

cross_tab_df = (
    data_df.group_by("fam_int")
    .agg([pl.col(v).mean() for v in binary_vars if v != "fam_int"])
    .sort("fam_int")
)

cross_tab_pdf = cross_tab_df.to_pandas().set_index("fam_int")
cross_tab_pdf.index = ["No intervention (fam_int=0)", "Intervention (fam_int=1)"]
cross_tab_pdf.style.format("{:.3f}").background_gradient(cmap="Blues", axis=0)

# %% [markdown]
# Intervention participants have lower rates across all outcomes, consistent with a protective effect.

# %% [markdown]
# ## Causal DAG
#
# The causal DAG encodes our structural assumptions. `gender` and `conflict` are exogenous
# covariates that influence all downstream variables. `fam_int` (treatment) affects both
# mediators (`dev_peer`, `sub_exp`) and the outcome (`sub_disorder`) directly. The mediators
# also affect the outcome.
#
# The direct edge `fam_int → sub_disorder` is included to capture the direct effect of
# family intervention on substance use disorder that does not operate through the mediators.
# This matches the regression specification in the
# [StatsNotebook blogpost](https://statsnotebook.io/blog/analysis/mediation/).

# %%
dag = gr.Digraph()

dag.node("gender")
dag.node("conflict")
dag.node("fam_int", style="filled", color="#2a2eec80")
dag.node("dev_peer", style="filled", color="#ff7f0e80")
dag.node("sub_exp", style="filled", color="#ff7f0e80")
dag.node("sub_disorder", style="filled", color="#328c0680")

for target in ["fam_int", "dev_peer", "sub_exp", "sub_disorder"]:
    dag.edge("gender", target)
    dag.edge("conflict", target)

dag.edge("fam_int", "dev_peer")
dag.edge("fam_int", "sub_exp")
dag.edge("fam_int", "sub_disorder")
dag.edge("dev_peer", "sub_disorder")
dag.edge("sub_exp", "sub_disorder")

dag

# %% [markdown]
# ## PyMC Model Specification
#
# We build a joint generative model with **four Bernoulli likelihoods**, each using a logistic link function. The model encodes the causal structure from the DAG above:
#
# 1. `fam_int ~ Bernoulli(logistic(gender, conflict))`
# 2. `dev_peer ~ Bernoulli(logistic(gender, conflict, fam_int))`
# 3. `sub_exp ~ Bernoulli(logistic(gender, conflict, fam_int))`
# 4. `sub_disorder ~ Bernoulli(logistic(gender, conflict, dev_peer, sub_exp, fam_int))`
#
# This corresponds to the three regression models described in the blogpost, plus a model for the treatment assignment mechanism. All four are needed for the full generative model that enables counterfactual reasoning via the `do` operator.
#
# **Remark (priors):** All regression coefficients and intercepts use `Normal(0, 1)` priors. On the log-odds scale, this is moderately informative: it places most prior mass on effects between roughly $-2$ and $+2$ log-odds, covering a wide but plausible range of effect sizes for binary outcomes.
#
# **Notation:** Throughout this notebook we use $M_1$ = `dev_peer`, $M_2$ = `sub_exp`, $Y$ = `sub_disorder`, and $T$ = `fam_int`.

# %%
gender_obs = data_df["gender"].to_numpy()
conflict_obs = data_df["conflict"].to_numpy()
fam_int_obs = data_df["fam_int"].to_numpy()
dev_peer_obs = data_df["dev_peer"].to_numpy()
sub_exp_obs = data_df["sub_exp"].to_numpy()
sub_disorder_obs = data_df["sub_disorder"].to_numpy()

# %%
coords = {"obs_idx": range(len(data_df))}


def _add_logistic_component(
    outcome_name: str,
    param_suffix: str,
    predictors: dict[str, pt.TensorVariable],
) -> tuple[pt.TensorVariable, pt.TensorVariable]:
    """Add a logistic regression sub-model inside a pm.Model context.

    Parameters
    ----------
    outcome_name : str
        Name for the outcome variable (e.g. "fam_int", "dev_peer").
    param_suffix : str
        Short suffix used for parameter names (e.g. "fi", "dp").
    predictors : dict
        Mapping from covariate suffix → PyMC/tensor variable.

    Returns
    -------
    mu : pm.Deterministic
        Probability (expit of the linear predictor), named ``mu_{outcome_name}``.
    outcome : pm.Bernoulli
        Binary outcome variable, named from ``outcome_name``.
    """
    intercept = pm.Normal(f"intercept_{param_suffix}", mu=0, sigma=1)
    logit = intercept
    for covariate_suffix, variable in predictors.items():
        beta = pm.Normal(f"beta_{covariate_suffix}_{param_suffix}", mu=0, sigma=1)
        logit = logit + beta * variable
    mu = pm.Deterministic(
        f"mu_{outcome_name}", pt.expit(logit), dims=("obs_idx",)
    )
    outcome = pm.Bernoulli(outcome_name, p=mu, dims=("obs_idx",))
    return mu, outcome


with pm.Model(coords=coords) as mediation_model:
    gender_data = pm.Data("gender_data", gender_obs, dims=("obs_idx",))
    conflict_data = pm.Data("conflict_data", conflict_obs, dims=("obs_idx",))

    # (1) fam_int: logistic(gender, conflict)
    mu_fam_int, fam_int = _add_logistic_component(
        "fam_int", "fi",
        {"gender": gender_data, "conflict": conflict_data},
    )

    # (2) dev_peer: logistic(gender, conflict, fam_int)
    mu_dev_peer, dev_peer = _add_logistic_component(
        "dev_peer", "dp",
        {"gender": gender_data, "conflict": conflict_data, "fi": fam_int},
    )

    # (3) sub_exp: logistic(gender, conflict, fam_int)
    mu_sub_exp, sub_exp = _add_logistic_component(
        "sub_exp", "se",
        {"gender": gender_data, "conflict": conflict_data, "fi": fam_int},
    )

    # (4) sub_disorder: logistic(gender, conflict, dev_peer, sub_exp, fam_int)
    mu_sub_disorder, sub_disorder = _add_logistic_component(
        "sub_disorder", "sd",
        {
            "gender": gender_data,
            "conflict": conflict_data,
            "dp": dev_peer,
            "se": sub_exp,
            "fi": fam_int,
        },
    )

pm.model_to_graphviz(mediation_model)

# %% [markdown]
# ## Prior Predictive Checks

# %%
with mediation_model:
    prior_idata = pm.sample_prior_predictive(samples=2_000, random_seed=rng)

# %%
target_vars = [
    "fam_int",
    "dev_peer",
    "sub_exp",
    "sub_disorder",
]

fig, axes = plt.subplots(
    nrows=len(target_vars) // 2,
    ncols=2,
    figsize=(10, 6),
    sharex=True,
    sharey=True,
    layout="constrained",
)


for ax, var in zip(axes.flatten(), target_vars, strict=True):
    prior_samples = prior_idata["prior"][var].values.flatten()
    az.plot_dist(prior_samples, ax=ax)
    ax.set(title=var)

fig.suptitle("Prior Predictive Checks", fontsize=16, fontweight="bold")

# %% [markdown]
# ## Model Conditioning and MCMC Fit
#
# We condition (observe) the model on all four endogenous binary variables and fit using MCMC.

# %%
conditioned_model = observe(
    mediation_model,
    {
        "fam_int": fam_int_obs,
        "dev_peer": dev_peer_obs,
        "sub_exp": sub_exp_obs,
        "sub_disorder": sub_disorder_obs,
    },
)

pm.model_to_graphviz(conditioned_model)

# %%
sample_kwargs = {
    "draws": 2_000,
    "tune": 1_000,
    "chains": 4,
    "nuts_sampler": "nutpie",
    "random_seed": rng,
}

with conditioned_model:
    idata = pm.sample(**sample_kwargs)

# %% [markdown]
# ## Diagnostics

# %%
var_names = [
    v for v in idata.posterior.data_vars if v.startswith(("intercept_", "beta_"))
]

axes = az.plot_trace(
    data=idata,
    var_names=var_names,
    compact=True,
    backend_kwargs={"figsize": (12, 21), "layout": "constrained"},
)
plt.gcf().suptitle("Trace Plots", fontsize=18, fontweight="bold")

# %%
az.summary(idata, var_names=var_names, kind="diagnostics")

# %% [markdown]
# All parameters show R-hat values close to 1 and high effective sample sizes (ESS), indicating good convergence.

# %% [markdown]
# ## Posterior Predictive Checks

# %%
with conditioned_model:
    pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=rng)

# %%
fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(12, 8),
    layout="constrained",
)

observed_data = {
    "fam_int": fam_int_obs,
    "dev_peer": dev_peer_obs,
    "sub_exp": sub_exp_obs,
    "sub_disorder": sub_disorder_obs,
}

for ax, var in zip(axes.flatten(), target_vars, strict=True):
    pp_mean = idata.posterior_predictive[var].mean(dim="obs_idx")
    az.plot_posterior(
        pp_mean, ref_val=observed_data[var].mean(), ax=ax, hdi_prob=0.95
    )
    ax.set_title(f"{var} (posterior predictive mean)", fontsize=12)

fig.suptitle("Posterior Predictive Checks", fontsize=16, fontweight="bold")

# %% [markdown]
# ### Coefficient Interpretation
#
# Before moving to causal effect decomposition, it is worth briefly inspecting the fitted
# coefficients. The `beta_fi_*` parameters capture the association between family intervention
# and each downstream variable on the log-odds scale.

# %%
fi_coefs = [v for v in var_names if v.startswith("beta_fi_")]
axes = az.plot_forest(
    idata,
    var_names=fi_coefs,
    combined=True,
    hdi_prob=0.95,
    figsize=(8, 3),
)
plt.gcf().suptitle(
    "Family Intervention Coefficients (log-odds)", fontsize=14, fontweight="bold"
)
plt.tight_layout()

# %% [markdown]
# The negative coefficients for `beta_fi_dp` and `beta_fi_sd` suggest that family intervention
# reduces both deviant peer engagement and substance use disorder on the log-odds scale.
# `beta_fi_se` is smaller and closer to zero. However, these are *associations* conditional on
# covariates — the causal decomposition below disentangles the direct and indirect pathways.

# %% [markdown]
# ## Total Effect via the `do` Operator
#
# The total effect (TE) measures how much the probability of substance use disorder changes
# when we intervene to set family intervention to 1 (for everyone) versus 0 (for everyone):
#
# $$\text{TE} = \mathbb{E}[Y \mid do(\text{fam\_int}=1)] - \mathbb{E}[Y \mid do(\text{fam\_int}=0)]$$
#
# We compute this using PyMC's `do` operator, following the same pattern as the
# [backdoor adjustment tutorial](https://juanitorduz.github.io/intro_causal_inference_ppl_pymc/).

# %%
# We apply `do` to the conditioned model so covariates and outcome remain observed
# while only treatment is intervened upon.
do_0_model = do(conditioned_model, {"fam_int": np.zeros(n_obs, dtype=np.int32)})
do_1_model = do(conditioned_model, {"fam_int": np.ones(n_obs, dtype=np.int32)})


# %%
with do_0_model:
    do_0_idata = pm.sample_posterior_predictive(
        idata, random_seed=rng, var_names=["mu_sub_disorder"]
    )

with do_1_model:
    do_1_idata = pm.sample_posterior_predictive(
        idata, random_seed=rng, var_names=["mu_sub_disorder"]
    )

# %%
mu_sd_do_0 = do_0_idata["posterior_predictive"]["mu_sub_disorder"]
mu_sd_do_1 = do_1_idata["posterior_predictive"]["mu_sub_disorder"]

te_do = (mu_sd_do_1 - mu_sd_do_0).mean(dim="obs_idx")

# %%
fig, ax = plt.subplots()
az.plot_posterior(
    te_do.rename("Total Effect (do operator)"),
    hdi_prob=0.95,
    ref_val=0,
    ax=ax,
)
ax.set_title("Total Effect via do Operator", fontsize=18, fontweight="bold")

# %% [markdown]
# The total effect tells us that the intervention works, but not *how*. To understand the mechanisms, we decompose TE into contributions from each causal pathway.

# %% [markdown]
# ## Mediation Decomposition: Analytical Computation
#
# Since all mediators are binary and conditionally independent given the treatment and covariates, we can compute the interventional expectations **analytically** from the posterior parameter samples. This avoids Monte Carlo noise in the decomposition.
#
# **The core idea:** imagine mixing and matching treatment assignments across different parts of the causal model. What if the outcome equation "sees" treatment, but the mediators behave as if there were no treatment? By comparing these hypothetical scenarios, we isolate each pathway's contribution.
#
# **Remark (conditional independence):** The absence of a `dev_peer` $\to$ `sub_exp` edge in the DAG means the two mediators are conditionally independent given treatment and covariates. This makes the joint mediator probability factorize: $P(M_1, M_2 \mid T, X) = P(M_1 \mid T, X) \cdot P(M_2 \mid T, X)$. If this assumption were violated (e.g., peer engagement causally drives experimentation), we would need to model the joint distribution and adjust accordingly.
#
# ### Setup
#
# For each posterior draw $\theta$ and observation $i$ with covariates
# $x_i = (\text{gender}_i, \text{conflict}_i)$, define (where $\sigma(z) = 1/(1+e^{-z})$ is the logistic sigmoid function):
#
# - $p_1(t) = P(M_1=1 \mid do(T=t), x_i;\theta) = \sigma(\alpha_{dp} + \beta_{g,dp}\, \text{gender}_i + \beta_{c,dp}\, \text{conflict}_i + \beta_{f,dp}\, t)$
# - $p_2(t) = P(M_2=1 \mid do(T=t), x_i;\theta) = \sigma(\alpha_{se} + \beta_{g,se}\, \text{gender}_i + \beta_{c,se}\, \text{conflict}_i + \beta_{f,se}\, t)$
# - $q(t, m_1, m_2) = P(Y=1 \mid T=t, M_1=m_1, M_2=m_2, x_i;\theta)$
#
# The expected outcome under the interventional regime $(t, t', t'')$ — treatment set to $t$,
# mediator 1 drawn from its $do(T=t')$ distribution, mediator 2 drawn from its $do(T=t'')$
# distribution — is:
#
# $$E_{t,t',t''}(x_i;\theta) = \sum_{m_1 \in \{0,1\}} \sum_{m_2 \in \{0,1\}} P(M_1=m_1 \mid T=t') \, P(M_2=m_2 \mid T=t'') \, q(t, m_1, m_2)$$
#
# For example, $E_{1,0,0}$ answers: *"What would the outcome be if treatment directly affects the outcome ($t=1$), but both mediators behave as if there were no treatment ($t'=0, t''=0$)?"* This isolates the **direct effect**.
#
# The following table summarizes the six regimes needed for the decomposition:
#
# | Notation | Regime | Interpretation |
# |----------|--------|----------------|
# | $E_{0,0,0}$ | baseline | No treatment anywhere |
# | $E_{1,1,1}$ | full treatment | Treatment everywhere |
# | $E_{1,0,0}$ | direct only | Treatment in outcome eq., mediators from control |
# | $E_{0,1,0}$ | $M_1$ pathway | Mediator 1 from treatment, rest from control |
# | $E_{0,0,1}$ | $M_2$ pathway | Mediator 2 from treatment, rest from control |
# | $E_{0,1,1}$ | both mediators | Both mediators from treatment, outcome from control |
#
# ### Effects (matching the blogpost table)
#
# 1. **Total Effect**: $\text{TE} = \bar{E}_{1,1,1} - \bar{E}_{0,0,0}$
# 2. **Direct Effect**: $\text{DE} = \bar{E}_{1,0,0} - \bar{E}_{0,0,0}$
# 3. **Indirect through dev_peer**: $\text{IIE}_1 = \bar{E}_{0,1,0} - \bar{E}_{0,0,0}$
# 4. **Indirect through sub_exp**: $\text{IIE}_2 = \bar{E}_{0,0,1} - \bar{E}_{0,0,0}$
# 5. **Interaction**: $\text{INT} = \bar{E}_{0,1,1} - \bar{E}_{0,1,0} - \bar{E}_{0,0,1} + \bar{E}_{0,0,0}$
# 6. **Dependence**: $\text{DEP} = \text{TE} - \text{DE} - \text{IIE}_1 - \text{IIE}_2 - \text{INT}$
# 7. **Proportion through M1**: $\text{IIE}_1 / \text{TE}$
# 8. **Proportion through M2**: $\text{IIE}_2 / \text{TE}$
#
# where $\bar{E}$ denotes averaging over observations.
#
# The **dependence** term captures any remaining contribution from joint shifts in the mediator distributions that is not explained by the individual indirect effects or their interaction. Under the conditional independence assumption encoded in our DAG, this term should be small.

# %% [markdown]
# ### Extracting posterior samples for manual computation
#
# To compute the interventional expectations analytically, we need to evaluate the
# logistic regression equations from our model for each posterior draw. We extract
# all regression coefficients as NumPy arrays with shape `(n_chains, n_draws, 1)` so they
# broadcast naturally against the covariate arrays of shape `(1, 1, n_obs)`. This
# gives us `(n_chains, n_draws, n_obs)` arrays — one predicted probability per posterior
# draw and observation.

# %%
posterior = idata.posterior

# Covariates: shape (1, 1, n_obs) for broadcasting with (n_chains, n_draws, 1) parameters
gender_arr = gender_obs[np.newaxis, np.newaxis, :]
conflict_arr = conflict_obs[np.newaxis, np.newaxis, :]


def _get_param(name: str) -> np.ndarray:
    """Extract posterior samples as a (n_chains, n_draws, 1) array for broadcasting."""
    return posterior[name].values[:, :, np.newaxis]


# --- dev_peer sub-model parameters ---
intercept_dp_s = _get_param("intercept_dp")
beta_gender_dp_s = _get_param("beta_gender_dp")
beta_conflict_dp_s = _get_param("beta_conflict_dp")
beta_fi_dp_s = _get_param("beta_fi_dp")

# --- sub_exp sub-model parameters ---
intercept_se_s = _get_param("intercept_se")
beta_gender_se_s = _get_param("beta_gender_se")
beta_conflict_se_s = _get_param("beta_conflict_se")
beta_fi_se_s = _get_param("beta_fi_se")

# --- sub_disorder sub-model parameters ---
intercept_sd_s = _get_param("intercept_sd")
beta_gender_sd_s = _get_param("beta_gender_sd")
beta_conflict_sd_s = _get_param("beta_conflict_sd")
beta_dp_sd_s = _get_param("beta_dp_sd")
beta_se_sd_s = _get_param("beta_se_sd")
beta_fi_sd_s = _get_param("beta_fi_sd")

# %% [markdown]
# ### Mediator and outcome probability functions
#
# These three functions reconstruct the logistic regressions from the model,
# evaluating them at arbitrary treatment / mediator values using the posterior
# parameter samples. Each returns an `(n_chains, n_draws, n_obs)` array of probabilities.


# %%
def p_dev_peer(t: int) -> np.ndarray:
    """P(dev_peer=1 | do(fam_int=t), X) for each posterior draw and observation."""
    logit = (
        intercept_dp_s
        + beta_gender_dp_s * gender_arr
        + beta_conflict_dp_s * conflict_arr
        + beta_fi_dp_s * t
    )
    return expit(logit)


def p_sub_exp(t: int) -> np.ndarray:
    """P(sub_exp=1 | do(fam_int=t), X) for each posterior draw and observation."""
    logit = (
        intercept_se_s
        + beta_gender_se_s * gender_arr
        + beta_conflict_se_s * conflict_arr
        + beta_fi_se_s * t
    )
    return expit(logit)


def p_sub_disorder(t: int, m1: int, m2: int) -> np.ndarray:
    """P(sub_disorder=1 | fam_int=t, dev_peer=m1, sub_exp=m2, X)."""
    logit = (
        intercept_sd_s
        + beta_gender_sd_s * gender_arr
        + beta_conflict_sd_s * conflict_arr
        + beta_dp_sd_s * m1
        + beta_se_sd_s * m2
        + beta_fi_sd_s * t
    )
    return expit(logit)


# %% [markdown]
# ### Marginalizing over binary mediators
#
# The key insight is that because both mediators are **binary**, we can enumerate
# all four $(m_1, m_2) \in \{0,1\}^2$ combinations and weight the outcome
# probability by the joint mediator probability. Since the mediators are
# conditionally independent given treatment and covariates, the joint probability
# factorizes: $P(M_1=m_1, M_2=m_2) = P(M_1=m_1) \cdot P(M_2=m_2)$.
#
# The three subscripts in $E_{t, t', t''}$ control:
# - $t$: the treatment value plugged into the **outcome** equation
# - $t'$: the treatment value used to generate the **dev_peer** mediator distribution
# - $t''$: the treatment value used to generate the **sub_exp** mediator distribution
#
# By mixing and matching these subscripts we can isolate each causal pathway.


# %%
def expected_outcome(t: int, t_m1: int, t_m2: int) -> np.ndarray:
    """E[Y | do(T=t), M1 ~ do(T=t_m1), M2 ~ do(T=t_m2)].

    Returns an (n_chains, n_draws, n_obs) array. Analytically marginalizes over the
    four binary mediator combinations, weighted by their interventional
    probabilities.
    """
    p1 = p_dev_peer(t_m1)  # P(dev_peer=1 | do(T=t'))
    p2 = p_sub_exp(t_m2)  # P(sub_exp=1 | do(T=t''))
    # Enumerate all (m1, m2) ∈ {0,1}² and weight by joint mediator probability
    return (
        p1 * p2 * p_sub_disorder(t, 1, 1)
        + p1 * (1 - p2) * p_sub_disorder(t, 1, 0)
        + (1 - p1) * p2 * p_sub_disorder(t, 0, 1)
        + (1 - p1) * (1 - p2) * p_sub_disorder(t, 0, 0)
    )


# %% [markdown]
# ### Computing all interventional expectations and effects
#
# We now evaluate $E_{t,t',t''}$ for the six regimes from the table above.

# %%
expected_000 = expected_outcome(0, 0, 0)  # baseline: no treatment anywhere
expected_111 = expected_outcome(1, 1, 1)  # full treatment everywhere
expected_100 = expected_outcome(1, 0, 0)  # direct: treatment only in outcome equation
expected_010 = expected_outcome(0, 1, 0)  # indirect M1: dev_peer from treatment
expected_001 = expected_outcome(0, 0, 1)  # indirect M2: sub_exp from treatment
expected_011 = expected_outcome(0, 1, 1)  # both mediators from treatment

def _to_xarray(arr: np.ndarray, name: str) -> xr.DataArray:
    """Wrap a (n_chains, n_draws) array into a named xarray DataArray."""
    return xr.DataArray(
        arr,
        dims=("chain", "draw"),
        coords={"chain": posterior.chain, "draw": posterior.draw},
        name=name,
    )


te_analytical = _to_xarray((expected_111 - expected_000).mean(axis=-1), "TE")
de = _to_xarray((expected_100 - expected_000).mean(axis=-1), "DE")
iie_m1 = _to_xarray((expected_010 - expected_000).mean(axis=-1), "IIE_M1")
iie_m2 = _to_xarray((expected_001 - expected_000).mean(axis=-1), "IIE_M2")
interaction = _to_xarray(
    (expected_011 - expected_010 - expected_001 + expected_000).mean(axis=-1), "INT"
)
dependence = (te_analytical - de - iie_m1 - iie_m2 - interaction).rename("DEP")

# Note: proportions can be unstable when TE is near zero for individual
# posterior draws, producing extreme values. Interpret with caution.
prop_m1 = (iie_m1 / te_analytical).rename("prop_M1")
prop_m2 = (iie_m2 / te_analytical).rename("prop_M2")

# %%
effects = {
    "Indirect through dev_peer (M1)": iie_m1,
    "Indirect through sub_exp (M2)": iie_m2,
    "Interaction between mediators": interaction,
    "Dependence between mediators": dependence,
    "Direct effect": de,
    "Total effect": te_analytical,
}

fig, axes = plt.subplots(
    nrows=3, ncols=2, figsize=(14, 12), layout="constrained", sharex=True
)

for ax, (name, samples) in zip(axes.flatten(), effects.items()):
    az.plot_posterior(samples, hdi_prob=0.95, ax=ax, kind="hist", bins=50)
    ax.axvline(0, color="grey", linestyle="--", alpha=0.5)
    ax.set_title(name, fontsize=12)

fig.suptitle("Mediation Decomposition (Analytical)", fontsize=16, fontweight="bold")

# %% [markdown]
# ### Decomposition Summary
#
# The following bar chart shows how the total effect decomposes into its
# components — the "punchline plot" of the analysis.

# %%
decomp_effects = {
    "DE": de,
    "IIE (M1)": iie_m1,
    "IIE (M2)": iie_m2,
    "INT": interaction,
    "DEP": dependence,
}

fig, ax = plt.subplots(figsize=(10, 5))

names = list(decomp_effects.keys())
means = []
lower_err = []
upper_err = []
for v in decomp_effects.values():
    m = float(v.mean())
    hdi_ds = az.hdi(v, hdi_prob=0.95)
    lo = float(hdi_ds[v.name].sel(hdi="lower"))
    hi = float(hdi_ds[v.name].sel(hdi="higher"))
    means.append(m)
    lower_err.append(m - lo)
    upper_err.append(hi - m)

colors = ["C0", "C1", "C2", "C3", "C4"]
ax.barh(names, means, xerr=[lower_err, upper_err], color=colors, capsize=4)
ax.axvline(0, color="grey", linestyle="--", alpha=0.5)

te_mean = float(te_analytical.mean())
te_hdi_ds = az.hdi(te_analytical, hdi_prob=0.95)
te_lo = float(te_hdi_ds["TE"].sel(hdi="lower"))
te_hi = float(te_hdi_ds["TE"].sel(hdi="higher"))
ax.errorbar(
    te_mean,
    len(names),
    xerr=[[te_mean - te_lo], [te_hi - te_mean]],
    fmt="D",
    color="black",
    capsize=4,
    markersize=8,
)
ax.set_yticks(list(range(len(names) + 1)))
ax.set_yticklabels(names + ["TE (sum)"])
ax.set_xlabel("Effect (probability scale)")
ax.set_title(
    "TE = DE + IIE(M1) + IIE(M2) + INT + DEP",
    fontsize=14,
    fontweight="bold",
)
fig.tight_layout()

# %% [markdown]
# ## Mediation Decomposition via `do` Operator
#
# The analytical decomposition above required us to **manually extract** posterior parameters
# and reconstruct the logistic regression equations in NumPy. This works, but it is tightly
# coupled to the model specification: if we changed the link function, added interactions, or
# used continuous mediators, we would need to rewrite the helper functions from scratch.
#
# The `do` operator offers a **model-agnostic** alternative. Instead of pulling out parameters,
# we intervene directly on the generative model and let PyMC's `sample_posterior_predictive`
# compute the quantities we need. The only requirement is that we can enumerate (or sample from)
# the mediator support — which is trivial for binary mediators.
#
# **Strategy:** We assemble $E_{t,t',t''}$ from two ingredients, both obtained via `do`:
#
# 1. **Mediator probabilities** — From `do(fam_int=t)` models we read off `mu_dev_peer` and
#    `mu_sub_exp`, giving $P(M_k=1 \mid do(T=t))$ per posterior draw and observation.
# 2. **Outcome corner values** — For each of the 8 combinations $(t, m_1, m_2) \in \{0,1\}^3$,
#    we intervene on all three variables via `do(fam_int=t, dev_peer=m_1, sub_exp=m_2)` and
#    read off `mu_sub_disorder`, giving $q(t, m_1, m_2)$.
#
# The marginalization formula is the same as before:
#
# $$E_{t,t',t''} = \sum_{m_1, m_2} P(M_1=m_1 \mid do(T=t'))\, P(M_2=m_2 \mid do(T=t''))\, q(t, m_1, m_2)$$

# %% [markdown]
# ### Step 1: Mediator probabilities under each treatment level

# %%
with do_0_model:
    do_0_mediators = pm.sample_posterior_predictive(
        idata, random_seed=rng, var_names=["mu_dev_peer", "mu_sub_exp"]
    )

with do_1_model:
    do_1_mediators = pm.sample_posterior_predictive(
        idata, random_seed=rng, var_names=["mu_dev_peer", "mu_sub_exp"]
    )

mu_dp_do = {
    0: do_0_mediators["posterior_predictive"]["mu_dev_peer"],
    1: do_1_mediators["posterior_predictive"]["mu_dev_peer"],
}
mu_se_do = {
    0: do_0_mediators["posterior_predictive"]["mu_sub_exp"],
    1: do_1_mediators["posterior_predictive"]["mu_sub_exp"],
}

# %% [markdown]
# We now have `mu_dp_do[t]` and `mu_se_do[t]` — the probability that each mediator
# equals 1 under `do(fam_int=t)`, for every posterior draw and observation. These are
# the weights in our marginalization formula. Next we need the **outcome probabilities**
# at every corner of the mediator space.

# %% [markdown]
# ### Step 2: Outcome probabilities for all $(t, m_1, m_2)$ corners

# %%
q_do = {}
# 8 interventions: all (fam_int, dev_peer, sub_exp) in {0,1}^3
for t, m1, m2 in product([0, 1], repeat=3):
    interventions = {
        "fam_int": np.full(n_obs, t, dtype=np.int32),
        "dev_peer": np.full(n_obs, m1, dtype=np.int32),
        "sub_exp": np.full(n_obs, m2, dtype=np.int32),
    }
    model_tmm = do(conditioned_model, interventions)
    with model_tmm:
        pp = pm.sample_posterior_predictive(
            idata, random_seed=rng, var_names=["mu_sub_disorder"]
        )
    q_do[(t, m1, m2)] = pp["posterior_predictive"]["mu_sub_disorder"]

# %% [markdown]
# ### Step 3: Compute all interventional expectations and effects
#
# We now have all building blocks: mediator probabilities under each treatment level
# (`mu_dp_do`, `mu_se_do`) and outcome probabilities for every $(t, m_1, m_2)$ corner
# (`q_do`). We combine them using the same marginalization formula as in the analytical
# approach — the only difference is that the quantities come from `do` operator
# interventions rather than manually reconstructed logistic functions.


# %%
def expected_do_op(t: int, t_m1: int, t_m2: int) -> xr.DataArray:
    """Compute E_{t,t',t''} via do operator building blocks.

    Returns an xarray DataArray with dimensions (chain, draw), averaged over
    observations.
    """
    p1 = mu_dp_do[t_m1]
    p2 = mu_se_do[t_m2]
    # Same enumeration over binary mediator corners, now using xarray DataArrays
    return (
        p1 * p2 * q_do[(t, 1, 1)]
        + p1 * (1 - p2) * q_do[(t, 1, 0)]
        + (1 - p1) * p2 * q_do[(t, 0, 1)]
        + (1 - p1) * (1 - p2) * q_do[(t, 0, 0)]
    ).mean(dim="obs_idx")  # average over individuals


expected_do_000 = expected_do_op(0, 0, 0)
expected_do_111 = expected_do_op(1, 1, 1)
expected_do_100 = expected_do_op(1, 0, 0)
expected_do_010 = expected_do_op(0, 1, 0)
expected_do_001 = expected_do_op(0, 0, 1)
expected_do_011 = expected_do_op(0, 1, 1)

te_do_decomp = expected_do_111 - expected_do_000
de_do = expected_do_100 - expected_do_000
iie_m1_do = expected_do_010 - expected_do_000
iie_m2_do = expected_do_001 - expected_do_000
interaction_do = expected_do_011 - expected_do_010 - expected_do_001 + expected_do_000
dependence_do = te_do_decomp - de_do - iie_m1_do - iie_m2_do - interaction_do

prop_m1_do = iie_m1_do / te_do_decomp
prop_m2_do = iie_m2_do / te_do_decomp

# %%
effects_do = {
    "Indirect through dev_peer (M1)": iie_m1_do,
    "Indirect through sub_exp (M2)": iie_m2_do,
    "Interaction between mediators": interaction_do,
    "Dependence between mediators": dependence_do,
    "Direct effect": de_do,
    "Total effect": te_do_decomp,
}

fig, axes = plt.subplots(
    nrows=3, ncols=2, figsize=(14, 12), layout="constrained", sharex=True
)

for ax, (name, samples) in zip(axes.flatten(), effects_do.items()):
    az.plot_posterior(samples, hdi_prob=0.95, ax=ax, kind="hist", bins=50)
    ax.axvline(0, color="grey", linestyle="--", alpha=0.5)
    ax.set_title(name, fontsize=12)

fig.suptitle("Mediation Decomposition (do Operator)", fontsize=16, fontweight="bold")

# %% [markdown]
# ## Cross-validation: Analytical vs `do` Operator
#
# Since both approaches compute the same interventional expectations from the same posterior, they should agree exactly (up to floating-point precision). Let's verify this for all 6 effects.

# %%
effects_comparison = {
    "Indirect through dev_peer (M1)": (iie_m1, iie_m1_do),
    "Indirect through sub_exp (M2)": (iie_m2, iie_m2_do),
    "Interaction between mediators": (interaction, interaction_do),
    "Dependence between mediators": (dependence, dependence_do),
    "Direct effect": (de, de_do),
    "Total effect": (te_analytical, te_do_decomp),
}

fig, axes = plt.subplots(
    nrows=3, ncols=2, figsize=(14, 12), layout="constrained", sharex=True
)

for ax, (name, (analytical_s, do_s)) in zip(axes.flatten(), effects_comparison.items()):
    az.plot_dist(analytical_s, ax=ax, label="Analytical")
    az.plot_dist(do_s, ax=ax, label="do operator")
    ax.axvline(0, color="grey", linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)
    ax.set_title(name, fontsize=12)

fig.suptitle(
    "Cross-validation: Analytical vs do Operator",
    fontsize=16,
    fontweight="bold",
)

# %%
print("Effect comparison (posterior mean): Analytical vs do operator")
print("=" * 70)
for name, (analytical_s, do_s) in effects_comparison.items():
    a_mean = float(analytical_s.mean())
    d_mean = float(do_s.mean())
    print(f"  {name:40s}: {a_mean:+.5f} vs {d_mean:+.5f}")

# %%
# Three-way consistency check: compare TE from the initial do operator
# computation (te_do) against the analytical and do-decomposition TEs.
print("Three-way TE consistency check (posterior mean):")
print(f"  TE (do operator, initial):  {float(te_do.mean()):+.5f}")
print(f"  TE (analytical decomp):     {float(te_analytical.mean()):+.5f}")
print(f"  TE (do operator, decomp):   {float(te_do_decomp.mean()):+.5f}")

# %% [markdown]
# Both approaches yield identical results, confirming correctness. The `do` operator approach is more general: it does not require manually extracting posterior parameters or constructing logistic functions. All building blocks come from `sample_posterior_predictive` on appropriately intervened models.

# %% [markdown]
# ## Summary: Comparison with Blogpost Results
#
# The blogpost used the `intmed` R package with multiple imputation and 1000 Monte Carlo
# simulations. Our Bayesian approach uses PyMC with MCMC inference, dropping missing values
# instead of imputing. Despite these methodological differences, we expect qualitatively
# similar results.

# %%
blogpost_results = pl.DataFrame(
    {
        "effect": [
            "Indirect through dev_peer (M1)",
            "Indirect through sub_exp (M2)",
            "Interaction between mediators",
            "Dependence between mediators",
            "Direct effect",
            "Total effect",
            "Proportion through M1",
            "Proportion through M2",
        ],
        "blogpost_est": [-0.018, -0.007, 0.001, 0.000, -0.055, -0.077, 0.218, 0.078],
        "blogpost_ci_lower": [
            -0.037,
            -0.021,
            -0.002,
            -0.009,
            -0.120,
            -0.143,
            None,
            None,
        ],
        "blogpost_ci_upper": [
            -0.004,
            0.004,
            0.005,
            0.009,
            0.010,
            -0.016,
            None,
            None,
        ],
    }
)

all_effects = {
    "Indirect through dev_peer (M1)": iie_m1,
    "Indirect through sub_exp (M2)": iie_m2,
    "Interaction between mediators": interaction,
    "Dependence between mediators": dependence,
    "Direct effect": de,
    "Total effect": te_analytical,
    "Proportion through M1": prop_m1,
    "Proportion through M2": prop_m2,
}

pymc_means = []
pymc_hdi_lower = []
pymc_hdi_upper = []

for name in blogpost_results["effect"].to_list():
    samples = all_effects[name]
    mean_val = float(samples.mean())
    hdi_ds = az.hdi(samples, hdi_prob=0.95)
    hdi_low = float(hdi_ds[samples.name].sel(hdi="lower"))
    hdi_high = float(hdi_ds[samples.name].sel(hdi="higher"))
    pymc_means.append(mean_val)
    pymc_hdi_lower.append(hdi_low)
    pymc_hdi_upper.append(hdi_high)

blogpost_results = blogpost_results.with_columns(
    pl.Series("pymc_mean", pymc_means),
    pl.Series("pymc_hdi_lower", pymc_hdi_lower),
    pl.Series("pymc_hdi_upper", pymc_hdi_upper),
)

blogpost_results.to_pandas().style.format(
    {
        "blogpost_est": "{:.3f}",
        "blogpost_ci_lower": "{:.3f}",
        "blogpost_ci_upper": "{:.3f}",
        "pymc_mean": "{:.3f}",
        "pymc_hdi_lower": "{:.3f}",
        "pymc_hdi_upper": "{:.3f}",
    },
    na_rep="—",
)

# %%
causal_effects = {k: v for k, v in all_effects.items() if "Proportion" not in k}

fig, ax = plt.subplots(figsize=(10, 6))

y_positions = np.arange(len(causal_effects))
effect_names = list(causal_effects.keys())

for i, name in enumerate(effect_names):
    samples = causal_effects[name]
    mean_val = float(samples.mean())
    hdi_ds = az.hdi(samples, hdi_prob=0.95)
    hdi_low = float(hdi_ds[samples.name].sel(hdi="lower"))
    hdi_high = float(hdi_ds[samples.name].sel(hdi="higher"))
    ax.errorbar(
        mean_val,
        i,
        xerr=[[mean_val - hdi_low], [hdi_high - mean_val]],
        fmt="o",
        color="C0",
        capsize=4,
        label="PyMC (95% HDI)" if i == 0 else None,
    )
    bp_row = blogpost_results.filter(pl.col("effect") == name)
    bp_est = bp_row["blogpost_est"][0]
    bp_lo = bp_row["blogpost_ci_lower"][0]
    bp_hi = bp_row["blogpost_ci_upper"][0]
    if bp_lo is not None and bp_hi is not None:
        ax.errorbar(
            bp_est,
            i + 0.15,
            xerr=[[bp_est - bp_lo], [bp_hi - bp_est]],
            fmt="s",
            color="C1",
            capsize=4,
            markersize=7,
            label="Blogpost (95% CI)" if i == 0 else None,
        )
    else:
        ax.plot(
            bp_est,
            i + 0.15,
            "s",
            color="C1",
            markersize=7,
            label="Blogpost" if i == 0 else None,
        )

ax.axvline(0, color="grey", linestyle="--", alpha=0.5)
ax.set_yticks(y_positions)
ax.set_yticklabels(effect_names)
ax.set_xlabel("Effect (probability scale)")
ax.legend(loc="lower left")
ax.set_title("Mediation Effects: PyMC vs Blogpost", fontsize=16, fontweight="bold")
fig.tight_layout()

# %% [markdown]
# ## Conclusion
#
# The intervention reduced substance use disorder primarily through its direct effect, with a meaningful indirect pathway through deviant peer engagement but negligible contribution through substance experimentation.
#
# In this notebook, we demonstrated how to perform **causal mediation analysis** using PyMC's
# probabilistic programming framework and the `do` operator. The key takeaways are:
#
# 1. **Full generative models enable counterfactual reasoning.** By specifying all four
#    structural equations as a joint PyMC model, we can use the `do` operator to intervene on
#    any variable and compute counterfactual outcomes.
#
# 2. **Two complementary decomposition approaches.** We computed all mediation effects both
#    analytically (extracting posterior parameters) and via the `do` operator (intervening on
#    variables and forward-sampling). Both yield identical results, cross-validating each other.
#
# 3. **The `do` operator as a general-purpose tool.** The `do` operator approach does not
#    require manually constructing logistic functions from posterior parameters. It works by
#    composing interventions on the generative model, making it applicable even when analytical
#    marginalization is intractable (e.g., continuous mediators, nonlinear interactions).
#
# 4. **Bayesian uncertainty quantification.** Unlike the frequentist approach in the blogpost,
#    our Bayesian framework provides full posterior distributions over each mediation effect,
#    giving a richer picture of uncertainty.
#
# 5. **Qualitative agreement with the blogpost.** Despite methodological differences (Bayesian
#    vs. frequentist, dropping NAs vs. multiple imputation), our estimates are in the same
#    direction and order of magnitude as the blogpost results.
#
# ### Extensions
#
# - **Multiple imputation**: Handle missing data using PyMC's built-in capabilities or
#   external imputation before fitting.
# - **Non-binary mediators**: For continuous or ordinal mediators, the `do` operator approach
#   generalizes naturally — replace the exhaustive enumeration over mediator corners with Monte
#   Carlo samples from the mediator distributions under `do(T=t)`.
# - **Sensitivity analysis**: Assess robustness to unmeasured confounding between mediators
#   and outcome.

# %%
# %load_ext watermark
# %watermark -n -u -v -iv -w -p pymc,pytensor,nutpie
