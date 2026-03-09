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
# Mediation analysis goes beyond asking *"does the treatment work?"* to ask *"how does the treatment work?"* Understanding the mechanisms by which an intervention achieves its effect can have serious consequences for what treatments or policy changes are preferable. For instance, a family intervention program during adolescence might reduce substance use disorder in young adulthood — but through which pathways? Does it work by reducing engagement with deviant peer groups? By reducing experimentation with drugs? Or does it have a direct effect independent of these mediators?
#
# This notebook demonstrates how to perform **causal mediation analysis** using [PyMC](https://docs.pymc.io/en/stable/) and the `do` operator. We decompose the total causal effect of a treatment into direct and indirect components, quantifying each pathway's contribution with full Bayesian uncertainty.
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
import arviz as az
import graphviz as gr
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pymc as pm
import pytensor.tensor as pt
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
# **Note on missing data:** The original blogpost handles missing data using multiple imputation (20 imputations via MICE). For simplicity, we drop rows with any missing values. This reduces the sample from 553 to ~410 observations. Results will be qualitatively similar to the blogpost but not numerically identical.

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
    else:
        ax.hist(data_df[col].to_numpy(), bins=20, edgecolor="white")
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

# %%
gender_obs = data_df["gender"].to_numpy()
conflict_obs = data_df["conflict"].to_numpy()
fam_int_obs = data_df["fam_int"].to_numpy()
dev_peer_obs = data_df["dev_peer"].to_numpy()
sub_exp_obs = data_df["sub_exp"].to_numpy()
sub_disorder_obs = data_df["sub_disorder"].to_numpy()

# %%
coords = {"obs_idx": range(len(data_df))}

with pm.Model(coords=coords) as mediation_model:
    # --- Covariates as Data ---
    gender_data = pm.Data("gender_data", gender_obs, dims=("obs_idx",))
    conflict_data = pm.Data("conflict_data", conflict_obs, dims=("obs_idx",))

    # --- (1) fam_int model: logistic(gender, conflict) ---
    intercept_fi = pm.Normal("intercept_fi", mu=0, sigma=1)
    beta_gender_fi = pm.Normal("beta_gender_fi", mu=0, sigma=1)
    beta_conflict_fi = pm.Normal("beta_conflict_fi", mu=0, sigma=1)
    logit_fi = (
        intercept_fi + beta_gender_fi * gender_data + beta_conflict_fi * conflict_data
    )
    mu_fam_int = pm.Deterministic("mu_fam_int", pt.expit(logit_fi), dims=("obs_idx",))
    fam_int = pm.Bernoulli("fam_int", p=mu_fam_int, dims=("obs_idx",))

    # --- (2) dev_peer model: logistic(gender, conflict, fam_int) ---
    intercept_dp = pm.Normal("intercept_dp", mu=0, sigma=1)
    beta_gender_dp = pm.Normal("beta_gender_dp", mu=0, sigma=1)
    beta_conflict_dp = pm.Normal("beta_conflict_dp", mu=0, sigma=1)
    beta_fi_dp = pm.Normal("beta_fi_dp", mu=0, sigma=1)
    logit_dp = (
        intercept_dp
        + beta_gender_dp * gender_data
        + beta_conflict_dp * conflict_data
        + beta_fi_dp * fam_int
    )
    mu_dev_peer = pm.Deterministic("mu_dev_peer", pt.expit(logit_dp), dims=("obs_idx",))
    dev_peer = pm.Bernoulli("dev_peer", p=mu_dev_peer, dims=("obs_idx",))

    # --- (3) sub_exp model: logistic(gender, conflict, fam_int) ---
    intercept_se = pm.Normal("intercept_se", mu=0, sigma=1)
    beta_gender_se = pm.Normal("beta_gender_se", mu=0, sigma=1)
    beta_conflict_se = pm.Normal("beta_conflict_se", mu=0, sigma=1)
    beta_fi_se = pm.Normal("beta_fi_se", mu=0, sigma=1)
    logit_se = (
        intercept_se
        + beta_gender_se * gender_data
        + beta_conflict_se * conflict_data
        + beta_fi_se * fam_int
    )
    mu_sub_exp = pm.Deterministic("mu_sub_exp", pt.expit(logit_se), dims=("obs_idx",))
    sub_exp = pm.Bernoulli("sub_exp", p=mu_sub_exp, dims=("obs_idx",))

    # --- (4) sub_disorder model: logistic(gender, conflict, dev_peer, sub_exp, fam_int) --- # noqa: E501
    intercept_sd = pm.Normal("intercept_sd", mu=0, sigma=1)
    beta_gender_sd = pm.Normal("beta_gender_sd", mu=0, sigma=1)
    beta_conflict_sd = pm.Normal("beta_conflict_sd", mu=0, sigma=1)
    beta_dp_sd = pm.Normal("beta_dp_sd", mu=0, sigma=1)
    beta_se_sd = pm.Normal("beta_se_sd", mu=0, sigma=1)
    beta_fi_sd = pm.Normal("beta_fi_sd", mu=0, sigma=1)
    logit_sd = (
        intercept_sd
        + beta_gender_sd * gender_data
        + beta_conflict_sd * conflict_data
        + beta_dp_sd * dev_peer
        + beta_se_sd * sub_exp
        + beta_fi_sd * fam_int
    )
    mu_sub_disorder = pm.Deterministic(
        "mu_sub_disorder", pt.expit(logit_sd), dims=("obs_idx",)
    )
    sub_disorder = pm.Bernoulli("sub_disorder", p=mu_sub_disorder, dims=("obs_idx",))

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

# %% [markdown]
# ## Posterior Predictive Checks

# %%
with conditioned_model:
    pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=rng)

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
# ## Mediation Decomposition: Analytical Computation
#
# Since all mediators are binary and conditionally independent given the treatment and covariates, we can compute the interventional expectations **analytically** from the posterior parameter samples. This avoids Monte Carlo noise in the decomposition.
#
# ### Setup
#
# For each posterior draw $\theta$ and observation $i$ with covariates
# $x_i = (\text{gender}_i, \text{conflict}_i)$, define:
#
# - $p_1(t) = P(\text{dev\_peer}=1 \mid do(\text{fam\_int}=t), x_i;\theta) = \sigma(\alpha_{dp} + \beta_{g,dp}\, \text{gender}_i + \beta_{c,dp}\, \text{conflict}_i + \beta_{f,dp}\, t)$
# - $p_2(t) = P(\text{sub\_exp}=1 \mid do(\text{fam\_int}=t), x_i;\theta) = \sigma(\alpha_{se} + \beta_{g,se}\, \text{gender}_i + \beta_{c,se}\, \text{conflict}_i + \beta_{f,se}\, t)$
# - $q(t, m_1, m_2) = P(\text{sub\_disorder}=1 \mid \text{fam\_int}=t, \text{dev\_peer}=m_1, \text{sub\_exp}=m_2, x_i;\theta)$
#
# The expected outcome under the interventional regime $(t, t', t'')$ — treatment set to $t$,
# mediator 1 drawn from its $do(T=t')$ distribution, mediator 2 drawn from its $do(T=t'')$
# distribution — is:
#
# $$E_{t,t',t''}(x_i;\theta) = \sum_{m_1 \in \{0,1\}} \sum_{m_2 \in \{0,1\}} P(M_1=m_1 \mid T=t') \, P(M_2=m_2 \mid T=t'') \, q(t, m_1, m_2)$$
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

# %%
posterior = idata.posterior.stack(sample=("chain", "draw"))

gender_arr = gender_obs[np.newaxis, :]  # (1, n_obs)
conflict_arr = conflict_obs[np.newaxis, :]  # (1, n_obs)

intercept_dp_s = posterior["intercept_dp"].values[:, np.newaxis]
beta_gender_dp_s = posterior["beta_gender_dp"].values[:, np.newaxis]
beta_conflict_dp_s = posterior["beta_conflict_dp"].values[:, np.newaxis]
beta_fi_dp_s = posterior["beta_fi_dp"].values[:, np.newaxis]

intercept_se_s = posterior["intercept_se"].values[:, np.newaxis]
beta_gender_se_s = posterior["beta_gender_se"].values[:, np.newaxis]
beta_conflict_se_s = posterior["beta_conflict_se"].values[:, np.newaxis]
beta_fi_se_s = posterior["beta_fi_se"].values[:, np.newaxis]

intercept_sd_s = posterior["intercept_sd"].values[:, np.newaxis]
beta_gender_sd_s = posterior["beta_gender_sd"].values[:, np.newaxis]
beta_conflict_sd_s = posterior["beta_conflict_sd"].values[:, np.newaxis]
beta_dp_sd_s = posterior["beta_dp_sd"].values[:, np.newaxis]
beta_se_sd_s = posterior["beta_se_sd"].values[:, np.newaxis]
beta_fi_sd_s = posterior["beta_fi_sd"].values[:, np.newaxis]


def p_dev_peer(t):
    """P(dev_peer=1 | do(fam_int=t), X) for each (sample, obs)."""
    logit = (
        intercept_dp_s
        + beta_gender_dp_s * gender_arr
        + beta_conflict_dp_s * conflict_arr
        + beta_fi_dp_s * t
    )
    return expit(logit)


def p_sub_exp(t):
    """P(sub_exp=1 | do(fam_int=t), X) for each (sample, obs)."""
    logit = (
        intercept_se_s
        + beta_gender_se_s * gender_arr
        + beta_conflict_se_s * conflict_arr
        + beta_fi_se_s * t
    )
    return expit(logit)


def p_sub_disorder(t, m1, m2):
    """P(sub_disorder=1 | fam_int=t, dev_peer=m1, sub_exp=m2, X) for each (sample, obs)."""
    logit = (
        intercept_sd_s
        + beta_gender_sd_s * gender_arr
        + beta_conflict_sd_s * conflict_arr
        + beta_dp_sd_s * m1
        + beta_se_sd_s * m2
        + beta_fi_sd_s * t
    )
    return expit(logit)


def expected_outcome(t, t_m1, t_m2):
    """E[Y | do(T=t), M1 ~ do(T=t_m1), M2 ~ do(T=t_m2)] for each (sample, obs).

    Analytically marginalizes over binary mediators.
    """
    p1 = p_dev_peer(t_m1)
    p2 = p_sub_exp(t_m2)
    return (
        p1 * p2 * p_sub_disorder(t, 1, 1)
        + p1 * (1 - p2) * p_sub_disorder(t, 1, 0)
        + (1 - p1) * p2 * p_sub_disorder(t, 0, 1)
        + (1 - p1) * (1 - p2) * p_sub_disorder(t, 0, 0)
    )


# %%
E_000 = expected_outcome(0, 0, 0)  # baseline
E_111 = expected_outcome(1, 1, 1)  # full treatment
E_100 = expected_outcome(1, 0, 0)  # direct effect reference
E_010 = expected_outcome(0, 1, 0)  # indirect through M1
E_001 = expected_outcome(0, 0, 1)  # indirect through M2
E_011 = expected_outcome(0, 1, 1)  # interaction reference

te_analytical = (E_111 - E_000).mean(axis=1)
de = (E_100 - E_000).mean(axis=1)
iie_m1 = (E_010 - E_000).mean(axis=1)
iie_m2 = (E_001 - E_000).mean(axis=1)
interaction = (E_011 - E_010 - E_001 + E_000).mean(axis=1)
dependence = te_analytical - de - iie_m1 - iie_m2 - interaction

prop_m1 = iie_m1 / te_analytical
prop_m2 = iie_m2 / te_analytical

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
# ## Mediation Decomposition via `do` Operator
#
# The analytical decomposition above required extracting posterior parameters and manually constructing the logistic functions. As an alternative, we can compute all the building blocks using the `do` operator directly. This approach is more general: it works with any model structure without requiring closed-form marginalization, and it only requires that we can enumerate (or sample from) the mediator support.
#
# **Strategy:** We compute `E_{t,t',t''}` by combining two ingredients from the `do` operator:
#
# 1. **Mediator probabilities**: `mu_dev_peer` and `mu_sub_exp` from `do(fam_int=t)` models give us $P(M_k=1 \mid do(T=t))$ for each posterior draw and observation.
# 2. **Outcome corner values**: For each combination $(t, m_1, m_2)$ with $m_k \in \{0,1\}$, we intervene on all three variables via `do(fam_int=t, dev_peer=m_1, sub_exp=m_2)` to get $q(t, m_1, m_2) = P(Y=1 \mid T=t, M_1=m_1, M_2=m_2, X)$.
#
# Then: $E_{t,t',t''} = \sum_{m_1, m_2} P(M_1=m_1 \mid do(T=t'))\, P(M_2=m_2 \mid do(T=t''))\, q(t, m_1, m_2)$

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
# ### Step 2: Outcome probabilities for all $(t, m_1, m_2)$ corners

# %%
q_do = {}
for t in [0, 1]:
    for m1 in [0, 1]:
        for m2 in [0, 1]:
            model_tmm = do(
                conditioned_model,
                {
                    "fam_int": np.full(n_obs, t, dtype=np.int32),
                    "dev_peer": np.full(n_obs, m1, dtype=np.int32),
                    "sub_exp": np.full(n_obs, m2, dtype=np.int32),
                },
            )
            with model_tmm:
                pp = pm.sample_posterior_predictive(
                    idata, random_seed=rng, var_names=["mu_sub_disorder"]
                )
            q_do[(t, m1, m2)] = pp["posterior_predictive"]["mu_sub_disorder"]

# %% [markdown]
# ### Step 3: Compute all interventional expectations and effects


# %%
def E_do_op(t, t_m1, t_m2):
    """Compute E_{t,t',t''} via do operator building blocks."""
    p1 = mu_dp_do[t_m1]
    p2 = mu_se_do[t_m2]
    return (
        p1 * p2 * q_do[(t, 1, 1)]
        + p1 * (1 - p2) * q_do[(t, 1, 0)]
        + (1 - p1) * p2 * q_do[(t, 0, 1)]
        + (1 - p1) * (1 - p2) * q_do[(t, 0, 0)]
    ).mean(dim="obs_idx")


E_do_000 = E_do_op(0, 0, 0)
E_do_111 = E_do_op(1, 1, 1)
E_do_100 = E_do_op(1, 0, 0)
E_do_010 = E_do_op(0, 1, 0)
E_do_001 = E_do_op(0, 0, 1)
E_do_011 = E_do_op(0, 1, 1)

te_do_decomp = E_do_111 - E_do_000
de_do = E_do_100 - E_do_000
iie_m1_do = E_do_010 - E_do_000
iie_m2_do = E_do_001 - E_do_000
interaction_do = E_do_011 - E_do_010 - E_do_001 + E_do_000
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
    ax.hist(analytical_s, bins=50, alpha=0.5, density=True, label="Analytical")
    ax.hist(
        do_s.values.flatten(), bins=50, alpha=0.5, density=True, label="do operator"
    )
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
    a_mean = np.mean(analytical_s)
    d_mean = float(do_s.mean())
    print(f"  {name:40s}: {a_mean:+.5f} vs {d_mean:+.5f}")

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
    mean_val = np.mean(samples)
    hdi = az.hdi(samples, hdi_prob=0.95)
    pymc_means.append(mean_val)
    pymc_hdi_lower.append(hdi[0])
    pymc_hdi_upper.append(hdi[1])

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
    mean_val = np.mean(samples)
    hdi = az.hdi(samples, hdi_prob=0.95)
    ax.errorbar(
        mean_val,
        i,
        xerr=[[mean_val - hdi[0]], [hdi[1] - mean_val]],
        fmt="o",
        color="C0",
        capsize=4,
        label="PyMC (95% HDI)" if i == 0 else None,
    )
    bp_est = blogpost_results.filter(pl.col("effect") == name)["blogpost_est"][0]
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
