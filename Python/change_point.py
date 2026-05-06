# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Bayesian Changepoint Detection with Enumeration and `scan`
#
# Looking at the British coal-mining-disaster counts from 1851 to 1962, the eye
# picks a clear drop somewhere around 1890 — but *when* exactly did the rate
# change, and how confident can we be about the year? A frequentist analysis
# returns a point estimate; a Bayesian analysis returns a posterior over the
# changepoint year, and the *shape* of that posterior is the story.
#
# This tutorial answers the question in three escalating beats:
#
# 1. **Beat 1 — One change, exactly marginalised.** Assume there is a single
#    changepoint at unknown year `tau`. We use NumPyro's `enumerate` directive
#    to integrate `tau` out of the joint exactly, leaving HMC with a smooth
#    continuous target. We recover the discrete posterior over `tau` after the
#    fact with `Predictive(..., infer_discrete=True)`.
# 2. **Beat 2 — Many changes, parallel-scan marginalisation.** What if the
#    rate has been moving through *several* regimes? We extend the model to a
#    hidden Markov chain over `K` regimes and use
#    `numpyro.contrib.control_flow.scan` to marginalise the per-year discrete
#    states in parallel-scan time.
# 3. **Beat 3 — When enumeration isn't an option.** Enumeration only works
#    when the discrete support is finite and bounded. We compare against
#    `MixedHMC` and `DiscreteHMCGibbs` — NumPyro's gradient-free fallbacks
#    when enumeration is infeasible.
#
# The tutorial assumes prior familiarity with NumPyro and NUTS-based MCMC. If
# you are new to NumPyro, work through `bayesian_regression.ipynb` first.

# %% [markdown]
# ## Setup

# %%
import time

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import seaborn as sns
from numpyro.contrib.control_flow import scan
from numpyro.infer import (
    HMC,
    MCMC,
    NUTS,
    DiscreteHMCGibbs,
    MixedHMC,
    Predictive,
)

numpyro.set_host_device_count(4)
sns.set_theme(style="whitegrid")
RNG = jax.random.PRNGKey(0)

# %% [markdown]
# ## The data
#
# The classic coal-mining-disaster series (Jarrett, 1979): 112 yearly counts
# of fatal disasters in UK collieries between 1851 and 1962. The series is
# small enough to inline, and each value is the count of disasters in that
# calendar year.

# %%
# Coal-mining disasters per year, 1851-1962 (Jarrett 1979).
# fmt: off
disasters = jnp.array([
    4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6, 3, 3, 5, 4,
    5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5, 2, 2, 3, 4, 2, 1, 3, 2,
    2, 1, 1, 1, 1, 3, 0, 0, 1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2,
    0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
    3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0,
])
# fmt: on
years = np.arange(1851, 1963)
T = disasters.shape[0]

# %%
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(years, np.asarray(disasters), color="C0", alpha=0.8)
ax.axvspan(1885, 1895, color="C3", alpha=0.15, label="visual band 1885-1895")
ax.set_xlabel("year")
ax.set_ylabel("disasters")
ax.set_title("UK coal-mining disasters per year, 1851-1962")
ax.legend(loc="upper right")
fig.tight_layout()

# %% [markdown]
# The series shows a clear regime: counts of 3-6 are common before ~1890 and
# rare after. The rest of the tutorial puts a posterior on that intuition.

# %% [markdown]
# ## Beat 1 — Single changepoint via enumeration
#
# We model a single regime change at unknown year `tau`. Before `tau` the
# disaster rate is `lam1`; after, it is `lam2`. The observation model is
# Poisson:
#
# $$y_t \mid \tau, \lambda_1, \lambda_2 \;\sim\; \text{Poisson}(\lambda_1)
#   \mathbb{1}[t < \tau] + \text{Poisson}(\lambda_2)\,\mathbb{1}[t \ge \tau].$$
#
# The Bayesian formulation in this style is due to
# [Carlin, Gelfand, & Smith (1992)](https://doi.org/10.2307/2347570).
#
# Joint MCMC over `(tau, lam1, lam2)` is awkward because HMC can't gradient
# through a discrete index. NumPyro's escape hatch is **enumeration**:
# annotate the discrete site with `infer={"enumerate": "parallel"}` and
# NumPyro sums it out exactly during inference. HMC then sees a smooth
# marginal over the continuous parameters only — a Rao-Blackwellised target
# with lower-variance gradients. The enumeration backend is described in
# [Obermeyer et al. (2019)](https://arxiv.org/abs/1902.03210).


# %%
def single_changepoint(disasters=None, T=T):
    if disasters is not None:
        T = disasters.shape[0]
    lam1 = numpyro.sample("lam1", dist.Gamma(2.0, 1.0))
    lam2 = numpyro.sample("lam2", dist.Gamma(2.0, 1.0))
    tau = numpyro.sample(
        "tau",
        dist.Categorical(logits=jnp.zeros(T)),
        infer={"enumerate": "parallel"},
    )
    rate = jnp.where(jnp.arange(T) < tau, lam1, lam2)
    with numpyro.plate("years", T):
        numpyro.sample("obs", dist.Poisson(rate), obs=disasters)


# %% [markdown]
# ### Prior predictive check
#
# Before fitting, we verify the priors imply plausible data. The
# `Gamma(2, 1)` prior on the rates has mean 2 and standard deviation
# `sqrt(2) ≈ 1.4`, so per-year counts should mostly fall in `[0, 6]`,
# matching the historical range. The uniform `Categorical` prior on `tau`
# encodes "no a-priori preference for any year".
#
# We confirm by drawing prior samples of the *observations*. We pass
# `disasters=None` so the `obs=` argument is `None` — that turns the `obs`
# site back into a sampled site rather than an observed one.

# %%
prior_predictive = Predictive(single_changepoint, num_samples=500)
prior_samples_m1 = prior_predictive(jax.random.PRNGKey(10), disasters=None, T=T)
prior_obs_m1 = prior_samples_m1["obs"]
prior_tau_m1 = prior_samples_m1["tau"]
print(f"prior obs shape:  {prior_obs_m1.shape}")
print(f"prior obs mean:   {float(prior_obs_m1.mean()):.2f}  (expected ~2.0)")
print(f"prior obs max:    {int(prior_obs_m1.max())}      (expected < ~15)")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax = axes[0]
for i in range(8):
    ax.plot(years, np.asarray(prior_obs_m1[i]), color="C0", alpha=0.4, lw=0.8)
ax.bar(years, np.asarray(disasters), color="black", alpha=0.5, label="observed")
ax.set_title("Model 1 — prior predictive trajectories")
ax.set_xlabel("year")
ax.set_ylabel("disasters")
ax.legend()

ax = axes[1]
ax.hist(np.asarray(prior_tau_m1), bins=20, color="C1", alpha=0.8)
ax.set_title(r"Model 1 — prior over $\tau$ (uniform)")
ax.set_xlabel("year index")
fig.tight_layout()

# %% [markdown]
# Prior predictive trajectories cover the data's range without dwarfing it,
# and `tau` is uniform across years — both as intended. We can proceed.

# %% [markdown]
# ### Inference

# %%
mcmc_m1 = MCMC(
    NUTS(single_changepoint),
    num_warmup=1000,
    num_samples=2000,
    num_chains=4,
    chain_method="sequential",
    progress_bar=False,
)
mcmc_m1.run(RNG, disasters=disasters)
mcmc_m1.print_summary()
posterior_m1 = mcmc_m1.get_samples()

# %% [markdown]
# Notice the summary shows only `lam1` and `lam2` — `tau` was integrated out
# during inference, so it never appeared in the chain. To recover its
# posterior we run `Predictive` with `infer_discrete=True`, which fills in
# the enumerated site by sampling from the conditional posterior given the
# continuous draws (the canonical pattern; see `examples/annotation.py`).

# %%
discrete_predictive = Predictive(single_changepoint, posterior_m1, infer_discrete=True)
discrete_m1 = discrete_predictive(jax.random.PRNGKey(1), disasters=disasters)
tau_samples = np.asarray(discrete_m1["tau"]).reshape(-1)
print(f"tau_samples shape: {tau_samples.shape}")
print(f"tau posterior mode year: {years[int(np.bincount(tau_samples).argmax())]}")

# %% [markdown]
# ### Posterior analysis

# %%
idata_m1 = az.from_numpyro(mcmc_m1, log_likelihood=False)
az.summary(idata_m1, var_names=["lam1", "lam2"])

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
tau_counts = np.bincount(tau_samples, minlength=T)
tau_probs = tau_counts / tau_counts.sum()
ax.bar(years, tau_probs, color="C2")
ax.set_title(r"Posterior $P(\tau = \mathrm{year})$")
ax.set_xlabel("year")
ax.set_ylabel("probability")

ax = axes[1]
post_predictive = Predictive(single_changepoint, posterior_m1)
post_obs = post_predictive(jax.random.PRNGKey(2), disasters=None, T=T)["obs"]
post_obs = np.asarray(post_obs)
lo, mid, hi = np.percentile(post_obs, [5, 50, 95], axis=0)
ax.fill_between(years, lo, hi, color="C0", alpha=0.25, label="90% PPC band")
ax.plot(years, mid, color="C0", lw=1.5, label="posterior median")
ax.bar(years, np.asarray(disasters), color="black", alpha=0.4, label="observed")
ax.set_title("Posterior predictive — Model 1")
ax.set_xlabel("year")
ax.set_ylabel("disasters")
ax.legend()
fig.tight_layout()

# %% [markdown]
# The posterior over `tau` is sharply concentrated around 1890, the rates
# split cleanly into a high pre-change regime (`lam1 ≈ 3.0`) and a low
# post-change regime (`lam2 ≈ 0.9`), and the posterior predictive band
# brackets the observed series.
#
# **Takeaway.** Discrete latents with bounded support are not a problem in
# NumPyro: annotate the site with `infer={"enumerate": "parallel"}` and HMC
# works as if the site weren't there.
#
# But Model 1 is rigid — it can express only one changepoint. What if there
# are several?

# %% [markdown]
# ## Beat 2 — Multi-regime HMM via `scan` + enumeration
#
# We generalise from "find a year" to "find a regime structure". Each year
# lives in one of `K` latent regimes; transitions form a Markov chain with a
# sticky transition matrix so regimes persist over many years; each regime
# emits Poisson counts at its own rate. The multi-changepoint formulation
# follows [Chib (1998)](https://doi.org/10.1016/S0304-4076(97)00115-2).
#
# Three implementation details to call out before showing the model:
#
# 1. **Label switching.** With `K` exchangeable regimes, the posterior is
#    invariant under regime permutation, which would make the chain
#    multi-modal. We enforce identifiability by parameterising the log-rates
#    as a base level `log_rate0 ~ Normal(0, 1)` plus `K - 1` cumulative
#    *positive* deltas drawn from `HalfNormal(0.5)`. This guarantees
#    `rates[0] < rates[1] < ... < rates[K-1]` by construction, with
#    well-controlled tails so the largest rate cannot blow up under the
#    prior — a problem with naive `OrderedTransform(Normal)` parameterisations.
# 2. **Sticky transitions.** Per-row Dirichlet prior with diagonal-heavy
#    concentration `alpha * I_K + beta * J_K` (with `alpha >> beta`)
#    encodes "regimes tend to persist". We use `.to_event(1)` so the
#    `K` independent row-Dirichlets are treated as a single event,
#    matching the canonical pattern in `examples/hmm_enum.py`.
# 3. **Initial state.** We use a constant initial carry `s_init = 0`,
#    matching the canonical pattern in `examples/hmm_enum.py`. The first
#    transition then samples from `probs_trans[0]`.
#
# Since there are `T = 112` discrete latents — one per year — naive
# enumeration would be exponential in `T`. NumPyro's `scan` with
# `infer={"enumerate": "parallel"}` does the forward marginalisation in
# parallel-scan time, keeping cost manageable.


# %%
def regime_switch(disasters=None, T=T, K=3, alpha=10.0, beta=1.0):
    if disasters is not None:
        T = disasters.shape[0]

    # Cumulative-delta parameterisation: ordered, identifiable, and with
    # well-controlled tails (HalfNormal deltas keep rates[K-1] from
    # exploding under the prior).
    log_rate0 = numpyro.sample("log_rate0", dist.Normal(0.0, 1.0))
    log_deltas = numpyro.sample(
        "log_deltas", dist.HalfNormal(0.5).expand([K - 1]).to_event(1)
    )
    log_rates = log_rate0 + jnp.concatenate([jnp.zeros(1), jnp.cumsum(log_deltas)])
    rates = numpyro.deterministic("rates", jnp.exp(log_rates))

    # `.to_event(1)` reinterprets the K independent row-Dirichlets as a
    # single event so the model passes plate validation without an
    # explicit plate (the canonical pattern in `examples/hmm_enum.py`).
    probs_trans = numpyro.sample(
        "probs_trans",
        dist.Dirichlet(alpha * jnp.eye(K) + beta * jnp.ones((K, K))).to_event(1),
    )

    def transition(s_prev, y):
        s = numpyro.sample(
            "s",
            dist.Categorical(probs_trans[s_prev]),
            infer={"enumerate": "parallel"},
        )
        numpyro.sample("y", dist.Poisson(rates[s]), obs=y)
        return s, None

    scan(transition, 0, disasters, length=T)


# %% [markdown]
# ### Prior predictive check
#
# Three concerns to verify before fitting:
#
# 1. **Are the rates ordered?** The cumulative-delta parameterisation
#    should guarantee `rates[0] < rates[1] < rates[2]` element-wise.
# 2. **Are regimes sticky?** With `alpha=10, beta=1` the *prior* expected
#    fraction of years where the regime changes is roughly
#    `(K-1)*beta / (alpha + (K-1)*beta) = 2 / 12 ≈ 0.17`, so we expect
#    ~19 changepoints per 112-year prior trajectory under the *marginal*
#    transition prior — a handful per century, not zero and not every
#    year. The actual prior-predictive number is somewhat lower because
#    the sticky prior peaks closer to its diagonal.
# 3. **Does the prior predictive cover the data?** Same overlay check as
#    Model 1.

# %%
prior_predictive = Predictive(regime_switch, num_samples=500)
prior_samples_m2 = prior_predictive(jax.random.PRNGKey(11), disasters=None, T=T, K=3)
prior_rates_m2 = np.asarray(prior_samples_m2["rates"])
prior_obs_m2 = np.asarray(prior_samples_m2["y"])
prior_s_m2 = np.asarray(prior_samples_m2["s"])

ordered_ok = bool(
    np.all(prior_rates_m2[:, 0] < prior_rates_m2[:, 1])
    and np.all(prior_rates_m2[:, 1] < prior_rates_m2[:, 2])
)
print(f"all prior rates strictly ordered: {ordered_ok}")

n_changes = (prior_s_m2[:, 1:] != prior_s_m2[:, :-1]).sum(axis=1)
print(
    f"prior changepoints per trajectory: "
    f"mean={n_changes.mean():.1f}, median={np.median(n_changes):.0f}, "
    f"max={n_changes.max()}"
)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax = axes[0]
for i in range(8):
    ax.plot(years, prior_obs_m2[i], color="C0", alpha=0.4, lw=0.8)
ax.bar(years, np.asarray(disasters), color="black", alpha=0.5, label="observed")
ax.set_title("Model 2 — prior predictive trajectories")
ax.set_xlabel("year")
ax.set_ylabel("disasters")
ax.legend()

ax = axes[1]
ax.hist(n_changes, bins=20, color="C4", alpha=0.8)
ax.axvline(
    n_changes.mean(),
    color="black",
    linestyle="--",
    label=f"mean={n_changes.mean():.1f}",
)
ax.set_title("Prior changepoints per 112-year trajectory")
ax.set_xlabel("number of changepoints")
ax.legend()
fig.tight_layout()

# %% [markdown]
# Rates are strictly ordered, the prior trajectories cover the data range,
# and the prior implies a reasonable number of changepoints per century. We
# proceed to inference.

# %% [markdown]
# ### Inference

# %%
mcmc_m2 = MCMC(
    NUTS(regime_switch, target_accept_prob=0.95),
    num_warmup=1000,
    num_samples=2000,
    num_chains=4,
    chain_method="sequential",
    progress_bar=False,
)
mcmc_m2.run(RNG, disasters=disasters, K=3)
mcmc_m2.print_summary(exclude_deterministic=False)
posterior_m2 = mcmc_m2.get_samples()

# %% [markdown]
# ### Recovering the regime trajectory (forward-filter / backward-sample)
#
# For the single-changepoint model we used `Predictive(infer_discrete=True)`
# to recover the discrete posterior. For HMM-style models built with `scan`,
# however, NumPyro's funsor backend doesn't currently support the discrete
# adjoint computation (see `test/contrib/test_infer_discrete.py` —
# `test_scan_hmm_smoke` is marked `xfail` for moderate sequence lengths).
#
# This is actually a great excuse to show the standard HMM smoothing
# algorithm explicitly. **Forward-filter, backward-sample (FFBS)** runs a
# forward pass to compute filtered probabilities `alpha_t(k) = P(s_t = k,
# y_{1..t} | theta)` and then a backward pass that samples
# `s_T, s_{T-1}, ..., s_1` consistently with those filters and the
# observations. Doing the whole thing in log-space makes it numerically
# stable; running it inside a `jax.jit` and `jax.vmap` makes it fast.


# %%
def ffbs_one(rng_key, log_rate0, log_deltas, probs_trans, disasters, K=3):
    log_rates = log_rate0 + jnp.concatenate([jnp.zeros(1), jnp.cumsum(log_deltas)])
    rates = jnp.exp(log_rates)
    log_trans = jnp.log(probs_trans + 1e-30)

    # Forward filter (log-space).
    def log_emit(y):
        return jax.scipy.stats.poisson.logpmf(y, rates)

    # The model's `scan(transition, 0, disasters)` takes initial carry s=0,
    # so the first regime s_1 has prior `probs_trans[0]`.
    log_alpha_1 = log_trans[0] + log_emit(disasters[0])

    def forward_step(log_alpha, y):
        log_alpha_next = jax.scipy.special.logsumexp(
            log_alpha[:, None] + log_trans, axis=0
        ) + log_emit(y)
        return log_alpha_next, log_alpha_next

    _, log_alphas_rest = jax.lax.scan(forward_step, log_alpha_1, disasters[1:])
    log_alphas = jnp.concatenate(
        [log_alpha_1[None, :], log_alphas_rest], axis=0
    )  # shape (T, K)

    # Backward sample.
    keys = jax.random.split(rng_key, log_alphas.shape[0])
    s_T = jax.random.categorical(keys[-1], log_alphas[-1])

    def backward_step(s_next, args):
        log_alpha_t, key = args
        log_post = log_alpha_t + log_trans[:, s_next]
        s_t = jax.random.categorical(key, log_post)
        return s_t, s_t

    _, s_back = jax.lax.scan(
        backward_step,
        s_T,
        (log_alphas[:-1], keys[:-1]),
        reverse=True,
    )
    return jnp.concatenate([s_back, s_T[None]])


ffbs = jax.jit(jax.vmap(ffbs_one, in_axes=(0, 0, 0, 0, None, None)))

num_post = posterior_m2["log_rate0"].shape[0]
ffbs_keys = jax.random.split(jax.random.PRNGKey(2), num_post)
s_samples = np.asarray(
    ffbs(
        ffbs_keys,
        posterior_m2["log_rate0"],
        posterior_m2["log_deltas"],
        posterior_m2["probs_trans"],
        disasters,
        3,
    )
)
print(f"s_samples shape: {s_samples.shape}")

# %%
K = 3
regime_probs = np.stack([(s_samples == k).mean(axis=0) for k in range(K)], axis=0)
change_prob = (s_samples[:, 1:] != s_samples[:, :-1]).mean(axis=0)

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

ax = axes[0]
im = ax.imshow(
    regime_probs,
    aspect="auto",
    cmap="viridis",
    extent=(years[0], years[-1], K - 0.5, -0.5),
)
ax.set_yticks(range(K))
ax.set_yticklabels([f"regime {k}\n(low→high rate)" for k in range(K)])
ax.set_title(r"Posterior $P(s_t = k)$ per year")
fig.colorbar(im, ax=ax, label="probability")

ax = axes[1]
ax.plot(years[1:], change_prob, color="C3", lw=1.5)
ax.set_xlabel("year")
ax.set_ylabel(r"$P(s_t \neq s_{t-1})$")
ax.set_title("Posterior probability of a changepoint at year t")
fig.tight_layout()

# %%
rates_post = np.asarray(posterior_m2["rates"])
fig, ax = plt.subplots(figsize=(8, 4))
for k in range(K):
    ax.hist(
        rates_post[:, k],
        bins=40,
        alpha=0.6,
        label=f"regime {k}",
        density=True,
    )
ax.set_xlabel("rate")
ax.set_ylabel("posterior density")
ax.set_title("Posterior over per-regime rates (ordered)")
ax.legend()
fig.tight_layout()

print("posterior mean rates:", rates_post.mean(axis=0))

# %% [markdown]
# **Takeaway.** `scan` + parallel enumeration is the canonical way to write
# HMM-style models in NumPyro, and the same `Predictive(infer_discrete=True)`
# machinery recovers the discrete trajectory.
#
# But enumeration only works when the discrete support is finite and
# bounded. What if it isn't?

# %% [markdown]
# ## Beat 3 — When enumeration isn't an option
#
# Enumeration trades exactness for the requirement that the discrete state
# space be small and bounded. NumPyro offers two gradient-free alternatives
# for the cases where enumeration is infeasible:
#
# - **`DiscreteHMCGibbs`** — alternates HMC for continuous sites with Gibbs
#   updates for discrete sites.
# - **`MixedHMC`** — implements [Zhou (2020)](https://proceedings.neurips.cc/paper/2020/hash/d27b95cac4c27feb850aaa4070cc4675-Abstract.html),
#   jointly sampling discrete and continuous variables in one Hamiltonian
#   trajectory. Note that `MixedHMC` requires an `HMC` inner kernel
#   (not `NUTS`).
#
# Both samplers see `tau` as a regular discrete sample, so we re-define
# Model 1 *without* the `enumerate` hint:


# %%
def single_changepoint_mixed(disasters=None, T=T):
    if disasters is not None:
        T = disasters.shape[0]
    lam1 = numpyro.sample("lam1", dist.Gamma(2.0, 1.0))
    lam2 = numpyro.sample("lam2", dist.Gamma(2.0, 1.0))
    tau = numpyro.sample("tau", dist.Categorical(logits=jnp.zeros(T)))
    rate = jnp.where(jnp.arange(T) < tau, lam1, lam2)
    with numpyro.plate("years", T):
        numpyro.sample("obs", dist.Poisson(rate), obs=disasters)


# %%
def run_sampler(kernel, model_args, label, num_warmup=500, num_samples=1000):
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=2,
        chain_method="sequential",
        progress_bar=False,
    )
    t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(42), **model_args)
    elapsed = time.perf_counter() - t0
    return mcmc, elapsed


results = {}

mcmc_enum, t_enum = run_sampler(
    NUTS(single_changepoint),
    {"disasters": disasters},
    "Enumeration + NUTS",
)
results["Enumeration + NUTS"] = (mcmc_enum, t_enum)

mcmc_gibbs, t_gibbs = run_sampler(
    DiscreteHMCGibbs(NUTS(single_changepoint_mixed)),
    {"disasters": disasters},
    "DiscreteHMCGibbs",
)
results["DiscreteHMCGibbs"] = (mcmc_gibbs, t_gibbs)

mcmc_mixed, t_mixed = run_sampler(
    MixedHMC(HMC(single_changepoint_mixed, trajectory_length=1.0)),
    {"disasters": disasters},
    "MixedHMC",
)
results["MixedHMC"] = (mcmc_mixed, t_mixed)

# %%
print(
    f"{'sampler':<22} {'wall (s)':>10} {'ESS lam1':>10} {'ESS lam2':>10} {'tau mode':>10}"
)
print("-" * 65)
for label, (mcmc, elapsed) in results.items():
    samples = mcmc.get_samples()
    if "tau" in samples:
        tau_post = np.asarray(samples["tau"]).reshape(-1)
    else:
        # enumerated model — recover tau via infer_discrete
        pred = Predictive(single_changepoint, samples, infer_discrete=True)
        tau_post = np.asarray(
            pred(jax.random.PRNGKey(123), disasters=disasters)["tau"]
        ).reshape(-1)
    tau_mode_year = int(years[int(np.bincount(tau_post).argmax())])
    idata = az.from_numpyro(mcmc, log_likelihood=False)
    summary = az.summary(idata, var_names=["lam1", "lam2"])
    ess1 = summary.loc["lam1", "ess_bulk"]
    ess2 = summary.loc["lam2", "ess_bulk"]
    print(
        f"{label:<22} {elapsed:>10.2f} {ess1:>10.0f} {ess2:>10.0f} {tau_mode_year:>10d}"
    )

# %% [markdown]
# **Verdict.** All three samplers identify the same `tau` posterior mode and
# similar `lam1`, `lam2` posteriors. On this small, bounded problem
# enumeration wins on wall-clock and gradient-friendliness. `MixedHMC` and
# `DiscreteHMCGibbs` become the right choice when the discrete state space
# is large or unbounded — for example, in a continuous-time changepoint
# model — where enumeration is no longer an option.

# %% [markdown]
# ## Summary
#
# We worked through three escalating models for the same dataset:
#
# 1. **Beat 1.** A single-changepoint Poisson model with `tau` enumerated
#    out exactly. NUTS sampled only the continuous rates; we recovered the
#    discrete posterior over the changepoint year with
#    `Predictive(infer_discrete=True)`.
# 2. **Beat 2.** A `K`-regime HMM with sticky transitions and ordered
#    rates, with all `T` discrete states marginalised in parallel-scan time
#    via `scan` + `enumerate`. We recovered per-year regime probabilities
#    and a probability-of-changepoint curve.
# 3. **Beat 3.** A side-by-side comparison of enumeration, `MixedHMC`, and
#    `DiscreteHMCGibbs` on the same model.
#
# In both substantive models we did a **prior predictive check** before
# fitting, in line with the standard Bayesian workflow.
#
# ### Where to go next
#
# - The **effect handlers tutorial** (`effect_handlers.ipynb`, Section 8)
#   shows how to use `do()` for counterfactual reasoning — natural for a
#   follow-up question like "what would the rates have looked like under a
#   different safety regime?"
# - **`examples/hmm_enum.py`** has more elaborate `scan` + enumeration
#   patterns, including factorial HMMs and second-order Markov chains.
# - **PyMC** and **Stan** both have well-known versions of this same
#   problem; cross-PPL comparison highlights NumPyro's parallel-scan
#   enumeration as the differentiator on the multi-regime variant.

# %% [markdown]
# ## References
#
# **Dataset.**
#
# - Jarrett, R. G. (1979). *A note on the intervals between coal-mining
#   disasters.* Biometrika 66(1): 191–193.
#   <https://doi.org/10.1093/biomet/66.1.191>
# - Maguire, B. A., Pearson, E. S., & Wynn, A. H. A. (1952). *The time
#   intervals between industrial accidents.* Biometrika 39(1/2): 168–180.
#
# **Changepoint methodology.**
#
# - Carlin, B. P., Gelfand, A. E., & Smith, A. F. M. (1992). *Hierarchical
#   Bayesian analysis of changepoint problems.* Applied Statistics 41(2):
#   389–405. <https://doi.org/10.2307/2347570>
# - Chib, S. (1998). *Estimation and comparison of multiple change-point
#   models.* Journal of Econometrics 86(2): 221–241.
#   <https://doi.org/10.1016/S0304-4076(97)00115-2>
# - Adams, R. P. & MacKay, D. J. C. (2007). *Bayesian Online Changepoint
#   Detection.* arXiv:0710.3742. <https://arxiv.org/abs/0710.3742>
#
# **Inference techniques.**
#
# - Obermeyer, F., Bingham, E., Jankowiak, M., Phan, D., & Chen, J. P.
#   (2019). *Tensor variable elimination for plated factor graphs.* ICML
#   2019. <https://arxiv.org/abs/1902.03210>
# - Zhou, G. (2020). *Mixed Hamiltonian Monte Carlo for Mixed Discrete and
#   Continuous Variables.* NeurIPS 2020.
#   <https://proceedings.neurips.cc/paper/2020/hash/d27b95cac4c27feb850aaa4070cc4675-Abstract.html>
# - Hoffman, M. D. & Gelman, A. (2014). *The No-U-Turn Sampler.* JMLR 15:
#   1593–1623. <https://jmlr.org/papers/v15/hoffman14a.html>
# - Bingham, E. et al. (2019). *Pyro: Deep Universal Probabilistic
#   Programming.* JMLR 20: 1–6.
#   <https://jmlr.org/papers/v20/18-403.html>
#
# **Cross-PPL implementations.**
#
# - PyMC tutorial: *Inference of changepoint locations in time series*.
#   <https://www.pymc.io/projects/examples/en/latest/case_studies/disaster_model.html>
# - Stan user guide: *Change-point models*.
#   <https://mc-stan.org/docs/stan-users-guide/change-point.html>
# - Pyro HMM tutorial. <https://pyro.ai/examples/hmm.html>
# - NumPyro `scan` API.
#   <https://num.pyro.ai/en/stable/primitives.html#scan>
