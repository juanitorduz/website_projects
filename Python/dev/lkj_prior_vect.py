import numpy as np
import polars as pl
import pymc as pm
import pytensor.tensor as pt

df = pl.DataFrame()
x_df = pl.DataFrame()

coords = {"effect": ["intercept", "slope"], "effect_copy": ["intercept", "slope"]}
coords.update({"corr_dim": ["corr_dim_1"]})


def vectorized_correlation_matrices(corr_values, size=2):
    n_matrices = corr_values.shape[0]

    # Reshape for broadcasting
    # Use reshape or expand_dims instead of dimshuffle
    corr_expanded = pt.reshape(corr_values, (n_matrices, 1, 1))

    # Create base: all elements are correlation values
    base = corr_expanded * pt.ones((n_matrices, size, size))

    # Create diagonal mask
    diag_mask = pt.eye(size, dtype="bool")

    # Set diagonal to 1
    return pt.where(diag_mask, 1.0, base)


def vectorized_diagonal_matrices_v4(values):
    k = values.shape[1]  # 2

    # Create identity matrix (2, 2)
    identity_matrix = pt.eye(k)

    # Reshape values for broadcasting: (4, 2) -> (4, 2, 1)
    values_expanded = values[:, :, None]

    # Multiply: (4, 2, 1) * (2, 2) -> (4, 2, 2)
    # This puts values[i, j] at position [i, j, j]
    return values_expanded * identity_matrix


with pm.Model(coords=coords) as cov_model:
    # --- Data Containers ---
    # covariates
    x_data = pm.Data("x_data", x_df, dims=("obs_idx", "covariates"))
    # grade
    grade_idx_data = pm.Data("grade_idx_data", df["grade"].to_numpy(), dims="obs_idx")
    # object categories
    pair_idx_data = pm.Data("pair_idx_data", df["pair_id"].to_numpy(), dims="obs_idx")
    # treatment
    treatment_data = pm.Data(
        "treatment_data", df["treatment"].to_numpy(), dims=("obs_idx")
    )
    # outcome
    post_test_data = pm.Data(
        "post_test_data", df["post_test"].to_numpy(), dims="obs_idx"
    )

    # --- Priors ---

    beta_x = pm.Normal("beta_x", mu=0, sigma=1, dims=("grade", "covariates"))
    sigma_outcome = pm.HalfNormal("sigma_outcome", sigma=1, dims=("grade"))

    mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=1, dims=("grade"))
    mu_theta = pm.Normal("mu_theta", mu=0, sigma=1, dims=("grade"))

    # Group-level standard deviations
    sigma_u = pm.HalfNormal("sigma_u", sigma=np.array([1, 1]), dims=("grade", "effect"))

    # Triangular upper part of the correlation matrix
    omega_triu = pm.LKJCorr("omega_triu", eta=1, n=2, dims=("grade", "corr_dim"))

    # Construct correlation matrix
    omega = pm.Deterministic(
        "omega",
        vectorized_correlation_matrices(omega_triu.eval()),
        dims=("grade", "effect", "effect_copy"),
    )

    # Construct diagonal matrix of standard deviation
    sigma_diagonal = pm.Deterministic(
        "sigma_diagonal",
        vectorized_diagonal_matrices_v4(sigma_u),
        dims=("grade", "effect", "effect_copy"),
    )

    # Compute covariance matrix
    cov = pm.Deterministic(
        "cov",
        pt.einsum("bij,bjk,bkl->bil", sigma_diagonal, omega, sigma_diagonal),
        dims=("grade", "effect", "effect_copy"),
    )

    # Cholesky decomposition of covariance matrix
    cholesky_cov = pm.Deterministic(
        "cholesky_cov",
        pt.slinalg.cholesky(cov),
        dims=("grade", "effect", "effect_copy"),
    )

    # And finally get group-specific coefficients
    u_raw = pm.Normal("u_raw", mu=0, sigma=1, dims=("grade", "effect", "pair_id"))
    u = pm.Deterministic(
        "u",
        pt.einsum("bik,bkj->bji", cholesky_cov, u_raw),
        dims=("grade", "pair_id", "effect"),
    )

    u0 = pm.Deterministic("u0", u[:, :, 0], dims=("grade", "pair_id"))
    sigma_u0 = pm.Deterministic("sigma_u0", sigma_u[:, 0], dims="grade")

    u1 = pm.Deterministic("u1", u[:, :, 1], dims=("grade", "pair_id"))
    sigma_u1 = pm.Deterministic("sigma_u1", sigma_u[:, 1], dims="grade")

    alpha = pm.Deterministic("alpha", mu_alpha + u0.T, dims=("pair_id", "grade"))
    theta = pm.Deterministic("theta", mu_theta + u1.T, dims=("pair_id", "grade"))

    mu_outcome = pm.Deterministic(
        "mu_outcome",
        alpha[pair_idx_data, grade_idx_data]
        + theta[pair_idx_data, grade_idx_data] * treatment_data
        + (beta_x[grade_idx_data] * x_data).sum(axis=-1),
        dims=("obs_idx"),
    )

    # --- Likelihood ---
    pm.Normal(
        "post_test_obs",
        mu=mu_outcome,
        sigma=sigma_outcome[grade_idx_data],
        observed=post_test_data,
        dims="obs_idx",
    )
