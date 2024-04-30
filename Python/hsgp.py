"""
Code taken from the NumPyro documentation:
Example: Hilbert space approximation for Gaussian processes
https://num.pyro.ai/en/stable/examples/hsgp.html

These functions are used to approximate Gaussian processes using
a low rank approximation in a Hilbert space.
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from tensorflow_probability.substrates import jax as tfp


# --- modelling utility functions --- #
def spectral_density(w, alpha, length):
    c = alpha * jnp.sqrt(2 * jnp.pi) * length
    e = jnp.exp(-0.5 * (length**2) * (w**2))
    return c * e


def diag_spectral_density(alpha, length, ell, m):
    sqrt_eigenvalues = jnp.arange(1, 1 + m) * jnp.pi / 2 / ell
    return spectral_density(sqrt_eigenvalues, alpha, length)


def eigenfunctions(x, ell, m):
    """
    The first `m` eigenfunctions of the laplacian operator in `[-ell, ell]`
    evaluated at `x`. These are used for the approximation of the
    squared exponential kernel.
    """
    m1 = (jnp.pi / (2 * ell)) * jnp.tile(ell + x[:, None], m)
    m2 = jnp.diag(jnp.linspace(1, m, num=m))
    num = jnp.sin(m1 @ m2)
    den = jnp.sqrt(ell)
    return num / den


def modified_bessel_first_kind(v, z):
    v = jnp.asarray(v, dtype=float)
    return jnp.exp(jnp.abs(z)) * tfp.math.bessel_ive(v, z)


def diag_spectral_density_periodic(alpha, length, m):
    """
    Not actually a spectral density but these are used in the same
    way. These are simply the first `m` coefficients of the low rank
    approximation for the periodic kernel.
    """
    a = length ** (-2)
    j = jnp.arange(0, m)
    c = jnp.where(j > 0, 2, 1)
    return (c * alpha**2 / jnp.exp(a)) * modified_bessel_first_kind(j, a)


def eigenfunctions_periodic(x, w0, m):
    """
    Basis functions for the approximation of the periodic kernel.
    """
    m1 = jnp.tile(w0 * x[:, None], m)
    m2 = jnp.diag(jnp.arange(m, dtype=jnp.float32))
    mw0x = m1 @ m2
    cosines = jnp.cos(mw0x)
    sines = jnp.sin(mw0x)
    return cosines, sines


# --- Approximate Gaussian processes --- #
def approx_se_ncp(x, alpha, length, ell, m):
    """
    Hilbert space approximation for the squared
    exponential kernel in the non-centered parametrisation.
    """
    phi = eigenfunctions(x, ell, m)
    spd = jnp.sqrt(diag_spectral_density(alpha, length, ell, m))
    with numpyro.plate("basis", m):
        beta = numpyro.sample("beta", dist.Normal(0, 1))

    return numpyro.deterministic("f", phi @ (spd * beta))


def approx_periodic_gp_ncp(x, alpha, length, w0, m):
    """
    low rank approximation for the periodic squared
    exponential kernel in the non-centered parametrisation.
    """
    q2 = diag_spectral_density_periodic(alpha, length, m)
    cosines, sines = eigenfunctions_periodic(x, w0, m)

    with numpyro.plate("cos_basis", m):
        beta_cos = numpyro.sample("beta_cos", dist.Normal(0, 1))

    with numpyro.plate("sin_basis", m - 1):
        beta_sin = numpyro.sample("beta_sin", dist.Normal(0, 1))

    # The first eigenfunction for the sine component
    # is zero, so the first parameter wouldn't contribute to the approximation.
    # We set it to zero to identify the model and avoid divergences.
    zero = jnp.array([0.0])
    beta_sin = jnp.concatenate((zero, beta_sin))

    return numpyro.deterministic(
        "f", cosines @ (q2 * beta_cos) + sines @ (q2 * beta_sin)
    )
