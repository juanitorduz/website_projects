# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Introduction to Stochastic Variational Inference (SVI) with NumPyro
#
# ## Overview
#
# **Stochastic Variational Inference (SVI)** is a scalable approximate inference method that transforms the problem of posterior inference into an optimization problem. Instead of sampling from the posterior distribution (like MCMC), SVI finds the best approximation to the posterior within a family of simpler distributions.
#
# ### Why Use SVI?
#
# 1. **Scalability**: SVI can handle large datasets through mini-batching and stochastic optimization
# 2. **Speed**: Generally faster than MCMC for large models and datasets
# 3. **Deterministic**: Produces consistent results (unlike MCMC which is stochastic)
# 4. **Memory Efficient**: Doesn't store samples, just optimized parameters
#
# ### Key Concepts
#
# - **Variational Family**: A family of simple distributions (e.g., Normal) parameterized by variational parameters
# - **ELBO (Evidence Lower BOund)**: The objective function we maximize, which lower-bounds the log marginal likelihood
# - **Guide Function**: Defines the variational approximation to the posterior
# - **Amortized Inference**: Using neural networks to parameterize the variational distribution
#
# ### The Mathematical Foundation
#
# SVI maximizes the Evidence Lower BOund (ELBO). To understand this, let's first establish our notation:
#
# **Notation:**
# - $\theta$: Model parameters (neural network weights and biases)
# - $\phi$: Variational parameters (parameters of our approximate posterior)
# - $x$: Observed input data
# - $y$: Observed output data
# - $D = \{(x_i, y_i)\}_{i=1}^N$: Our complete dataset
# - $p(\theta|D)$: True posterior distribution (what we want but can't compute easily)
# - $q_\phi(\theta)$: Variational approximation to the posterior (what we'll optimize)
#
# The ELBO can be written as:
#
# $$\text{ELBO}(\phi) = \mathbb{E}_{q_\phi(\theta)}[\log p(y|x, \theta) + \log p(\theta) - \log q_\phi(\theta)]$$
#
# This decomposes into three intuitive terms:
# - $\mathbb{E}_{q_\phi(\theta)}[\log p(y|x, \theta)]$: Expected log-likelihood (how well we explain the data)
# - $\mathbb{E}_{q_\phi(\theta)}[\log p(\theta)]$: Expected log-prior (staying close to prior beliefs)
# - $-\mathbb{E}_{q_\phi(\theta)}[\log q_\phi(\theta)]$: Entropy of variational distribution (encouraging exploration)
#
# ## Example: Bayesian Neural Network Classification
#
# In this notebook, we'll implement a Bayesian Neural Network (BNN) for binary classification using SVI. We'll:
#
# 1. Generate synthetic data (two moons dataset)
# 2. Define a Bayesian neural network model
# 3. Create a variational guide (approximate posterior)
# 4. Train using SVI optimization
# 5. Evaluate the model and quantify uncertainty

# %% [markdown]
# ## Prepare Notebook

# %%
from itertools import pairwise

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
import seaborn as sns
import tqdm
import xarray as xr
from flax import nnx
from jax import random
from jaxtyping import Array, Float, Int
from numpyro.contrib.module import random_nnx_module
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.svi import SVIRunResult
from sklearn.datasets import make_moons
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

seed = 42
rng_key = random.PRNGKey(seed=seed)

az.style.use("arviz-darkgrid")
plt.rcParams["figure.figsize"] = [10, 6]
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.facecolor"] = "white"

# %% [markdown]
# ## Generate Synthetic Data
#
# We'll use the classic "two moons" dataset, which consists of two interleaving half-moon shapes. This dataset is:
#
# - **Non-linearly separable**: Requires a non-linear classifier
# - **Well-defined decision boundary**: Good for visualizing model performance
# - **Moderate complexity**: Not too easy, not too hard for demonstration
#
# ### Data Splitting Strategy
#
# We'll use a three-way split:
# - **Training set (58.8%)**: For learning model parameters
# - **Validation set (25.2%)**: For early stopping and hyperparameter tuning
# - **Test set (30%)**: For final evaluation (held out during training)

# %%
# Generate synthetic two moons dataset
# The moons dataset creates two interleaving half-moon shapes with controllable noise
n_samples = 1200
x, y = make_moons(
    n_samples=n_samples,  # Total number of samples
    noise=0.25,  # Standard deviation of Gaussian noise added to data
    random_state=seed,  # For reproducible results
)

# First split: separate test set (30% of total)
x_train_all, x_test, y_train_all, y_test = train_test_split(
    x,
    y,
    test_size=0.3,  # 30% for testing
    random_state=seed,  # Reproducible split
    stratify=y,  # Maintain class balance across splits
)

# Second split: create validation set from remaining training data (30% of 70% = 21% of total)
x_train, x_val, y_train, y_val = train_test_split(
    x_train_all,
    y_train_all,
    test_size=0.3,  # 30% of remaining for validation
    random_state=seed,
    stratify=y_train_all,
)

# Calculate sample sizes for each split
n_train = x_train.shape[0]  # ~588 samples (49% of total)
n_val = x_val.shape[0]  # ~252 samples (21% of total)
n_test = x_test.shape[0]  # ~360 samples (30% of total)
n = n_train + n_val + n_test

print("Dataset sizes:")
print(f"  Training: {n_train} samples ({n_train / n:.1%})")
print(f"  Validation: {n_val} samples ({n_val / n:.1%})")
print(f"  Test: {n_test} samples ({n_test / n:.1%})")

# Convert to JAX arrays with explicit type annotations
# JAX arrays are immutable and can be compiled/optimized by JAX
x_train: Float[Array, "n_train 2"] = jnp.array(x_train)
x_val: Float[Array, "n_val 2"] = jnp.array(x_val)
x_test: Float[Array, "n_test 2"] = jnp.array(x_test)
y_train: Int[Array, "n_train"] = jnp.array(y_train)
y_val: Int[Array, "n_val"] = jnp.array(y_val)
y_test: Int[Array, "n_test"] = jnp.array(y_test)

# Create index ranges for each dataset split (useful for plotting and analysis)
idx_train = range(n_train)
idx_val = range(n_train, n_train + n_val)
idx_test = range(n_train + n_val, n_train + n_val + n_test)

# %% [markdown]
# Let's visualize our data to understand the classification challenge we're facing.

# %%
cmap = mpl.colormaps["coolwarm"]
colors = list(cmap(np.linspace(0, 1, 2)))

fig, ax = plt.subplots(
    nrows=1, ncols=2, figsize=(12, 5), sharex=True, sharey=True, layout="constrained"
)

sns.scatterplot(
    x=x_train[:, 0], y=x_train[:, 1], s=50, hue=y_train, palette=colors, ax=ax[0]
)
ax[0].set_title("Raw Data - Training Set", fontsize=18, fontweight="bold")
ax[0].set_xlabel("Feature 1")
ax[0].set_ylabel("Feature 2")

sns.scatterplot(
    x=x_test[:, 0], y=x_test[:, 1], s=50, hue=y_test, palette=colors, ax=ax[1]
)
ax[1].set_title("Raw Data - Test Set", fontsize=18, fontweight="bold")
ax[1].set_xlabel("Feature 1")

# %% [markdown]
# **Observations:**
# - The data consists of two interleaving half-moon shapes
# - A linear classifier would fail completely on this dataset
# - We need a non-linear model to separate the classes
# - The decision boundary should curve around each moon
#
# The idea is to develop a **Bayesian neural network classifier** that can:
# 1. Learn the non-linear decision boundary
# 2. Quantify uncertainty in its predictions
# 3. Provide probabilistic outputs rather than hard classifications

# %% [markdown]
# ## Model Specification
#
# ### Bayesian Neural Networks (BNNs)
#
# Unlike traditional neural networks with fixed weights, **Bayesian Neural Networks** place probability distributions over the weights. This allows us to:
#
# 1. **Quantify uncertainty**: Different weight samples lead to different predictions
# 2. **Avoid overfitting**: The prior acts as regularization
# 3. **Make calibrated predictions**: Output probabilities reflect true confidence
#
# ### Architecture Design
#
# Our BNN architecture consists of:
# - **Input layer**: 2 features (x, y coordinates from the two moons dataset)
# - **Hidden layer 1**: 4 neurons with tanh activation
# - **Hidden layer 2**: 3 neurons with tanh activation
# - **Output layer**: 1 neuron with sigmoid activation (for binary classification probabilities)
#
# **Prior distributions** over all network parameters:
# - **Weights** $W_\ell$: SoftLaplace(0, 1) - encourages sparsity and robust learning
# - **Biases** $b_\ell$: Normal(0, 1) - standard regularization with moderate spread
#
# ### Mathematical Formulation
#
# **Forward pass through the network:**
#
# Let $z_0 = x$ be the input features. For hidden layers $\ell = 1, 2$:
# $$z_\ell = \tanh(W_\ell z_{\ell-1} + b_\ell)$$
#
# where:
# - $W_\ell \in \mathbb{R}^{d_{\ell-1} \times d_\ell}$ is the weight matrix for layer $\ell$
# - $b_\ell \in \mathbb{R}^{d_\ell}$ is the bias vector for layer $\ell$
# - $d_0 = 2$, $d_1 = 4$, $d_2 = 3$, $d_3 = 1$ are the layer dimensions
#
# **Final output (classification probability):**
# $$p(y=1|x, \theta) = \sigma(W_3 z_2 + b_3)$$
#
# where $\sigma(t) = \frac{1}{1 + e^{-t}}$ is the sigmoid function and $\theta = \{W_\ell, b_\ell\}_{\ell=1}^3$ represents all network parameters.
#
# **Prior distributions:**
# $$W_\ell \sim \text{SoftLaplace}(0, 1), \quad b_\ell \sim \mathcal{N}(0, 1) \quad \text{for } \ell = 1, 2, 3$$


# %%
class MLP(nnx.Module):
    """
    Multi-Layer Perceptron implemented with Flax NNX.

    This class defines the architecture of our neural network using Flax NNX,
    which integrates seamlessly with NumPyro for Bayesian inference.

    Parameters
    ----------
    din : int
        Input dimension (number of features)
    dout : int
        Output dimension (1 for binary classification)
    hidden_layers : list of int
        List of hidden layer sizes
    rngs : nnx.Rngs
        Random number generator for parameter initialization
    """

    def __init__(self, din, dout, hidden_layers, *, rngs):
        self.layers = []

        # Create layer dimensions: [input_size, hidden1, hidden2, ..., output_size]
        layer_dims = [din, *hidden_layers, dout]

        # Build layers sequentially using pairwise iteration
        for in_dim, out_dim in pairwise(layer_dims):
            # Each layer is a linear transformation: y = Wx + b
            self.layers.append(nnx.Linear(in_dim, out_dim, rngs=rngs))

    def __call__(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : jax.numpy.ndarray
            Input tensor of shape (batch_size, input_dim)

        Returns
        -------
        jax.numpy.ndarray
            Sigmoid-activated output for binary classification
        """
        # Apply tanh activation to all hidden layers
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))

        # Apply sigmoid to final layer for probability output
        return jax.nn.sigmoid(self.layers[-1](x))


# %%
# Split random key for neural network initialization
rng_key, rng_subkey = random.split(rng_key)

# Define network architecture
# This creates a network with structure: 2 -> 4 -> 3 -> 1
hidden_layers = [4, 3]  # Hidden layer sizes only
dout = 1  # Output layer size

# Initialize the neural network module
nnx_module = MLP(
    din=x_train.shape[1],  # Input dimension (2 features)
    dout=dout,  # Output dimension (1 for binary classification)
    hidden_layers=hidden_layers,
    rngs=nnx.Rngs(rng_subkey),  # Flax NNX random number generator
)

print(
    f"Network architecture: {x_train.shape[1]} -> {' -> '.join(map(str, hidden_layers))}"
)
print(
    f"Total parameters: {sum(p.size for p in jax.tree_leaves(nnx.state(nnx_module)))}"
)

# %% [markdown]
# ### The Model Function
#
# In NumPyro, the **model function** defines our generative story - how we believe the data was generated. Our model assumes:
#
# 1. **Neural network weights and biases** are drawn from prior distributions
# 2. **Network forward pass** transforms inputs to probabilities
# 3. **Observed labels** are drawn from Bernoulli distributions with those probabilities
#
# This creates a hierarchical model:
# ```
# θ ~ Prior()                    # Network parameters
# p = NN(x; θ)                  # Forward pass
# y ~ Bernoulli(p)              # Likelihood
# ```


# %%
def model(
    x: Float[Array, "n_obs features"], y: Int[Array, " n_obs"] | None = None
) -> None:
    """
    NumPyro model function defining the Bayesian Neural Network.

    This function specifies the generative process:
    1. Sample neural network parameters from priors
    2. Compute predictions via forward pass
    3. Sample observations from Bernoulli likelihood

    Parameters
    ----------
    x : Float[Array, "n_obs features"]
        Input features of shape (n_obs, 2)
    y : Int[Array, " n_obs"] or None, optional
        Target labels of shape (n_obs,). None during prediction.
    """
    n_obs: int = x.shape[0]  # Number of observations

    def prior(name, shape):
        """
        Prior distribution factory for network parameters.

        We use different priors for weights vs biases:
        - Weights: SoftLaplace(0, 1) - heavy tails, encourages sparsity
        - Biases: Normal(0, 1) - standard Gaussian prior

        Parameters
        ----------
        name : str
            Parameter name (contains 'bias' for bias parameters)
        shape : tuple
            Parameter shape

        Returns
        -------
        numpyro.distributions.Distribution
            Prior distribution for the parameter
        """
        if "bias" in name:
            return dist.Normal(loc=0, scale=1)
        return dist.SoftLaplace(loc=0, scale=1)

    # Create a NumPyro-wrapped version of our neural network
    # This automatically assigns priors to all parameters
    nn = random_nnx_module(
        "nn",  # Name prefix for all parameters
        nnx_module,  # Our Flax NNX module
        prior=prior,  # Prior distribution factory
    )

    # Forward pass: compute probabilities for each observation
    # squeeze(-1) removes the last dimension to get shape (n_obs,)
    p = numpyro.deterministic("p", nn(x).squeeze(-1))

    # Likelihood: each label is drawn from a Bernoulli distribution
    # numpyro.plate creates conditional independence across observations
    with numpyro.plate("data", n_obs):
        numpyro.sample("y", dist.Bernoulli(probs=p), obs=y)


# Test the model by rendering its structure
print("Model structure visualization:")
numpyro.render_model(
    model=model,
    model_kwargs={"x": x_train},  # Pass training data for shape inference
    render_distributions=True,  # Show distribution details
    render_params=True,  # Show parameter nodes
)

# %% [markdown]
# ### The Guide Function (Variational Approximation)
#
# The **guide function** defines our variational approximation to the posterior. Instead of the complex true posterior $p(\theta|D)$, we use a simpler family of distributions $q_\phi(\theta)$.
#
# #### Mean-Field Variational Approximation
#
# We assume **mean-field independence**: each parameter has its own independent Normal or SoftLaplace distribution:
#
# $$q_\phi(\theta) = \prod_i q_{\phi_i}(\theta_i)$$
#
# Where each $q_{\phi_i}$ is parameterized by:
# - **Location parameter** $\mu_i$ (learnable)
# - **Scale parameter** $\sigma_i$ (learnable, constrained to be positive)
#
# #### The Mean-Field Assumption: Benefits and Limitations
#
# This factorization assumption dramatically simplifies the optimization landscape. Instead of searching over the space of all possible multivariate distributions, we restrict ourselves to products of univariate distributions. This brings several advantages:
#
# **Computational Benefits:**
# - **Tractable KL divergence**: The KL divergence between factorized distributions decomposes as $\text{KL}[q_\phi(\theta) \| p(\theta)] = \sum_i \text{KL}[q_{\phi_i}(\theta_i) \| p(\theta_i)]$
# - **Parallel computation**: Each factor can be optimized independently
# - **Memory efficiency**: Storage scales linearly $O(|\theta|)$ rather than quadratically $O(|\theta|^2)$ in the number of parameters
#
# **Theoretical Limitations:**
# - **Posterior correlations**: The approximation cannot capture correlations between parameters
# - **Multimodality**: Mean-field approximations struggle with multimodal posteriors
# - **Underestimation of uncertainty**: The independence assumption typically leads to overconfident (too narrow) posterior approximations
#
# Despite these limitations, mean-field VI often provides excellent approximations for many practical problems, especially when the true posterior is reasonably close to unimodal and when parameter correlations are not too strong.


# %%
def layer_guide(
    loc_shape: tuple[int, ...],  # Shape of location parameters
    loc_amplitude: float,  # Initial scale for location parameters
    scale_shape: tuple[int, ...],  # Shape of scale parameters
    scale_amplitude: float,  # Initial scale for scale parameters
    loc_name: str,  # Name for location parameters
    scale_name: str,  # Name for scale parameters
    layer_name: str,  # Name of the layer being approximated
    event_shape: int = 1,  # Event dimension for to_event()
    seed: int = 42,  # Random seed for initialization
) -> None:
    """
    Create a variational approximation for a single layer's parameters.

    This function defines the guide (variational approximation) for one layer's
    weights or biases. It creates learnable location and scale parameters for
    either Normal or SoftLaplace distributions.

    Parameters
    ----------
    loc_shape : tuple of int
        Shape of the location (mean) parameters
    loc_amplitude : float
        Initialization scale for location parameters
    scale_shape : tuple of int
        Shape of the scale (std) parameters
    scale_amplitude : float
        Initialization scale for scale parameters
    loc_name : str
        Parameter name for location
    scale_name : str
        Parameter name for scale
    layer_name : str
        Name of the layer (used to choose distribution type)
    event_shape : int, optional
        Dimensionality for to_event() transformation, by default 1
    seed : int, optional
        Random seed for reproducible initialization, by default 42
    """
    # Create local random key for this layer
    rng_key = random.PRNGKey(seed)

    # Initialize location parameters with random values
    rng_key, rng_subkey = random.split(rng_key)
    loc = numpyro.param(
        loc_name, loc_amplitude * random.uniform(rng_subkey, shape=loc_shape)
    )

    # Initialize scale parameters with positive random values
    rng_key, rng_subkey = random.split(rng_key)
    scale = numpyro.param(
        scale_name,
        scale_amplitude * random.uniform(rng_subkey, shape=scale_shape),
        constraint=dist.constraints.positive,  # Ensure scale > 0
    )

    # Choose distribution type based on layer name
    if "bias" in layer_name:
        # Bias parameters use Normal distribution (matching model prior)
        numpyro.sample(
            layer_name,
            dist.Normal(loc=loc, scale=scale).to_event(event_shape),
        )
    else:
        # Weight parameters use SoftLaplace distribution (matching model prior)
        numpyro.sample(
            layer_name,
            dist.SoftLaplace(loc=loc, scale=scale).to_event(event_shape),
        )


def guide(
    x: Float[Array, "n_obs features"], y: Int[Array, " n_obs"] | None = None
) -> None:
    """
    Variational guide function that approximates the posterior distribution.

    This function defines the variational family q_φ(θ) that approximates
    the true posterior p(θ|data). We use mean-field independence with
    separate Normal/SoftLaplace distributions for each parameter.

    Parameters
    ----------
    x : Float[Array, "n_obs features"]
        Input features (same as model, but may not be used in guide)
    y : Int[Array, " n_obs"] or None, optional
        Target labels (same as model, but may not be used in guide)
    """
    output_dim = 1  # Output dimension

    # Create variational approximations for all bias parameters
    # Biases have shape (layer_size,) so event_shape=1
    for i, hl in enumerate([*hidden_layers, output_dim]):
        layer_guide(
            loc_shape=(hl,),  # Bias vector shape
            loc_amplitude=1.0,  # Initialize around [-1, 1]
            scale_shape=(hl,),  # One scale per bias
            scale_amplitude=1.0,  # Initialize scales around [0, 1]
            loc_name=f"nn/layers.{i}.bias_auto_loc",  # NumPyro parameter name
            scale_name=f"nn/layers.{i}.bias_auto_scale",  # NumPyro parameter name
            layer_name=f"nn/layers.{i}.bias",  # Layer parameter name
            event_shape=1,  # Vector parameter
            seed=42 + i,  # Unique seed per layer
        )

    # Create variational approximations for all weight parameters
    # Weights have shape (input_size, output_size) so event_shape=2
    for j, (hl_in, hl_out) in enumerate(
        pairwise([x.shape[1], *hidden_layers, output_dim])
    ):
        layer_guide(
            loc_shape=(hl_in, hl_out),  # Weight matrix shape
            loc_amplitude=1.0,  # Initialize around [-1, 1]
            scale_shape=(hl_in, hl_out),  # One scale per weight
            scale_amplitude=1.0,  # Initialize scales around [0, 1]
            loc_name=f"nn/layers.{j}.kernel_auto_loc",  # NumPyro parameter name
            scale_name=f"nn/layers.{j}.kernel_auto_scale",  # NumPyro parameter name
            layer_name=f"nn/layers.{j}.kernel",  # Layer parameter name
            event_shape=2,  # Matrix parameter
            seed=1 + j,  # Unique seed per layer
        )


print("Guide function created successfully!")
print("Variational parameters:")
print(f"  - Bias layers: {len(hidden_layers) + 1}")
print(f"  - Weight layers: {len(hidden_layers) + 1}")
print(
    f"  - Total variational params: {2 * (len(hidden_layers) + 1) * 2}"
)  # 2 params per layer × 2 types

# %% [markdown]
# ## SVI Training Setup
#
# ### Optimization Strategy
#
# We'll use a sophisticated optimization setup with:
#
# 1. **OneCycle Learning Rate Schedule**:
#    - Starts low, increases to peak, then decreases
#    - Helps escape local minima and achieve better convergence
#    - Based on Leslie Smith's research on cyclical learning rates
#
# 2. **Adam Optimizer**:
#    - Adaptive learning rates for each parameter
#    - Good for training neural networks
#    - Combines momentum with adaptive step sizes
#
# 3. **Reduce on Plateau**:
#    - Automatically reduces learning rate when loss plateaus
#    - Helps with fine-tuning in later stages
#
# 4. **Early Stopping**:
#    - Monitors validation loss to prevent overfitting
#    - Stops training when validation performance degrades
#
# ### The ELBO: Our Optimization Target
#
# SVI maximizes the Evidence Lower BOund (ELBO), which provides a tractable lower bound on the log marginal likelihood. Using our established notation, the ELBO can be written in the standard form:
#
# $$\text{ELBO}(\phi) = \mathbb{E}_{q_\phi(\theta)}[\log p(y|x, \theta)] - \text{KL}[q_\phi(\theta) \| p(\theta)]$$
#
# This formulation clearly shows the two competing objectives:
# - **Reconstruction term** $\mathbb{E}_{q_\phi(\theta)}[\log p(y|x, \theta)]$: Rewards the model for explaining the observed data well
# - **KL regularization** $\text{KL}[q_\phi(\theta) \| p(\theta)]$: Penalizes the approximate posterior for deviating from the prior
#
# #### The Gradient Estimation Challenge
#
# The key computational challenge in SVI is estimating gradients of the ELBO with respect to the variational parameters $\phi$. The reconstruction term involves an expectation over the variational distribution, which we need to differentiate:
#
# $$\nabla_\phi \mathbb{E}_{q_\phi(\theta)}[\log p(y|x, \theta)]$$
#
# We can't simply move the gradient inside the expectation because $q_\phi(\theta)$ depends on $\phi$. This is where the **reparameterization trick** becomes crucial. For distributions like Normal($\mu_\phi, \sigma_\phi^2$), we can write:
#
# $$\theta = \mu_\phi + \sigma_\phi \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
#
# This transforms the stochastic gradient into a deterministic one:
# $$\nabla_\phi \mathbb{E}_{\epsilon}[\log p(y|x, g_\phi(\epsilon))]$$
#
# where $g_\phi(\epsilon) = \mu_\phi + \sigma_\phi \cdot \epsilon$. Now we can use Monte Carlo estimation with low variance by sampling $\epsilon$ and computing gradients through the deterministic function $g_\phi$.

# %%
# Configure the learning rate scheduler
# OneCycle schedule: low -> high -> low with specific timing
scheduler = optax.linear_onecycle_schedule(
    transition_steps=8_000,  # Total number of optimization steps
    peak_value=0.008,  # Maximum learning rate (reached at pct_start)
    pct_start=0.008,  # Percent of training to reach peak (0.8%)
    pct_final=0.8,  # Percent of training for final phase (80%)
    div_factor=3,  # Initial LR = peak_value / div_factor
    final_div_factor=4,  # Final LR = initial_LR / final_div_factor
)

# Chain multiple optimizers for sophisticated training
optimizer = optax.chain(
    # Primary optimizer: Adam with scheduled learning rate
    optax.adam(learning_rate=scheduler),
    # Secondary optimizer: Reduce LR when loss plateaus
    optax.contrib.reduce_on_plateau(
        factor=0.1,  # Multiply LR by 0.1 when plateau detected
        patience=10,  # Wait 10 evaluations before reducing
        accumulation_size=100,  # Window size for detecting plateaus
    ),
)

# Create the SVI object that coordinates model, guide, optimizer, and loss
svi = SVI(
    model=model,  # Our BNN model
    guide=guide,  # Our variational approximation
    optim=optimizer,  # Optimization algorithm
    loss=Trace_ELBO(),  # ELBO loss function
)

# Initialize SVI state with random parameters
rng_key, rng_subkey = random.split(key=rng_key)
svi_state = svi.init(rng_subkey, x=x_train, y=y_train)

print("SVI setup complete!")
print("Optimizer: Adam + OneCycle + ReduceOnPlateau")
print("Loss function: Trace_ELBO")
print(f"Initial learning rate: {0.008 / 3:.4f}")
print(f"Peak learning rate: {0.008:.4f}")

# %% [markdown]
# ### Training Loop Implementation
#
# Our training loop includes several important components:
#
# 1. **JAX Compilation**: Using `jax.jit` for fast execution
# 2. **Validation Monitoring**: Track validation loss for early stopping
# 3. **Early Stopping**: Prevent overfitting by stopping when validation loss increases
# 4. **Progress Tracking**: Monitor training progress and loss evolution
#
# The training loop alternates between:
# - **Forward pass**: Compute ELBO loss on training data
# - **Backward pass**: Update variational parameters via gradients
# - **Validation**: Evaluate performance on held-out validation set

# %%
# Define functions for efficient training loop execution


def body_fn(svi_state, _):
    """
    Single training step: compute gradients and update parameters.

    Parameters
    ----------
    svi_state : numpyro.infer.svi.SVIState
        Current SVI state containing parameters and optimizer state
    _ : Any
        Unused (for scan compatibility)

    Returns
    -------
    tuple
        Updated SVI state and training loss
    """
    svi_state, loss = svi.update(svi_state, x=x_train, y=y_train)
    return svi_state, loss


def get_val_loss(svi_state, x_val, y_val):
    """
    Compute validation loss without updating parameters.

    Parameters
    ----------
    svi_state : numpyro.infer.svi.SVIState
        Current SVI state
    x_val : jax.numpy.ndarray
        Validation features
    y_val : jax.numpy.ndarray
        Validation labels

    Returns
    -------
    jax.numpy.ndarray
        Validation ELBO loss
    """
    _, rng_subkey = random.split(svi_state.rng_key)
    params = svi.get_params(svi_state)  # Extract current parameter values

    # Compute loss without gradients or parameter updates
    return svi.loss.loss(
        rng_subkey,
        params,
        svi.model,  # Model function
        svi.guide,  # Guide function
        x=x_val,
        y=y_val,
    )


# Training configuration
num_steps = 8_000  # Maximum number of training steps
patience = 200  # Early stopping patience (steps)

# Storage for loss tracking
train_losses = []  # Raw training losses
norm_train_losses = []  # Training losses normalized by dataset size
val_losses = []  # Raw validation losses
norm_val_losses = []  # Validation losses normalized by dataset size

print("Starting SVI training...")
print(f"Max steps: {num_steps}")
print(f"Early stopping patience: {patience}")
print(f"Training set size: {n_train}")
print(f"Validation set size: {n_val}")

# Main training loop with progress bar
with tqdm.trange(1, num_steps + 1) as t:
    batch = max(num_steps // 20, 1)  # Batch size for progress updates
    patience_counter = 0  # Counter for early stopping

    for i in t:
        # Perform one training step (JIT compiled for speed)
        svi_state, train_loss = jax.jit(body_fn)(svi_state, None)

        # Normalize loss by dataset size for fair comparison
        norm_train_loss = jax.device_get(train_loss) / x_train.shape[0]
        train_losses.append(jax.device_get(train_loss))
        norm_train_losses.append(norm_train_loss)

        # Compute validation loss (JIT compiled for speed)
        val_loss = jax.jit(get_val_loss)(svi_state, x_val, y_val)
        norm_val_loss = jax.device_get(val_loss) / x_val.shape[0]
        val_losses.append(jax.device_get(val_loss))
        norm_val_losses.append(norm_val_loss)

        # Early stopping logic: stop if validation loss > training loss consistently
        condition = norm_val_loss > norm_train_loss
        patience_counter = patience_counter + 1 if condition else 0

        if patience_counter >= patience:
            print(
                f"\nEarly stopping at step {i} (validation loss exceeding training loss)"
            )
            break

        # Update progress bar with recent average losses
        if i % batch == 0:
            avg_train_loss = sum(train_losses[i - batch :]) / batch
            avg_val_loss = sum(val_losses[i - batch :]) / batch

            t.set_postfix_str(
                f"train: {avg_train_loss:.4f}, val: {avg_val_loss:.4f}",
                refresh=False,
            )

# Convert loss lists to JAX arrays for efficient computation
train_losses = jnp.stack(train_losses)
val_losses = jnp.stack(val_losses)

# Create result object containing final parameters and training history
svi_result = SVIRunResult(
    params=svi.get_params(svi_state),  # Final optimized parameters
    state=svi_state,  # Final SVI state
    losses=train_losses,  # Training loss history
)

print(f"\nTraining completed after {len(train_losses)} steps")
print(f"Final training loss: {train_losses[-1]:.4f}")
print(f"Final validation loss: {val_losses[-1]:.4f}")

# %% [markdown]
# ### Training Progress Visualization
#
# Let's examine how the training progressed by plotting both the raw ELBO loss and the normalized loss (divided by dataset size). This helps us understand:
#
# 1. **Convergence behavior**: How quickly the model learned
# 2. **Overfitting detection**: Whether validation loss diverged from training loss
# 3. **Training efficiency**: Whether early stopping was necessary

# %%
fig, ax = plt.subplots(
    nrows=2, ncols=1, figsize=(10, 8), sharex=True, sharey=False, layout="constrained"
)

ax = ax.flatten()

ax[0].plot(train_losses, c="C0", label="Training", linewidth=2, alpha=0.8)
ax[0].plot(val_losses, c="C1", label="Validation", linewidth=2, alpha=0.8)
ax[0].legend(loc="upper right")
ax[0].set(yscale="log")
ax[0].set_title("ELBO Loss (Raw)", fontsize=16, fontweight="bold")
ax[0].grid(True, alpha=0.3)

ax[1].plot(norm_train_losses, c="C0", label="Training", linewidth=2, alpha=0.8)
ax[1].plot(norm_val_losses, c="C1", label="Validation", linewidth=2, alpha=0.8)
ax[1].legend(loc="upper right")
ax[1].set(yscale="log")
ax[1].set_xlabel("Training Step")
ax[1].set_title("Normalized ELBO Loss (Per Sample)", fontsize=16, fontweight="bold")
ax[1].grid(True, alpha=0.3)

final_train = train_losses[-1]
final_val = val_losses[-1]
ax[0].text(
    0.02,
    0.98,
    f"Final Train: {final_train:.3f}\nFinal Val: {final_val:.3f}",
    transform=ax[0].transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

plt.suptitle("SVI Training Progress", fontsize=18, fontweight="bold")
plt.show()

# Print training summary
print("Training Summary:")
print(f"  Steps completed: {len(train_losses)}")
print(f"  Initial train loss: {train_losses[0]:.4f}")
print(f"  Final train loss: {train_losses[-1]:.4f}")
print(f"  Loss reduction: {(train_losses[0] - train_losses[-1]) / train_losses[0]:.2%}")
print(f"  Validation tracking: {'✓' if val_losses[-1] < 2 * train_losses[-1] else '⚠'}")

# %% [markdown]
# ## Posterior Predictive Analysis
#
# Now that we've trained our variational approximation, let's use it to make predictions and analyze the results.
#
# ### Posterior Predictive Sampling
#
# The **posterior predictive distribution** tells us what new data points would look like according to our trained model:
#
# $$p(y_{\text{new}}|x_{\text{new}}, D) = \int p(y_{\text{new}}|x_{\text{new}}, \theta) p(\theta|D) \, d\theta$$
#
# Since we can't compute the true posterior $p(\theta|D)$ exactly, we approximate it using our trained variational distribution $q_\phi(\theta)$:
#
# $$p(y_{\text{new}}|x_{\text{new}}, D) \approx \int p(y_{\text{new}}|x_{\text{new}}, \theta) q_\phi(\theta) \, d\theta$$
#
# In practice, we implement this via Monte Carlo sampling:
# 1. **Sample parameters** $\theta^{(s)} \sim q_\phi(\theta)$ from our trained variational approximation
# 2. **Forward pass** each parameter sample through the network to get $p(y_{\text{new}}|x_{\text{new}}, \theta^{(s)})$
# 3. **Sample predictions** from the resulting Bernoulli distributions
#
# This Monte Carlo approximation gives us both **point estimates** (mean predictions) and **uncertainty quantification** (variance across samples).

# %%
# Extract the optimized variational parameters
params = svi_result.params

print("Generating posterior predictive samples...")
print(f"Variational parameters optimized: {len(jax.tree_leaves(params))}")

# Create posterior predictive sampler for training data
train_posterior_predictive = Predictive(
    model=model,  # Our BNN model
    guide=guide,  # Our trained variational guide
    params=params,  # Optimized variational parameters
    num_samples=2_000,  # Number of posterior samples to draw
    return_sites=["p", "y"],  # Return both probabilities and predictions
)

# Generate samples for training data
rng_key, rng_subkey = random.split(key=rng_key)
train_posterior_predictive_samples = train_posterior_predictive(rng_subkey, x_train)

print("Training predictions shape:")
print(f"  Probabilities (p): {train_posterior_predictive_samples['p'].shape}")
print(f"  Predictions (y): {train_posterior_predictive_samples['y'].shape}")

# Convert to ArviZ InferenceData for analysis and visualization
train_idata = az.from_dict(
    posterior_predictive={
        # Add chain dimension for ArviZ compatibility
        k: np.expand_dims(a=np.asarray(v), axis=0)
        for k, v in train_posterior_predictive_samples.items()
    },
    coords={"obs_idx": idx_train},  # Coordinate labels for observations
    dims={
        "p": ["obs_idx"],  # Probability predictions
        "y": ["obs_idx"],  # Binary predictions
    },
)

print("Training posterior predictive samples created successfully!")

# %%
# Generate posterior predictive samples for test data
print("Generating test set predictions...")

test_posterior_predictive = Predictive(
    model=model,
    guide=guide,
    params=params,
    num_samples=2_000,
    return_sites=["p", "y"],
)

# Generate samples for test data
rng_key, rng_subkey = random.split(key=rng_key)
test_posterior_predictive_samples = test_posterior_predictive(rng_subkey, x_test)

print("Test predictions shape:")
print(f"  Probabilities (p): {test_posterior_predictive_samples['p'].shape}")
print(f"  Predictions (y): {test_posterior_predictive_samples['y'].shape}")

# Convert to ArviZ InferenceData
test_idata = az.from_dict(
    posterior_predictive={
        k: np.expand_dims(a=np.asarray(v), axis=0)
        for k, v in test_posterior_predictive_samples.items()
    },
    coords={"obs_idx": idx_test},
    dims={
        "p": ["obs_idx"],
        "y": ["obs_idx"],
    },
)

print("Test posterior predictive samples created successfully!")
print(f"Total samples generated: {2_000 * (n_train + n_test):,}")

# %% [markdown]
# ### Model Performance Evaluation
#
# We'll evaluate our Bayesian Neural Network using the **Area Under the ROC Curve (AUC)** metric. The beauty of the Bayesian approach is that we get a **distribution** of AUC scores rather than a single point estimate.
#
# #### Why AUC?
#
# 1. **Threshold-independent**: Evaluates performance across all classification thresholds
# 2. **Probability-aware**: Uses predicted probabilities, not just hard classifications
# 3. **Balanced metric**: Accounts for both sensitivity and specificity
# 4. **Uncertainty-friendly**: Can be computed for each posterior sample
#
# #### Bayesian Model Evaluation
#
# For each posterior sample $\theta^{(s)}$, we compute:
# $$\text{AUC}^{(s)} = \text{AUC}(y_{true}, p^{(s)})$$
#
# This gives us a **distribution** of performance metrics, allowing us to quantify uncertainty in model performance itself!

# %%
# Compute AUC score for each posterior sample on training data
print("Computing AUC distributions...")

auc_train = xr.apply_ufunc(
    roc_auc_score,  # Function to apply
    y_train,  # True labels (same for all samples)
    train_idata["posterior_predictive"][
        "p"
    ],  # Predicted probabilities (varies by sample)
    input_core_dims=[["obs_idx"], ["obs_idx"]],  # Dimensions to apply function over
    output_core_dims=[[]],  # Output is scalar
    vectorize=True,  # Apply to each sample independently
)

# Compute AUC score for each posterior sample on test data
auc_test = xr.apply_ufunc(
    roc_auc_score,
    y_test,
    test_idata["posterior_predictive"]["p"],
    input_core_dims=[["obs_idx"], ["obs_idx"]],
    output_core_dims=[[]],
    vectorize=True,
)

print("AUC distributions computed:")
print(f"  Training AUC: {auc_train.mean():.3f} ± {auc_train.std():.3f}")
print(f"  Test AUC: {auc_test.mean():.3f} ± {auc_test.std():.3f}")

# Compute point estimates using posterior mean predictions
train_mean_auc = roc_auc_score(
    y_train, train_idata["posterior_predictive"]["p"].mean(dim=("chain", "draw"))
)

test_mean_auc = roc_auc_score(
    y_test, test_idata["posterior_predictive"]["p"].mean(dim=("chain", "draw"))
)

print("Point estimates (using posterior mean):")
print(f"  Training AUC: {train_mean_auc:.3f}")
print(f"  Test AUC: {test_mean_auc:.3f}")

# %% [markdown]
# Let's visualize the **distribution** of AUC scores. This shows us not just how well our model performs on average, but also how uncertain we are about that performance.

# %%
# Create comprehensive AUC visualization
fig, ax = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(10, 10),
    sharex=True,  # Share x-axis for easy comparison
    sharey=True,  # Share y-axis for consistent scale
    layout="constrained",
)

# Plot training AUC distribution
az.plot_posterior(data=auc_train, ax=ax[0])
ax[0].axvline(
    train_mean_auc,
    color="C3",
    linestyle="--",
    linewidth=2,
    label="AUC using posterior mean",
)
ax[0].legend(loc="upper left")
ax[0].set_title("AUC Posterior Distribution (Training)", fontsize=18, fontweight="bold")
ax[0].grid(True, alpha=0.3)

# Plot test AUC distribution
az.plot_posterior(data=auc_test, ax=ax[1])
ax[1].axvline(
    test_mean_auc,
    color="C3",
    linestyle="--",
    linewidth=2,
    label="AUC using posterior mean",
)
ax[1].legend(loc="upper left")
ax[1].set_xlabel("AUC Score")
ax[1].set_title("AUC Posterior Distribution (Test)", fontsize=18, fontweight="bold")
ax[1].grid(True, alpha=0.3)

plt.suptitle("Bayesian Model Performance Evaluation", fontsize=20, fontweight="bold")
plt.show()

# Print detailed performance summary
print("Detailed Performance Analysis:")
print("Training Set:")
print(f"  Mean AUC: {auc_train.mean():.3f}")
print(f"  Std AUC: {auc_train.std():.3f}")
print(f"  95% CI: [{auc_train.quantile(0.025):.3f}, {auc_train.quantile(0.975):.3f}]")
print("Test Set:")
print(f"  Mean AUC: {auc_test.mean():.3f}")
print(f"  Std AUC: {auc_test.std():.3f}")
print(f"  95% CI: [{auc_test.quantile(0.025):.3f}, {auc_test.quantile(0.975):.3f}]")

# Assess generalization
performance_gap = auc_train.mean() - auc_test.mean()
print(f"Generalization Gap: {performance_gap:.3f}")
print(
    f"Overfitting Assessment: {'Minimal' if performance_gap < 0.05 else 'Moderate' if performance_gap < 0.1 else 'Significant'}"
)

# %% [markdown]
# ### ROC Curve Analysis
#
# The **Receiver Operating Characteristic (ROC)** curve shows the trade-off between true positive rate and false positive rate across all classification thresholds.
#
# #### Bayesian ROC Analysis
#
# Since we have multiple posterior samples, we can compute a **distribution** of ROC curves. This shows:
#
# 1. **Average performance**: The central tendency of ROC curves
# 2. **Uncertainty bands**: How much the performance varies across parameter samples
# 3. **Robustness**: Whether performance is consistent across the posterior
#
# Each curve represents the ROC for one set of sampled network parameters.


# %%
def _roc_curve(y_true, y_score):
    """
    Compute ROC curve with truncation for consistent array sizes.

    This helper function ensures all ROC curves have the same length
    for easier visualization and analysis.

    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_score : array-like
        Predicted probabilities

    Returns
    -------
    tuple
        Truncated false positive rates, true positive rates, and thresholds
    """
    fpr, tpr, thresholds = roc_curve(
        y_true=y_true,
        y_score=y_score,
        drop_intermediate=False,  # Keep all points for smoother curves
    )

    # Truncate to consistent length (avoids xarray size mismatch issues)
    n = y_true.shape[0] - 3
    return fpr[:n], tpr[:n], thresholds[:n]


print("Computing ROC curves for all posterior samples...")

# Compute ROC curves for training data across all posterior samples
fpr_train, tpr_train, thresholds_train = xr.apply_ufunc(
    lambda x, y: _roc_curve(y_true=x, y_score=y),
    y_train,  # True labels
    train_idata["posterior_predictive"]["p"],  # Predicted probabilities
    input_core_dims=[["obs_idx"], ["obs_idx"]],  # Input dimensions
    output_core_dims=[["threshld"], ["threshld"], ["threshld"]],  # Output dimensions
    vectorize=True,  # Apply to each sample
)

# Compute ROC curves for test data across all posterior samples
fpr_test, tpr_test, thresholds_test = xr.apply_ufunc(
    lambda x, y: _roc_curve(y_true=x, y_score=y),
    y_test,
    test_idata["posterior_predictive"]["p"],
    input_core_dims=[["obs_idx"], ["obs_idx"]],
    output_core_dims=[["threshld"], ["threshld"], ["threshld"]],
    vectorize=True,
)

print("ROC curves computed:")
print(f"  Training curves: {fpr_train.shape}")
print(f"  Test curves: {fpr_test.shape}")
print(f"  Total curves: {fpr_train.shape[1] + fpr_test.shape[1]}")

# %% [markdown]
# Now let's visualize the **ensemble** of ROC curves. Each light-colored line represents one posterior sample, while the overall pattern shows the model's consistency.

# %%
fig, ax = plt.subplots(
    nrows=1, ncols=2, figsize=(14, 6), sharex=True, sharey=True, layout="constrained"
)

sample_indices = range(0, 2_000, 50)
for i in sample_indices:
    ax[0].plot(
        fpr_train.sel(chain=0, draw=i),
        tpr_train.sel(chain=0, draw=i),
        c="C0",
        alpha=0.15,
        linewidth=1,
    )

for i in sample_indices:
    ax[1].plot(
        fpr_test.sel(chain=0, draw=i),
        tpr_test.sel(chain=0, draw=i),
        c="C1",
        alpha=0.15,
        linewidth=1,
    )

for i in range(2):
    ax[i].axline(
        (0, 0),
        (1, 1),
        color="black",
        linestyle="--",
        alpha=0.8,
        linewidth=2,
        label="Random Classifier",
    )

ax[0].set_xlabel("False Positive Rate")
ax[0].set_ylabel("True Positive Rate")
ax[0].set_title("Training Set ROC Curves", fontsize=16, fontweight="bold")
ax[0].legend()
ax[0].grid(True, alpha=0.3)

ax[1].set_xlabel("False Positive Rate")
ax[1].set_title("Test Set ROC Curves", fontsize=16, fontweight="bold")
ax[1].legend()
ax[1].grid(True, alpha=0.3)

plt.suptitle("Posterior Ensemble of ROC Curves", fontsize=18, fontweight="bold")
plt.show()

# Analyze ROC curve consistency
print("ROC Curve Analysis:")
print(f"  Curves plotted: {len(sample_indices)} per dataset")
print(f"  Training AUC range: [{auc_train.min():.3f}, {auc_train.max():.3f}]")
print(f"  Test AUC range: [{auc_test.min():.3f}, {auc_test.max():.3f}]")
print(
    f"  Consistency: {'High' if auc_test.std() < 0.01 else 'Moderate' if auc_test.std() < 0.02 else 'Low'}"
)

# %% [markdown]
# ### Prediction Visualization
#
# Finally, let's visualize our model's predictions in the original feature space. This will show us:
#
# 1. **Decision boundary**: How the model separates the two classes
# 2. **Prediction confidence**: Areas where the model is more/less certain
# 3. **Uncertainty patterns**: Where Bayesian uncertainty is highest
#
# We'll plot the **posterior mean predictions** - the average probability across all posterior samples. The color intensity represents the predicted probability of belonging to class 1.

# %%
fig, ax = plt.subplots(
    nrows=1, ncols=2, figsize=(14, 6), sharex=True, sharey=True, layout="constrained"
)

(
    train_idata["posterior_predictive"]["p"]
    .mean(dim=("chain", "draw"))
    .to_pandas()
    .to_frame()
    .assign(x1=x_train[:, 0], x2=x_train[:, 1])
    .pipe(
        (sns.scatterplot, "data"),
        x="x1",
        y="x2",
        hue="p",
        hue_norm=(0, 1),
        palette="coolwarm",
        s=50,
        ax=ax[0],
    )
)

ax[0].set_xlabel("Feature 1")
ax[0].set_ylabel("Feature 2")
ax[0].set_title("Training Set Predictions", fontsize=16, fontweight="bold")

(
    test_idata["posterior_predictive"]["p"]
    .mean(dim=("chain", "draw"))
    .to_pandas()
    .to_frame()
    .assign(x1=x_test[:, 0], x2=x_test[:, 1])
    .pipe(
        (sns.scatterplot, "data"),
        x="x1",
        y="x2",
        hue="p",
        hue_norm=(0, 1),
        palette="coolwarm",
        s=50,
        ax=ax[1],
    )
)

ax[1].set_xlabel("Feature 1")
ax[1].set_title("Test Set Predictions", fontsize=16, fontweight="bold")

for i in range(2):
    ax[i].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=4,
        title="Predicted Probability",
    )
    ax[i].grid(True, alpha=0.3)

plt.suptitle("Bayesian Neural Network Predictions", fontsize=18, fontweight="bold")
plt.show()

# Analyze prediction patterns
train_probs = train_idata["posterior_predictive"]["p"].mean(dim=("chain", "draw"))
test_probs = test_idata["posterior_predictive"]["p"].mean(dim=("chain", "draw"))

print("Prediction Analysis:")
print("Training Set:")
print(f"  Mean probability: {train_probs.mean():.3f}")
print(f"  Probability range: [{train_probs.min():.3f}, {train_probs.max():.3f}]")
print(
    f"  Decision boundary quality: {'Sharp' if (train_probs.std() > 0.3) else 'Soft'}"
)

print("Test Set:")
print(f"  Mean probability: {test_probs.mean():.3f}")
print(f"  Probability range: [{test_probs.min():.3f}, {test_probs.max():.3f}]")
print(
    f"  Consistency with training: {'Good' if abs(train_probs.mean() - test_probs.mean()) < 0.05 else 'Poor'}"
)

# %% [markdown]
# ## Summary and Key Takeaways
#
# ### What We've Accomplished
#
# 1. **Implemented a Bayesian Neural Network** using NumPyro's SVI framework
# 2. **Learned complex non-linear decision boundaries** for the two moons dataset
# 3. **Quantified uncertainty** in both parameters and predictions
# 4. **Evaluated performance** using probabilistic metrics (AUC distributions)
# 5. **Visualized results** including ROC curves and prediction landscapes
#
# ### Key Advantages of SVI
#
# ✅ **Scalability**: Can handle large datasets through mini-batching
# ✅ **Speed**: Faster than MCMC for most applications
# ✅ **Uncertainty Quantification**: Provides meaningful uncertainty estimates
# ✅ **Deterministic**: Reproducible results for deployment
# ✅ **Flexible**: Works with complex models (neural networks, etc.)
#
# ### When to Use SVI vs MCMC
#
# **Use SVI when:**
# - You have large datasets (>10k samples)
# - You need fast inference for production systems
# - Your model has many parameters (deep neural networks)
# - You can accept approximate (vs exact) posterior inference
#
# **Use MCMC when:**
# - You have small-medium datasets (<10k samples)
# - You need exact posterior samples
# - Your model is relatively simple
# - You have time for longer computation
#
# ### Further Exploration
#
# To extend this example, consider:
#
# 1. **Different architectures**: Try deeper networks or different activation functions
# 2. **Alternative guides**: Experiment with normalizing flows for more flexible posteriors
# 3. **Mini-batching**: Scale to larger datasets using subsample plates
# 4. **Hierarchical models**: Add group-level parameters for multi-level data
# 5. **Model comparison**: Use marginal likelihood estimation to compare models
#
# ### Resources for Learning More
#
# - [NumPyro Documentation](https://num.pyro.ai/)
# - [Pyro SVI Tutorial](https://pyro.ai/examples/svi_part_i.html)
# - [Blei et al. (2017): Variational Inference Review](https://arxiv.org/abs/1601.00670)
# - [Hoffman et al. (2013): Stochastic Variational Inference](https://arxiv.org/abs/1206.7051)
