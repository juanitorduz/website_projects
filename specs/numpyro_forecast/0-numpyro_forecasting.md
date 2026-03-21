# Forecasting with NumPyro - Open Source Package Design

I want to build a flexible and powerfull forecasting library based on nympyro based on all of my previous work.

## Name

`probcast`

## NumPyro Forecasting Models

- In `Python/.` you can find many Jupyter notebooks with custom forecasting models in numpyro (they have numpyro in the name)
- An entry point for the contenct can be found in `Presentations/probabilistic_forecasting`
- In paricular we should be able to support the new custom model: https://juanitorduz.github.io/availability_tsb/ which highloights the custom models in numpyro
- A core model should be a extension of the unobserved componets model https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.structural.UnobservedComponents.html (we already have loval level but we should suppor all components in a modular way)
- Also, all the models should vectorize through batch dimensions so that the extension from 1 time series to many should work seamesly.

### Advanced Models

- We should be able to provide a simple DeepAR model (does not need to have super general version), maybe another one with attention components. We should use nnx for the neural network parts.

- We also should allow time verying covariates using hilbert space GPS like https://github.com/pyro-ppl/numpyro/tree/master/numpyro/contrib/hsgp 

- We must be able support all the mdoels from the M% competition repository https://github.com/pyro-ppl/Pyro-M5-Starter-Kit

## Open Source Package Vision

- The idea that I want to refine is to create a python package to provide these models in a similar way as the forecasting module in Pyro, see https://github.com/pyro-ppl/pyro/tree/dev/pyro/contrib/forecast. Please read the exmples like https://pyro.ai/examples/forecasting_iii.html, https://pyro.ai/examples/forecast_simple.html and https://pyro.ai/examples/forecasting_dlm.html. 


### Requirements

- This package has to be modular and follww the functional paradigms of JAX.
- Should be build on to of NumPyro
- Should support both MCMC and SVI with custom optimizers and samples.
- Provide metrics like crps for probabilistic forecasting models
- Have helper timeslice cross validation rutines
- We should offer simpler wrappers for most comon models like exponential smoothing, SARIMAX, VAR, croston, and the ones found in the examples I ahve done with the ability to add custom priors and hierarchies. 
- But the strength of the package should be the customizability and just reducing the boilerplate of writting custom models like https://num.pyro.ai/en/stable/tutorials/time_series_forecasting.html


## Documentation

Provide light weight version using Sphinx and readthedocs. See for example https://github.com/pyro-ppl/numpyro/tree/master/docs

## Plotting

- The plotting module should use `matplotlib` directly.
- Do not add `seaborn` as a dependency.
- Any ArviZ integration used by plotting or diagnostics should target `ArviZ > 1.0.0`.
- Plotting helpers should stay thin and focus on forecast visualization, uncertainty bands, CV views, and posterior diagnostics built on top of Matplotlib.

## Dev Stack

- Type hints
- Install and CI with uv
- test with pytest
- Use https://github.com/patrick-kidger/jaxtyping
- Use https://github.com/beartype/beartype
- NumPy style docstrings
- Add contributing guide and code of conduct
- Apache licence 2.0
- pre-commit hooks with ruff and mypy
- it should also be AI friendly so we should add AGENTS.md and SKILLS.md