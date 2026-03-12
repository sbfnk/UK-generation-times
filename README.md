# UKGenerationTimes.jl

Julia port of the SARS-CoV-2 generation time inference model from:

> Hart et al. 2022. "Inference of the SARS-CoV-2 generation time using UK household data". *eLife* 11:e70767. DOI: https://doi.org/10.7554/eLife.70767

The [original MATLAB code](https://github.com/will-s-hart/UK-generation-times) uses data-augmentation MCMC to infer generation time distributions from UK household transmission data. This package reimplements that in Julia.

Two models are included:

- **Independent model**: lognormal generation time distribution (3 parameters: mean, sd, transmission rate)
- **Mechanistic model**: staged E->P->I infection process (4 parameters: proportion of latent period that is pre-infectious, mean infectious period, relative pre-symptomatic infectiousness, transmission rate)

See [`UKGenerationTimes/README.md`](UKGenerationTimes/README.md) for installation and usage.
