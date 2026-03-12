# UKGenerationTimes.jl

Julia port of the SARS-CoV-2 generation time inference model from [Hart et al. (2022)](https://doi.org/10.7554/eLife.70767). The [original MATLAB code](https://github.com/will-s-hart/UK-generation-times) uses data-augmentation MCMC to infer generation time distributions from UK household transmission data. This port swaps the Metropolis-Hastings parameter update for NUTS ([AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl)), which needs far fewer samples to converge because it uses gradient information to explore the posterior.

## The model in brief

The data are household transmission chains: who was infected, when symptoms appeared (within a time window), and household membership. The MCMC jointly infers the epidemiological parameters (generation time shape, transmission rate, etc.) and the latent variables (exact infection times, onset times, who infected whom).

Two models are fitted:

- **Independent model** -- the generation time is lognormal, independent of the incubation period. Three parameters: mean generation time, SD, and baseline transmission rate.
- **Mechanistic model** -- a staged E->P->I infection where individuals progress through Exposed (latent), Presymptomatic (infectious before symptoms), and symptomatic Infectious stages. Four parameters: fraction of incubation spent in E, mean infectious duration, relative presymptomatic infectiousness, and baseline transmission rate.

## Reproducing results

### Setup

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

The data file `Supplementary_Data.xlsx` (from the [eLife paper](https://doi.org/10.7554/eLife.70767)) should be in `data/`.

### Step 1: Fit models

```bash
julia --project=. scripts/fit_indep.jl    # ~hours, 1M MCMC steps
julia --project=. scripts/fit_mech.jl     # ~hours, 1M MCMC steps
```

Each script loads the data, configures the sampler, runs the chain, and saves output (parameter samples, log-likelihoods, empirical summaries) to `results/param_fit_*.jld2`.

### Step 2: Posterior summaries

```bash
julia --project=. scripts/posterior_indep.jl
julia --project=. scripts/posterior_mech.jl
```

These compute posterior medians, credible intervals, and the presymptomatic transmission probability from the chains. Output goes to `results/posterior_*.jld2`.

### Step 3: Distribution curves

```bash
julia --project=. scripts/gen_tost_serial.jl
```

Computes the generation time, TOST (time from onset to transmission), and serial interval distributions at posterior mean parameter values, using numerical quadrature. Saves to `results/gen_tost_serial_*.jld2`.

### Step 4: Figures

```bash
julia --project=. scripts/figures.jl
```

Produces posterior densities, distribution curves, model comparisons, and trace plots as PDFs in `figures/`.

## How the code fits together

The package has three layers. Each depends only on the ones above it.

### Data layer

| File | What it does |
|------|-------------|
| `src/types.jl` | Data structures. `ObservedData` holds the fixed observations (onset time windows, household membership, possible-infector matrices). `AugmentedData` wraps that plus the latent variables (infection times, onset times) that the MCMC updates. `PossibleInfectors` stores the sparse matrices that map between individual-level and pair-level arrays. |
| `src/data.jl` | Reads `Supplementary_Data.xlsx`, filters to the 28-day follow-up period, removes inconclusive cases, and builds the `ObservedData` struct with all the sparse indicator matrices. |
| `src/parameters.jl` | Fixed quantities not estimated by the MCMC: incubation period distribution (lognormal, from McAloon et al.), relative asymptomatic infectiousness `x_A = 0.35`, and household-size exponent `rho = 1`. |

### Model layer

| File | What it does |
|------|-------------|
| `src/likelihood_indep.jl` | Household log-likelihood for the independent model. Each household's contribution has three parts: (1) how well the latent onset/infection times match the incubation distribution, (2) generation time density at the inferred transmission times, weighted by transmission rate, (3) evasion -- the probability each individual avoided earlier infection. Uses sparse matrix multiplication to sum over possible infector-infectee pairs. |
| `src/likelihood_mech.jl` | Same structure for the mechanistic model, but replaces the generation time PDF/CDF with the conditional infectiousness profile from the E->P->I staged infection. |
| `src/infectiousness.jl` | The mechanistic infectiousness functions. `b_cond_mech` gives instantaneous infectiousness at a given time relative to symptom onset. `b_int_cond_mech` is its cumulative integral (for the evasion term). `mean_transmissions_mech` is the lifetime total. `f_tost_mech` is the TOST density. `get_gen_mean_sd_mech` computes the analytical mean and SD of the generation time from the stage duration moments. Also defines `MechParams`, a named tuple that replaces the raw parameter vector so you can write `p.α` instead of `params[6]`. |
| `src/priors.jl` | Prior distributions for both models. The independent model has LogNormal priors on all three parameters. The mechanistic model has a Beta prior on `p_E` (fraction of incubation in the latent stage) and LogNormal priors on the rest. |
| `src/generation_time.jl` | Post-hoc computation of population-level distributions (generation time, TOST, serial interval) by numerically convolving the component distributions with QuadGK. Not used during MCMC -- only for the final analysis. |
| `src/summary.jl` | Empirical summaries computed from the augmented data at each saved MCMC step: realised mean/SD of household generation times and the proportion of presymptomatic transmissions. |
| `src/plotting.jl` | CairoMakie figure generation: posterior density plots, distribution curves, trace plots, model comparisons. |

### MCMC layer

The sampler (`src/mcmc/sampler.jl`) runs a 4-step Gibbs cycle, repeating:

1. **Update parameters** (`src/mcmc/update_theta.jl`) -- NUTS via AdvancedHMC.jl. The log-density wraps the household likelihood + prior, transformed to unconstrained space (log for positive parameters, logit for `p_E`). ForwardDiff computes gradients automatically. Step size and mass matrix adapt during the first 20% of parameter updates.

2. **Update infection times** (`src/mcmc/update_infection_indep.jl` or `update_infection_mech.jl`) -- Metropolis-Hastings proposals for each individual's infection time, accepted/rejected per household. After proposing, individuals are re-sorted by infection order within each household.

3. **Update onset times** (`src/mcmc/update_onset.jl`) -- resample symptom onset times uniformly within each individual's observed onset window.

4. **Update asymptomatic status** (`src/mcmc/update_asymp_indep.jl` or `update_asymp_mech.jl`) -- for individuals with uncertain symptom status, propose toggling symptomatic/asymptomatic and adjusting times accordingly.

`src/mcmc/initialise.jl` sets up the initial augmented data (random infection times within bounds, onset times at interval midpoints).

### Reading order

If you want to understand the model, start with:

1. `src/types.jl` -- what data the model works with
2. `src/likelihood_indep.jl` -- the simpler likelihood (same three-component structure as the mechanistic one, but with a standard generation time distribution instead of the staged infection)
3. `src/infectiousness.jl` -- the E->P->I model and what the mechanistic infectiousness functions compute
4. `src/mcmc/sampler.jl` -- how the four Gibbs steps fit together

If you want to understand the NUTS integration specifically, read `src/mcmc/update_theta.jl`.

## Differences from the MATLAB original

- NUTS replaces Metropolis-Hastings for the continuous parameter update, with automatic step size and mass matrix adaptation. The augmented data steps (infection times, onsets, asymptomatic status) remain as Metropolis-within-Gibbs since they are discrete/constrained.
- Mechanistic model parameters are stored in a `MechParams` named tuple instead of a raw vector, so field access is self-documenting.
- [QuadGK.jl](https://github.com/JuliaMath/QuadGK.jl) replaces Chebfun for distribution convolutions.
- [CairoMakie.jl](https://docs.makie.org/stable/) replaces MATLAB's plotting functions.

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

65 tests cover data import, both likelihood functions, the infectiousness profiles, and prior distributions.
