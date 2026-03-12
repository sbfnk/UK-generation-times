# UKGenerationTimes.jl

Julia port of the SARS-CoV-2 generation time inference model from [Hart et al. (2022)](https://doi.org/10.7554/eLife.70767). The [original MATLAB code](https://github.com/will-s-hart/UK-generation-times) uses data-augmentation MCMC to infer generation time distributions from UK household transmission data. This port replaces the Metropolis-Hastings parameter update with NUTS ([AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl)) for more efficient sampling of the continuous parameters.

Two models are included:

- **Independent model**: lognormal generation time distribution (3 parameters: mean, sd, transmission rate)
- **Mechanistic model**: staged E->P->I infection process (4 parameters: proportion of latent period that is pre-infectious, mean infectious period, relative pre-symptomatic infectiousness, transmission rate)

## Installation

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

### Data

Place `Supplementary_Data.xlsx` (from the [eLife paper](https://doi.org/10.7554/eLife.70767)) in the `data/` directory.

## Usage

### Fitting models

```bash
julia --project=. scripts/fit_indep.jl    # Independent model (1M MCMC steps)
julia --project=. scripts/fit_mech.jl     # Mechanistic model (1M MCMC steps)
```

Results are saved as JLD2 files in `results/`.

### Post-processing

```bash
julia --project=. scripts/posterior_indep.jl   # Posterior summaries
julia --project=. scripts/posterior_mech.jl
julia --project=. scripts/gen_tost_serial.jl   # Generation time, TOST, serial interval distributions
julia --project=. scripts/figures.jl            # Publication figures
```

## Package structure

```
src/
  UKGenerationTimes.jl     # Main module
  types.jl                 # ObservedData, AugmentedData structs
  data.jl                  # Excel import and data formatting
  parameters.jl            # Fixed/assumed parameters (incubation period, etc.)
  likelihood_indep.jl      # Independent model household log-likelihood
  likelihood_mech.jl       # Mechanistic model household log-likelihood
  priors.jl                # Prior distributions (both models)
  infectiousness.jl        # Mechanistic infectiousness profile functions
  generation_time.jl       # GT/serial interval via QuadGK numerical integration
  summary.jl               # Empirical summaries and presymptomatic transmission
  plotting.jl              # Figure generation (CairoMakie)
  mcmc/
    sampler.jl             # Main MCMC loop (4-step data augmentation)
    update_theta.jl        # NUTS parameter update (AdvancedHMC.jl)
    update_infection_indep.jl  # Infection time updates (independent)
    update_infection_mech.jl   # Infection time updates (mechanistic)
    update_onset.jl        # Onset time resampling
    update_asymp_indep.jl  # Asymptomatic updates (independent)
    update_asymp_mech.jl   # Asymptomatic updates (mechanistic)
    initialise.jl          # Augmented data initialisation
```

## Differences from the MATLAB original

- NUTS ([AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl)) replaces Metropolis-Hastings for the continuous parameter update, with automatic step size and mass matrix adaptation. The augmented data steps (infection times, onsets, asymptomatic status) remain as Metropolis-within-Gibbs updates since they are discrete/constrained
- [QuadGK.jl](https://github.com/JuliaMath/QuadGK.jl) replaces Chebfun for distribution convolutions (generation time, serial interval)
- [CairoMakie.jl](https://docs.makie.org/stable/) replaces MATLAB's plotting functions and export_fig
- Reads the same Excel data file via [XLSX.jl](https://github.com/felipenoris/XLSX.jl)

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

65 tests covering data import, likelihood functions, and infectiousness profile calculations.
