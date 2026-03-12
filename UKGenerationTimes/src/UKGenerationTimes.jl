module UKGenerationTimes

using AdvancedHMC
using CairoMakie
using DataFrames
using Distributions
using ForwardDiff
using LinearAlgebra
using LogDensityProblems
using LogDensityProblemsAD
using ProgressMeter
using QuadGK
using Random
using SparseArrays
using Statistics
using XLSX

# Data structures
include("types.jl")
export ObservedData, AugmentedData, PossibleInfectors

# Fixed parameters
include("parameters.jl")
export AssumedParameters, f_inc_logn, f_inc_gam

# Data import
include("data.jl")
export import_and_format_data

# Likelihood functions
include("likelihood_indep.jl")
export log_likelihood_household_indep

include("likelihood_mech.jl")
export log_likelihood_household_mech

# Infectiousness functions (mechanistic model)
include("infectiousness.jl")
export get_params_mech, b_cond_mech, b_int_cond_mech,
       mean_transmissions_mech, f_tost_mech, get_gen_mean_sd_mech

# Prior distributions
include("priors.jl")
export prior_indep, logprior_indep, prior_mech, logprior_mech

# Summary statistics
include("summary.jl")
export empirical_summary_indep, empirical_summary_mech,
       get_presymp_trans_probs_indep_logn

# Generation time / serial interval distributions
include("generation_time.jl")
export gen_tost_serial_indep, gen_tost_serial_mech

# MCMC sampler
include("mcmc/initialise.jl")
export initialise_augmented_data_indep, initialise_augmented_data_mech

include("mcmc/update_theta.jl")
export ParameterTransform, LogTransform, LogitLogTransform,
       to_unconstrained, to_constrained,
       NUTSTarget, NUTSState, initialise_nuts, update_theta_nuts!

include("mcmc/update_infection_indep.jl")
export update_infection_indep!

include("mcmc/update_infection_mech.jl")
export update_infection_mech!

include("mcmc/update_onset.jl")
export update_onset!

include("mcmc/update_asymp_indep.jl")
export update_asymp_indep!

include("mcmc/update_asymp_mech.jl")
export update_asymp_mech!

include("mcmc/sampler.jl")
export MCMCConfig, MCMCResult, fit_params

# Plotting (loaded on demand via extension or explicit import)
include("plotting.jl")
export plot_posterior_densities, plot_gen_tost_serial,
       plot_trace, plot_comparison

end # module
