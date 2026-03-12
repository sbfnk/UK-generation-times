"""
Fit the mechanistic model using data augmentation MCMC.

theta = [p_E, 1/mu, alpha, beta0]
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using UKGenerationTimes
using Distributions
using Random
using JLD2

# Initialise random number generator
Random.seed!(10)

# Load data
data_path = joinpath(@__DIR__, "..", "data", "Supplementary_Data.xlsx")
obs = import_and_format_data(data_path)
ap = AssumedParameters()

# MCMC settings
no_steps = 1_000_000
burn_in = no_steps ÷ 5
thin = 10
steps_keep = collect((burn_in + 1):thin:no_steps)

# Likelihood function
function ll_household_form(theta, aug)
    (theta[1] <= 0 || theta[1] >= 1) && return fill(-Inf, length(obs.household_sizes_incl))
    any(theta[2:end] .<= 0) && return fill(-Inf, length(obs.household_sizes_incl))

    p = get_params_mech(theta, ap.params_known)
    log_likelihood_household_mech(
        t -> f_inc_gam(t, ap),
        (x, t_inc, hh_size, asymp) -> b_cond_mech(x, t_inc, hh_size, asymp, p),
        (x, t_inc, hh_size, asymp) -> b_int_cond_mech(x, t_inc, hh_size, asymp, p),
        (t_inc, hh_size, asymp) -> mean_transmissions_mech(t_inc, hh_size, asymp, p),
        aug
    )
end

# Empirical summary function
function empirical_summary_form(theta, aug)
    p = get_params_mech(theta, ap.params_known)
    empirical_summary_mech(
        (x, t_inc, hh_size, asymp) -> b_cond_mech(x, t_inc, hh_size, asymp, p),
        aug
    )
end

# Initial values
theta_init = [0.5, 1 / 0.18, 3.5, 2.0]

# Proposal SDs
t_i_prop_sd = 9.0
t_prop_sd_asymp = 18.0

# MCMC configuration (NUTS for theta, MH for augmented data)
config = MCMCConfig(no_steps, steps_keep,
                    t_i_prop_sd, t_prop_sd_asymp,
                    logprior_mech, LogitLogTransform())

# Initialise augmented data
aug_init = initialise_augmented_data_mech(obs)

# Run MCMC
@info "Starting mechanistic model MCMC..."
@time result = fit_params(config, ll_household_form, empirical_summary_form,
                          theta_init, aug_init; model=:mech)

# Save results
results_path = joinpath(@__DIR__, "..", "results")
mkpath(results_path)
@save joinpath(results_path, "param_fit_mech.jld2") result
@info "Results saved to $(joinpath(results_path, "param_fit_mech.jld2"))"
