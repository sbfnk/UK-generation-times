"""
Fit the independent transmission and symptoms model using data augmentation MCMC.

theta = [mean_gen, sd_gen, beta0]
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using UKGenerationTimes
using Distributions
using Random
using JLD2

# Initialise random number generator
Random.seed!(3)

# Load data
data_path = joinpath(@__DIR__, "..", "data", "Supplementary_Data.xlsx")
obs = import_and_format_data(data_path)
ap = AssumedParameters()

# MCMC settings
# NUTS makes the parameter update much cheaper than the original MH, so fewer
# total steps are needed. The bottleneck is now the augmented data MH steps.
# 200K total = 50K of each Gibbs step type; thin by 4 to keep one per cycle.
no_steps = 200_000
burn_in = no_steps ÷ 5
thin = 4
steps_keep = collect((burn_in + 1):thin:no_steps)

# Generation time distribution functions (lognormal parameterised by mean, sd)
function logn_mu(m, s)
    log(m^2 / sqrt(s^2 + m^2))
end

function logn_sigma(m, s)
    sqrt(log(1 + s^2 / m^2))
end

function f_gen(t_gen, theta)
    mu = logn_mu(theta[1], theta[2])
    sigma = logn_sigma(theta[1], theta[2])
    return pdf.(LogNormal(mu, sigma), t_gen)
end

function F_gen(t_gen, theta)
    mu = logn_mu(theta[1], theta[2])
    sigma = logn_sigma(theta[1], theta[2])
    return cdf.(LogNormal(mu, sigma), t_gen)
end

# Log-likelihood function
function ll_household_form(theta, aug)
    any(theta .<= 0) && return fill(-Inf, length(obs.household_sizes_incl))
    log_likelihood_household_indep(
        t -> f_inc_logn(t, ap),
        theta[3], ap.rho, ap.x_A,
        t -> f_gen(t, theta),
        t -> F_gen(t, theta),
        aug
    )
end

# Empirical summary function
function empirical_summary_form(theta, aug)
    empirical_summary_indep(
        theta[3], ap.rho, ap.x_A,
        t -> f_gen(t, theta),
        aug
    )
end

# Initial values
theta_init = [5.0, 5.0, 2.0]

# Proposal SDs for infection times
t_i_prop_sd_symp = 8.0
t_i_prop_sd_asymp = 13.0

# MCMC configuration (NUTS for theta, MH for augmented data)
config = MCMCConfig(no_steps, steps_keep,
                    t_i_prop_sd_symp, t_i_prop_sd_asymp,
                    logprior_indep, LogTransform())

# Initialise augmented data
aug_init = initialise_augmented_data_indep(obs)

# Run MCMC
@info "Starting independent model MCMC..."
@time result = fit_params(config, ll_household_form, empirical_summary_form,
                          theta_init, aug_init; model=:indep)

# Save results
results_path = joinpath(@__DIR__, "..", "results")
mkpath(results_path)
@save joinpath(results_path, "param_fit_indep.jld2") result
@info "Results saved to $(joinpath(results_path, "param_fit_indep.jld2"))"
