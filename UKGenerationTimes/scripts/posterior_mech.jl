"""
Posterior analysis for the mechanistic model.

Based on [mcmc_posterior_mech.m](https://github.com/will-s-hart/UK-generation-times/blob/main/Scripts/Fitted%20model%20analysis/mcmc_posterior_mech.m) from the original MATLAB implementation.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using UKGenerationTimes
using Distributions
using Statistics
using JLD2

# Load assumed parameters
ap = AssumedParameters()

# Load MCMC results
results_path = joinpath(@__DIR__, "..", "results")
@load joinpath(results_path, "param_fit_mech.jld2") result

theta_mat = result.theta_mat

# Posterior distributions of parameters
p_E_post = theta_mat[:, 1]
k_E_post = ap.k_inc .* p_E_post
k_P_post = ap.k_inc .- k_E_post
mu_inv_post = theta_mat[:, 2]
alpha_post = theta_mat[:, 3]
beta_post = theta_mat[:, 4]

# Point estimates
p_E_best = mean(p_E_post)
k_E_best = ap.k_inc * p_E_best
k_P_best = ap.k_inc - k_E_best
mu_inv_best = mean(mu_inv_post)
alpha_best = mean(alpha_post)
beta_best = mean(beta_post)

theta_best = [p_E_best, mu_inv_best, alpha_best, beta_best]
params_best = get_params_mech(theta_best, ap.params_known)

@info "Mechanistic model posteriors:" p_E_best mu_inv_best alpha_best beta_best

# Presymptomatic transmission probability
prob_presymp_post = (alpha_post .* k_P_post ./ (ap.k_inc * ap.gamma)) ./
    ((alpha_post .* k_P_post ./ (ap.k_inc * ap.gamma)) .+ mu_inv_post)
prob_presymp_best = (alpha_best * k_P_best / (ap.k_inc * ap.gamma)) /
    ((alpha_best * k_P_best / (ap.k_inc * ap.gamma)) + mu_inv_best)

# Generation time mean and SD
no_steps_kept = length(k_E_post)
params_post = hcat(
    fill(ap.gamma, no_steps_kept),
    1 ./ mu_inv_post,
    fill(ap.k_inc, no_steps_kept),
    k_E_post,
    fill(ap.k_I, no_steps_kept),
    alpha_post,
    beta_post,
    fill(ap.rho, no_steps_kept),
    fill(ap.x_A, no_steps_kept)
)

mean_post, sd_post = get_gen_mean_sd_mech(params_post)
mean_best, sd_best = get_gen_mean_sd_mech(params_best)

# Print quantiles
for (name, vals) in [("Mean gen time", mean_post),
                      ("SD gen time", sd_post),
                      ("p_E", p_E_post),
                      ("1/mu", mu_inv_post),
                      ("alpha", alpha_post),
                      ("beta", beta_post),
                      ("P(presymp)", prob_presymp_post)]
    q = quantile(vals, [0.025, 0.5, 0.975])
    @info "$name: median=$(round(q[2]; digits=2)), 95% CI [$(round(q[1]; digits=2)), $(round(q[3]; digits=2))]"
end

# Save
empirical_summary_mat = result.empirical_summary_mat
@save joinpath(results_path, "posterior_mech.jld2") p_E_post mu_inv_post alpha_post beta_post prob_presymp_post params_best mean_post mean_best sd_post sd_best empirical_summary_mat
