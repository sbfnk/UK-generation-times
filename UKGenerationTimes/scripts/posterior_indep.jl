"""
Posterior analysis for the independent model.

Based on [mcmc_posterior_indep.m](https://github.com/will-s-hart/UK-generation-times/blob/main/Scripts/Fitted%20model%20analysis/mcmc_posterior_indep.m) from the original MATLAB implementation.
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
@load joinpath(results_path, "param_fit_indep.jld2") result

theta_mat = result.theta_mat

# Posterior distributions of parameters
mean_post = theta_mat[:, 1]
sd_post = theta_mat[:, 2]
beta_post = theta_mat[:, 3]

# Point estimates
mean_best = mean(mean_post)
sd_best = mean(sd_post)
beta_best = mean(beta_post)

@info "Independent model posteriors:" mean_best sd_best beta_best

# Presymptomatic transmission probability
logn_mu_fn(m, s) = log(m^2 / sqrt(s^2 + m^2))
logn_sigma_fn(m, s) = sqrt(log(1 + s^2 / m^2))

F_inc(t) = cdf(LogNormal(ap.inc_mu, ap.inc_sigma), t)

mu_post = logn_mu_fn.(mean_post, sd_post)
sigma_post = logn_sigma_fn.(mean_post, sd_post)
prob_presymp_post = get_presymp_trans_probs_indep_logn(mu_post, sigma_post, F_inc)

mu_best = logn_mu_fn(mean_best, sd_best)
sigma_best = logn_sigma_fn(mean_best, sd_best)
prob_presymp_best = get_presymp_trans_probs_indep_logn([mu_best], [sigma_best], F_inc)[1]

@info "Presymptomatic transmission:" prob_presymp_best

# Empirical summary
empirical_summary_mat = result.empirical_summary_mat

# Print quantiles
for (name, vals) in [("Mean gen time", mean_post),
                      ("SD gen time", sd_post),
                      ("Beta", beta_post),
                      ("P(presymp)", prob_presymp_post)]
    q = quantile(vals, [0.025, 0.5, 0.975])
    @info "$name: median=$(round(q[2]; digits=2)), 95% CI [$(round(q[1]; digits=2)), $(round(q[3]; digits=2))]"
end

# Save
@save joinpath(results_path, "posterior_indep.jld2") mean_post sd_post beta_post mean_best sd_best beta_best prob_presymp_post prob_presymp_best mu_post sigma_post empirical_summary_mat
