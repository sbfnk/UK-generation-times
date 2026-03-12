"""
Prior distributions for both models.

Translates:
- Functions/Indep/prior_fun_indep.m
- Functions/Mech/prior_fun_mech.m
"""

using Distributions

"""
Prior density for the independent model.

theta = [mean_gen, sd_gen, beta0]
"""
function prior_indep(theta)
    p_mean = pdf(LogNormal(1.6, 0.35), theta[1])   # median 5, 95% CI [2.5, 10]
    p_sd = pdf(LogNormal(0.7, 0.65), theta[2])     # median 2, 95% CI [0.6, 7]
    p_beta = pdf(LogNormal(0.7, 0.8), theta[3])    # median 2, 95% CI [0.4, 10]
    return p_mean * p_sd * p_beta
end

"""
Log-prior density for the independent model.

All three parameters must be positive (LogNormal support); logpdf returns -Inf
for non-positive arguments, so an explicit guard is added for consistency with
`logprior_mech`.
"""
function logprior_indep(theta)
    any(x -> x <= 0, theta) && return -Inf
    lp = logpdf(LogNormal(1.6, 0.35), theta[1])
    lp += logpdf(LogNormal(0.7, 0.65), theta[2])
    lp += logpdf(LogNormal(0.7, 0.8), theta[3])
    return lp
end

"""
Prior density for the mechanistic model.

theta = [p_E, 1/mu, alpha, beta0]
"""
function prior_mech(theta)
    p_p_E = pdf(Beta(2.1, 2.1), theta[1])          # median 0.5, 95% CI [0.1, 0.9]
    p_mu_inv = pdf(LogNormal(1.6, 0.8), theta[2])  # median 5, 95% CI [1, 24]
    p_alpha = pdf(LogNormal(0.0, 0.8), theta[3])    # median 1, 95% CI [0.2, 5]
    p_beta = pdf(LogNormal(0.7, 0.8), theta[4])     # median 2, 95% CI [0.4, 10]
    return p_p_E * p_mu_inv * p_alpha * p_beta
end

"""
Log-prior density for the mechanistic model.
"""
function logprior_mech(theta)
    # p_E must be in (0, 1)
    if theta[1] <= 0 || theta[1] >= 1
        return -Inf
    end
    lp = logpdf(Beta(2.1, 2.1), theta[1])
    lp += logpdf(LogNormal(1.6, 0.8), theta[2])
    lp += logpdf(LogNormal(0.0, 0.8), theta[3])
    lp += logpdf(LogNormal(0.7, 0.8), theta[4])
    return lp
end
