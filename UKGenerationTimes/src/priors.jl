"""
Prior distributions for both models.

Independent model: theta = [mean_gen, sd_gen, beta0]
  All three are positive, with LogNormal priors chosen to give broad support
  over plausible epidemiological ranges.

Mechanistic model: theta = [p_E, 1/mu, alpha, beta0]
  p_E (fraction of incubation spent in E) is in (0,1) with a Beta prior.
  The remaining three are positive with LogNormal priors.
"""

"""
Prior density for the independent model.

theta = [mean_gen, sd_gen, beta0]:
- mean_gen: mean generation time (days). Prior median ~5, 95% CI ~[2.5, 10].
- sd_gen:   SD of generation time (days). Prior median ~2, 95% CI ~[0.6, 7].
- beta0:    baseline transmission rate. Prior median ~2, 95% CI ~[0.4, 10].
"""
function prior_indep(theta)
    p_mean = pdf(LogNormal(1.6, 0.35), theta[1])
    p_sd = pdf(LogNormal(0.7, 0.65), theta[2])
    p_beta = pdf(LogNormal(0.7, 0.8), theta[3])
    p_mean * p_sd * p_beta
end

"""
Log-prior density for the independent model.
Returns -Inf for non-positive parameters.
"""
function logprior_indep(theta)
    any(x -> x <= 0, theta) && return -Inf
    lp = logpdf(LogNormal(1.6, 0.35), theta[1])
    lp += logpdf(LogNormal(0.7, 0.65), theta[2])
    lp += logpdf(LogNormal(0.7, 0.8), theta[3])
    lp
end

"""
Prior density for the mechanistic model.

theta = [p_E, 1/mu, alpha, beta0]:
- p_E:   fraction of incubation in the exposed (latent) stage. Prior: Beta(2.1, 2.1),
          symmetric around 0.5 with 95% CI ~[0.1, 0.9].
- 1/mu:  mean symptomatic infectious duration (days). Prior median ~5, 95% CI ~[1, 24].
- alpha: relative presymptomatic infectiousness. Prior median ~1, 95% CI ~[0.2, 5].
- beta0: baseline transmission rate. Prior median ~2, 95% CI ~[0.4, 10].
"""
function prior_mech(theta)
    p_p_E = pdf(Beta(2.1, 2.1), theta[1])
    p_mu_inv = pdf(LogNormal(1.6, 0.8), theta[2])
    p_alpha = pdf(LogNormal(0.0, 0.8), theta[3])
    p_beta = pdf(LogNormal(0.7, 0.8), theta[4])
    p_p_E * p_mu_inv * p_alpha * p_beta
end

"""
Log-prior density for the mechanistic model.
Returns -Inf if p_E is outside (0,1) or other parameters are non-positive.
"""
function logprior_mech(theta)
    if theta[1] <= 0 || theta[1] >= 1
        return -Inf
    end
    lp = logpdf(Beta(2.1, 2.1), theta[1])
    lp += logpdf(LogNormal(1.6, 0.8), theta[2])
    lp += logpdf(LogNormal(0.0, 0.8), theta[3])
    lp += logpdf(LogNormal(0.7, 0.8), theta[4])
    lp
end
