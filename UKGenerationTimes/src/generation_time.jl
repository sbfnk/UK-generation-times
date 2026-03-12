"""
Generation time, TOST, and serial interval distribution computation via
numerical quadrature (QuadGK).

These are post-hoc analyses run on posterior parameter estimates, not part of the
MCMC loop. They compute the population-level distributions by convolving the
component distributions of the infection process.

Key relationships:
- Generation time = time from infector's infection to infectee's infection
- TOST = time from infector's symptom onset to infectee's infection
- Serial interval = time from infector's symptom onset to infectee's symptom onset

So:  generation time = incubation period + TOST
     serial interval = TOST + infectee's incubation period
"""

# ──────────────────────────────────────────────────────────────────────────────
# Independent model
# ──────────────────────────────────────────────────────────────────────────────

"""
    gen_tost_serial_indep(mean_gen, sd_gen, inc_mu, inc_sigma; t_grid, atol, rtol)

Compute generation time, TOST, and serial interval distributions for the
independent model, where the generation time is lognormally distributed
(independent of the incubation period).

Returns named tuple `(f_gen, f_tost, f_serial, t_grid)`.
"""
function gen_tost_serial_indep(mean_gen, sd_gen, inc_mu, inc_sigma;
                                t_grid=range(-25, 50; length=751),
                                atol=1e-10, rtol=1e-8)
    # Convert (mean, sd) to lognormal (mu, sigma) parameterisation
    logn_mu = log(mean_gen^2 / sqrt(sd_gen^2 + mean_gen^2))
    logn_sigma = sqrt(log(1 + sd_gen^2 / mean_gen^2))

    gen_dist = LogNormal(logn_mu, logn_sigma)
    inc_dist = LogNormal(inc_mu, inc_sigma)

    # Generation time density (positive support only)
    f_gen_vals = [t > 0 ? pdf(gen_dist, t) : 0.0 for t in t_grid]

    # TOST = generation time minus incubation period of the infector
    # f_tost(t) = ∫ f_gen(t + s) * f_inc(s) ds,  s > 0
    # (shift generation time back by the incubation period)
    f_tost_vals = map(t_grid) do t
        val, _ = quadgk(s -> pdf(gen_dist, t + s) * pdf(inc_dist, s),
                        0.0, Inf; atol, rtol)
        val
    end

    # Serial interval = TOST + infectee's incubation period
    # f_serial(t) = ∫ f_tost(t - s) * f_inc(s) ds,  s > 0
    f_tost_interp = _linear_interp(collect(t_grid), f_tost_vals)

    f_serial_vals = map(t_grid) do t
        val, _ = quadgk(s -> f_tost_interp(t - s) * pdf(inc_dist, s),
                        0.0, Inf; atol, rtol)
        val
    end

    (f_gen=f_gen_vals, f_tost=f_tost_vals,
     f_serial=f_serial_vals, t_grid=collect(t_grid))
end

# ──────────────────────────────────────────────────────────────────────────────
# Mechanistic model
# ──────────────────────────────────────────────────────────────────────────────

"""
    gen_tost_serial_mech(p; t_grid, atol, rtol)

Compute generation time, TOST, and serial interval distributions for the
mechanistic model (E→P→I staged infection).

The generation time is *not* independent of the incubation period here: it
equals the latent period (E) plus an infectiousness-weighted draw from the
presymptomatic (P) and symptomatic infectious (I) stages.

`p` should be a `MechParams` named tuple (from `get_params_mech`).

Returns named tuple `(f_gen, f_tost, f_serial, t_grid)`.
"""
function gen_tost_serial_mech(p;
                               t_grid=range(-25, 50; length=751),
                               atol=1e-10, rtol=1e-8)
    (; γ, μ, k_inc, k_E, k_I, α) = p
    k_P = k_inc - k_E

    W = infectiousness_weight(p)
    inc_rate = k_inc * γ

    latent_dist  = Gamma(k_E, 1 / inc_rate)   # E-stage duration
    presymp_dist = Gamma(k_P, 1 / inc_rate)   # P-stage duration
    symp_dist    = Gamma(k_I, 1 / (k_I * μ))  # I-stage duration
    inc_dist     = Gamma(k_inc, 1 / inc_rate)  # total incubation (E + P)

    # f_star(t): infectiousness-weighted transmission timing after leaving E.
    # Combines presymptomatic (P) and symptomatic (I) stage contributions:
    #   f_star(t) = α*W*(1 - F_P(t)) + W*(F_P(t) - (f_P * F_I)(t))
    # where * denotes convolution and F denotes CDF.
    function f_star(t)
        t <= 0 && return 0.0
        # Convolution: ∫ f_P(s) * F_I(t-s) ds from 0 to t
        conv_val, _ = quadgk(s -> pdf(presymp_dist, s) * cdf(symp_dist, t - s),
                             0.0, t; atol, rtol)
        α * W * (1 - cdf(presymp_dist, t)) + W * (cdf(presymp_dist, t) - conv_val)
    end

    # Generation time = E-stage duration + star
    # f_gen(t) = ∫ f_E(s) * f_star(t - s) ds
    f_gen_vals = map(t_grid) do t
        t <= 0 && return 0.0
        val, _ = quadgk(s -> pdf(latent_dist, s) * f_star(t - s),
                        0.0, t; atol, rtol)
        val
    end

    # TOST has a direct analytical form (see infectiousness.jl)
    f_tost_vals = f_tost_mech(collect(t_grid), p)

    # Serial interval = TOST + infectee's incubation period
    f_tost_interp = _linear_interp(collect(t_grid), f_tost_vals)
    f_serial_vals = map(t_grid) do t
        val, _ = quadgk(s -> pdf(inc_dist, s) * f_tost_interp(t - s),
                        0.0, Inf; atol, rtol)
        val
    end

    (f_gen=f_gen_vals, f_tost=f_tost_vals,
     f_serial=f_serial_vals, t_grid=collect(t_grid))
end

# ──────────────────────────────────────────────────────────────────────────────
# Helper: simple linear interpolation
# ──────────────────────────────────────────────────────────────────────────────

"""
Create a linear interpolation closure from sorted (x, y) arrays.
Returns 0 outside the domain of x.
"""
function _linear_interp(x::Vector{Float64}, y::Vector{Float64})
    function interp(t)
        t < x[1] && return 0.0
        t > x[end] && return 0.0
        i = searchsortedlast(x, t)
        i >= length(x) && return y[end]
        i < 1 && return y[1]
        frac = (t - x[i]) / (x[i+1] - x[i])
        y[i] + frac * (y[i+1] - y[i])
    end
    interp
end
