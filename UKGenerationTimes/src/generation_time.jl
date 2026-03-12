"""
Generation time, TOST, and serial interval distribution computation via
numerical quadrature (QuadGK), replacing MATLAB's Chebfun convolutions.

Translates:
- Scripts/Fitted model analysis/gen_tost_serial_indep.m
- Scripts/Fitted model analysis/gen_tost_serial_mech.m
- Functions/Mech/get_gen_dist_mech.m
- Functions/Mech/get_serial_dist_mech.m
"""

# ──────────────────────────────────────────────────────────────────────────────
# Independent model
# ──────────────────────────────────────────────────────────────────────────────

"""
    gen_tost_serial_indep(mean_gen, sd_gen, inc_mu, inc_sigma; t_grid, atol, rtol)

Compute generation time, TOST, and serial interval distributions for the
independent model at posterior mean parameter values.

Returns named tuple (f_gen, f_tost, f_serial, t_grid).
"""
function gen_tost_serial_indep(mean_gen, sd_gen, inc_mu, inc_sigma;
                                t_grid=range(-25, 50; length=751),
                                atol=1e-10, rtol=1e-8)
    # Lognormal parameterisation from mean and SD
    logn_mu = log(mean_gen^2 / sqrt(sd_gen^2 + mean_gen^2))
    logn_sigma = sqrt(log(1 + sd_gen^2 / mean_gen^2))

    f_gen_dist = LogNormal(logn_mu, logn_sigma)
    f_inc_dist = LogNormal(inc_mu, inc_sigma)

    # Generation time density (only positive support)
    f_gen_vals = [t > 0 ? pdf(f_gen_dist, t) : 0.0 for t in t_grid]

    # TOST = generation time - incubation period
    # f_tost(x) = integral f_gen(x + s) * f_inc(s) ds, s > 0
    f_tost_vals = map(t_grid) do t
        val, _ = quadgk(s -> pdf(f_gen_dist, t + s) * pdf(f_inc_dist, s),
                        0.0, Inf; atol, rtol)
        val
    end

    # Serial interval = TOST + incubation period
    # f_serial(t) = integral f_tost(t - s) * f_inc(s) ds, s > 0
    # Use interpolation of f_tost for inner evaluation
    dt = Float64(step(t_grid))
    f_tost_interp = _linear_interp(collect(t_grid), f_tost_vals)

    f_serial_vals = map(t_grid) do t
        val, _ = quadgk(s -> f_tost_interp(t - s) * pdf(f_inc_dist, s),
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
    gen_tost_serial_mech(params; t_grid, atol, rtol)

Compute generation time, TOST, and serial interval distributions for the
mechanistic model at given parameter values.

Returns named tuple (f_gen, f_tost, f_serial, t_grid).
"""
function gen_tost_serial_mech(params;
                               t_grid=range(-25, 50; length=751),
                               atol=1e-10, rtol=1e-8)
    gamma, mu, k_inc, k_E, k_I, alpha = params[1:6]
    k_P = k_inc - k_E

    C = k_inc * gamma * mu / (alpha * k_P * mu + k_inc * gamma)

    f_E_dist = Gamma(k_E, 1 / (k_inc * gamma))
    f_P_dist = Gamma(k_P, 1 / (k_inc * gamma))
    F_I_dist = Gamma(k_I, 1 / (k_I * mu))
    f_inc_dist = Gamma(k_inc, 1 / (k_inc * gamma))

    # f_star(t) for t > 0: the infectiousness profile convolved with incubation
    # f_star(t) = alpha*C*(1 - F_P(t)) + C*(F_P(t) - conv(f_P, F_I)(t))
    function f_star(t)
        t <= 0 && return 0.0
        conv_val, _ = quadgk(s -> pdf(f_P_dist, s) * cdf(F_I_dist, t - s),
                             0.0, t; atol, rtol)
        alpha * C * (1 - cdf(f_P_dist, t)) + C * (cdf(f_P_dist, t) - conv_val)
    end

    # Generation time = E + star
    # f_gen(t) = integral f_E(s) * f_star(t - s) ds
    f_gen_vals = map(t_grid) do t
        t <= 0 && return 0.0
        val, _ = quadgk(s -> pdf(f_E_dist, s) * f_star(t - s),
                        0.0, t; atol, rtol)
        val
    end

    # TOST (direct analytical form)
    f_tost_vals = f_tost_mech(collect(t_grid), params)

    # Serial interval = incubation + TOST
    f_tost_interp = _linear_interp(collect(t_grid), f_tost_vals)
    f_serial_vals = map(t_grid) do t
        val, _ = quadgk(s -> pdf(f_inc_dist, s) * f_tost_interp(t - s),
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
Create a linear interpolation function from sorted x, y arrays.
Returns 0 outside the domain.
"""
function _linear_interp(x::Vector{Float64}, y::Vector{Float64})
    function interp(t)
        t < x[1] && return 0.0
        t > x[end] && return 0.0
        # Find interval
        i = searchsortedlast(x, t)
        i >= length(x) && return y[end]
        i < 1 && return y[1]
        # Linear interpolation
        frac = (t - x[i]) / (x[i+1] - x[i])
        y[i] + frac * (y[i+1] - y[i])
    end
    interp
end
