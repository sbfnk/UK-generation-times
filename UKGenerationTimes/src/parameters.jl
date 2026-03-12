"""
Fixed/assumed parameter values from McAloon et al. and Hart et al. (2022).
"""

"""
Assumed parameters common to both models.
"""
struct AssumedParameters
    # Lognormal incubation period (McAloon et al.)
    inc_mu::Float64
    inc_sigma::Float64
    inc_mean::Float64
    inc_var::Float64

    # Gamma incubation period (same mean and SD)
    inc_shape::Float64
    inc_scale::Float64

    # Common parameters
    x_A::Float64   # relative infectiousness of asymptomatic hosts
    rho::Float64   # transmission scales with household_size^(-rho)

    # Mechanistic model known parameters
    k_inc::Float64
    gamma::Float64
    k_I::Float64
    params_known::Vector{Float64}  # [k_inc, gamma, k_I, rho, x_A]
end

function AssumedParameters()
    inc_mu = 1.63
    inc_sigma = 0.5
    inc_mean = exp(inc_mu + 0.5 * inc_sigma^2)
    inc_var = (exp(inc_sigma^2) - 1) * exp(2 * inc_mu + inc_sigma^2)

    inc_shape = inc_mean^2 / inc_var
    inc_scale = inc_var / inc_mean

    x_A = 0.35
    rho = 1.0

    k_inc = inc_shape
    gamma = 1 / (k_inc * inc_scale)
    k_I = 1.0

    params_known = [k_inc, gamma, k_I, rho, x_A]

    AssumedParameters(inc_mu, inc_sigma, inc_mean, inc_var,
                      inc_shape, inc_scale, x_A, rho,
                      k_inc, gamma, k_I, params_known)
end

"""
Lognormal incubation period PDF.
"""
f_inc_logn(t_inc, ap::AssumedParameters) =
    pdf.(LogNormal(ap.inc_mu, ap.inc_sigma), t_inc)

"""
Gamma incubation period PDF.
"""
f_inc_gam(t_inc, ap::AssumedParameters) =
    pdf.(Gamma(ap.inc_shape, ap.inc_scale), t_inc)
