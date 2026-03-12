"""
Infectiousness profile functions for the mechanistic model.

Translates:
- Functions/Mech/get_params_mech.m
- Functions/Mech/b_cond_form_mech.m
- Functions/Mech/b_int_cond_form_mech.m
- Functions/Mech/mean_transmissions_form_mech.m
- Functions/Mech/f_tost_form_mech.m
- Functions/Mech/get_gen_mean_sd_mech.m
"""

"""
Recover the full parameter vector from fitted and known parameters.

theta = [p_E, 1/mu, alpha, beta0]
params_known = [k_inc, gamma, k_I, rho, x_A]

Returns params = [gamma, mu, k_inc, k_E, k_I, alpha, beta0, rho, x_A]
"""
function get_params_mech(theta, params_known)
    k_inc = params_known[1]
    gamma = params_known[2]
    k_I = params_known[3]
    rho = params_known[4]
    x_A = params_known[5]

    k_E = theta[1] * k_inc
    mu = 1 / theta[2]
    alpha = theta[3]
    beta0 = theta[4]

    [gamma, mu, k_inc, k_E, k_I, alpha, beta0, rho, x_A]
end

"""
    b_cond_mech(x, t_inc, household_size, asymp, params)

Expected infectiousness at time `x` since symptom onset, conditional on
incubation period `t_inc`.
"""
function b_cond_mech(x, t_inc, household_size, asymp, params)
    gamma, mu, k_inc, k_E, k_I, alpha, beta0, rho, x_A = params
    k_P = k_inc - k_E

    C = k_inc * gamma * mu / (alpha * k_P * mu + k_inc * gamma)
    beta_indiv = beta0 ./ (household_size .^ rho)
    beta_indiv[asymp] .*= x_A

    result = zeros(eltype(beta_indiv), length(x))

    ind_m = x .< 0
    ind_p = .!ind_m

    beta_dist = Beta(k_P, k_E)
    gamma_dist = Gamma(k_I, 1 / (k_I * mu))

    if any(ind_m)
        x_m = x[ind_m]
        t_inc_m = t_inc[ind_m]
        beta_m = beta_indiv[ind_m]
        result[ind_m] .= alpha * C .* beta_m .*
            (1 .- cdf.(Ref(beta_dist), -x_m ./ t_inc_m))
    end

    if any(ind_p)
        x_p = x[ind_p]
        beta_p = beta_indiv[ind_p]
        result[ind_p] .= C .* beta_p .*
            (1 .- cdf.(Ref(gamma_dist), x_p))
    end

    result
end

"""
    b_int_cond_mech(x, t_inc, household_size, asymp, params)

Cumulative infectiousness integrated from -infinity to `x` since symptom onset,
conditional on incubation period `t_inc`.
"""
function b_int_cond_mech(x, t_inc, household_size, asymp, params)
    gamma, mu, k_inc, k_E, k_I, alpha, beta0, rho, x_A = params
    k_P = k_inc - k_E

    C = k_inc * gamma * mu / (alpha * k_P * mu + k_inc * gamma)
    beta_indiv = beta0 ./ (household_size .^ rho)
    beta_indiv[asymp] .*= x_A

    result = zeros(eltype(beta_indiv), length(x))

    ind_m = x .< 0
    ind_p = .!ind_m

    beta_dist = Beta(k_P, k_E)
    beta_dist_p1 = Beta(k_P + 1, k_E)
    gamma_dist = Gamma(k_I, 1 / (k_I * mu))
    gamma_dist_p1 = Gamma(k_I + 1, 1 / (k_I * mu))

    if any(ind_m)
        x_m = x[ind_m]
        t_inc_m = t_inc[ind_m]
        beta_m = beta_indiv[ind_m]

        f_m1 = alpha * C .* beta_m .* x_m .*
            (1 .- cdf.(Ref(beta_dist), -x_m ./ t_inc_m))
        f_m2 = alpha * C .* beta_m .* (k_P .* t_inc_m ./ k_inc) .*
            (1 .- cdf.(Ref(beta_dist_p1), -x_m ./ t_inc_m))
        result[ind_m] .= f_m1 .+ f_m2
    end

    if any(ind_p)
        x_p = x[ind_p]
        t_inc_p = t_inc[ind_p]
        beta_p = beta_indiv[ind_p]

        f_p1 = C .* beta_p .* alpha .* k_P .* t_inc_p ./ k_inc
        f_p2 = C .* beta_p .* x_p .*
            (1 .- cdf.(Ref(gamma_dist), x_p))
        f_p3 = (C .* beta_p ./ mu) .*
            cdf.(Ref(gamma_dist_p1), x_p)
        result[ind_p] .= f_p1 .+ f_p2 .+ f_p3
    end

    result
end

"""
    mean_transmissions_mech(t_inc, household_size, asymp, params)

Total expected transmissions over the entire course of infection, conditional
on incubation period `t_inc`.
"""
function mean_transmissions_mech(t_inc, household_size, asymp, params)
    gamma, mu, k_inc, k_E, k_I, alpha, beta0, rho, x_A = params
    k_P = k_inc - k_E

    beta_indiv = beta0 ./ (household_size .^ rho)
    beta_indiv[asymp] .*= x_A

    gamma .* beta_indiv .* (alpha .* k_P .* mu .* t_inc .+ k_inc) ./
    (alpha .* k_P .* mu .+ k_inc .* gamma)
end

"""
    f_tost_mech(t_tost, params)

TOST (time from onset of symptoms to transmission) distribution density.
"""
function f_tost_mech(t_tost, params)
    gamma, mu, k_inc, k_E, k_I, alpha = params[1:6]
    k_P = k_inc - k_E

    C = k_inc * gamma * mu / (alpha * k_P * mu + k_inc * gamma)

    gamma_dist_pre = Gamma(k_P, 1 / (k_inc * gamma))
    gamma_dist_post = Gamma(k_I, 1 / (k_I * mu))

    result = zeros(float(typeof(C)), length(t_tost))

    ind_m = t_tost .< 0
    ind_p = .!ind_m

    if any(ind_m)
        result[ind_m] .= alpha * C .*
            (1 .- cdf.(Ref(gamma_dist_pre), .-t_tost[ind_m]))
    end

    if any(ind_p)
        result[ind_p] .= C .*
            (1 .- cdf.(Ref(gamma_dist_post), t_tost[ind_p]))
    end

    result
end

"""
    get_gen_mean_sd_mech(params)

Analytical mean and standard deviation of the generation time distribution.
`params` can be a single vector or a matrix with one parameter set per row.
"""
function get_gen_mean_sd_mech(params::AbstractVector)
    gamma, mu, k_inc, k_E, k_I, alpha = params[1:6]
    k_P = k_inc - k_E

    C = k_inc * gamma * mu / (alpha * k_P * mu + k_inc * gamma)

    m_E = k_E / (k_inc * gamma)
    v_E = k_E / (k_inc * gamma)^2

    m_P = k_P / (k_inc * gamma)
    m_PP = k_P * (k_P + 1) / (k_inc * gamma)^2
    m_PPP = k_P * (k_P + 1) * (k_P + 2) / (k_inc * gamma)^3

    m_I = 1 / mu
    m_II = (k_I + 1) / (k_I * mu^2)
    m_III = (k_I + 1) * (k_I + 2) / (k_I^2 * mu^3)

    m_PI = m_P * m_I
    m_PPI = m_PP * m_I
    m_PII = m_P * m_II

    m_star = (C / 2) * (alpha * m_PP + 2 * m_PI + m_II)
    v_star = (C / 3) * (alpha * m_PPP + 3 * m_PPI + 3 * m_PII + m_III) - m_star^2

    m_gen = m_E + m_star
    v_gen = v_E + v_star
    s_gen = sqrt(v_gen)

    m_gen, s_gen
end

function get_gen_mean_sd_mech(params_mat::AbstractMatrix)
    n = size(params_mat, 1)
    m_gen = zeros(n)
    s_gen = zeros(n)
    for i in eachindex(m_gen)
        m_gen[i], s_gen[i] = get_gen_mean_sd_mech(params_mat[i, :])
    end
    m_gen, s_gen
end
