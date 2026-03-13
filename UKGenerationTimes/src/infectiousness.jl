"""
Infectiousness profile functions for the mechanistic model.

# Model overview

The mechanistic model uses a staged infection process:

    Exposed (E) → Presymptomatic (P) → Symptomatic Infectious (I)

The total incubation period (E + P) is Gamma-distributed with shape `k_inc` and
rate `k_inc * γ`. The exposed period has shape `k_E` and the presymptomatic
period has shape `k_P = k_inc - k_E`, both with the same rate `k_inc * γ`.
The symptomatic infectious period has shape `k_I` and rate `k_I * μ`.

Infectiousness starts during stage P (at relative level `α` compared to stage I)
and continues through stage I, giving a two-phase infectiousness profile. The
normalisation constant `W` ensures the profile integrates to the correct total
infectiousness over the infection course.

Household transmission follows a frequency-dependent contact model:
- `β₀` is the baseline per-contact transmission rate
- `ρ` scales transmission by household size: β = β₀ / (household_size ^ ρ)
- `x_A` scales transmission for asymptomatic infectors

Based on the following files from the original MATLAB implementation:
- [get_params_mech.m](https://github.com/will-s-hart/UK-generation-times/blob/main/Functions/Mech/get_params_mech.m)
- [b_cond_form_mech.m](https://github.com/will-s-hart/UK-generation-times/blob/main/Functions/Mech/b_cond_form_mech.m)
- [b_int_cond_form_mech.m](https://github.com/will-s-hart/UK-generation-times/blob/main/Functions/Mech/b_int_cond_form_mech.m)
- [mean_transmissions_form_mech.m](https://github.com/will-s-hart/UK-generation-times/blob/main/Functions/Mech/mean_transmissions_form_mech.m)
- [f_tost_form_mech.m](https://github.com/will-s-hart/UK-generation-times/blob/main/Functions/Mech/f_tost_form_mech.m)
- [get_gen_mean_sd_mech.m](https://github.com/will-s-hart/UK-generation-times/blob/main/Functions/Mech/get_gen_mean_sd_mech.m)
"""

"""
    MechParams

Named tuple holding the full mechanistic model parameter set.

# Fields
- `γ`:  incubation progression rate (shared across E and P stages)
- `μ`:  recovery rate (1/mean symptomatic infectious duration)
- `k_inc`: Erlang shape for total incubation period (E + P)
- `k_E`:   Erlang shape for exposed (latent, pre-infectious) period
- `k_I`:   Erlang shape for symptomatic infectious period
- `α`:  relative infectiousness during presymptomatic vs symptomatic stage
- `β₀`: baseline per-contact transmission rate
- `ρ`:  household-size frequency dependence exponent
- `x_A`:   relative infectiousness of asymptomatic cases
"""
const MechParams = NamedTuple{
    (:γ, :μ, :k_inc, :k_E, :k_I, :α, :β₀, :ρ, :x_A)}

"""
    mech_params_from_vector(v)

Construct a `MechParams` named tuple from a 9-element vector
[γ, μ, k_inc, k_E, k_I, α, β₀, ρ, x_A].
"""
function mech_params_from_vector(v)
    MechParams((v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9]))
end

"""
    get_params_mech(theta, params_known) -> MechParams

Recover the full parameter set from fitted and known parameters.

`theta = [p_E, 1/μ, α, β₀]` are the MCMC-estimated parameters.
`params_known = [k_inc, γ, k_I, ρ, x_A]` are fixed.
"""
function get_params_mech(theta, params_known)
    k_inc = params_known[1]
    γ     = params_known[2]
    k_I   = params_known[3]
    ρ     = params_known[4]
    x_A   = params_known[5]

    k_E = theta[1] * k_inc
    μ   = 1 / theta[2]
    α   = theta[3]
    β₀  = theta[4]

    MechParams((γ, μ, k_inc, k_E, k_I, α, β₀, ρ, x_A))
end

"""
    infectiousness_weight(p::MechParams) -> W

Normalisation constant ensuring the infectiousness profile (across stages P and I)
integrates to the correct total. Appears as `C` in Hart et al. (2022) eq. 3.
"""
function infectiousness_weight(p)
    k_P = p.k_inc - p.k_E
    p.k_inc * p.γ * p.μ / (p.α * k_P * p.μ + p.k_inc * p.γ)
end

"""
    individual_transmission_rate(β₀, ρ, x_A, household_size, asymp)

Per-individual transmission rate β, adjusted for household size and
asymptomatic status.
"""
function individual_transmission_rate(β₀, ρ, x_A, household_size, asymp)
    β = β₀ ./ (household_size .^ ρ)
    β[asymp] .*= x_A
    β
end

"""
    b_cond_mech(tost, t_inc, household_size, asymp, p)

Instantaneous infectiousness at time `tost` since symptom onset, conditional on
incubation period `t_inc`.

Before symptom onset (tost < 0): infectiousness comes from the presymptomatic
stage P, scaled by `α`. After onset (tost ≥ 0): from the symptomatic stage I.
"""
function b_cond_mech(tost, t_inc, household_size, asymp, p)
    (; γ, μ, k_inc, k_E, k_I, α, β₀, ρ, x_A) = p
    k_P = k_inc - k_E

    W = infectiousness_weight(p)
    β = individual_transmission_rate(β₀, ρ, x_A, household_size, asymp)

    result = zeros(eltype(β), length(tost))

    presymp = tost .< 0
    postsymp = .!presymp

    # Presymptomatic stage: fraction of P period already elapsed,
    # expressed via the Beta distribution (normalised Gamma ratio)
    presymp_shape = Beta(k_P, k_E)
    # Symptomatic stage: survival function of the infectious period
    inf_survival = Gamma(k_I, 1 / (k_I * μ))

    if any(presymp)
        t_pre = tost[presymp]
        t_inc_pre = t_inc[presymp]
        β_pre = β[presymp]
        # Fraction of presymptomatic period remaining at time tost before onset
        result[presymp] .= α * W .* β_pre .*
            (1 .- cdf.(Ref(presymp_shape), -t_pre ./ t_inc_pre))
    end

    if any(postsymp)
        t_post = tost[postsymp]
        β_post = β[postsymp]
        # Probability of still being infectious at time tost after onset
        result[postsymp] .= W .* β_post .*
            (1 .- cdf.(Ref(inf_survival), t_post))
    end

    result
end

"""
    b_int_cond_mech(tost, t_inc, household_size, asymp, p)

Cumulative infectiousness integrated from -∞ to `tost` since symptom onset,
conditional on incubation period `t_inc`.

Used for the evasion (survival) term in the household likelihood: the probability
that a susceptible has *not yet* been infected by this infector by time `tost`.
"""
function b_int_cond_mech(tost, t_inc, household_size, asymp, p)
    (; γ, μ, k_inc, k_E, k_I, α, β₀, ρ, x_A) = p
    k_P = k_inc - k_E

    W = infectiousness_weight(p)
    β = individual_transmission_rate(β₀, ρ, x_A, household_size, asymp)

    result = zeros(eltype(β), length(tost))

    presymp = tost .< 0
    postsymp = .!presymp

    presymp_shape = Beta(k_P, k_E)
    presymp_shape_shifted = Beta(k_P + 1, k_E)  # for integrating x * beta_pdf(x)
    inf_survival = Gamma(k_I, 1 / (k_I * μ))
    inf_survival_shifted = Gamma(k_I + 1, 1 / (k_I * μ))  # for mean residual life

    if any(presymp)
        t_pre = tost[presymp]
        t_inc_pre = t_inc[presymp]
        β_pre = β[presymp]

        # Two terms from integrating the presymptomatic infectiousness profile
        elapsed_fraction = α * W .* β_pre .* t_pre .*
            (1 .- cdf.(Ref(presymp_shape), -t_pre ./ t_inc_pre))
        mean_residual = α * W .* β_pre .* (k_P .* t_inc_pre ./ k_inc) .*
            (1 .- cdf.(Ref(presymp_shape_shifted), -t_pre ./ t_inc_pre))
        result[presymp] .= elapsed_fraction .+ mean_residual
    end

    if any(postsymp)
        t_post = tost[postsymp]
        t_inc_post = t_inc[postsymp]
        β_post = β[postsymp]

        # Total presymptomatic contribution (fully elapsed)
        total_presymp = W .* β_post .* α .* k_P .* t_inc_post ./ k_inc
        # Symptomatic contribution: time still infectious
        symp_remaining = W .* β_post .* t_post .*
            (1 .- cdf.(Ref(inf_survival), t_post))
        # Symptomatic contribution: expected time already recovered
        symp_recovered = (W .* β_post ./ μ) .*
            cdf.(Ref(inf_survival_shifted), t_post)
        result[postsymp] .= total_presymp .+ symp_remaining .+ symp_recovered
    end

    result
end

"""
    mean_transmissions_mech(t_inc, household_size, asymp, p)

Total expected number of transmissions over the entire infection course,
conditional on incubation period `t_inc`. This is the limit of
`b_int_cond_mech` as tost → ∞.
"""
function mean_transmissions_mech(t_inc, household_size, asymp, p)
    (; γ, μ, k_inc, k_E, k_I, α, β₀, ρ, x_A) = p
    k_P = k_inc - k_E

    β = individual_transmission_rate(β₀, ρ, x_A, household_size, asymp)

    γ .* β .* (α .* k_P .* μ .* t_inc .+ k_inc) ./
    (α .* k_P .* μ .+ k_inc .* γ)
end

"""
    f_tost_mech(tost, p)

TOST (time from onset of symptoms to transmission) density, marginalised over
the incubation period.

Before onset (tost < 0): probability of transmitting during the presymptomatic
stage, given by the survival function of the P-stage duration.
After onset (tost ≥ 0): probability of transmitting during the symptomatic
stage, given by the survival function of the I-stage duration.
"""
function f_tost_mech(tost, p)
    (; γ, μ, k_inc, k_E, k_I, α) = p
    k_P = k_inc - k_E

    W = infectiousness_weight(p)

    # P-stage duration distribution (presymptomatic)
    presymp_duration = Gamma(k_P, 1 / (k_inc * γ))
    # I-stage duration distribution (symptomatic infectious)
    symp_duration = Gamma(k_I, 1 / (k_I * μ))

    result = zeros(float(typeof(W)), length(tost))

    presymp = tost .< 0
    postsymp = .!presymp

    if any(presymp)
        # Survival function: probability P stage hasn't ended yet
        result[presymp] .= α * W .*
            (1 .- cdf.(Ref(presymp_duration), .-tost[presymp]))
    end

    if any(postsymp)
        # Survival function: probability I stage hasn't ended yet
        result[postsymp] .= W .*
            (1 .- cdf.(Ref(symp_duration), tost[postsymp]))
    end

    result
end

"""
    get_gen_mean_sd_mech(p) -> (mean, sd)

Analytical mean and standard deviation of the generation time distribution.

The generation time = time in E (latent) + time from becoming infectious to
transmitting. The second part ("star") is an infectiousness-weighted mixture
across the P and I stages. We compute its moments from the raw moments of the
P-stage and I-stage durations.
"""
function get_gen_mean_sd_mech(p)
    (; γ, μ, k_inc, k_E, k_I, α) = p
    k_P = k_inc - k_E
    W = infectiousness_weight(p)

    inc_rate = k_inc * γ  # common rate for E and P stages

    # E stage (latent period): Gamma(k_E, 1/inc_rate)
    mean_E = k_E / inc_rate
    var_E  = k_E / inc_rate^2

    # Raw moments of P-stage duration: Gamma(k_P, 1/inc_rate)
    # E[T^n] = k(k+1)...(k+n-1) / rate^n
    E_P    = k_P / inc_rate
    E_P_sq = k_P * (k_P + 1) / inc_rate^2
    E_P_cb = k_P * (k_P + 1) * (k_P + 2) / inc_rate^3

    # Raw moments of I-stage duration: Gamma(k_I, 1/(k_I * μ))
    E_I    = 1 / μ
    E_I_sq = (k_I + 1) / (k_I * μ^2)
    E_I_cb = (k_I + 1) * (k_I + 2) / (k_I^2 * μ^3)

    # Cross-moments (P and I are independent)
    E_PI   = E_P * E_I
    E_PsqI = E_P_sq * E_I
    E_PIsq = E_P * E_I_sq

    # "Star" distribution: infectiousness-weighted time from start of P to transmission.
    # Its moments come from combining P-stage (weighted by α) and I-stage contributions.
    mean_star = (W / 2) * (α * E_P_sq + 2 * E_PI + E_I_sq)
    var_star  = (W / 3) * (α * E_P_cb + 3 * E_PsqI + 3 * E_PIsq + E_I_cb) - mean_star^2

    # Generation time = E + star (independent sum)
    mean_gen = mean_E + mean_star
    sd_gen   = sqrt(var_E + var_star)

    mean_gen, sd_gen
end

function get_gen_mean_sd_mech(params_mat::AbstractMatrix)
    n = size(params_mat, 1)
    means = zeros(n)
    sds = zeros(n)
    for i in eachindex(means)
        means[i], sds[i] = get_gen_mean_sd_mech(mech_params_from_vector(params_mat[i, :]))
    end
    means, sds
end
