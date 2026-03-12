"""
Household log-likelihood for the independent transmission and symptoms model.

Translates Functions/Indep/log_likelihood_household_indep.m
"""

"""
    log_likelihood_household_indep(f_inc, beta0, rho, x_A, f_gen, F_gen, aug)

Calculate the per-household log-likelihood contributions for the independent
transmission and symptoms model.

# Arguments
- `f_inc`: incubation period PDF (vectorised)
- `beta0`: baseline transmission rate
- `rho`: household size scaling exponent
- `x_A`: relative infectiousness of asymptomatic hosts
- `f_gen`: generation time PDF (vectorised)
- `F_gen`: generation time CDF (vectorised)
- `aug`: AugmentedData struct
"""
function log_likelihood_household_indep(f_inc, beta0, rho, x_A, f_gen, F_gen,
                                        aug::AugmentedData)
    t_i_dir = aug.t_i_dir
    t_s_dir = aug.t_s_dir
    no_hosts = length(t_i_dir)

    infected_dir = aug.observed.infected_dir
    uninfected_dir = .!infected_dir
    symp_dir = aug.symp_dir
    asymp_dir = aug.asymp_dir
    primary_dir = aug.observed.primary_dir

    household_indicator_mat = aug.observed.household_indicator_mat

    poss = aug.observed.poss_infectors_dir
    v = poss.all
    M_from = poss.from_indicator_mat
    M_to = poss.to_indicator_mat
    to_uninfected_v = .!poss.to_infected_indicator
    to_primary_v = poss.to_primary_indicator
    to_recipient_v = poss.to_recipient_indicator
    household_size_v = poss.household_size

    from_asymp_v = M_from * asymp_dir .> 0

    T = float(typeof(beta0))

    # Per-pair transmission rate
    beta_v = beta0 ./ (household_size_v .^ rho)
    beta_v[to_primary_v] .= zero(T)
    beta_v[from_asymp_v] .*= x_A

    # Incubation period contribution (symptomatic only)
    l1_indiv = zeros(no_hosts)
    if any(symp_dir)
        t_inc_symp = t_s_dir[symp_dir] .- t_i_dir[symp_dir]
        l1_indiv[symp_dir] .= log.(f_inc(t_inc_symp))
    end

    # Generation time contributions
    t_gen_contribs = (M_to - M_from) * t_i_dir

    t_gen_recip = t_gen_contribs[to_recipient_v]
    beta_recip = beta_v[to_recipient_v]
    beta_uninf = beta_v[to_uninfected_v]

    # Transmission contribution
    nv = length(v)
    L2a_contribs = zeros(T, nv)
    L2a_contribs[to_recipient_v] .= beta_recip .* f_gen(t_gen_recip)

    L2a = M_to' * L2a_contribs
    L2a[primary_dir .| uninfected_dir] .= one(T)
    l2a = log.(L2a)

    # Evasion contribution
    l2b_contribs = zeros(T, nv)
    l2b_contribs[to_recipient_v] .= beta_recip .* F_gen(t_gen_recip)
    l2b_contribs[to_uninfected_v] .= beta_uninf
    l2b = -(M_to' * l2b_contribs)

    # Per-individual and per-household log-likelihood
    l_indiv = l1_indiv .+ l2a .+ l2b
    l_household = collect(household_indicator_mat' * l_indiv)

    l_household
end
