"""
Household log-likelihood for the mechanistic model.

Translates Functions/Mech/log_likelihood_household_mech.m
"""

"""
    log_likelihood_household_mech(f_inc, b_cond, B_cond, mean_transmissions, aug)

Calculate the per-household log-likelihood contributions for the mechanistic
model.

# Arguments
- `f_inc`: incubation period PDF (vectorised)
- `b_cond`: conditional infectiousness function b(tost, t_inc, hh_size, asymp)
- `B_cond`: integrated infectiousness function
- `mean_transmissions`: total expected transmissions function
- `aug`: AugmentedData struct
"""
function log_likelihood_household_mech(f_inc, b_cond, B_cond, mean_transmissions,
                                       aug::AugmentedData)
    t_i_dir = aug.t_i_dir
    t_s_dir = aug.t_s_dir
    no_hosts = length(t_i_dir)

    primary_dir = aug.observed.primary_dir
    infected_dir = aug.observed.infected_dir
    uninfected_dir = .!infected_dir
    asymp_dir = aug.asymp_dir

    household_indicator_mat = aug.observed.household_indicator_mat

    poss = aug.observed.poss_infectors_dir
    v = poss.all
    M_from = poss.from_indicator_mat
    M_to = poss.to_indicator_mat
    to_uninfected_v = .!poss.to_infected_indicator
    to_recipient_v = poss.to_recipient_indicator
    household_size_v = poss.household_size

    from_asymp_v = BitVector(Bool.(M_from * Float64.(asymp_dir)))

    # Incubation period contribution (all infected)
    l1_indiv = zeros(no_hosts)
    t_inc_inf = t_s_dir[infected_dir] .- t_i_dir[infected_dir]
    l1_indiv[infected_dir] .= log.(f_inc(t_inc_inf))

    # TOST and incubation period for each possible infector-infectee pair
    t_tost_contribs = M_to * t_i_dir .- M_from * t_s_dir
    t_inc_contribs = M_from * (t_s_dir .- t_i_dir)

    t_tost_recip = t_tost_contribs[to_recipient_v]
    t_inc_recip = t_inc_contribs[to_recipient_v]
    hh_size_recip = household_size_v[to_recipient_v]
    from_asymp_recip = from_asymp_v[to_recipient_v]

    t_inc_uninf = t_inc_contribs[to_uninfected_v]
    hh_size_uninf = household_size_v[to_uninfected_v]
    from_asymp_uninf = from_asymp_v[to_uninfected_v]

    # Transmission contribution
    nv = length(v)
    bcond_vals = b_cond(t_tost_recip, t_inc_recip,
                        hh_size_recip, from_asymp_recip)
    T = eltype(bcond_vals)
    L2a_contribs = zeros(T, nv)
    L2a_contribs[to_recipient_v] .= bcond_vals

    L2a = M_to' * L2a_contribs
    L2a[primary_dir .| uninfected_dir] .= one(T)
    l2a = log.(L2a)

    # Evasion contribution
    l2b_contribs = zeros(T, nv)
    l2b_contribs[to_recipient_v] .= B_cond(t_tost_recip, t_inc_recip,
                                            hh_size_recip, from_asymp_recip)
    l2b_contribs[to_uninfected_v] .= mean_transmissions(t_inc_uninf,
                                                         hh_size_uninf,
                                                         from_asymp_uninf)
    l2b = -(M_to' * l2b_contribs)

    # Per-individual and per-household log-likelihood
    l_indiv = l1_indiv .+ l2a .+ l2b
    l_household = collect(household_indicator_mat' * l_indiv)

    return l_household
end
