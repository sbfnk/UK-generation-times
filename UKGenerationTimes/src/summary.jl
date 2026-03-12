"""
Empirical summary statistics and presymptomatic transmission probabilities.

Translates:
- Functions/Indep/empirical_summary_indep.m
- Functions/Mech/empirical_summary_mech.m
- Functions/Indep/get_presymp_trans_probs_indep_logn.m
"""

using Distributions, Statistics

"""
    empirical_summary_indep(beta0, rho, x_A, f_gen, aug)

Estimate mean and SD of realised household generation times, and the
proportion of presymptomatic transmissions, for the independent model.
"""
function empirical_summary_indep(beta0, rho, x_A, f_gen, aug::AugmentedData)
    t_i_dir = aug.t_i_dir
    t_s_dir = aug.t_s_dir
    symp_dir = aug.symp_dir
    asymp_dir = aug.asymp_dir

    poss = aug.observed.poss_infectors_dir
    v = poss.all
    M_from = poss.from_indicator_mat
    M_to = poss.to_indicator_mat
    to_primary_v = poss.to_primary_indicator
    to_recipient_v = poss.to_recipient_indicator
    household_size_v = poss.household_size

    from_symp_v = BitVector(Bool.(M_from * Float64.(symp_dir)))
    from_asymp_v = BitVector(Bool.(M_from * Float64.(asymp_dir)))

    beta_v = beta0 ./ (household_size_v .^ rho)
    beta_v[to_primary_v] .= 0.0
    beta_v[from_asymp_v] .*= x_A

    t_gen_contribs = (M_to - M_from) * t_i_dir
    t_inc_contribs = M_from * (t_s_dir .- t_i_dir)

    t_gen_recip = t_gen_contribs[to_recipient_v]
    beta_recip = beta_v[to_recipient_v]

    nv = length(v)
    L2a_contribs = zeros(nv)
    L2a_contribs[to_recipient_v] .= beta_recip .* f_gen(t_gen_recip)
    L2a = M_to' * L2a_contribs

    tost_mask = from_symp_v .& to_recipient_v
    t_tost = t_gen_contribs[tost_mask] .- t_inc_contribs[tost_mask]

    return _compute_summary_stats(L2a_contribs, M_to, L2a, t_gen_recip,
                                  to_recipient_v, tost_mask, t_tost)
end

"""
    empirical_summary_mech(b_cond, aug)

Estimate mean and SD of realised household generation times, and the
proportion of presymptomatic transmissions, for the mechanistic model.
"""
function empirical_summary_mech(b_cond, aug::AugmentedData)
    t_i_dir = aug.t_i_dir
    t_s_dir = aug.t_s_dir
    symp_dir = aug.symp_dir
    asymp_dir = aug.asymp_dir

    poss = aug.observed.poss_infectors_dir
    v = poss.all
    M_from = poss.from_indicator_mat
    M_to = poss.to_indicator_mat
    to_recipient_v = poss.to_recipient_indicator
    household_size_v = poss.household_size

    from_symp_v = BitVector(Bool.(M_from * Float64.(symp_dir)))
    from_asymp_v = BitVector(Bool.(M_from * Float64.(asymp_dir)))

    t_gen_contribs = (M_to - M_from) * t_i_dir
    t_tost_contribs = M_to * t_i_dir .- M_from * t_s_dir
    t_inc_contribs = M_from * (t_s_dir .- t_i_dir)

    t_gen_recip = t_gen_contribs[to_recipient_v]
    t_tost_recip = t_tost_contribs[to_recipient_v]
    t_inc_recip = t_inc_contribs[to_recipient_v]
    hh_size_recip = household_size_v[to_recipient_v]
    from_asymp_recip = from_asymp_v[to_recipient_v]

    nv = length(v)
    L2a_contribs = zeros(nv)
    L2a_contribs[to_recipient_v] .= b_cond(t_tost_recip, t_inc_recip,
                                            hh_size_recip, from_asymp_recip)
    L2a = M_to' * L2a_contribs

    tost_mask = from_symp_v .& to_recipient_v
    t_tost = t_tost_contribs[tost_mask]

    return _compute_summary_stats(L2a_contribs, M_to, L2a, t_gen_recip,
                                  to_recipient_v, tost_mask, t_tost)
end

"""
Compute weighted mean, SD, and presymptomatic proportion from unnormalised
per-pair likelihood contributions.

- `L2a_contribs`: unnormalised weight for every possible infector-infectee pair
- `M_to`:         matrix mapping pairs to infectees
- `L2a`:          per-infectee sum of `L2a_contribs` (= `M_to' * L2a_contribs`)
- `t_gen_recip`:  generation times for recipient pairs
- `to_recipient_v`: mask selecting recipient pairs
- `tost_mask`:    mask selecting symptomatic-infector recipient pairs
- `t_tost`:       TOST values for those pairs
"""
function _compute_summary_stats(L2a_contribs, M_to, L2a, t_gen_recip,
                                 to_recipient_v, tost_mask, t_tost)
    weights_all = L2a_contribs ./ (M_to * L2a)
    weights_gen = weights_all[to_recipient_v]
    weights_tost = weights_all[tost_mask]

    weights_gen ./= sum(weights_gen)
    weights_tost ./= sum(weights_tost)

    m = sum(t_gen_recip .* weights_gen)
    s = sqrt(sum(t_gen_recip .^ 2 .* weights_gen) - m^2)
    p = sum((t_tost .< 0) .* weights_tost)

    return [m, s, p]
end

"""
    get_presymp_trans_probs_indep_logn(gen_mu_vec, gen_sigma_vec, F_inc)

Numerically estimate the proportion of presymptomatic transmissions for the
independent model, for each set of lognormal generation time parameters.
"""
function get_presymp_trans_probs_indep_logn(gen_mu_vec, gen_sigma_vec, F_inc)
    t_max = 50.0
    dt = 0.1
    t_vec = collect(range(dt / 2, t_max - dt / 2; step=dt))

    F_inc_vec = F_inc.(t_vec)

    p_vec = zeros(length(gen_mu_vec))
    for i in eachindex(gen_mu_vec)
        f_gen_vec = pdf.(LogNormal(gen_mu_vec[i], gen_sigma_vec[i]), t_vec)
        p_vec[i] = 1 - dot(f_gen_vec, F_inc_vec) * dt
    end

    return p_vec
end
