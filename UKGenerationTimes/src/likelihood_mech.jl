"""
Household log-likelihood for the mechanistic model.

The structure mirrors the independent model (see `likelihood_indep.jl`) but
replaces the generation time PDF/CDF with the conditional infectiousness profile
from the E→P→I staged infection model:

1. **Incubation period** (l1): log f_inc(t_onset - t_infection) for *all* infected
   hosts (not just symptomatic, since the mechanistic model draws incubation
   periods for asymptomatic hosts too).

2. **Transmission** (l2a): for each non-primary infectee, sum over possible
   infectors of b_cond(tost, t_inc_infector, hh_size, asymp). This is the
   instantaneous infectiousness of the infector at the time of transmission,
   conditional on the infector's incubation period.

3. **Evasion** (l2b): cumulative infectiousness B_cond for infected pairs, or
   total expected transmissions (mean_transmissions) for uninfected pairs.
"""

"""
    log_likelihood_household_mech(f_inc, b_cond, B_cond, mean_transmissions, aug)

Per-household log-likelihood for the mechanistic model.

# Arguments
- `f_inc`:  incubation period PDF (Gamma), vectorised
- `b_cond`: instantaneous infectiousness b(tost, t_inc, hh_size, asymp)
- `B_cond`: cumulative infectiousness ∫b from -∞ to tost
- `mean_transmissions`: total expected transmissions over infection course
- `aug`:    augmented data (latent infection/onset times + observed structure)
"""
function log_likelihood_household_mech(f_inc, b_cond, B_cond, mean_transmissions,
                                       aug::AugmentedData)
    t_i_dir = aug.t_i_dir       # infection times (infection order)
    t_s_dir = aug.t_s_dir       # symptom onset times (infection order)
    no_hosts = length(t_i_dir)

    primary_dir = aug.observed.primary_dir
    infected_dir = aug.observed.infected_dir
    uninfected_dir = .!infected_dir
    asymp_dir = aug.asymp_dir

    household_indicator_mat = aug.observed.household_indicator_mat

    # Possible-infector structure
    poss = aug.observed.poss_infectors_dir
    v = poss.all
    M_from = poss.from_indicator_mat      # maps pairs → infector index
    M_to = poss.to_indicator_mat          # maps pairs → infectee index
    to_uninfected_v = .!poss.to_infected_indicator
    to_recipient_v = poss.to_recipient_indicator
    household_size_v = poss.household_size

    from_asymp_v = M_from * asymp_dir .> 0

    # --- Component 1: incubation period (all infected individuals) ---
    l1_indiv = zeros(no_hosts)
    t_inc_inf = t_s_dir[infected_dir] .- t_i_dir[infected_dir]
    l1_indiv[infected_dir] .= log.(f_inc(t_inc_inf))

    # TOST = time from infector's symptom onset to infectee's infection
    # t_inc = infector's incubation period
    t_tost_contribs = M_to * t_i_dir .- M_from * t_s_dir
    t_inc_contribs = M_from * (t_s_dir .- t_i_dir)

    # Subset for recipient (non-primary infected) pairs
    t_tost_recip = t_tost_contribs[to_recipient_v]
    t_inc_recip = t_inc_contribs[to_recipient_v]
    hh_size_recip = household_size_v[to_recipient_v]
    from_asymp_recip = from_asymp_v[to_recipient_v]

    # Subset for uninfected pairs (survived entire exposure)
    t_inc_uninf = t_inc_contribs[to_uninfected_v]
    hh_size_uninf = household_size_v[to_uninfected_v]
    from_asymp_uninf = from_asymp_v[to_uninfected_v]

    # --- Component 2: transmission (conditional infectiousness at time of infection) ---
    nv = length(v)
    bcond_vals = b_cond(t_tost_recip, t_inc_recip,
                        hh_size_recip, from_asymp_recip)
    T = eltype(bcond_vals)
    L2a_contribs = zeros(T, nv)
    L2a_contribs[to_recipient_v] .= bcond_vals

    # Sum pair-level contributions to infectee level
    L2a = M_to' * L2a_contribs
    L2a[primary_dir .| uninfected_dir] .= one(T)
    l2a = log.(L2a)

    # --- Component 3: evasion (cumulative exposure without infection) ---
    # For infected pairs: cumulative infectiousness up to time of infection
    # For uninfected pairs: total expected transmissions over full infection
    l2b_contribs = zeros(T, nv)
    l2b_contribs[to_recipient_v] .= B_cond(t_tost_recip, t_inc_recip,
                                            hh_size_recip, from_asymp_recip)
    l2b_contribs[to_uninfected_v] .= mean_transmissions(t_inc_uninf,
                                                         hh_size_uninf,
                                                         from_asymp_uninf)
    l2b = -(M_to' * l2b_contribs)

    # Sum the three components per individual, then aggregate to household level
    l_indiv = l1_indiv .+ l2a .+ l2b
    l_household = collect(household_indicator_mat' * l_indiv)

    l_household
end
