"""
Household log-likelihood for the independent transmission and symptoms model.

# Likelihood structure

Each household's log-likelihood has three additive components per individual:

1. **Incubation period** (l1): log f_inc(t_onset - t_infection) for symptomatic hosts.
   How well the augmented infection/onset times match the assumed incubation distribution.

2. **Transmission** (l2a): for each non-primary infected individual, log of the
   sum over possible infectors of β * f_gen(t_gen). The generation time density
   evaluated at the time between infector's and infectee's infection times,
   weighted by the transmission rate.

3. **Evasion** (l2b): for each individual, the negative cumulative hazard of
   infection from all potential infectors. This captures the probability that
   the individual avoided infection until the time they were actually infected
   (or remained uninfected throughout).

The sparse matrices M_from and M_to map between the flattened list of all
possible infector-infectee pairs and the individual-level arrays. M_to' sums
contributions back to the infectee level.
"""

"""
    log_likelihood_household_indep(f_inc, beta0, rho, x_A, f_gen, F_gen, aug)

Per-household log-likelihood for the independent model.

# Arguments
- `f_inc`:  incubation period PDF, vectorised over times
- `beta0`:  baseline per-contact transmission rate
- `rho`:    household-size frequency dependence exponent
- `x_A`:    relative infectiousness of asymptomatic infectors
- `f_gen`:  generation time PDF, vectorised over times
- `F_gen`:  generation time CDF, vectorised over times
- `aug`:    augmented data (latent infection/onset times + observed structure)
"""
function log_likelihood_household_indep(f_inc, beta0, rho, x_A, f_gen, F_gen,
                                        aug::AugmentedData)
    t_i_dir = aug.t_i_dir       # infection times (sorted by infection order)
    t_s_dir = aug.t_s_dir       # symptom onset times (infection order)
    no_hosts = length(t_i_dir)

    infected_dir = aug.observed.infected_dir
    uninfected_dir = .!infected_dir
    symp_dir = aug.symp_dir
    asymp_dir = aug.asymp_dir
    primary_dir = aug.observed.primary_dir

    household_indicator_mat = aug.observed.household_indicator_mat

    # Possible-infector structure: one entry per (infector, infectee) pair
    poss = aug.observed.poss_infectors_dir
    v = poss.all                          # flat list of pair indices
    M_from = poss.from_indicator_mat      # maps pairs → infector index
    M_to = poss.to_indicator_mat          # maps pairs → infectee index
    to_uninfected_v = .!poss.to_infected_indicator
    to_primary_v = poss.to_primary_indicator
    to_recipient_v = poss.to_recipient_indicator  # non-primary infected
    household_size_v = poss.household_size

    from_asymp_v = M_from * asymp_dir .> 0

    T = float(typeof(beta0))

    # Per-pair transmission rate: β₀ / hh_size^ρ, zeroed for primary cases,
    # reduced by x_A for asymptomatic infectors
    beta_v = beta0 ./ (household_size_v .^ rho)
    beta_v[to_primary_v] .= zero(T)
    beta_v[from_asymp_v] .*= x_A

    # --- Component 1: incubation period ---
    # log f_inc(t_onset - t_infection) for symptomatic individuals
    l1_indiv = zeros(no_hosts)
    if any(symp_dir)
        t_inc_symp = t_s_dir[symp_dir] .- t_i_dir[symp_dir]
        l1_indiv[symp_dir] .= log.(f_inc(t_inc_symp))
    end

    # Generation times: t_infectee - t_infector for each pair
    t_gen_contribs = (M_to - M_from) * t_i_dir

    t_gen_recip = t_gen_contribs[to_recipient_v]
    beta_recip = beta_v[to_recipient_v]
    beta_uninf = beta_v[to_uninfected_v]

    # --- Component 2: transmission (who infected whom) ---
    # For each non-primary infectee, sum β * f_gen(t_gen) over possible infectors.
    # The infectee's contribution is log of this sum.
    nv = length(v)
    L2a_contribs = zeros(T, nv)
    L2a_contribs[to_recipient_v] .= beta_recip .* f_gen(t_gen_recip)

    # Sum pair-level contributions to infectee level
    L2a = M_to' * L2a_contribs
    L2a[primary_dir .| uninfected_dir] .= one(T)  # no transmission term for these
    l2a = log.(L2a)

    # --- Component 3: evasion (survival/escape from infection) ---
    # Cumulative hazard: β * F_gen(t_gen) for infected pairs, β * 1 for uninfected
    # (uninfected survived the entire infectious period of each potential infector)
    l2b_contribs = zeros(T, nv)
    l2b_contribs[to_recipient_v] .= beta_recip .* F_gen(t_gen_recip)
    l2b_contribs[to_uninfected_v] .= beta_uninf
    l2b = -(M_to' * l2b_contribs)

    # Sum the three components per individual, then aggregate to household level
    l_indiv = l1_indiv .+ l2a .+ l2b
    l_household = collect(household_indicator_mat' * l_indiv)

    l_household
end
