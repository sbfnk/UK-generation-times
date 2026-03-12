"""
Update infection times of asymptomatic hosts (independent model).

Translates Functions/MCMC/update_asymp_fun_indep.m
"""

"""
    update_asymp_indep!(theta, aug, ll_household, ll_household_form,
                        t_i_prop_sd_asymp)

Propose new infection times for one randomly chosen asymptomatic host per
household (where applicable), accept/reject per household.
"""
function update_asymp_indep!(theta, aug::AugmentedData, ll_household,
                             ll_household_form, t_i_prop_sd_asymp)
    obs = aug.observed
    no_infected_in_household = obs.no_infected_in_household
    no_symp_in_household = obs.no_symp_in_household
    no_asymp_in_household = obs.no_asymp_in_household
    asymp_in_household = obs.asymp_in_household

    n_with_asymp = sum(asymp_in_household)
    if n_with_asymp == 0
        acceptance = (overall=NaN, infection=NaN)
        return aug, ll_household, acceptance
    end

    update_hosts = _select_asymp_hosts(obs, asymp_in_household, n_with_asymp,
                                       no_symp_in_household, no_asymp_in_household)

    t_i_prop = copy(aug.t_i)
    t_i_prop[update_hosts] .+= t_i_prop_sd_asymp .* randn(n_with_asymp)

    aug_new, ll_new, accept_hh = _propose_infection_times(
        theta, aug, ll_household, ll_household_form, t_i_prop, aug.t_s)

    nontrivial = asymp_in_household .& (no_infected_in_household .> 1)
    acc_rate = any(nontrivial) ? mean(accept_hh[nontrivial]) : NaN
    acceptance = (overall=acc_rate, infection=acc_rate)

    aug_new, ll_new, acceptance
end
