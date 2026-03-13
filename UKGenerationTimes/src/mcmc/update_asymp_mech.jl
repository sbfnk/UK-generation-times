"""
Update infection times and I-stage entry times of asymptomatic hosts
(mechanistic model).

Based on [update_asymp_fun_mech.m](https://github.com/will-s-hart/UK-generation-times/blob/main/Functions/MCMC/update_asymp_fun_mech.m) from the original MATLAB implementation.

In the mechanistic model, both t_i and t_s are shifted by the same amount
for asymptomatic hosts, preserving the "incubation period" (time between
infection and entry into the I stage).
"""

"""
    update_asymp_mech!(theta, aug, ll_household, ll_household_form,
                       t_prop_sd_asymp)

Propose joint shifts of infection and I-stage entry times for one randomly
chosen asymptomatic host per household, accept/reject per household.
"""
function update_asymp_mech!(theta, aug::AugmentedData, ll_household,
                            ll_household_form, t_prop_sd_asymp)
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

    # Propose joint shifts: same delta applied to both t_i and t_s
    t_updates = t_prop_sd_asymp .* randn(n_with_asymp)

    t_i_prop = copy(aug.t_i)
    t_s_prop = copy(aug.t_s)
    t_i_prop[update_hosts] .+= t_updates
    t_s_prop[update_hosts] .+= t_updates

    # update_t_s=true so accepted households also have t_s replaced
    aug_new, ll_new, accept_hh = _propose_infection_times(
        theta, aug, ll_household, ll_household_form, t_i_prop, t_s_prop;
        update_t_s=true)

    nontrivial = asymp_in_household .& (no_infected_in_household .> 1)
    acc_rate = any(nontrivial) ? mean(accept_hh[nontrivial]) : NaN
    acceptance = (overall=acc_rate, infection=NaN)

    aug_new, ll_new, acceptance
end
