"""
Update infection times of all infected hosts (mechanistic model).

Based on [update_infection_fun_mech.m](https://github.com/will-s-hart/UK-generation-times/blob/main/Functions/MCMC/update_infection_fun_mech.m) from the original MATLAB implementation.
"""

"""
    update_infection_mech!(theta, aug, ll_household, ll_household_form,
                           t_i_prop_sd)

Propose new infection times for one randomly chosen infected host per
household, accept/reject per household.
"""
function update_infection_mech!(theta, aug::AugmentedData, ll_household,
                                ll_household_form, t_i_prop_sd)
    obs = aug.observed
    household_sizes = obs.household_sizes_incl
    no_households = length(household_sizes)
    no_infected_in_household = obs.no_infected_in_household
    no_symp_in_household = obs.no_symp_in_household

    # Select one infected host per household
    update_hosts_in_hh = ceil.(Int, no_infected_in_household .* rand(no_households))
    household_start = cumsum(household_sizes) .- household_sizes
    update_hosts = household_start .+ update_hosts_in_hh

    t_i_prop = copy(aug.t_i)
    t_i_prop[update_hosts] .+= t_i_prop_sd .* randn(no_households)

    aug_new, ll_new, accept_hh = _propose_infection_times(
        theta, aug, ll_household, ll_household_form, t_i_prop, aug.t_s)

    # Track acceptance for symptomatic vs asymptomatic updates
    update_symp = update_hosts_in_hh .<= no_symp_in_household
    update_asymp = .!update_symp

    acc_symp = any(update_symp) ? mean(accept_hh[update_symp]) : NaN
    acc_asymp = any(update_asymp) ? mean(accept_hh[update_asymp]) : NaN

    acceptance = (overall=mean(accept_hh), symp=acc_symp, asymp=acc_asymp)

    aug_new, ll_new, acceptance
end
