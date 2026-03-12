"""
Update symptom onset times for symptomatic hosts.

Translates Functions/MCMC/update_onset_fun.m
"""

using Random

"""
    update_onset!(theta, aug, ll_household, ll_household_form)

Resample onset times from Uniform(t_sL, t_sR) for all symptomatic hosts,
accept/reject per household.
"""
function update_onset!(theta, aug::AugmentedData, ll_household,
                       ll_household_form)
    obs = aug.observed
    household_sizes = obs.household_sizes_incl
    no_households = length(household_sizes)
    household_indicator_mat = obs.household_indicator_mat
    symp_in_household = obs.symp_in_household

    # Propose new onset times for all symptomatic hosts
    t_s_prop = copy(aug.t_s)
    r = rand(sum(obs.symp))
    t_s_prop[obs.symp] .= obs.t_sL[obs.symp] .+
        (obs.t_sR[obs.symp] .- obs.t_sL[obs.symp]) .* r

    # Reorder in infection-time order
    t_s_dir_prop = t_s_prop[aug.t_dir_host_inds]

    aug_prop = AugmentedData(obs, aug.t_i, t_s_prop,
                             aug.t_i_dir, t_s_dir_prop,
                             aug.t_dir_host_inds, aug.symp_dir, aug.asymp_dir)

    ll_household_prop = ll_household_form(theta, aug_prop)

    # Per-household accept/reject
    la_vec = ll_household_prop .- ll_household
    accept_hh = log.(rand(no_households)) .< la_vec
    accept_hosts = BitVector(Bool.(household_indicator_mat * Float64.(accept_hh)))

    # Apply updates
    t_s_new = copy(aug.t_s)
    t_s_new[accept_hosts] .= t_s_prop[accept_hosts]

    t_s_dir_new = copy(aug.t_s_dir)
    t_s_dir_new[accept_hosts] .= t_s_dir_prop[accept_hosts]

    aug_new = AugmentedData(obs, aug.t_i, t_s_new,
                            aug.t_i_dir, t_s_dir_new,
                            aug.t_dir_host_inds, aug.symp_dir, aug.asymp_dir)

    ll_new = copy(ll_household)
    ll_new[accept_hh] .= ll_household_prop[accept_hh]

    acc_rate = any(symp_in_household) ? mean(accept_hh[symp_in_household]) : 0.0
    acceptance = (overall=acc_rate,)

    return aug_new, ll_new, acceptance
end
