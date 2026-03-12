"""
Update infection times of symptomatic hosts (independent model).

Translates Functions/MCMC/update_infection_fun_indep.m
"""

"""
    update_infection_indep!(theta, aug, ll_household, ll_household_form,
                            t_i_prop_sd_symp)

Propose new infection times for one randomly chosen symptomatic host per
household, accept/reject per household. Returns (aug_new, ll_new, acceptance).
"""
function update_infection_indep!(theta, aug::AugmentedData, ll_household,
                                 ll_household_form, t_i_prop_sd_symp)
    obs = aug.observed
    household_sizes = obs.household_sizes_incl
    no_households = length(household_sizes)
    symp_in_household = obs.symp_in_household
    no_symp_in_household = obs.no_symp_in_household

    # Select one symptomatic host per household (for households with symptomatic hosts)
    n_with_symp = sum(symp_in_household)
    update_hosts_in_hh = ceil.(Int, no_symp_in_household[symp_in_household] .* rand(n_with_symp))
    household_start = cumsum(household_sizes) .- household_sizes
    update_hosts = household_start[symp_in_household] .+ update_hosts_in_hh

    t_i_prop = copy(aug.t_i)
    t_i_prop[update_hosts] .+= t_i_prop_sd_symp .* randn(n_with_symp)

    aug_new, ll_new, accept_hh = _propose_infection_times(
        theta, aug, ll_household, ll_household_form, t_i_prop, aug.t_s)

    acc_rate = any(symp_in_household) ? mean(accept_hh[symp_in_household]) : 0.0
    acceptance = (overall=acc_rate, symp=acc_rate, asymp=NaN)

    aug_new, ll_new, acceptance
end

"""
Select one asymptomatic host per household (for households that have one).

Returns a vector of global host indices to update.
"""
function _select_asymp_hosts(obs::ObservedData, asymp_in_household::BitVector,
                              n_with_asymp::Int, no_symp_in_household::Vector{Int},
                              no_asymp_in_household::Vector{Int})
    household_sizes = obs.household_sizes_incl
    update_hosts_in_hh = ceil.(Int, no_asymp_in_household[asymp_in_household] .* rand(n_with_asymp))
    household_start = cumsum(household_sizes) .- household_sizes
    household_asymp_start = household_start .+ no_symp_in_household
    household_asymp_start[asymp_in_household] .+ update_hosts_in_hh
end

"""
Apply accepted infection time updates, merging old and new augmented data.

`update_t_s` controls whether `t_s` (symptom-order onset times) is also updated
from the proposal. Pass `true` for the mechanistic asymptomatic update, which
shifts both `t_i` and `t_s` jointly.
"""
function _apply_infection_update(aug_old::AugmentedData, aug_prop::AugmentedData,
                                 accept_hosts::BitVector; update_t_s::Bool=false)
    t_i_new = copy(aug_old.t_i)
    t_i_new[accept_hosts] .= aug_prop.t_i[accept_hosts]

    t_s_new = update_t_s ? copy(aug_old.t_s) : aug_old.t_s
    if update_t_s
        t_s_new[accept_hosts] .= aug_prop.t_s[accept_hosts]
    end

    t_i_dir_new = copy(aug_old.t_i_dir)
    t_i_dir_new[accept_hosts] .= aug_prop.t_i_dir[accept_hosts]

    t_s_dir_new = copy(aug_old.t_s_dir)
    t_s_dir_new[accept_hosts] .= aug_prop.t_s_dir[accept_hosts]

    t_dir_host_inds_new = copy(aug_old.t_dir_host_inds)
    t_dir_host_inds_new[accept_hosts] .= aug_prop.t_dir_host_inds[accept_hosts]

    symp_dir_new = copy(aug_old.symp_dir)
    symp_dir_new[accept_hosts] .= aug_prop.symp_dir[accept_hosts]

    asymp_dir_new = copy(aug_old.asymp_dir)
    asymp_dir_new[accept_hosts] .= aug_prop.asymp_dir[accept_hosts]

    AugmentedData(aug_old.observed, t_i_new, t_s_new,
                  t_i_dir_new, t_s_dir_new,
                  t_dir_host_inds_new, symp_dir_new, asymp_dir_new)
end

"""
Shared proposal/accept/reject logic for infection time updates.

Proposes new infection times `t_i_prop` (and matching `t_s_prop`), checks
bounds, sorts within households, and applies per-household MH acceptance.
Returns `(aug_new, ll_new, accept_hh)`.

When `update_t_s=true` the accepted households also have their `t_s` replaced
from the proposal (needed for the mechanistic asymptomatic update).
"""
function _propose_infection_times(theta, aug::AugmentedData, ll_household,
                                  ll_household_form, t_i_prop, t_s_prop;
                                  update_t_s::Bool=false)
    obs = aug.observed
    no_households = length(obs.household_sizes_incl)
    household_indicator_mat = obs.household_indicator_mat

    # Check bounds
    t_i_bdry_hosts = (t_i_prop .< obs.t_iL) .| (t_i_prop .> obs.t_iR)
    t_i_bdry_hh = BitVector(Bool.(household_indicator_mat' * Float64.(t_i_bdry_hosts)))

    # Sort proposed data by infection time within households
    t_dir_host_inds_prop = _sort_by_household_and_time(obs.household_no, t_i_prop)
    t_i_dir_prop = t_i_prop[t_dir_host_inds_prop]
    t_s_dir_prop = t_s_prop[t_dir_host_inds_prop]
    symp_dir_prop = obs.symp[t_dir_host_inds_prop]
    asymp_dir_prop = obs.asymp[t_dir_host_inds_prop]

    aug_prop = AugmentedData(obs, t_i_prop, t_s_prop,
                             t_i_dir_prop, t_s_dir_prop,
                             t_dir_host_inds_prop, symp_dir_prop, asymp_dir_prop)

    ll_household_prop = ll_household_form(theta, aug_prop)
    ll_household_prop[t_i_bdry_hh] .= -Inf

    # Per-household accept/reject
    la_vec = ll_household_prop .- ll_household
    accept_hh = log.(rand(no_households)) .< la_vec
    accept_hosts = BitVector(Bool.(household_indicator_mat * Float64.(accept_hh)))

    aug_new = _apply_infection_update(aug, aug_prop, accept_hosts; update_t_s)
    ll_new = copy(ll_household)
    ll_new[accept_hh] .= ll_household_prop[accept_hh]

    aug_new, ll_new, accept_hh
end
