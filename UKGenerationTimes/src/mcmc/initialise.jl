"""
Initialisation of augmented data for both models.

Based on the following files from the original MATLAB implementation:
- [initialise_augmented_data_indep.m](https://github.com/will-s-hart/UK-generation-times/blob/main/Functions/MCMC/initialise_augmented_data_indep.m)
- [initialise_augmented_data_mech.m](https://github.com/will-s-hart/UK-generation-times/blob/main/Functions/MCMC/initialise_augmented_data_mech.m)
"""

"""
    initialise_augmented_data_indep(obs::ObservedData) -> AugmentedData

Initialise augmented data for the independent model. Infection times of
symptomatic hosts are set to 6 days before onset; asymptomatic infection
times are drawn uniformly between the earliest and latest symptomatic onsets.
"""
function initialise_augmented_data_indep(obs::ObservedData)
    _initialise_augmented_data(obs; assign_asymp_onset=false)
end

"""
    initialise_augmented_data_mech(obs::ObservedData) -> AugmentedData

Initialise augmented data for the mechanistic model. Same as independent
model but also assigns "onset" times for asymptomatic hosts (time of entry
into the I stage).
"""
function initialise_augmented_data_mech(obs::ObservedData)
    _initialise_augmented_data(obs; assign_asymp_onset=true)
end

function _initialise_augmented_data(obs::ObservedData; assign_asymp_onset::Bool)
    no_hosts = length(obs.household_no)

    t_s = fill(Inf, no_hosts)
    t_i = fill(Inf, no_hosts)

    # Symptomatic: draw onset uniformly within [t_sL, t_sR], set infection 6 days before
    t_s[obs.symp] .= obs.t_sL[obs.symp] .+
        (obs.t_sR[obs.symp] .- obs.t_sL[obs.symp]) .* rand(sum(obs.symp))
    t_i[obs.symp] .= t_s[obs.symp] .- 6.0

    # Asymptomatic: draw infection time uniformly between min and max symptomatic onset
    if any(obs.symp)
        t_s_min = minimum(t_s[obs.symp])
        t_s_max = maximum(t_s[obs.symp])
        t_i[obs.asymp] .= t_s_min .+ (t_s_max - t_s_min) .* rand(sum(obs.asymp))
    else
        t_i[obs.asymp] .= rand(sum(obs.asymp))
    end

    # Clamp to bounds
    t_i .= max.(min.(t_i, obs.t_iR), obs.t_iL)

    # In the mechanistic model, asymptomatic hosts also have an "onset" time
    # (entry into the I stage), initialised 6 days after infection
    if assign_asymp_onset
        t_s[obs.asymp] .= t_i[obs.asymp] .+ 6.0
    end

    # Sort by infection time within households
    t_dir_host_inds = _sort_by_household_and_time(obs.household_no, t_i)
    t_i_dir = t_i[t_dir_host_inds]
    t_s_dir = t_s[t_dir_host_inds]
    symp_dir = obs.symp[t_dir_host_inds]
    asymp_dir = obs.asymp[t_dir_host_inds]

    AugmentedData(obs, t_i, t_s, t_i_dir, t_s_dir,
                  t_dir_host_inds, symp_dir, asymp_dir)
end

"""
Sort indices by household number then by time (replicating MATLAB's sortrows).
Uses a `by` function to avoid allocating a zipped array.
"""
function _sort_by_household_and_time(household_no, t_i)
    sortperm(1:length(household_no); by=i -> (household_no[i], t_i[i]))
end
