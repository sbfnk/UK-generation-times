"""
Main MCMC sampler implementing the 4-step data augmentation algorithm.

Based on [fit_params.m](https://github.com/will-s-hart/UK-generation-times/blob/main/Functions/MCMC/fit_params.m) from the original MATLAB implementation.

Uses NUTS (via AdvancedHMC.jl) for the parameter update step and custom
Metropolis-within-Gibbs steps for the augmented data updates.
"""


"""
    MCMCConfig

Configuration for the MCMC sampler.
"""
struct MCMCConfig
    no_steps::Int
    steps_keep::Vector{Int}
    t_i_prop_sd_symp::Float64
    t_i_prop_sd_asymp::Float64
    logprior_fun::Function
    transform::ParameterTransform
    nuts_target_accept::Float64
    nuts_adapt_steps::Int  # number of theta updates (not total steps) for adaptation
end

function MCMCConfig(no_steps, steps_keep, t_i_prop_sd_symp, t_i_prop_sd_asymp,
                    logprior_fun, transform;
                    nuts_target_accept=0.8, nuts_adapt_fraction=0.2)
    nuts_adapt_steps = round(Int, (no_steps / 4) * nuts_adapt_fraction)
    MCMCConfig(no_steps, steps_keep, t_i_prop_sd_symp, t_i_prop_sd_asymp,
               logprior_fun, transform, nuts_target_accept, nuts_adapt_steps)
end

"""
    MCMCResult

Output from the MCMC sampler.
"""
struct MCMCResult
    theta_mat::Matrix{Float64}
    ll_vec::Vector{Float64}
    empirical_summary_mat::Matrix{Float64}
    acceptance_rates::Dict{Symbol,Float64}
    aug_final::AugmentedData
end

"""
    fit_params(config, ll_household_form, empirical_summary_form,
               theta_init, aug_init; model=:indep)

Run the data augmentation MCMC sampler.

# Arguments
- `config`: MCMCConfig with sampler settings
- `ll_household_form`: function (theta, aug) -> per-household log-likelihoods
- `empirical_summary_form`: function (theta, aug) -> [mean, sd, presymp_prop]
- `theta_init`: initial parameter vector
- `aug_init`: initial AugmentedData
- `model`: `:indep` or `:mech`
"""
function fit_params(config::MCMCConfig, ll_household_form, empirical_summary_form,
                    theta_init, aug_init; model=:indep)

    no_steps = config.no_steps
    steps_keep = config.steps_keep
    no_params = length(theta_init)
    no_steps_kept = length(steps_keep)

    # Kept output
    theta_mat = zeros(no_steps_kept, no_params)
    ll_vec = zeros(no_steps_kept)
    empirical_summary_mat = zeros(no_steps_kept, 3)

    # Acceptance tracking
    acc_infection_symp = Float64[]
    acc_infection_asymp = Float64[]
    acc_onset = Float64[]
    acc_asymp = Float64[]
    nuts_tree_depths = Int[]
    nuts_divergences = Int[]

    # Current state
    theta = copy(theta_init)
    aug = aug_init
    ll_household = ll_household_form(theta, aug)

    # Initialise NUTS
    target = NUTSTarget(ll_household_form, config.logprior_fun,
                        config.transform, Ref(aug), no_params)
    nuts_state = initialise_nuts(theta_init, target, config.transform;
                                 target_accept=config.nuts_target_accept)

    step_no_kept = 0
    steps_keep_set = Set(steps_keep)
    theta_update_count = 0

    p = Progress(no_steps; dt=1.0, desc="MCMC: ")

    for step_no in 1:no_steps
        step_type = mod(step_no - 1, 4)

        if step_type == 0
            # Update parameters via NUTS
            theta_update_count += 1
            do_adapt = theta_update_count <= config.nuts_adapt_steps
            theta, ll_household, nuts_state, nstat = update_theta_nuts!(
                nuts_state, target, config.transform, aug, ll_household_form;
                adapt=do_adapt,
                adapt_step=theta_update_count,
                n_adapts=config.nuts_adapt_steps)
            push!(nuts_tree_depths, nstat.n_steps)
            push!(nuts_divergences, nstat.numerical_error ? 1 : 0)

        elseif step_type == 1
            # Update infection times
            if model == :indep
                aug, ll_household, acceptance = update_infection_indep!(
                    theta, aug, ll_household, ll_household_form,
                    config.t_i_prop_sd_symp)
            else
                aug, ll_household, acceptance = update_infection_mech!(
                    theta, aug, ll_household, ll_household_form,
                    config.t_i_prop_sd_symp)
            end
            push!(acc_infection_symp, acceptance.symp)
            push!(acc_infection_asymp, acceptance.asymp)

        elseif step_type == 2
            # Update onset times
            aug, ll_household, acceptance = update_onset!(
                theta, aug, ll_household, ll_household_form)
            push!(acc_onset, acceptance.overall)

        else
            # Update asymptomatic data
            if model == :indep
                aug, ll_household, acceptance = update_asymp_indep!(
                    theta, aug, ll_household, ll_household_form,
                    config.t_i_prop_sd_asymp)
            else
                aug, ll_household, acceptance = update_asymp_mech!(
                    theta, aug, ll_household, ll_household_form,
                    config.t_i_prop_sd_asymp)
            end
            push!(acc_asymp, acceptance.overall)
            if hasproperty(acceptance, :infection) && !isnan(something(acceptance.infection, NaN))
                push!(acc_infection_asymp, acceptance.infection)
            end
        end

        # Record kept steps
        if step_no in steps_keep_set
            step_no_kept += 1
            theta_mat[step_no_kept, :] .= theta
            ll_vec[step_no_kept] = sum(ll_household)
            empirical_summary_mat[step_no_kept, :] .=
                empirical_summary_form(theta, aug)
        end

        next!(p)
    end

    # Compute acceptance rates
    _nanmean(x) = isempty(x) ? NaN : mean(filter(!isnan, x))

    acceptance_rates = Dict{Symbol,Float64}(
        :infection_symp => _nanmean(acc_infection_symp),
        :onset => _nanmean(acc_onset),
        :asymp => _nanmean(acc_asymp),
        :infection_asymp => _nanmean(acc_infection_asymp),
    )

    n_divergent = sum(nuts_divergences)
    mean_tree_depth = isempty(nuts_tree_depths) ? NaN : mean(nuts_tree_depths)

    @info "NUTS diagnostics:" mean_tree_depth n_divergent
    @info "Augmented data acceptance rates:" acceptance_rates

    MCMCResult(theta_mat, ll_vec, empirical_summary_mat,
               acceptance_rates, aug)
end
