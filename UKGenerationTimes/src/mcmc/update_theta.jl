"""
NUTS-based update of model parameters theta.

Uses AdvancedHMC.jl for Hamiltonian Monte Carlo (NUTS) sampling of the
continuous parameters, replacing the original Metropolis-Hastings update.
"""

# --- Parameter transforms ---

"""
    ParameterTransform

Maps between constrained (model) and unconstrained (NUTS) parameter spaces.
"""
abstract type ParameterTransform end

"""All-positive parameters (independent model): log transform."""
struct LogTransform <: ParameterTransform end

function to_unconstrained(::LogTransform, theta)
    log.(theta)
end

function to_constrained(::LogTransform, phi)
    exp.(phi)
end

function log_jacobian(::LogTransform, phi)
    sum(phi)
end

"""First parameter in (0,1), rest positive (mechanistic model): logit + log."""
struct LogitLogTransform <: ParameterTransform end

function to_unconstrained(::LogitLogTransform, theta)
    vcat(log(theta[1] / (1 - theta[1])), log.(theta[2:end]))
end

function to_constrained(::LogitLogTransform, phi)
    p = 1 / (1 + exp(-phi[1]))
    vcat(p, exp.(phi[2:end]))
end

function log_jacobian(::LogitLogTransform, phi)
    lj = phi[1] - 2 * log(1 + exp(phi[1]))
    lj += sum(phi[2:end])
    lj
end

# --- Log-density target for NUTS ---

"""
    NUTSTarget(ll_household_form, logprior_fun, transform, aug_ref, dim)

Log-density target wrapping the likelihood and prior, operating in unconstrained
space. `aug_ref` is a `Ref{AugmentedData}` so the target sees the current
augmented data without rebuilding the struct.
"""
struct NUTSTarget{T<:ParameterTransform, F1, F2}
    ll_household_form::F1
    logprior_fun::F2
    transform::T
    aug_ref::Ref
    dim::Int
end

function LogDensityProblems.logdensity(target::NUTSTarget, phi)
    theta = to_constrained(target.transform, phi)
    lp = target.logprior_fun(theta)
    lp == -Inf && return -Inf
    lp += sum(target.ll_household_form(theta, target.aug_ref[]))
    lp += log_jacobian(target.transform, phi)
    lp
end

LogDensityProblems.dimension(target::NUTSTarget) = target.dim
LogDensityProblems.capabilities(::Type{<:NUTSTarget}) = LogDensityProblems.LogDensityOrder{0}()

# --- NUTS state management ---

"""
    NUTSState

Holds the AdvancedHMC objects needed between NUTS steps.
"""
mutable struct NUTSState
    hamiltonian::Any
    kernel::Any
    adaptor::Any
    phi::Vector{Float64}
end

"""
    initialise_nuts(theta_init, target, transform; target_accept=0.8)

Set up AdvancedHMC objects for NUTS sampling.
"""
function initialise_nuts(theta_init, target, transform; target_accept=0.8)
    D = length(theta_init)
    phi_init = to_unconstrained(transform, theta_init)

    ad_target = LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), target)

    metric = DiagEuclideanMetric(D)
    hamiltonian = Hamiltonian(metric, ad_target)

    eps = find_good_stepsize(hamiltonian, phi_init)
    integrator = Leapfrog(eps)
    kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(target_accept, integrator))

    NUTSState(hamiltonian, kernel, adaptor, phi_init)
end

"""
    update_theta_nuts!(nuts_state, target, transform, aug, ll_household_form;
                       adapt=true, adapt_step=1, n_adapts=1000)

Take one NUTS step for theta, returning (theta, ll_household, nuts_state, stats).
"""
function update_theta_nuts!(nuts_state::NUTSState, target::NUTSTarget, transform,
                            aug, ll_household_form;
                            adapt=true, adapt_step=1, n_adapts=1000)
    # Update the target's view of the augmented data
    target.aug_ref[] = aug

    # Rebuild hamiltonian (aug changed, so the log-density closure is different)
    ad_target = LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), target)
    nuts_state.hamiltonian = Hamiltonian(nuts_state.hamiltonian.metric, ad_target)

    # Create phase point with fresh momentum
    rng = Random.default_rng()
    z = AdvancedHMC.phasepoint(rng, nuts_state.phi, nuts_state.hamiltonian)

    # Take one NUTS transition
    t = AdvancedHMC.transition(rng, nuts_state.hamiltonian, nuts_state.kernel, z)
    tstat = AdvancedHMC.stat(t)

    # Adapt step size and mass matrix
    if adapt
        nuts_state.hamiltonian, nuts_state.kernel, _ =
            AdvancedHMC.adapt!(nuts_state.hamiltonian, nuts_state.kernel,
                               nuts_state.adaptor,
                               adapt_step, n_adapts,
                               t.z.θ, tstat.acceptance_rate)
    end

    # Extract new position
    nuts_state.phi = t.z.θ

    theta_new = to_constrained(transform, nuts_state.phi)
    ll_household_new = ll_household_form(theta_new, aug)

    theta_new, ll_household_new, nuts_state, tstat
end
