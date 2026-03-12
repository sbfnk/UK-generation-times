"""
Data structures for the UK generation times model.

The household transmission model uses data augmentation MCMC, meaning some
quantities are observed (symptom onset windows, household membership) and others
are latent (exact infection times, exact onset times, who infected whom).

Arrays with the suffix `_dir` are sorted by (household, infection time) — the
inferred infection order. Arrays without `_dir` are sorted by (household,
symptom onset time) — the observed order. The mapping between these orderings
changes as the MCMC updates latent infection times.
"""

"""
Possible infectors for each individual within their household.

For each individual, every household member who was infected earlier is a
possible infector. This structure stores the flattened list of all such
(infector, infectee) pairs and sparse indicator matrices that map between
the pair-level and individual-level representations.
"""
struct PossibleInfectors
    cell::Vector{Vector{Int}}     # possible infectors for each individual (ragged)
    all::Vector{Int}              # flattened pair list: infector indices
    from_indicator_mat::SparseMatrixCSC{Float64,Int}  # pairs → infector (M_from)
    to_indicator_mat::SparseMatrixCSC{Float64,Int}    # pairs → infectee (M_to)
    to_primary_indicator::BitVector          # true for pairs where infectee is primary
    to_infected_indicator::BitVector         # true for pairs where infectee is infected
    to_recipient_indicator::BitVector        # true for non-primary infected infectees
    household_size::Vector{Float64}          # household size for each pair
end

"""
Immutable observed data: everything known before MCMC begins.

Includes symptom onset time windows (t_sL, t_sR), infection time bounds
(t_iL, t_iR), household structure, and the possible-infector matrices.
"""
struct ObservedData
    # Bounds on latent times (from symptom onset windows and epidemiological constraints)
    t_iL::Vector{Float64}         # lower bound on infection time
    t_iR::Vector{Float64}         # upper bound on infection time
    t_sL::Vector{Float64}         # lower bound on symptom onset time
    t_sR::Vector{Float64}         # upper bound on symptom onset time

    # Household structure
    household_sizes_incl::Vector{Int}    # household size (excluding inconclusive cases)
    household_sizes_full::Vector{Int}    # original household size (before exclusions)
    household_size_indiv_incl::Vector{Int}  # per-individual household size (excl.)
    household_size_indiv_full::Vector{Int}  # per-individual household size (full)
    household_no::Vector{Int}            # which household each individual belongs to
    household_indicator_mat::SparseMatrixCSC{Float64,Int}  # block-diagonal: sums individuals → households

    # Per-household infection counts
    no_infected_in_household::Vector{Int}
    no_symp_in_household::Vector{Int}
    symp_in_household::BitVector         # does this household have symptomatic cases?
    no_asymp_in_household::Vector{Int}
    asymp_in_household::BitVector

    # Per-individual infection/symptom status
    primary_dir::BitVector        # first infected in each household
    infected_dir::BitVector       # infected (primary + secondary)
    symp::BitVector               # symptomatic
    asymp::BitVector              # asymptomatic (infected but no symptoms)

    # Possible infectors (infection-order structure)
    no_poss_infectors_dir::Vector{Int}
    poss_infectors_dir::PossibleInfectors

    # Recruitment timing
    household_months::Vector{Int}
end

"""
Mutable augmented (latent) data, updated during MCMC.

Wraps `ObservedData` plus the latent quantities that the sampler proposes
and updates: exact infection times, exact onset times, and the current
infection ordering.
"""
mutable struct AugmentedData
    observed::ObservedData

    # Latent times in symptom-onset order
    t_i::Vector{Float64}          # infection times
    t_s::Vector{Float64}          # symptom onset times

    # Same times in infection order (re-sorted each MCMC step)
    t_i_dir::Vector{Float64}
    t_s_dir::Vector{Float64}

    # Permutation from infection order back to symptom order
    t_dir_host_inds::Vector{Int}

    # Symptom status in infection order
    symp_dir::BitVector
    asymp_dir::BitVector
end

# Forward field access: aug.household_no etc. look through to aug.observed
function Base.getproperty(a::AugmentedData, s::Symbol)
    if s === :observed || s === :t_i || s === :t_s || s === :t_i_dir ||
       s === :t_s_dir || s === :t_dir_host_inds || s === :symp_dir || s === :asymp_dir
        getfield(a, s)
    else
        getfield(getfield(a, :observed), s)
    end
end
