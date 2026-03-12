"""
Data structures for the UK generation times model.

Arrays with the suffix `_dir` are ordered according to (i) household number,
and (ii) the (unknown) order in which household members became infected.
Otherwise, arrays are ordered according to (i) household number, and (ii) the
(known) order in which household members developed symptoms.
"""

"""
Information about possible infectors for each individual, ordered by
infection time within households.
"""
struct PossibleInfectors
    cell::Vector{Vector{Int}}     # possible infectors for each individual
    all::Vector{Int}              # flattened list of all possible infectors
    from_indicator_mat::SparseMatrixCSC{Float64,Int}  # M1: index of potential infector
    to_indicator_mat::SparseMatrixCSC{Float64,Int}    # M2: index of potential infectee
    to_primary_indicator::BitVector          # entries of `all` corresponding to primary cases
    to_infected_indicator::BitVector         # entries corresponding to infected individuals
    to_recipient_indicator::BitVector        # entries corresponding to non-primary infected
    household_size::Vector{Float64}          # household size for each entry
end

"""
Immutable struct containing all observed data.
"""
struct ObservedData
    # Bounds for infection and onset times
    t_iL::Vector{Float64}
    t_iR::Vector{Float64}
    t_sL::Vector{Float64}
    t_sR::Vector{Float64}

    # Household structure
    household_sizes_incl::Vector{Int}    # size of each household (excluding inconclusive)
    household_sizes_full::Vector{Int}    # original household sizes
    household_size_indiv_incl::Vector{Int}  # household size repeated per individual
    household_size_indiv_full::Vector{Int}  # original size repeated per individual
    household_no::Vector{Int}            # household number for each individual
    household_indicator_mat::SparseMatrixCSC{Float64,Int}  # block-diagonal indicator

    # Counts per household
    no_infected_in_household::Vector{Int}
    no_symp_in_household::Vector{Int}
    symp_in_household::BitVector
    no_asymp_in_household::Vector{Int}
    asymp_in_household::BitVector

    # Individual status
    primary_dir::BitVector
    infected_dir::BitVector
    symp::BitVector
    asymp::BitVector

    # Possible infectors
    no_poss_infectors_dir::Vector{Int}
    poss_infectors_dir::PossibleInfectors

    # Recruitment month
    household_months::Vector{Int}
end

"""
Mutable struct containing augmented (latent) data on top of observed data.
"""
mutable struct AugmentedData
    # Reference to observed data
    observed::ObservedData

    # Augmented times (symptom-onset order)
    t_i::Vector{Float64}
    t_s::Vector{Float64}

    # Augmented times (infection order)
    t_i_dir::Vector{Float64}
    t_s_dir::Vector{Float64}

    # Mapping from infection order to symptom order
    t_dir_host_inds::Vector{Int}

    # Status in infection order
    symp_dir::BitVector
    asymp_dir::BitVector
end

# Convenience accessors: fields not on AugmentedData itself are forwarded to observed data.
# Uses === on Symbol literals for type stability (the compiler can infer the return type).
function Base.getproperty(a::AugmentedData, s::Symbol)
    if s === :observed || s === :t_i || s === :t_s || s === :t_i_dir ||
       s === :t_s_dir || s === :t_dir_host_inds || s === :symp_dir || s === :asymp_dir
        getfield(a, s)
    else
        getfield(getfield(a, :observed), s)
    end
end
