"""
Data import and formatting from Supplementary_Data.xlsx.

Implements the logic of import_data.m and format_data.m.
"""

using XLSX, DataFrames, SparseArrays

"""
    import_and_format_data(filepath::String) -> ObservedData

Import household transmission data from the Excel file and construct the
observed data structure. Combines the steps of import_data.m and format_data.m.
"""
function import_and_format_data(filepath::String)
    # Read Excel file
    df = DataFrame(XLSX.readtable(filepath, 1))

    # Extract columns
    household_no_all = df.hoconumber
    household_size_all = df.householdsize
    d_s_all = Float64.(coalesce.(df.case_swab_ill, NaN))
    month_all = df.month_case_swab

    # Infection status
    inf_all = df.infected .== 1
    uninf_all = df.infected .== 0
    symp_all = inf_all .& (df.symptoms .== 1)
    asymp_all = inf_all .& (df.symptoms .== 0)

    # Uninfected or asymptomatic individuals: onset at infinity
    d_s_all[asymp_all] .= Inf
    d_s_all[uninf_all] .= Inf

    # Find index cases (status string length == 4, i.e. "ndex" from "index")
    status_all = string.(df.status)
    index_all = length.(status_all) .== 4

    # Right bounds for infection date
    d_iR_all = fill(Inf, length(d_s_all))
    d_iR_all[symp_all] .= d_s_all[symp_all]
    d_iR_all[index_all] .= min.(d_iR_all[index_all], 0.0)

    # Positive swab results tighten infection bounds
    for (swab_col, date_col) in [(:swab1, :case_swab_swab1), (:swab2, :case_swab_swab2)]
        if hasproperty(df, swab_col)
            positive = coalesce.(getproperty(df, swab_col), 0) .== 1
            dates = Float64.(coalesce.(getproperty(df, date_col), Inf))
            d_iR_all[positive] .= min.(d_iR_all[positive], dates[positive])
        end
    end

    # Discard inconclusive hosts
    keep = symp_all .| asymp_all .| uninf_all

    household_no_all = household_no_all[keep]
    household_size_all = household_size_all[keep]
    month_all = month_all[keep]
    d_s_all = d_s_all[keep]
    d_iR_all = d_iR_all[keep]
    symp_all = symp_all[keep]
    asymp_all = asymp_all[keep]
    uninf_all = uninf_all[keep]
    inf_all = .!uninf_all
    index_all = index_all[keep]

    # Sort by household, then uninfected status, then onset date
    order = sortperm(collect(zip(household_no_all, uninf_all, d_s_all)))

    household_no_all = household_no_all[order]
    household_size_all = household_size_all[order]
    month_all = month_all[order]
    d_s_all = d_s_all[order]
    d_iR_all = d_iR_all[order]
    symp_all = symp_all[order]
    asymp_all = asymp_all[order]
    uninf_all = uninf_all[order]
    inf_all = inf_all[order]

    # Apply 28-day filter and separate clusters
    max_onsetdiff = 28
    households_vec = unique(household_no_all)

    # Accumulation arrays
    household_no_new = Int[]
    household_sizes_old = Int[]
    household_sizes_new = Int[]
    household_size_indiv_old = Int[]
    household_size_indiv_new = Int[]
    household_months = Int[]
    d_s_out = Float64[]
    d_iR_out = Float64[]
    symp_out = Bool[]
    asymp_out = Bool[]
    uninf_out = Bool[]

    no_households_incl = 0

    for household in households_vec
        inds = findall(household_no_all .== household)
        h_d_s = d_s_all[inds]
        h_d_iR = d_iR_all[inds]
        h_symp = symp_all[inds]
        h_asymp = asymp_all[inds]
        h_uninf = uninf_all[inds]

        no_in_household_old = household_size_all[inds[1]]
        no_in_household = length(inds)

        # Check onset diffs for symptomatic individuals
        symp_onsets = h_d_s[h_symp]
        onset_diffs = diff(symp_onsets)
        max_diff = isempty(onset_diffs) ? -Inf : maximum(onset_diffs)

        if no_in_household > 1 && (isempty(onset_diffs) || max_diff <= max_onsetdiff)
            no_households_incl += 1
            n = no_in_household

            append!(household_no_new, fill(no_households_incl, n))
            push!(household_sizes_old, no_in_household_old)
            push!(household_sizes_new, n)
            append!(household_size_indiv_old, fill(no_in_household_old, n))
            append!(household_size_indiv_new, fill(n, n))
            push!(household_months, month_all[inds[1]])
            append!(d_s_out, h_d_s)
            append!(d_iR_out, h_d_iR)
            append!(symp_out, h_symp)
            append!(asymp_out, h_asymp)
            append!(uninf_out, h_uninf)
        end
    end

    symp = BitVector(symp_out)
    asymp = BitVector(asymp_out)
    uninf = BitVector(uninf_out)
    infected_dir = .!uninf

    # Compute bounds
    no_hosts = length(d_s_out)
    t_sL = d_s_out .- 0.5
    t_sR = d_s_out .+ 0.5
    t_iL = fill(-Inf, no_hosts)
    t_iR = d_iR_out .+ 0.5

    no_households = length(household_sizes_new)
    household_no = household_no_new

    # Build household indicator matrix (block-diagonal)
    household_indicator_mat = _build_block_diagonal(household_sizes_new, no_hosts)

    # Compute per-household counts and possible infectors
    no_infected_in_household = zeros(Int, no_households)
    no_symp_in_household = zeros(Int, no_households)
    no_asymp_in_household = zeros(Int, no_households)
    no_poss_infectors_dir = zeros(Int, no_hosts)
    poss_infectors_dir_cell = [Int[] for _ in 1:no_hosts]
    primary_dir = falses(no_hosts)

    for i in 1:no_households
        in_household = household_no .== i
        infected_hosts = findall(in_household .& infected_dir)
        symp_hosts = findall(in_household .& symp)
        asymp_hosts = findall(in_household .& asymp)
        uninfected_hosts = findall(in_household .& uninf)

        no_infected_in_household[i] = length(infected_hosts)
        no_symp_in_household[i] = length(symp_hosts)
        no_asymp_in_household[i] = length(asymp_hosts)

        if !isempty(infected_hosts)
            # Primary case
            no_poss_infectors_dir[infected_hosts[1]] = 1
            poss_infectors_dir_cell[infected_hosts[1]] = [0]
            primary_dir[infected_hosts[1]] = true

            # Secondary cases
            for j in 2:length(infected_hosts)
                poss_infectors_dir_cell[infected_hosts[j]] = infected_hosts[1:(j-1)]
                no_poss_infectors_dir[infected_hosts[j]] = j - 1
            end
        end

        # Uninfected hosts can be infected by any infected host
        for j in eachindex(uninfected_hosts)
            poss_infectors_dir_cell[uninfected_hosts[j]] = infected_hosts
            no_poss_infectors_dir[uninfected_hosts[j]] = length(infected_hosts)
        end
    end

    symp_in_household = no_symp_in_household .> 0
    asymp_in_household = no_asymp_in_household .> 0

    # Flatten possible infectors
    v = reduce(vcat, poss_infectors_dir_cell)
    nv = length(v)

    # Build M1 (from-indicator matrix)
    I_idx = Int[]
    J_idx = Int[]
    for i in 1:nv
        if v[i] > 0
            push!(I_idx, i)
            push!(J_idx, v[i])
        end
    end
    M1 = sparse(I_idx, J_idx, ones(length(I_idx)), nv, no_hosts)

    # Build M2 (to-indicator matrix)
    M2 = _build_block_diagonal(no_poss_infectors_dir, nv)

    # Indicator vectors
    to_infected_indicator = BitVector(Bool.(M2 * Float64.(infected_dir)))
    to_primary_indicator = BitVector(v .== 0)
    to_recipient_indicator = to_infected_indicator .& .!to_primary_indicator

    # Household size for each entry of v
    household_size_v = Float64.(M2 * Float64.(household_size_indiv_old))

    poss_infectors = PossibleInfectors(
        poss_infectors_dir_cell, v, M1, M2,
        to_primary_indicator, to_infected_indicator, to_recipient_indicator,
        household_size_v
    )

    ObservedData(
        t_iL, t_iR, t_sL, t_sR,
        household_sizes_new, household_sizes_old,
        household_size_indiv_new, household_size_indiv_old,
        household_no, household_indicator_mat,
        no_infected_in_household, no_symp_in_household, symp_in_household,
        no_asymp_in_household, asymp_in_household,
        primary_dir, infected_dir, symp, asymp,
        no_poss_infectors_dir, poss_infectors,
        household_months
    )
end

"""
Build a block-diagonal sparse matrix from block sizes.
"""
function _build_block_diagonal(sizes::Vector{Int}, total_rows::Int)
    I_idx = Int[]
    J_idx = Int[]
    row_offset = 0
    for (j, sz) in enumerate(sizes)
        for k in 1:sz
            push!(I_idx, row_offset + k)
            push!(J_idx, j)
        end
        row_offset += sz
    end
    sparse(I_idx, J_idx, ones(length(I_idx)), total_rows, length(sizes))
end
