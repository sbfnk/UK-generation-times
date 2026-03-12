using Test
using UKGenerationTimes

@testset "Data import" begin
    data_path = joinpath(@__DIR__, "..", "data", "Supplementary_Data.xlsx")

    if isfile(data_path)
        obs = import_and_format_data(data_path)

        @test length(obs.household_sizes_incl) > 0
        @test length(obs.t_iL) == length(obs.t_iR)
        @test length(obs.t_sL) == length(obs.t_sR)
        @test length(obs.household_no) == length(obs.t_iL)

        # All infection left bounds should be -Inf
        @test all(obs.t_iL .== -Inf)

        # Symptomatic individuals have finite onset bounds
        @test all(isfinite.(obs.t_sL[obs.symp]))
        @test all(isfinite.(obs.t_sR[obs.symp]))

        # Uninfected individuals have infinite onset bounds
        uninf = .!obs.infected_dir
        @test all(obs.t_sL[uninf] .== Inf)

        # Household indicator matrix should have correct dimensions
        no_hosts = length(obs.household_no)
        no_households = length(obs.household_sizes_incl)
        @test size(obs.household_indicator_mat) == (no_hosts, no_households)

        # Each host belongs to exactly one household
        @test all(sum(obs.household_indicator_mat; dims=2) .== 1)

        # Household sizes should sum to total hosts
        @test sum(obs.household_sizes_incl) == no_hosts

        # M1 and M2 dimensions
        poss = obs.poss_infectors_dir
        nv = length(poss.all)
        @test size(poss.from_indicator_mat) == (nv, no_hosts)
        @test size(poss.to_indicator_mat) == (nv, no_hosts)

        @info "Data import test passed: $(no_hosts) hosts in $(no_households) households"
    else
        @warn "Supplementary_Data.xlsx not found, skipping data import tests"
    end
end

@testset "Assumed parameters" begin
    ap = AssumedParameters()

    @test ap.inc_mu == 1.63
    @test ap.inc_sigma == 0.5
    @test ap.x_A == 0.35
    @test ap.rho == 1.0
    @test ap.k_I == 1.0
    @test length(ap.params_known) == 5

    # Gamma parameterisation should have same mean as lognormal
    @test ap.inc_mean ≈ exp(ap.inc_mu + 0.5 * ap.inc_sigma^2)
    @test ap.inc_shape * ap.inc_scale ≈ ap.inc_mean
end
