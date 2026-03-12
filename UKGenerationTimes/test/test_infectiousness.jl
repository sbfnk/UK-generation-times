using Test
using UKGenerationTimes
using Distributions

@testset "Infectiousness functions" begin
    ap = AssumedParameters()
    theta = [0.5, 1 / 0.18, 3.5, 2.0]
    p = get_params_mech(theta, ap.params_known)

    @testset "get_params_mech" begin
        @test p isa NamedTuple
        @test p.γ == ap.gamma
        @test p.μ ≈ 0.18
        @test p.k_inc == ap.k_inc
        @test p.k_E ≈ 0.5 * ap.k_inc
        @test p.k_I == ap.k_I
        @test p.α == 3.5
        @test p.β₀ == 2.0
        @test p.ρ == ap.rho
        @test p.x_A == ap.x_A
    end

    @testset "b_cond_mech" begin
        # Test presymptomatic period (x < 0)
        x = [-5.0, -2.0, -1.0]
        t_inc = [6.0, 6.0, 6.0]
        hh_size = [3.0, 3.0, 3.0]
        asymp = BitVector([false, false, false])

        result = b_cond_mech(x, t_inc, hh_size, asymp, p)
        @test all(result .>= 0)
        @test length(result) == 3

        # Test symptomatic period (x >= 0)
        x_p = [0.0, 1.0, 5.0]
        result_p = b_cond_mech(x_p, t_inc, hh_size, asymp, p)
        @test all(result_p .>= 0)

        # Infectiousness should decrease over time after onset
        @test result_p[1] >= result_p[3]
    end

    @testset "b_int_cond_mech" begin
        x = [-2.0, 0.0, 5.0, 10.0]
        t_inc = fill(6.0, 4)
        hh_size = fill(3.0, 4)
        asymp = falses(4)

        result = b_int_cond_mech(x, t_inc, hh_size, asymp, p)
        @test all(result .>= 0)
        # Cumulative should be monotonically increasing
        @test issorted(result)
    end

    @testset "mean_transmissions_mech" begin
        t_inc = [5.0, 8.0]
        hh_size = [3.0, 3.0]
        asymp = BitVector([false, true])

        result = mean_transmissions_mech(t_inc, hh_size, asymp, p)
        @test all(result .> 0)
        # Asymptomatic should have lower mean transmissions
        @test result[2] < result[1]
    end

    @testset "f_tost_mech" begin
        t = collect(range(-10, 15; length=26))
        result = f_tost_mech(t, p)
        @test all(result .>= 0)
        @test length(result) == 26
    end

    @testset "get_gen_mean_sd_mech" begin
        m, s = get_gen_mean_sd_mech(p)
        @test m > 0
        @test s > 0
        @test isfinite(m)
        @test isfinite(s)
    end

    @testset "Priors" begin
        theta_indep = [5.0, 3.0, 2.0]
        @test prior_indep(theta_indep) > 0
        @test isfinite(logprior_indep(theta_indep))

        @test prior_mech(theta) > 0
        @test isfinite(logprior_mech(theta))

        # Invalid p_E should give -Inf log-prior
        @test logprior_mech([1.5, 5.0, 1.0, 2.0]) == -Inf
        @test logprior_mech([-0.1, 5.0, 1.0, 2.0]) == -Inf
    end
end
