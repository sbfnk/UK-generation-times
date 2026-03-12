using Test
using UKGenerationTimes
using Distributions

@testset "Infectiousness functions" begin
    ap = AssumedParameters()
    theta = [0.5, 1 / 0.18, 3.5, 2.0]
    params = get_params_mech(theta, ap.params_known)

    @testset "get_params_mech" begin
        @test length(params) == 9
        @test params[1] == ap.gamma   # gamma
        @test params[2] ≈ 0.18        # mu
        @test params[3] == ap.k_inc   # k_inc
        @test params[4] ≈ 0.5 * ap.k_inc  # k_E
        @test params[5] == ap.k_I     # k_I
        @test params[6] == 3.5        # alpha
        @test params[7] == 2.0        # beta0
        @test params[8] == ap.rho     # rho
        @test params[9] == ap.x_A     # x_A
    end

    @testset "b_cond_mech" begin
        # Test presymptomatic period (x < 0)
        x = [-5.0, -2.0, -1.0]
        t_inc = [6.0, 6.0, 6.0]
        hh_size = [3.0, 3.0, 3.0]
        asymp = BitVector([false, false, false])

        result = b_cond_mech(x, t_inc, hh_size, asymp, params)
        @test all(result .>= 0)
        @test length(result) == 3

        # Test symptomatic period (x >= 0)
        x_p = [0.0, 1.0, 5.0]
        result_p = b_cond_mech(x_p, t_inc, hh_size, asymp, params)
        @test all(result_p .>= 0)

        # Infectiousness should decrease over time after onset
        @test result_p[1] >= result_p[3]
    end

    @testset "b_int_cond_mech" begin
        x = [-2.0, 0.0, 5.0, 10.0]
        t_inc = fill(6.0, 4)
        hh_size = fill(3.0, 4)
        asymp = falses(4)

        result = b_int_cond_mech(x, t_inc, hh_size, asymp, params)
        @test all(result .>= 0)
        # Cumulative should be monotonically increasing
        @test issorted(result)
    end

    @testset "mean_transmissions_mech" begin
        t_inc = [5.0, 8.0]
        hh_size = [3.0, 3.0]
        asymp = BitVector([false, true])

        result = mean_transmissions_mech(t_inc, hh_size, asymp, params)
        @test all(result .> 0)
        # Asymptomatic should have lower mean transmissions
        @test result[2] < result[1]
    end

    @testset "f_tost_mech" begin
        t = collect(range(-10, 15; length=26))
        result = f_tost_mech(t, params)
        @test all(result .>= 0)
        @test length(result) == 26
    end

    @testset "get_gen_mean_sd_mech" begin
        m, s = get_gen_mean_sd_mech(params)
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
