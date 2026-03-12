using Test
using UKGenerationTimes
using Distributions
using Random

@testset "Likelihood functions" begin
    data_path = joinpath(@__DIR__, "..", "data", "Supplementary_Data.xlsx")

    if !isfile(data_path)
        @warn "Supplementary_Data.xlsx not found, skipping likelihood tests"
        return
    end

    Random.seed!(42)
    obs = import_and_format_data(data_path)
    ap = AssumedParameters()

    @testset "Independent model likelihood" begin
        aug = initialise_augmented_data_indep(obs)

        # Lognormal generation time with mean=5, sd=3
        m, s = 5.0, 3.0
        mu = log(m^2 / sqrt(s^2 + m^2))
        sigma = sqrt(log(1 + s^2 / m^2))
        f_gen(t) = pdf.(LogNormal(mu, sigma), t)
        F_gen(t) = cdf.(LogNormal(mu, sigma), t)

        beta0 = 2.0
        f_inc(t) = f_inc_logn(t, ap)

        ll = log_likelihood_household_indep(f_inc, beta0, ap.rho, ap.x_A,
                                            f_gen, F_gen, aug)

        @test length(ll) == length(obs.household_sizes_incl)
        @test all(isfinite.(ll))
        @test all(ll .<= 0)  # log-likelihoods should be non-positive

        @info "Independent model likelihood: sum = $(sum(ll))"
    end

    @testset "Mechanistic model likelihood" begin
        aug = initialise_augmented_data_mech(obs)

        theta = [0.5, 1 / 0.18, 3.5, 2.0]
        p = get_params_mech(theta, ap.params_known)

        f_inc(t) = f_inc_gam(t, ap)
        b_cond(x, t_inc, hh, a) = b_cond_mech(x, t_inc, hh, a, p)
        B_cond(x, t_inc, hh, a) = b_int_cond_mech(x, t_inc, hh, a, p)
        mt(t_inc, hh, a) = mean_transmissions_mech(t_inc, hh, a, p)

        ll = log_likelihood_household_mech(f_inc, b_cond, B_cond, mt, aug)

        @test length(ll) == length(obs.household_sizes_incl)
        @test all(isfinite.(ll))
        @test all(ll .<= 0)

        @info "Mechanistic model likelihood: sum = $(sum(ll))"
    end

    @testset "MCMC initialisation" begin
        aug_indep = initialise_augmented_data_indep(obs)
        aug_mech = initialise_augmented_data_mech(obs)

        n = length(obs.household_no)
        @test length(aug_indep.t_i) == n
        @test length(aug_indep.t_s) == n
        @test length(aug_indep.t_i_dir) == n
        @test length(aug_indep.t_dir_host_inds) == n

        # Symptomatic hosts should have finite infection and onset times
        @test all(isfinite.(aug_indep.t_i[obs.symp]))
        @test all(isfinite.(aug_indep.t_s[obs.symp]))

        # Uninfected hosts should have infinite times
        uninf = .!obs.infected_dir
        @test all(aug_indep.t_i[uninf] .== Inf)

        # Mechanistic model: asymptomatic hosts have finite "onset" times
        @test all(isfinite.(aug_mech.t_s[obs.asymp]))
    end
end
