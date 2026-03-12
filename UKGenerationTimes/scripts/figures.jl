"""
Generate publication figures from fitted results.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using UKGenerationTimes
using JLD2
using CairoMakie

results_path = joinpath(@__DIR__, "..", "results")
figures_path = joinpath(@__DIR__, "..", "figures")
mkpath(figures_path)

# ──────────────────────────────────────────────────────────────────────────────
# Load results
# ──────────────────────────────────────────────────────────────────────────────

@load joinpath(results_path, "param_fit_indep.jld2") result
result_indep_mcmc = result

@load joinpath(results_path, "param_fit_mech.jld2") result
result_mech_mcmc = result

@load joinpath(results_path, "gen_tost_serial_indep.jld2") result_indep
@load joinpath(results_path, "gen_tost_serial_mech.jld2") result_mech

# ──────────────────────────────────────────────────────────────────────────────
# Figure 1: Posterior densities
# ──────────────────────────────────────────────────────────────────────────────

fig1a = plot_posterior_densities(
    result_indep_mcmc.theta_mat,
    ["Mean generation time (days)", "SD generation time (days)", "β₀"]
)
save(joinpath(figures_path, "figure1a_posterior_indep.pdf"), fig1a)
@info "Saved figure1a"

fig1b = plot_posterior_densities(
    result_mech_mcmc.theta_mat,
    ["pE", "1/μ (days)", "α", "β₀"]
)
save(joinpath(figures_path, "figure1b_posterior_mech.pdf"), fig1b)
@info "Saved figure1b"

# ──────────────────────────────────────────────────────────────────────────────
# Figure 2: Generation time, TOST, serial interval
# ──────────────────────────────────────────────────────────────────────────────

fig2a = plot_gen_tost_serial(result_indep; title="Independent model")
save(joinpath(figures_path, "figure2a_distributions_indep.pdf"), fig2a)

fig2b = plot_gen_tost_serial(result_mech; title="Mechanistic model")
save(joinpath(figures_path, "figure2b_distributions_mech.pdf"), fig2b)

# ──────────────────────────────────────────────────────────────────────────────
# Figure 3: Model comparison
# ──────────────────────────────────────────────────────────────────────────────

fig3 = plot_comparison(result_indep, result_mech)
save(joinpath(figures_path, "figure3_comparison.pdf"), fig3)
@info "Saved figure3"

# ──────────────────────────────────────────────────────────────────────────────
# Trace plots (diagnostic)
# ──────────────────────────────────────────────────────────────────────────────

fig_trace_indep = plot_trace(
    result_indep_mcmc.theta_mat, result_indep_mcmc.ll_vec,
    ["Mean gen time", "SD gen time", "β₀"]
)
save(joinpath(figures_path, "trace_indep.pdf"), fig_trace_indep)

fig_trace_mech = plot_trace(
    result_mech_mcmc.theta_mat, result_mech_mcmc.ll_vec,
    ["pE", "1/μ", "α", "β₀"]
)
save(joinpath(figures_path, "trace_mech.pdf"), fig_trace_mech)

@info "All figures saved to $(figures_path)"
