"""
Compute generation time, TOST, and serial interval distributions for both
models using posterior mean parameters.

Based on the following files from the original MATLAB implementation:
- [gen_tost_serial_indep.m](https://github.com/will-s-hart/UK-generation-times/blob/main/Scripts/Fitted%20model%20analysis/gen_tost_serial_indep.m)
- [gen_tost_serial_mech.m](https://github.com/will-s-hart/UK-generation-times/blob/main/Scripts/Fitted%20model%20analysis/gen_tost_serial_mech.m)
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using UKGenerationTimes
using JLD2

# Load assumed parameters
ap = AssumedParameters()

results_path = joinpath(@__DIR__, "..", "results")

# ──────────────────────────────────────────────────────────────────────────────
# Independent model
# ──────────────────────────────────────────────────────────────────────────────

@info "Loading independent model posteriors..."
@load joinpath(results_path, "posterior_indep.jld2") mean_best sd_best

@info "Computing independent model distributions..."
@time result_indep = gen_tost_serial_indep(mean_best, sd_best,
                                            ap.inc_mu, ap.inc_sigma)

@save joinpath(results_path, "gen_tost_serial_indep.jld2") result_indep

# ──────────────────────────────────────────────────────────────────────────────
# Mechanistic model
# ──────────────────────────────────────────────────────────────────────────────

@info "Loading mechanistic model posteriors..."
@load joinpath(results_path, "posterior_mech.jld2") params_best

@info "Computing mechanistic model distributions..."
@time result_mech = gen_tost_serial_mech(params_best)

@save joinpath(results_path, "gen_tost_serial_mech.jld2") result_mech

@info "Done."
