"""
Publication figure generation using CairoMakie.

Reproduces the key figures from Hart et al. (2022).
"""


"""
    plot_posterior_densities(theta_mat, param_names; kwargs...)

Plot posterior density estimates for each parameter.
"""
function plot_posterior_densities(theta_mat::Matrix{Float64},
                                  param_names::Vector{String};
                                  figsize=(800, 200 * size(theta_mat, 2)))
    n_params = size(theta_mat, 2)
    fig = Figure(; size=figsize)

    for i in 1:n_params
        ax = Axis(fig[i, 1]; xlabel=param_names[i], ylabel="Density")
        density!(ax, theta_mat[:, i]; color=(:steelblue, 0.6),
                 strokewidth=1.5, strokecolor=:steelblue)
    end

    fig
end

"""
    plot_gen_tost_serial(result; kwargs...)

Plot generation time, TOST, and serial interval distributions.
`result` should be a named tuple with fields f_gen, f_tost, f_serial, t_grid.
"""
function plot_gen_tost_serial(result; figsize=(900, 300),
                               title="Distribution estimates")
    fig = Figure(; size=figsize)
    t = result.t_grid

    ax1 = Axis(fig[1, 1]; xlabel="Time (days)", ylabel="Density",
               title="Generation time")
    lines!(ax1, t, result.f_gen; color=:steelblue, linewidth=2)

    ax2 = Axis(fig[1, 2]; xlabel="Time (days)", ylabel="Density",
               title="TOST")
    lines!(ax2, t, result.f_tost; color=:coral, linewidth=2)
    vlines!(ax2, [0.0]; color=:grey, linestyle=:dash)

    ax3 = Axis(fig[1, 3]; xlabel="Time (days)", ylabel="Density",
               title="Serial interval")
    lines!(ax3, t, result.f_serial; color=:seagreen, linewidth=2)
    vlines!(ax3, [0.0]; color=:grey, linestyle=:dash)

    fig
end

"""
    plot_trace(theta_mat, ll_vec, param_names; kwargs...)

Plot trace plots for parameters and log-likelihood.
"""
function plot_trace(theta_mat::Matrix{Float64}, ll_vec::Vector{Float64},
                    param_names::Vector{String}; figsize=(800, 150 * (size(theta_mat, 2) + 1)))
    n_params = size(theta_mat, 2)
    fig = Figure(; size=figsize)

    for i in 1:n_params
        ax = Axis(fig[i, 1]; ylabel=param_names[i])
        lines!(ax, 1:size(theta_mat, 1), theta_mat[:, i];
               color=(:steelblue, 0.5), linewidth=0.5)
    end

    ax = Axis(fig[n_params + 1, 1]; ylabel="Log-likelihood", xlabel="Iteration")
    lines!(ax, 1:length(ll_vec), ll_vec; color=(:coral, 0.5), linewidth=0.5)

    fig
end

"""
    plot_comparison(result_indep, result_mech; kwargs...)

Compare generation time, TOST, and serial interval between models.
"""
function plot_comparison(result_indep, result_mech;
                          figsize=(900, 300))
    fig = Figure(; size=figsize)

    for (i, (field, title)) in enumerate(zip(
            [:f_gen, :f_tost, :f_serial],
            ["Generation time", "TOST", "Serial interval"]))
        ax = Axis(fig[1, i]; xlabel="Time (days)", ylabel="Density", title=title)
        lines!(ax, result_indep.t_grid, getfield(result_indep, field);
               color=:steelblue, linewidth=2, label="Independent")
        lines!(ax, result_mech.t_grid, getfield(result_mech, field);
               color=:coral, linewidth=2, label="Mechanistic")
        if field != :f_gen
            vlines!(ax, [0.0]; color=:grey, linestyle=:dash)
        end
        if i == 1
            axislegend(ax; position=:rt)
        end
    end

    fig
end
