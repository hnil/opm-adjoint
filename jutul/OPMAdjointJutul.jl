# Helpers for cross-validating the opm-adjoint module against
# Jutul/JutulDarcy (test level T6 of the adjoint plan).
#
# Usage (with the dedicated environment in this directory):
#   julia --project=<repo>/jutul
#   include(joinpath(<repo>, "jutul", "OPMAdjointJutul.jl"))
#   using .OPMAdjointJutul
#
# Workflow:
#   case, result = run_forward("CASE.DATA")
#   save_well_curves_csv("out.csv", result, "PROD")
#   obj = well_rate_misfit_objective(result, :PROD, :grat)   # twin misfit
#   sens = parameter_sensitivities(case, result, obj)
#
# All values are SI internally (Jutul convention), matching the
# opm-adjoint internal convention; CSV output converts to days / sm3/day
# / bar for human consumption.

module OPMAdjointJutul

using Jutul
using JutulDarcy

export run_forward, well_curves, save_well_curves_csv,
       well_rate_misfit_objective, parameter_sensitivities

"""
    run_forward(deck_path; kwargs...)

Set up a case from an OPM .DATA file and simulate it. Returns
`(case, result)`. Extra keyword arguments are passed to
`simulate_reservoir` (e.g. `info_level = -1`).
"""
function run_forward(deck_path::AbstractString; kwargs...)
    case = setup_case_from_data_file(deck_path)
    result = simulate_reservoir(case; info_level = -1, kwargs...)
    return case, result
end

"""
    well_curves(result, well)

Extract time [s] and oil/gas/water surface rates [m^3/s] plus bhp [Pa]
for `well` (Symbol or String) from a `simulate_reservoir` result.
Rates follow Jutul's sign convention (producers negative).
"""
function well_curves(result, well)
    ws = result.wells
    w = Symbol(well)
    return (time = ws.time,
            orat = ws[w, :orat],
            grat = ws[w, :grat],
            wrat = ws[w, :wrat],
            bhp  = ws[w, :bhp])
end

"""
    save_well_curves_csv(path, result, well)

Write the well curves to CSV in display units
(days, sm3/day with positive production, bar).
"""
function save_well_curves_csv(path::AbstractString, result, well)
    c = well_curves(result, well)
    open(path, "w") do io
        println(io, "time_days,orat_sm3_d,grat_sm3_d,wrat_sm3_d,bhp_bar")
        for i in eachindex(c.time)
            println(io, join((c.time[i] / 86400.0,
                              abs(c.orat[i]) * 86400.0,
                              abs(c.grat[i]) * 86400.0,
                              abs(c.wrat[i]) * 86400.0,
                              c.bhp[i] / 1e5), ","))
        end
    end
    return path
end

# Jutul rate symbol for a phase symbol.
const PHASE_RATE = Dict(:oil => :orat, :gas => :grat, :water => :wrat)

"""
    well_rate_misfit_objective(result_obs, well, phase; w = 1.0)

Build a Jutul sum-objective

    G = sum_k w * dt_k * (q(t_k) - q_obs(t_k))^2

matching the opm-adjoint `matchref` objective, with the observed curve
taken from another simulation result (`result_obs`, e.g. a reference
run). Time alignment is by linear interpolation, all in SI.
"""
function well_rate_misfit_objective(result_obs, well, phase; w = 1.0)
    cobs = well_curves(result_obs, well)
    rate_sym = PHASE_RATE[Symbol(phase)]
    obs_t = collect(cobs.time)
    obs_q = collect(getproperty(cobs, rate_sym))
    wname = Symbol(well)

    function interp_obs(t)
        if t <= obs_t[1]
            return obs_q[1]
        elseif t >= obs_t[end]
            return obs_q[end]
        end
        i = searchsortedfirst(obs_t, t)
        a = (t - obs_t[i-1]) / (obs_t[i] - obs_t[i-1])
        return (1 - a) * obs_q[i-1] + a * obs_q[i]
    end

    function objective(model, state, dt, step_info, forces)
        t = step_info[:time] + dt
        q = JutulDarcy.compute_well_qoi(model, state, forces, wname, rate_sym)
        dq = q - interp_obs(t)
        return w * dt * dq^2
    end
    return objective
end

"""
    parameter_sensitivities(case, result, objective)

Adjoint sensitivities of `objective` with respect to all differentiable
model parameters (Jutul's parameter-AD machinery). Returns the
data-domain sensitivity object; e.g. `sens[:model][:porosity]` for the
porosity gradient. Compare with the opm-adjoint
`<CASE>.ADJOINT_GRADIENTS_PV.txt` via
dJ/dporo_i = g_pvmult_i / poro_i.
"""
function parameter_sensitivities(case, result, objective)
    return JutulDarcy.reservoir_sensitivities(case, result, objective,
                                              include_parameters = true)
end

end # module
