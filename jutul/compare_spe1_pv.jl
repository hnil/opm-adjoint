# Compare JutulDarcy vs opm-adjoint: gradient of the final-step average
# reservoir pressure w.r.t. porosity, on SPE1CASE1.
#
#   julia +lts --project=<this dir> compare_spe1_pv.jl <SPE1.DATA> <out.csv>
#
# OPM side (run separately):
#   flow_adjoint SPE1CASE1.DATA --enable-storage-cache=false \
#       --enable-adaptive-time-stepping=false --adjoint-objective=pressure-average \
#       --adjoint-save=true
#   flow_adjoint ... --adjoint-mode=gradient
#   # dJ/dporo_i = g_pvmult_i / poro_i  from SPE1CASE1.ADJOINT_GRADIENTS_PV.txt

using JutulDarcy, Jutul
using DelimitedFiles

deck = length(ARGS) >= 1 ? ARGS[1] : error("usage: compare_spe1_pv.jl <DATA> <out.csv>")
outcsv = length(ARGS) >= 2 ? ARGS[2] : "jutul_pv.csv"

println("setup_case_from_data_file: ", deck)
case = setup_case_from_data_file(deck)
result = simulate_reservoir(case; info_level = -1)

# number of report steps
nstep = length(case.dt)
println("report steps: ", nstep)

# Objective: average reservoir pressure at the FINAL step only (to match
# the opm-adjoint pressure-average objective). reservoir_sensitivities
# sums the per-step objective, so gate on the last step.
function final_pressure_average(model, state, dt, step_info, forces)
    if step_info[:step] != nstep
        return 0.0
    end
    p = state[:Reservoir][:Pressure]
    return sum(p) / length(p)
end

println("reservoir_sensitivities ...")
sens = JutulDarcy.reservoir_sensitivities(case, result, final_pressure_average,
                                          include_parameters = true)

poro_grad = sens[:porosity]
println("porosity gradient: n=", length(poro_grad),
        " min=", minimum(poro_grad), " max=", maximum(poro_grad))
writedlm(outcsv, poro_grad)
println("wrote ", outcsv)
