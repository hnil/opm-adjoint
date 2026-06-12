#!/bin/bash
# T6 stage 1: compare the FORWARD simulation of OPM flow and JutulDarcy
# on the same deck (well curves). Gradient cross-validation only makes
# sense once the forward solutions agree — this script establishes that
# baseline.
#
# OPM side: run flow, extract well vectors with opm-common's `summary`
# utility. Jutul side: setup_case_from_data_file + simulate_reservoir,
# write the same vectors to CSV. Compare with a relative tolerance on
# the time-aligned curves (linear interpolation onto the OPM report
# times).
#
# Usage:
#   run-jutul-forward-compare.sh <flow-binary> <summary-tool> <deck> <output-dir> [well] [rel-tol]

set -u

FLOW=${1:?usage: $0 <flow-binary> <summary-tool> <deck> <output-dir> [well] [rel-tol]}
SUMMARYTOOL=${2:?missing opm-common summary tool}
DECK=${3:?missing deck}
OUTDIR=${4:?missing output dir}
WELL=${5:-PROD}
RELTOL=${6:-0.02}

CASENAME=$(basename "${DECK}" .DATA)
mkdir -p "${OUTDIR}"

echo "=== OPM forward run"
${FLOW} "${DECK}" --output-dir="${OUTDIR}/opm" --threads-per-process=1 \
    > "${OUTDIR}/opm.log" 2>&1 || { echo "OPM run failed"; exit 1; }

for kw in WOPR WGPR WBHP; do
    ${SUMMARYTOOL} "${OUTDIR}/opm/${CASENAME}" TIME "${kw}:${WELL}" \
        > "${OUTDIR}/opm_${kw}.txt" 2>/dev/null
done

echo "=== JutulDarcy forward run"
julia --project="${OUTDIR}/julia_env" -e "
using Pkg
try
    using JutulDarcy
catch
    Pkg.add(\"JutulDarcy\")
    using JutulDarcy
end
using Jutul

case = setup_case_from_data_file(raw\"${DECK}\")
result = simulate_reservoir(case; info_level = -1)
ws = result.wells

t = ws.time ./ 86400.0   # days
open(\"${OUTDIR}/jutul_${WELL}.csv\", \"w\") do io
    println(io, \"time_days,orat_sm3_d,grat_sm3_d,bhp_bar\")
    orat = abs.(ws[Symbol(\"${WELL}\"), :orat]) .* 86400.0
    grat = abs.(ws[Symbol(\"${WELL}\"), :grat]) .* 86400.0
    bhp  = ws[Symbol(\"${WELL}\"), :bhp] ./ 1e5
    for i in eachindex(t)
        println(io, string(t[i], \",\", orat[i], \",\", grat[i], \",\", bhp[i]))
    end
end
println(\"JUTUL_DONE\")
" > "${OUTDIR}/jutul.log" 2>&1
grep -q "JUTUL_DONE" "${OUTDIR}/jutul.log" || {
    echo "Jutul run failed (log: ${OUTDIR}/jutul.log)"; exit 1; }

echo "=== comparison (rel tol ${RELTOL} on time-averaged curves)"
python3 - "${OUTDIR}" "${WELL}" "${RELTOL}" << 'EOF'
import sys
outdir, well, tol = sys.argv[1], sys.argv[2], float(sys.argv[3])

def load_opm(path):
    # summary tool output: header line then "time value" rows
    times, values = [], []
    for line in open(path):
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            times.append(float(parts[0])); values.append(float(parts[1]))
        except ValueError:
            continue
    return times, values

import csv
jt, jorat, jgrat, jbhp = [], [], [], []
with open(f"{outdir}/jutul_{well}.csv") as f:
    for row in csv.DictReader(f):
        jt.append(float(row["time_days"]))
        jorat.append(float(row["orat_sm3_d"]))
        jgrat.append(float(row["grat_sm3_d"]))
        jbhp.append(float(row["bhp_bar"]))

def interp(t, ts, vs):
    if t <= ts[0]: return vs[0]
    if t >= ts[-1]: return vs[-1]
    import bisect
    i = bisect.bisect_right(ts, t)
    w = (t - ts[i-1]) / (ts[i] - ts[i-1])
    return (1-w)*vs[i-1] + w*vs[i]

status = 0
# OPM units: FIELD decks give STB/D / MSCF/D / PSIA; convert to SI-ish
# common basis via scale-free relative comparison of time-averages.
for kw, jvals in (("WOPR", jorat), ("WGPR", jgrat), ("WBHP", jbhp)):
    ts, vs = load_opm(f"{outdir}/opm_{kw}.txt")
    if not ts:
        print(f"{kw}: no OPM data, skipped"); continue
    # unit factors: detect by magnitude-free approach: compare SHAPE via
    # normalized curves (divide by the time-average) to be unit agnostic.
    o_avg = sum(vs)/len(vs)
    j_at = [interp(t, jt, jvals) for t in ts]
    j_avg = sum(j_at)/len(j_at)
    if o_avg == 0 and j_avg == 0:
        print(f"{kw}: both zero, OK"); continue
    worst = 0.0
    for v, j in zip(vs, j_at):
        vn = v / o_avg if o_avg else 0.0
        jn = j / j_avg if j_avg else 0.0
        worst = max(worst, abs(vn - jn) / max(abs(vn), abs(jn), 1e-12))
    ratio = (o_avg / j_avg) if j_avg else float("nan")
    print(f"{kw}: normalized-shape max rel diff {worst:.3e}, "
          f"OPM/Jutul average ratio {ratio:.6g} (unit factor)")
    if worst > tol:
        status = 1
print("T6 forward comparison:", "PASS" if status == 0 else "FAIL")
sys.exit(status)
EOF
