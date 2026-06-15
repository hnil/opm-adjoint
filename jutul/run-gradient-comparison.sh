#!/bin/bash
# Compare an opm-adjoint gradient against the JutulDarcy adjoint gradient
# on the same deck, for the porosity sensitivity of the final-step average
# reservoir pressure. This is the T6 stage-2 cross-check: an independent
# adjoint implementation on the same case.
#
# Usage:
#   run-gradient-comparison.sh <flow_adjoint> <CASE.DATA> <output-dir>
#
# Requirements:
#   - a built flow_adjoint binary (opm-adjoint module);
#   - JulieDarcy installed in this directory's Julia project
#     (juliaup add lts; julia +lts --project=. -e 'using Pkg;
#      Pkg.add(["Jutul","JutulDarcy"]); Pkg.precompile()').
#
# What it does:
#   1. OPM: record the forward run and run the adjoint with the
#      pressure-average objective; read dJ/dporo_i = g_pvmult_i / poro_i.
#   2. Jutul: setup_case_from_data_file + reservoir_sensitivities of the
#      same objective -> sens[:porosity].
#   3. Compare cell-by-cell: Pearson correlation, sign agreement on the
#      dominant cells, and the best-fit scale.
#
# Note: the two forward simulations are NOT identical (well-model
# differences: connection factors, control-switch timing) and the
# "average pressure" objective is defined on different pressure variables
# (OPM: oil-phase primary variable; Jutul: reservoir reference pressure).
# So expect a pattern/sign cross-check, not a tight numerical match. The
# authoritative quantitative test is the in-tree finite-difference check
# (run-adjoint-fd-test.sh and friends).

set -u

FLOW=${1:?usage: $0 <flow_adjoint> <CASE.DATA> <output-dir>}
DECK=${2:?missing deck}
OUTDIR=${3:?missing output dir}

SCRIPTDIR=$(cd "$(dirname "$0")" && pwd)
CASENAME=$(basename "${DECK}" .DATA)
mkdir -p "${OUTDIR}"

# Strict, fixed-stepping flags so the adjoint is well-posed.
STRICT="--enable-storage-cache=false --enable-adaptive-time-stepping=false \
 --threads-per-process=1 --tolerance-cnv=1e-7 --tolerance-cnv-relaxed=1e-7 \
 --tolerance-mb=1e-9 --tolerance-mb-relaxed=1e-9 \
 --adjoint-objective=pressure-average"

echo "=== [1/3] OPM forward + adjoint gradient (pressure-average) ==="
${FLOW} "${DECK}" --output-dir="${OUTDIR}" ${STRICT} --adjoint-file="${OUTDIR}/arch" \
    --adjoint-save=true > "${OUTDIR}/opm_fwd.log" 2>&1 \
    || { echo "OPM forward failed (see ${OUTDIR}/opm_fwd.log)"; exit 1; }
${FLOW} "${DECK}" --output-dir="${OUTDIR}" ${STRICT} --adjoint-file="${OUTDIR}/arch" \
    --adjoint-mode=gradient > "${OUTDIR}/opm_grad.log" 2>&1 \
    || { echo "OPM gradient failed (see ${OUTDIR}/opm_grad.log)"; exit 1; }
OPM_PV="${OUTDIR}/${CASENAME}.ADJOINT_GRADIENTS_PV.txt"
[ -f "${OPM_PV}" ] || { echo "no OPM gradient file ${OPM_PV}"; exit 1; }

echo "=== [2/3] JutulDarcy forward + porosity sensitivity ==="
julia +lts --project="${SCRIPTDIR}" "${SCRIPTDIR}/compare_spe1_pv.jl" \
    "${DECK}" "${OUTDIR}/jutul_poro.csv" > "${OUTDIR}/jutul.log" 2>&1 \
    || { echo "Jutul run failed (see ${OUTDIR}/jutul.log)"; tail -5 "${OUTDIR}/jutul.log"; exit 1; }

echo "=== [3/3] compare dJ/dporo (OPM g_pvmult/poro) vs Jutul sens[:porosity] ==="
# Read the deck's PORO; here we assume a single constant value is fine for
# uniform-porosity decks (SPE1). For non-uniform decks, pass the per-cell
# poro to divide; for the cross-check the correlation is poro-scaling
# invariant when poro is constant.
PORO=$(awk '/^PORO/{getline; while($0 ~ /^--/) getline; print; exit}' "${DECK}" \
        | grep -oE '[0-9]*\*?[0-9.]+' | tail -1 | sed 's/.*\*//')
PORO=${PORO:-0.3}
echo "using constant porosity ${PORO} for the OPM g_pvmult -> dJ/dporo conversion"

python3 - "${OPM_PV}" "${OUTDIR}/jutul_poro.csv" "${PORO}" << 'EOF'
import sys, math
opm_pv, jutul_csv, poro = sys.argv[1], sys.argv[2], float(sys.argv[3])
opm = [float(l)/poro for l in open(opm_pv) if not l.startswith('#')]
jut = [float(l) for l in open(jutul_csv)]
if len(opm) != len(jut):
    print(f"size mismatch: OPM {len(opm)} vs Jutul {len(jut)}"); sys.exit(1)
n = len(opm)
mo, mj = sum(opm)/n, sum(jut)/n
cov = sum((opm[i]-mo)*(jut[i]-mj) for i in range(n))
so = math.sqrt(sum((x-mo)**2 for x in opm))
sj = math.sqrt(sum((x-mj)**2 for x in jut))
corr = cov/(so*sj) if so*sj else float('nan')
scale = sum(opm[i]*jut[i] for i in range(n))/sum(x*x for x in jut)
# sign agreement on the top-decile cells by |OPM|
order = sorted(range(n), key=lambda i: -abs(opm[i]))[:max(1, n//10)]
sign_ok = sum(1 for i in order if (opm[i] > 0) == (jut[i] > 0))
print(f"cells                : {n}")
print(f"OPM  dJ/dporo range  : [{min(opm):.3e}, {max(opm):.3e}]")
print(f"Jutul sens[:porosity]: [{min(jut):.3e}, {max(jut):.3e}]")
print(f"Pearson correlation  : {corr:.4f}")
print(f"best-fit scale OPM~s*Jutul : s={scale:.4f}")
print(f"sign agreement (top {len(order)} cells by |OPM|): {sign_ok}/{len(order)}")
print()
print("Cross-check verdict: correlation > 0.7 and sign-consistent on the")
print("dominant cells indicates the opm-adjoint gradient matches the")
print("JutulDarcy adjoint in structure and sign (a tight numerical match")
print("is not expected; see the header note).")
EOF
