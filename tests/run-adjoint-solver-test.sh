#!/bin/bash
# Iterative-vs-direct verification of the adjoint linear solver (the
# plan's v1.5 spike): the same gradient computation run with
# --adjoint-linear-solver=umfpack (direct reference) and the iterative
# presets (ilu0, cpr, cprt on the explicitly transposed matrix), with
# the gradient files compared entry-wise. Also reports the iteration
# counts - on SPE1 cprt (CPR built for transposed systems) needs ~4x
# fewer iterations than ILU0/cpr, which is the cprt health check asked
# for in the plan.
#
# Usage:
#   run-adjoint-solver-test.sh <flow_adjoint> <deck.DATA> <output-dir> [rel-tol] [solvers...]
#
# Default rel-tol 1e-9 (m1d converges in 1-2 iterations and the
# default reduction 1e-14 gives near-machine agreement); default
# solvers "ilu0 cpr cprt".

set -u

FLOW=${1:?usage: $0 <flow_adjoint> <deck.DATA> <output-dir> [rel-tol] [solvers...]}
DECK=${2:?missing deck}
OUTDIR=${3:?missing output dir}
RELTOL=${4:-1e-9}
shift 4 2>/dev/null || shift $#
SOLVERS=${@:-ilu0 cpr cprt}

DECKNAME=$(basename "${DECK}" .DATA)
STRICT="--threads-per-process=1 --enable-storage-cache=false \
 --tolerance-cnv=1e-7 --tolerance-cnv-relaxed=1e-7 \
 --tolerance-mb=1e-9 --tolerance-mb-relaxed=1e-9 \
 --adjoint-objective=pressure-average"

mkdir -p "${OUTDIR}/run"

echo "=== forward (recording)"
${FLOW} "${DECK}" --output-dir="${OUTDIR}/run" ${STRICT} \
    --adjoint-save=true > "${OUTDIR}/fwd.log" 2>&1 \
    || { echo "forward FAILED"; exit 1; }

status=0
for solver in umfpack ${SOLVERS}; do
    ${FLOW} "${DECK}" --output-dir="${OUTDIR}/run" ${STRICT} \
        --adjoint-mode=gradient --adjoint-linear-solver="${solver}" \
        > "${OUTDIR}/grad_${solver}.log" 2>&1 \
        || { echo "${solver}: gradient run FAILED"; status=1; continue; }
    for kind in PV TRANS; do
        cp "${OUTDIR}/run/${DECKNAME}.ADJOINT_GRADIENTS_${kind}.txt" \
           "${OUTDIR}/${kind}_${solver}.txt"
    done
    iters=$(grep -o "Adjoint linear solver.*" "${OUTDIR}/grad_${solver}.log" || true)
    echo "${solver}: OK ${iters:+(${iters})}"
done

for solver in ${SOLVERS}; do
    [ -f "${OUTDIR}/PV_${solver}.txt" ] || continue
    for kind in PV TRANS; do
        result=$(python3 - "${OUTDIR}/${kind}_umfpack.txt" \
                           "${OUTDIR}/${kind}_${solver}.txt" "${RELTOL}" << 'EOF'
import sys
ref, test, tol = sys.argv[1], sys.argv[2], float(sys.argv[3])
def load(p):
    return [[float(v) for v in l.split()] for l in open(p)
            if not l.startswith('#')]
a, b = load(ref), load(test)
worst = max((abs(x - y) / max(abs(x), abs(y), 1e-30)
             for ra, rb in zip(a, b) for x, y in zip(ra, rb)), default=0.0)
print(f"{worst:.3e}")
sys.exit(0 if worst <= tol and len(a) == len(b) else 1)
EOF
)
        if [ $? -eq 0 ]; then
            echo "${solver}/${kind} vs umfpack: OK (max rel diff ${result})"
        else
            echo "${solver}/${kind} vs umfpack: FAIL (max rel diff ${result})"
            status=1
        fi
    done
done

if [ ${status} -eq 0 ]; then
    echo "adjoint solver comparison: PASS (rel tol ${RELTOL})"
else
    echo "adjoint solver comparison: FAIL"
fi
exit ${status}
