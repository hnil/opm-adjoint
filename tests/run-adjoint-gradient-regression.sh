#!/bin/bash
# Gradient regression test: re-runs the FD-verified gradient cases and
# compares the adjoint gradients against stored references
# (tests/references/). The references were generated from runs whose
# gradients were verified against central finite differences (see
# run-adjoint-fd-test.sh and friends), so this guards the whole chain --
# recording, replay, Bdiag, well coupling, transposed solves, gradient
# accumulation -- against regressions without the cost of FD loops.
#
# Usage:
#   run-adjoint-gradient-regression.sh <flow_adjoint> <m1d-inputfiles-dir> <output-dir> [rel-tol] [--update]
#
# With --update the references are regenerated instead of compared.

set -u

FLOW=${1:?usage: $0 <flow_adjoint> <m1d-inputfiles-dir> <output-dir> [rel-tol] [--update]}
INPUTDIR=${2:?missing MODEL_1D_DEBUG inputfiles dir}
OUTDIR=${3:?missing output dir}
RELTOL=${4:-1e-6}
UPDATE=${5:-}

SCRIPTDIR=$(cd "$(dirname "$0")" && pwd)
REFDIR=${SCRIPTDIR}/references

DECKNAME=MODEL_1D_DEBUG
STRICT="--threads-per-process=1 --enable-storage-cache=false \
 --enable-adaptive-time-stepping=false \
 --tolerance-cnv=1e-7 --tolerance-cnv-relaxed=1e-7 \
 --tolerance-mb=1e-9 --tolerance-mb-relaxed=1e-9"

# case-name objective orat(- for none)
CASES="
pressure_pv_trans pressure-average -
bhp_prod bhp:Prod 1.0
rate_oil rate:Prod:oil -
"

transform_orat() { # <dir> <orat>
    local dir=$1 orat=$2
    [ "$orat" = "-" ] && return
    local wcf wellname
    for wcf in "${dir}"/wconprodstep_*.txt; do
        wellname=$(awk 'NR==2 {print $1}' "${wcf}")
        printf 'WCONPROD\n%s OPEN ORAT %s 4* 50.0 /\n/\n' "${wellname}" "${orat}" > "${wcf}"
    done
}

compare_file() { # <test-file> <ref-file>
    python3 - "$1" "$2" "${RELTOL}" << 'EOF'
import sys
test, ref, tol = sys.argv[1], sys.argv[2], float(sys.argv[3])
def load(path):
    rows = []
    for line in open(path):
        if line.startswith('#'):
            continue
        rows.append([float(v) for v in line.split()])
    return rows
a, b = load(test), load(ref)
if len(a) != len(b):
    print(f"FAIL size {len(a)} vs {len(b)}")
    sys.exit(1)
worst = 0.0
for ra, rb in zip(a, b):
    for va, vb in zip(ra, rb):
        err = abs(va - vb) / max(abs(va), abs(vb), 1e-30)
        worst = max(worst, err)
print(f"max rel diff {worst:.3e}")
sys.exit(0 if worst <= tol else 1)
EOF
}

mkdir -p "${OUTDIR}" "${REFDIR}"
status=0

echo "${CASES}" | while read -r name objective orat; do
    [ -z "${name}" ] && continue
    rundir=${OUTDIR}/${name}
    indir=${OUTDIR}/input_${name}
    rm -rf "${rundir}" "${indir}"
    cp -r "${INPUTDIR}" "${indir}"
    transform_orat "${indir}" "${orat}"
    mkdir -p "${rundir}"
    ${FLOW} "${indir}/${DECKNAME}.DATA" --output-dir="${rundir}" ${STRICT} \
        --adjoint-save=true --adjoint-objective="${objective}" \
        > "${rundir}/fwd.log" 2>&1 || { echo "${name}: forward FAILED"; exit 1; }
    ${FLOW} "${indir}/${DECKNAME}.DATA" --output-dir="${rundir}" ${STRICT} \
        --adjoint-mode=gradient --adjoint-objective="${objective}" \
        > "${rundir}/grad.log" 2>&1 || { echo "${name}: gradient FAILED"; exit 1; }

    for kind in PV TRANS; do
        test_file=${rundir}/${DECKNAME}.ADJOINT_GRADIENTS_${kind}.txt
        ref_file=${REFDIR}/${name}_${kind}.txt
        if [ "${UPDATE}" = "--update" ]; then
            cp "${test_file}" "${ref_file}"
            echo "${name}/${kind}: reference updated"
        else
            [ -f "${ref_file}" ] || { echo "${name}/${kind}: MISSING reference"; exit 1; }
            result=$(compare_file "${test_file}" "${ref_file}") \
                || { echo "${name}/${kind}: FAIL (${result})"; exit 1; }
            echo "${name}/${kind}: OK (${result})"
        fi
    done
done
status=$?

if [ ${status} -eq 0 ] && [ "${UPDATE}" != "--update" ]; then
    echo "adjoint gradient regression: PASS (rel tol ${RELTOL})"
fi
exit ${status}
