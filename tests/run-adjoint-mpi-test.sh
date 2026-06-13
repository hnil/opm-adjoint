#!/bin/bash
# Parallel-vs-serial verification of the adjoint gradient (MPI
# milestone): the same gradient computation run on 1 rank and on N ranks
# must agree. Forward recording and backward replay are both per-rank
# (the HDF5 archive stores one dataset per rank via PROCESS_SPLIT; the
# directory store uses a per-rank sub-directory), and the transposed
# solve uses the parallel ghost-last operator. The gradient files are
# compared as a sorted value multiset, so the comparison is independent
# of the cell ordering (which differs between the serial grid and the
# partitioned one).
#
# Parallel runs use --adjoint-linear-solver=cpr: cprt (transpose-CPR)
# falsely converges in parallel and is rejected at runtime; cpr and ilu0
# are correct (verified to ~5e-6 vs serial on SPE1).
#
# Usage:
#   run-adjoint-mpi-test.sh <flow_adjoint> <deck.DATA> <output-dir> [nprocs] [rel-tol]

set -u

FLOW=${1:?usage: $0 <flow_adjoint> <deck.DATA> <output-dir> [nprocs] [rel-tol]}
DECK=${2:?missing deck}
OUTDIR=${3:?missing output dir}
NP=${4:-2}
RELTOL=${5:-1e-4}

MPIRUN=${MPIRUN:-mpirun}
if ! command -v "${MPIRUN}" >/dev/null 2>&1; then
    echo "SKIP: ${MPIRUN} not found (no MPI launcher)"
    exit 0
fi

DECKNAME=$(basename "${DECK}" .DATA)
# Directory store (--adjoint-file without .h5): dependency-free and
# per-rank, so the test does not require parallel HDF5.
STRICT="--enable-storage-cache=false \
 --tolerance-cnv=1e-7 --tolerance-cnv-relaxed=1e-7 \
 --tolerance-mb=1e-9 --tolerance-mb-relaxed=1e-9 \
 --adjoint-objective=pressure-average"

run() { # <nprocs> <subdir> <extra-flags...>
    local np=$1 sub=$2; shift 2
    local dir="${OUTDIR}/${sub}"
    mkdir -p "${dir}"
    local launch=""
    [ "${np}" -gt 1 ] && launch="${MPIRUN} -np ${np}"
    ${launch} ${FLOW} "${DECK}" --output-dir="${dir}" ${STRICT} \
        --adjoint-file="${dir}/arch" "$@" > "${dir}/run.log" 2>&1
    return $?
}

mkdir -p "${OUTDIR}"

echo "=== serial (np=1) record + gradient ==="
run 1 serial --adjoint-save=true --threads-per-process=1 \
    || { echo "serial forward FAILED"; tail -5 "${OUTDIR}/serial/run.log"; exit 1; }
run 1 serial --adjoint-mode=gradient --adjoint-linear-solver=cpr \
    --threads-per-process=1 \
    || { echo "serial gradient FAILED"; tail -5 "${OUTDIR}/serial/run.log"; exit 1; }
cp "${OUTDIR}/serial/${DECKNAME}.ADJOINT_GRADIENTS_PV.txt" "${OUTDIR}/pv_serial.txt"

echo "=== parallel (np=${NP}) record + gradient ==="
run "${NP}" parallel --adjoint-save=true \
    || { echo "parallel forward FAILED"; tail -8 "${OUTDIR}/parallel/run.log"; exit 1; }
run "${NP}" parallel --adjoint-mode=gradient --adjoint-linear-solver=cpr \
    || { echo "parallel gradient FAILED"; tail -8 "${OUTDIR}/parallel/run.log"; exit 1; }
cp "${OUTDIR}/parallel/${DECKNAME}.ADJOINT_GRADIENTS_PV.txt" "${OUTDIR}/pv_parallel.txt"

echo "=== compare (sorted value multiset) ==="
python3 - "${OUTDIR}/pv_serial.txt" "${OUTDIR}/pv_parallel.txt" "${RELTOL}" << 'EOF'
import sys
serial, parallel, tol = sys.argv[1], sys.argv[2], float(sys.argv[3])
def load(p):
    return sorted(float(l) for l in open(p) if not l.startswith('#'))
a, b = load(serial), load(parallel)
if len(a) != len(b):
    print(f"FAIL: cell count {len(a)} (serial) vs {len(b)} (parallel)")
    sys.exit(1)
worst = max((abs(x - y) / max(abs(x), abs(y), 1e-30) for x, y in zip(a, b)),
            default=0.0)
print(f"{len(a)} cells, max rel diff (sorted) = {worst:.3e}")
if worst <= tol:
    print(f"adjoint MPI np=N vs serial: PASS (rel tol {tol})")
    sys.exit(0)
print("adjoint MPI np=N vs serial: FAIL")
sys.exit(1)
EOF
