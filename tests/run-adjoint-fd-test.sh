#!/bin/bash
# Finite-difference verification of the adjoint pore-volume gradient
# (test level T5 of the adjoint plan), specialized for decks whose PORO
# comes from a single text include (e.g. adjoint_tests/MODEL_1D_DEBUG).
#
# For each cell i:
#   FD:      dJ/dporo_i ~ (J(poro_i + d) - J(poro_i - d)) / (2 d)
#   adjoint: dJ/dporo_i = g_pvmult_i / poro_i   (pore volume linear in poro)
#
# All runs use fixed report-step time stepping
# (--enable-adaptive-time-stepping=false): with adaptive stepping a tiny
# parameter perturbation can change the substep sequence, and the
# resulting discretization difference would swamp the FD signal.
#
# Usage:
#   run-adjoint-fd-test.sh <flow_adjoint-binary> <inputfiles-dir> <deck-name> <output-dir> [delta] [rel-tol] [objective] [orat]
#
# If [orat] is given, all WCONPROD step files are rewritten to ORAT
# control with that target (BHP floor 50 bar), so the producer BHP is a
# free variable - required for bhp:<WELL> objectives (under BHP control
# the BHP is pinned by the control equation and the gradient is
# legitimately zero).

set -u

FLOW=${1:?usage: $0 <flow_adjoint> <inputfiles-dir> <deck-name> <output-dir> [delta] [rel-tol]}
INPUTDIR=${2:?missing inputfiles dir}
DECKNAME=${3:?missing deck name (without .DATA)}
OUTDIR=${4:?missing output dir}
DELTA=${5:-1e-6}
RELTOL=${6:-1e-3}
OBJECTIVE=${7:-pressure-average}
ORAT=${8:-}

transform_input() { # <inputdir>
    [ -z "${ORAT}" ] && return
    local dir=$1
    local wcf wellname
    for wcf in "${dir}"/wconprodstep_*.txt; do
        wellname=$(awk 'NR==2 {print $1}' "${wcf}")
        printf 'WCONPROD\n%s OPEN ORAT %s 4* 50.0 /\n/\n' "${wellname}" "${ORAT}" > "${wcf}"
    done
}

STRICT="--threads-per-process=1 --enable-storage-cache=false \
 --enable-adaptive-time-stepping=false \
 --tolerance-cnv=1e-7 --tolerance-cnv-relaxed=1e-7 \
 --tolerance-mb=1e-9 --tolerance-mb-relaxed=1e-9 \
 --adjoint-objective=${OBJECTIVE}"

run_J() { # <inputdir> <rundir>  -> prints J
    local inputdir=$1 rundir=$2
    mkdir -p "${rundir}"
    ${FLOW} "${inputdir}/${DECKNAME}.DATA" --output-dir="${rundir}" ${STRICT} \
        --adjoint-save=true > "${rundir}/fwd.log" 2>&1 || { echo FAIL; return; }
    ${FLOW} "${inputdir}/${DECKNAME}.DATA" --output-dir="${rundir}" ${STRICT} \
        --adjoint-mode=objective > "${rundir}/obj.log" 2>&1 || { echo FAIL; return; }
    grep -o "Adjoint objective J = [0-9eE.+-]*" "${rundir}/obj.log" | awk '{print $5}'
}

mkdir -p "${OUTDIR}"

echo "=== base run: forward + adjoint gradient"
BASE=${OUTDIR}/base
mkdir -p "${BASE}"
cp -r "${INPUTDIR}" "${OUTDIR}/inputfiles_base"
transform_input "${OUTDIR}/inputfiles_base"
${FLOW} "${OUTDIR}/inputfiles_base/${DECKNAME}.DATA" --output-dir="${BASE}" ${STRICT} \
    --adjoint-save=true > "${BASE}/fwd.log" 2>&1 || { echo "base forward failed"; exit 1; }
${FLOW} "${OUTDIR}/inputfiles_base/${DECKNAME}.DATA" --output-dir="${BASE}" ${STRICT} \
    --adjoint-mode=gradient > "${BASE}/grad.log" 2>&1 || { echo "gradient run failed"; exit 1; }
GRADFILE=${BASE}/${DECKNAME}.ADJOINT_GRADIENTS_PV.txt
[ -f "${GRADFILE}" ] || { echo "no gradient file ${GRADFILE}"; exit 1; }

# Cell porosities and count from the include (skip keyword line and final /).
POROS=($(grep -vE "PORO|/" "${INPUTDIR}/poro.txt"))
NCELLS=${#POROS[@]}
echo "=== ${NCELLS} cells, FD delta ${DELTA}"

status=0
i=0
maxerr=0
while [ $i -lt ${NCELLS} ]; do
    for sign in + -; do
        dir=${OUTDIR}/fd_${i}_${sign}
        cp -r "${INPUTDIR}" "${OUTDIR}/inputfiles_${i}_${sign}"
        transform_input "${OUTDIR}/inputfiles_${i}_${sign}"
        python3 - "$INPUTDIR/poro.txt" "${OUTDIR}/inputfiles_${i}_${sign}/poro.txt" $i ${sign}${DELTA} << 'EOF'
import sys
src, dst, idx, delta = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
out, n = [], 0
for line in open(src):
    t = line.strip()
    if t and t != '/' and not t.startswith('PORO'):
        if n == idx:
            line = f"{float(t) + delta:.16e}\n"
        n += 1
    out.append(line)
open(dst, 'w').writelines(out)
EOF
        J=$(run_J "${OUTDIR}/inputfiles_${i}_${sign}" "${dir}")
        [ "$J" = "FAIL" ] && { echo "FD run cell $i $sign failed"; exit 1; }
        if [ "$sign" = "+" ]; then JP=$J; else JM=$J; fi
    done
    read FD ADJ ERR MAXOK << EOF
$(python3 -c "
fd = ($JP - $JM) / (2 * $DELTA)
g = float(open('$GRADFILE').read().splitlines()[$((i + 1))])
adj = g / ${POROS[$i]}
err = abs(fd - adj) / max(abs(fd), abs(adj), 1e-300)
print(fd, adj, err, 1 if err < $RELTOL else 0)")
EOF
    printf "cell %2d: FD %15.6e  adjoint %15.6e  rel.err %9.2e\n" $i $FD $ADJ $ERR
    [ "$MAXOK" = "1" ] || status=1
    i=$((i + 1))
done

if [ ${status} -eq 0 ]; then
    echo "T5 adjoint-vs-FD (pore volume): PASS (rel tol ${RELTOL})"
else
    echo "T5 adjoint-vs-FD (pore volume): FAIL"
fi
exit ${status}
