#!/bin/bash
# Finite-difference verification of the adjoint PERMEABILITY gradient
# (chain rule through the one-sided half transmissibilities),
# specialized for decks whose PERMX comes from a single text include
# (e.g. adjoint_tests/MODEL_1D_DEBUG, METRIC units -> PERMX in mD).
#
#   FD:      dJ/dPERMX_i ~ (J(K_i + d) - J(K_i - d)) / (2 d)   [per mD]
#   adjoint: column 1 of <CASE>.ADJOINT_GRADIENTS_PERM.txt (per m^2)
#            * 9.869233e-16 (mD -> m^2)
#
# Usage:
#   run-adjoint-fd-perm-test.sh <flow_adjoint> <inputfiles-dir> <deck-name> <output-dir> [delta_mD] [rel-tol] [objective]

set -u

FLOW=${1:?usage: $0 <flow_adjoint> <inputfiles-dir> <deck-name> <output-dir> [delta] [rel-tol] [objective]}
INPUTDIR=${2:?missing inputfiles dir}
DECKNAME=${3:?missing deck name}
OUTDIR=${4:?missing output dir}
DELTA=${5:-1e-3}
RELTOL=${6:-1e-3}
OBJECTIVE=${7:-pressure-average}

MD_TO_SI=9.869233e-16

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

echo "=== base run + adjoint gradient (objective ${OBJECTIVE})"
BASE=${OUTDIR}/base
mkdir -p "${BASE}"
cp -r "${INPUTDIR}" "${OUTDIR}/inputfiles_base"
${FLOW} "${OUTDIR}/inputfiles_base/${DECKNAME}.DATA" --output-dir="${BASE}" ${STRICT} \
    --adjoint-save=true > "${BASE}/fwd.log" 2>&1 || { echo "base forward failed"; exit 1; }
${FLOW} "${OUTDIR}/inputfiles_base/${DECKNAME}.DATA" --output-dir="${BASE}" ${STRICT} \
    --adjoint-mode=gradient > "${BASE}/grad.log" 2>&1 || { echo "gradient run failed"; exit 1; }
GRADFILE=${BASE}/${DECKNAME}.ADJOINT_GRADIENTS_PERM.txt
[ -f "${GRADFILE}" ] || { echo "no gradient file ${GRADFILE}"; exit 1; }

NCELLS=$(grep -cvE "PERMX|/" "${INPUTDIR}/permx.txt")
echo "=== ${NCELLS} cells, FD delta ${DELTA} mD"

status=0
i=0
while [ $i -lt ${NCELLS} ]; do
    for sign in + -; do
        dir=${OUTDIR}/fd_${i}_${sign}
        cp -r "${INPUTDIR}" "${OUTDIR}/inputfiles_${i}_${sign}"
        python3 - "$INPUTDIR/permx.txt" "${OUTDIR}/inputfiles_${i}_${sign}/permx.txt" $i ${sign}${DELTA} << 'EOF'
import sys
src, dst, idx, delta = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
out, n = [], 0
for line in open(src):
    t = line.strip()
    if t and t != '/' and not t.startswith('PERMX'):
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
    read FD ADJ ERR OK << EOF
$(python3 -c "
fd = ($JP - $JM) / (2 * $DELTA)                       # per mD
rows = [l.split() for l in open('$GRADFILE').read().splitlines()[1:]]
adj = float(rows[$i][0]) * $MD_TO_SI                  # per m^2 -> per mD
err = abs(fd - adj) / max(abs(fd), abs(adj), 1e-300)
print(fd, adj, err, 1 if err < $RELTOL else 0)")
EOF
    printf "cell %2d: FD %15.6e  adjoint %15.6e  rel.err %9.2e\n" $i $FD $ADJ $ERR
    [ "$OK" = "1" ] || status=1
    i=$((i + 1))
done

if [ ${status} -eq 0 ]; then
    echo "T5 adjoint-vs-FD (permeability): PASS (rel tol ${RELTOL})"
else
    echo "T5 adjoint-vs-FD (permeability): FAIL"
fi
exit ${status}
