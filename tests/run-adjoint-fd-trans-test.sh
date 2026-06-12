#!/bin/bash
# Finite-difference verification of the adjoint transmissibility-multiplier
# gradient (test level T5), specialized for the 1D MODEL_1D_DEBUG deck:
# MULTX on cell I scales exactly the transmissibility of the face between
# I and I+1, matching dJ/d(tmult_f) one to one.
#
# Usage:
#   run-adjoint-fd-trans-test.sh <flow_adjoint> <inputfiles-dir> <deck-name> <output-dir> [delta] [rel-tol] [objective]

set -u

FLOW=${1:?usage: $0 <flow_adjoint> <inputfiles-dir> <deck-name> <output-dir> [delta] [rel-tol] [objective]}
INPUTDIR=${2:?missing inputfiles dir}
DECKNAME=${3:?missing deck name}
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

# Copy the input files and add a MULTX include to the GRID section
# (inserted right after the poro.txt include).
prepare_input() { # <dstdir> <multx-values...>
    local dst=$1; shift
    rm -rf "${dst}"
    cp -r "${INPUTDIR}" "${dst}"
    transform_input "${dst}"
    {
        echo "MULTX"
        for v in "$@"; do echo "$v"; done
        echo "/"
    } > "${dst}/multx.txt"
    python3 - "${dst}/${DECKNAME}.DATA" << 'EOF'
import sys
path = sys.argv[1]
lines = open(path).read().splitlines(keepends=True)
out = []
for i, line in enumerate(lines):
    out.append(line)
    if line.strip() == 'poro.txt':
        # after "poro.txt" comes the closing "/" of its INCLUDE
        out.append(lines[i+1])
        out.append("\nINCLUDE\nmultx.txt\n/\n")
        lines[i+1] = ''
open(path, 'w').writelines(out)
EOF
}

run_J() { # <inputdir> <rundir> -> prints J
    local inputdir=$1 rundir=$2
    mkdir -p "${rundir}"
    ${FLOW} "${inputdir}/${DECKNAME}.DATA" --output-dir="${rundir}" ${STRICT} \
        --adjoint-save=true > "${rundir}/fwd.log" 2>&1 || { echo FAIL; return; }
    ${FLOW} "${inputdir}/${DECKNAME}.DATA" --output-dir="${rundir}" ${STRICT} \
        --adjoint-mode=objective > "${rundir}/obj.log" 2>&1 || { echo FAIL; return; }
    grep -o "Adjoint objective J = [0-9eE.+-]*" "${rundir}/obj.log" | awk '{print $5}'
}

NCELLS=$(grep -cvE "PORO|/" "${INPUTDIR}/poro.txt")
NFACES=$((NCELLS - 1))
mkdir -p "${OUTDIR}"
echo "=== ${NCELLS} cells, ${NFACES} x-faces, FD delta ${DELTA}, objective ${OBJECTIVE}"

# Base run with MULTX = 1 everywhere (so the FD decks differ only in one value).
BASEIN=${OUTDIR}/inputfiles_base
prepare_input "${BASEIN}" $(python3 -c "print(' '.join(['1.0']*${NCELLS}))")
BASE=${OUTDIR}/base
mkdir -p "${BASE}"
${FLOW} "${BASEIN}/${DECKNAME}.DATA" --output-dir="${BASE}" ${STRICT} \
    --adjoint-save=true > "${BASE}/fwd.log" 2>&1 || { echo "base forward failed"; exit 1; }
${FLOW} "${BASEIN}/${DECKNAME}.DATA" --output-dir="${BASE}" ${STRICT} \
    --adjoint-mode=gradient > "${BASE}/grad.log" 2>&1 || { echo "gradient run failed"; exit 1; }
GRADFILE=${BASE}/${DECKNAME}.ADJOINT_GRADIENTS_TRANS.txt
[ -f "${GRADFILE}" ] || { echo "no gradient file ${GRADFILE}"; exit 1; }

status=0
f=0
while [ $f -lt ${NFACES} ]; do
    for sign in + -; do
        vals=$(python3 -c "
v = ['1.0'] * ${NCELLS}
v[$f] = repr(1.0 ${sign} ${DELTA})
print(' '.join(v))")
        dir=${OUTDIR}/fd_${f}_${sign}
        prepare_input "${OUTDIR}/inputfiles_${f}_${sign}" ${vals}
        J=$(run_J "${OUTDIR}/inputfiles_${f}_${sign}" "${dir}")
        [ "$J" = "FAIL" ] && { echo "FD run face $f $sign failed"; exit 1; }
        if [ "$sign" = "+" ]; then JP=$J; else JM=$J; fi
    done
    read FD ADJ ERR OK << EOF
$(python3 -c "
fd = ($JP - $JM) / (2 * $DELTA)
# gradient file lines: '# ...' then 'I J g'
for line in open('$GRADFILE').read().splitlines()[1:]:
    parts = line.split()
    if int(parts[0]) == $f and int(parts[1]) == $f + 1:
        adj = float(parts[2])
        break
err = abs(fd - adj) / max(abs(fd), abs(adj), 1e-300)
print(fd, adj, err, 1 if err < $RELTOL else 0)")
EOF
    printf "face %d-%d: FD %15.6e  adjoint %15.6e  rel.err %9.2e\n" $f $((f+1)) $FD $ADJ $ERR
    [ "$OK" = "1" ] || status=1
    f=$((f + 1))
done

if [ ${status} -eq 0 ]; then
    echo "T5 adjoint-vs-FD (transmissibility): PASS (rel tol ${RELTOL})"
else
    echo "T5 adjoint-vs-FD (transmissibility): FAIL"
fi
exit ${status}
