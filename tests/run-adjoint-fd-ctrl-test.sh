#!/bin/bash
# Finite-difference verification of the well-CONTROL gradient dJ/du
# (towards rate optimization): the producer of MODEL_1D_DEBUG is put on
# ORAT control with target u, the objective is the time-integrated BHP
# of the producer (free under rate control), and
#   FD:      dJ/du ~ (J(u+d) - J(u-d)) / (2 d)        [per sm3/day]
#   adjoint: dJ/du = -sum_k lambda_w,k[last] / 86400   [converted to /sm3/day]
# (control equation R = rate - target in the last well-equation row).
#
# Usage:
#   run-adjoint-fd-ctrl-test.sh <flow_adjoint> <inputfiles-dir> <deck-name> <output-dir> [u] [delta] [rel-tol]

set -u

FLOW=${1:?usage: $0 <flow_adjoint> <inputfiles-dir> <deck-name> <output-dir> [u] [delta] [rel-tol]}
INPUTDIR=${2:?missing inputfiles dir}
DECKNAME=${3:?missing deck name}
OUTDIR=${4:?missing output dir}
URATE=${5:-1.0}
DELTA=${6:-1e-6}
RELTOL=${7:-1e-3}

OBJECTIVE="bhp:Prod"
STRICT="--threads-per-process=1 --enable-storage-cache=false \
 --enable-adaptive-time-stepping=false \
 --tolerance-cnv=1e-7 --tolerance-cnv-relaxed=1e-7 \
 --tolerance-mb=1e-9 --tolerance-mb-relaxed=1e-9 \
 --adjoint-objective=${OBJECTIVE}"

prepare_input() { # <dstdir> <orat>
    local dst=$1 orat=$2
    rm -rf "${dst}"
    cp -r "${INPUTDIR}" "${dst}"
    local wcf wellname
    for wcf in "${dst}"/wconprodstep_*.txt; do
        wellname=$(awk 'NR==2 {print $1}' "${wcf}")
        printf 'WCONPROD\n%s OPEN ORAT %s 4* 50.0 /\n/\n' "${wellname}" "${orat}" > "${wcf}"
    done
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

mkdir -p "${OUTDIR}"

echo "=== base run (ORAT ${URATE}) + adjoint gradient"
prepare_input "${OUTDIR}/input_base" "${URATE}"
BASE=${OUTDIR}/base
mkdir -p "${BASE}"
${FLOW} "${OUTDIR}/input_base/${DECKNAME}.DATA" --output-dir="${BASE}" ${STRICT} \
    --adjoint-save=true > "${BASE}/fwd.log" 2>&1 || { echo "base forward failed"; exit 1; }
${FLOW} "${OUTDIR}/input_base/${DECKNAME}.DATA" --output-dir="${BASE}" ${STRICT} \
    --adjoint-mode=gradient > "${BASE}/grad.log" 2>&1 || { echo "gradient run failed"; exit 1; }
CTRLFILE=${BASE}/${DECKNAME}.ADJOINT_GRADIENTS_WELLCTRL.txt
[ -f "${CTRLFILE}" ] || { echo "no control-gradient file"; exit 1; }
ADJ_SI=$(awk '$1 == "Prod" {print $2}' "${CTRLFILE}")
echo "adjoint dJ/du (SI) for Prod: ${ADJ_SI}"

echo "=== FD of the ORAT target"
for sign in + -; do
    U=$(python3 -c "print(repr(${URATE} ${sign} ${DELTA}))")
    prepare_input "${OUTDIR}/input_${sign}" "${U}"
    J=$(run_J "${OUTDIR}/input_${sign}" "${OUTDIR}/fd_${sign}")
    [ "$J" = "FAIL" ] && { echo "FD run ${sign} failed"; exit 1; }
    if [ "$sign" = "+" ]; then JP=$J; else JM=$J; fi
done

read FD ADJ ERR OK << EOF
$(python3 -c "
fd = ($JP - $JM) / (2 * $DELTA)        # per sm3/day
adj = $ADJ_SI / 86400.0                # SI (per m3/s) -> per sm3/day
err = abs(fd - adj) / max(abs(fd), abs(adj), 1e-300)
print(fd, adj, err, 1 if err < $RELTOL else 0)")
EOF
printf "dJ/dORAT: FD %15.6e  adjoint %15.6e  rel.err %9.2e\n" $FD $ADJ $ERR

if [ "$OK" = "1" ]; then
    echo "adjoint-vs-FD (well control target): PASS (rel tol ${RELTOL})"
    exit 0
else
    echo "adjoint-vs-FD (well control target): FAIL"
    exit 1
fi
