#!/bin/bash
# Verification of the matchref objective (well-curve matching against a
# reference summary read via ESmry), specialized for SPE1CASE1:
#
#  1. reference run with PORO = <ref-poro> (default 0.285) -> UNSMRY;
#  2. base run with PORO = 0.3, J = sum dt (WGPR - WGPR_ref)^2 via
#     --adjoint-objective=matchref:PROD:gas:<refcase> -> adjoint gradient;
#     (gas rate is free: PROD is ORAT-controlled in SPE1)
#  3. twin check: matching the base run against its own summary must give
#     J orders of magnitude below the misfit J of step 2;
#  4. FD check of the GLOBAL porosity scale: perturb PORO 0.3 -> 0.3 +- d;
#     dJ/dporo_scalar (FD) vs sum_i g_pvmult_i / 0.3 (adjoint).
#
# Usage:
#   run-adjoint-matchref-test.sh <flow_adjoint> <spe1-deck> <output-dir> [delta] [rel-tol] [ref-poro]

set -u

FLOW=${1:?usage: $0 <flow_adjoint> <spe1-deck> <output-dir> [delta] [rel-tol] [ref-poro]}
DECK=${2:?missing SPE1CASE1.DATA path}
OUTDIR=${3:?missing output dir}
DELTA=${4:-2e-5}
RELTOL=${5:-1e-3}
REFPORO=${6:-0.285}

STRICT="--threads-per-process=1 --enable-storage-cache=false \
 --enable-adaptive-time-stepping=false \
 --tolerance-cnv=1e-7 --tolerance-cnv-relaxed=1e-7 \
 --tolerance-mb=1e-9 --tolerance-mb-relaxed=1e-9"

make_deck() { # <poro> <dstdir>
    mkdir -p "$2"
    sed "s/300\*0\.3/300*$1/" "${DECK}" > "$2/SPE1CASE1.DATA"
}

run_forward() { # <deckdir> <rundir> <extra args...>
    local deckdir=$1 rundir=$2; shift 2
    mkdir -p "${rundir}"
    ${FLOW} "${deckdir}/SPE1CASE1.DATA" --output-dir="${rundir}" ${STRICT} "$@" \
        > "${rundir}/run.log" 2>&1
}

get_J() { grep -o "Adjoint objective J = [0-9eE.+-]*" "$1" | awk '{print $5}'; }

mkdir -p "${OUTDIR}"

echo "=== reference run (PORO ${REFPORO})"
make_deck "${REFPORO}" "${OUTDIR}/deck_ref"
run_forward "${OUTDIR}/deck_ref" "${OUTDIR}/ref" || { echo "reference run failed"; exit 1; }
REFCASE=${OUTDIR}/ref/SPE1CASE1
[ -f "${REFCASE}.UNSMRY" ] || { echo "no reference UNSMRY"; exit 1; }
OBJ="matchref:PROD:gas:${REFCASE}"

echo "=== base run (PORO 0.3) + adjoint gradient, objective ${OBJ}"
make_deck 0.3 "${OUTDIR}/deck_base"
run_forward "${OUTDIR}/deck_base" "${OUTDIR}/base" \
    --adjoint-save=true --adjoint-objective="${OBJ}" || { echo "base run failed"; exit 1; }
# Secure the base summary before any replay/gradient/objective run in the
# same output dir: those runs truncate the summary files at startup.
mkdir -p "${OUTDIR}/twinref"
cp "${OUTDIR}/base/SPE1CASE1.SMSPEC" "${OUTDIR}/base/SPE1CASE1.UNSMRY" "${OUTDIR}/twinref/"
${FLOW} "${OUTDIR}/deck_base/SPE1CASE1.DATA" --output-dir="${OUTDIR}/base" ${STRICT} \
    --adjoint-mode=gradient --adjoint-objective="${OBJ}" \
    > "${OUTDIR}/base/grad.log" 2>&1 || { echo "gradient run failed"; exit 1; }
J0=$(grep -o "J = [0-9eE.+-]*" "${OUTDIR}/base/grad.log" | head -1 | awk '{print $3}')
GRADSUM=$(awk 'NR>1 {s+=$1} END {printf "%.16e", s}' "${OUTDIR}/base/SPE1CASE1.ADJOINT_GRADIENTS_PV.txt")
echo "misfit J = ${J0}, sum of pvmult gradients = ${GRADSUM}"

echo "=== twin check (base matched against its own summary)"
${FLOW} "${OUTDIR}/deck_base/SPE1CASE1.DATA" --output-dir="${OUTDIR}/base" ${STRICT} \
    --adjoint-mode=objective \
    --adjoint-objective="matchref:PROD:gas:${OUTDIR}/twinref/SPE1CASE1" \
    > "${OUTDIR}/base/twin.log" 2>&1 || { echo "twin run failed"; exit 1; }
JTWIN=$(get_J "${OUTDIR}/base/twin.log")
TWINOK=$(python3 -c "print(1 if abs($JTWIN) < 1e-6 * abs($J0) else 0)")
echo "twin J = ${JTWIN} (misfit J = ${J0})"

echo "=== FD of the global porosity scale"
for sign in + -; do
    PORO=$(python3 -c "print(repr(0.3 ${sign} ${DELTA}))")
    make_deck "${PORO}" "${OUTDIR}/deck_${sign}"
    run_forward "${OUTDIR}/deck_${sign}" "${OUTDIR}/fd_${sign}" \
        --adjoint-save=true --adjoint-objective="${OBJ}" \
        || { echo "FD run ${sign} failed"; exit 1; }
    ${FLOW} "${OUTDIR}/deck_${sign}/SPE1CASE1.DATA" --output-dir="${OUTDIR}/fd_${sign}" \
        ${STRICT} --adjoint-mode=objective --adjoint-objective="${OBJ}" \
        > "${OUTDIR}/fd_${sign}/obj.log" 2>&1 || { echo "FD objective ${sign} failed"; exit 1; }
    if [ "$sign" = "+" ]; then JP=$(get_J "${OUTDIR}/fd_+/obj.log");
    else JM=$(get_J "${OUTDIR}/fd_-/obj.log"); fi
done

read FD ADJ ERR OK << EOF
$(python3 -c "
fd = ($JP - $JM) / (2 * $DELTA)
adj = $GRADSUM / 0.3
err = abs(fd - adj) / max(abs(fd), abs(adj), 1e-300)
print(fd, adj, err, 1 if err < $RELTOL else 0)")
EOF
printf "global poro scale: FD %15.6e  adjoint %15.6e  rel.err %9.2e\n" $FD $ADJ $ERR

status=0
[ "$OK" = "1" ] || { echo "FD comparison FAILED"; status=1; }
[ "$TWINOK" = "1" ] || { echo "twin check FAILED"; status=1; }
if [ ${status} -eq 0 ]; then
    echo "matchref objective (ESmry well-curve matching): PASS"
fi
exit ${status}
