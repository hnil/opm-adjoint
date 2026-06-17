/*
  Copyright 2026 SINTEF Digital, Mathematics and Cybernetics.

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/
/*!
 * \file
 * \brief End-of-run warning: scan the deck for quantities the adjoint
 *        does not reproduce/differentiate exactly.
 *
 * The adjoint v1 reproduces the forward residual/Jacobian bitwise and
 * differentiates it exactly for the common black-oil + (group-)well
 * control case. A number of *explicitly-updated* quantities are handled
 * approximately; this scan looks them up in the deck and warns at the
 * end of a gradient/replay run so a user knows whether their case is in
 * the exact regime. The categories mirror the catalog in
 * opm-adjoint/adjoint_plan.md (explicitly-updated-quantities table).
 *
 *  - approx-gradient: the forward state IS serialized so REPLAY is exact
 *    (J is exact), but the cross-step derivative of the explicitly-updated
 *    quantity is neglected, so dJ/dm is approximate;
 *  - approx-replay: the control target is re-derived during the replay
 *    assembly and may not reproduce the forward exactly (reservoir-volume
 *    / voidage-replacement group control);
 *  - neglected: not handled by the adjoint at all (aquifers, networks).
 */
#ifndef OPM_ADJOINT_DECK_WARNINGS_HPP
#define OPM_ADJOINT_DECK_WARNINGS_HPP

#include <opm/common/OpmLog/OpmLog.hpp>

#include <opm/input/eclipse/Schedule/Group/Group.hpp>
#include <opm/input/eclipse/Schedule/OilVaporizationProperties.hpp>

#include <fmt/format.h>

#include <string>
#include <utility>
#include <vector>

namespace Opm {

//! \brief Scan the deck and emit a single categorized warning listing the
//!        explicitly-updated / non-exact quantities present.
template<class Simulator>
inline void warnAdjointDeckApproximations(const Simulator& simulator)
{
    const auto& eclState = simulator.vanguard().eclState();
    const auto& schedule = simulator.vanguard().schedule();
    const std::size_t nstep = schedule.size();

    using Entry = std::pair<std::string, std::string>;
    std::vector<Entry> approxGradient;  // replay exact, dJ/dm cross-term neglected
    std::vector<Entry> approxReplay;    // re-derived control target, replay may drift
    std::vector<Entry> neglected;       // not handled at all

    // --- explicitly-updated quantities: replay exact, gradient approximate ---
    if (eclState.runspec().hysterPar().active()) {
        approxGradient.emplace_back(
            "saturation hysteresis (HYSTER)",
            "state serialized so replay is exact; the hysteresis-state "
            "cross-term in dJ/dm is neglected");
    }

    bool drsdt = false, drvdt = false, vappars = false;
    for (std::size_t i = 0; i < nstep; ++i) {
        const auto& ov = schedule[i].oilvap();
        // Only flag an *active* rate limit: DRSDT/DRVDT 0 (a frozen, i.e.
        // constant, bubble/dew point) carries no rate-dependent cross-term
        // and is reproduced exactly (e.g. SPE1CASE1 has DRSDT 0 and is
        // FD-verified exact).
        for (std::size_t r = 0; r < ov.numPvtRegions(); ++r) {
            if (ov.drsdtActive() && ov.getMaxDRSDT(r) > 0.0) drsdt = true;
            if (ov.drvdtActive() && ov.getMaxDRVDT(r) > 0.0) drvdt = true;
        }
        vappars |= (ov.getType() ==
                    OilVaporizationProperties::OilVaporization::VAPPARS);
    }
    if (drsdt) {
        approxGradient.emplace_back("DRSDT",
            "replay exact (cache-free storage); the dissolved-gas-rate-limit "
            "cross-term in dJ/dm is neglected");
    }
    if (drvdt) {
        approxGradient.emplace_back("DRVDT",
            "replay exact (cache-free storage); the vaporized-oil-rate-limit "
            "cross-term in dJ/dm is neglected");
    }
    if (vappars) {
        approxGradient.emplace_back("VAPPARS",
            "replay exact (cache-free storage); the vaporization-parameter "
            "cross-term in dJ/dm is neglected");
    }

    // --- reservoir-volume / voidage-replacement group control ---
    bool voidage = false;
    for (std::size_t i = 0; i < nstep && !voidage; ++i) {
        for (const auto& gname : schedule.groupNames(i)) {
            const auto& group = schedule.getGroup(gname, i);
            for (const auto& [phase, props] : group.injectionProperties()) {
                static_cast<void>(phase);
                using IM = Group::InjectionCMode;
                if (props.cmode == IM::VREP || props.cmode == IM::RESV ||
                    props.cmode == IM::REIN) {
                    voidage = true;
                    break;
                }
            }
            if (!voidage &&
                group.productionProperties().cmode == Group::ProductionCMode::RESV) {
                voidage = true;
            }
            if (voidage) {
                break;
            }
        }
    }
    if (voidage) {
        approxReplay.emplace_back(
            "reservoir-volume / voidage-replacement group control "
            "(RESV/VREP/REIN)",
            "the effective control target is re-derived from reservoir "
            "voidage during the replay assembly and may not reproduce the "
            "forward bitwise; the gradient is correspondingly approximate");
    }

    // --- neglected entirely ---
    if (eclState.aquifer().active()) {
        neglected.emplace_back("aquifers",
            "aquifer contributions are not assembled during replay and their "
            "cross-term is neglected; exclude aquifer cells from gradients");
    }
    bool network = false;
    for (std::size_t i = 0; i < nstep && !network; ++i) {
        network |= schedule[i].network().active();
    }
    if (network) {
        neglected.emplace_back("extended network (NODEPROP/BRANPROP)",
            "network nodal-pressure coupling is not handled by the adjoint");
    }

    if (approxGradient.empty() && approxReplay.empty() && neglected.empty()) {
        return;
    }

    std::string msg = "Adjoint accuracy notes for this deck — some "
                      "explicitly-updated quantities are not handled exactly:\n";
    auto append = [&msg](const std::vector<Entry>& list, const char* tag) {
        for (const auto& [feature, treatment] : list) {
            msg += fmt::format("  [{}] {}: {}\n", tag, feature, treatment);
        }
    };
    append(approxGradient, "approx-gradient");
    append(approxReplay, "approx-replay");
    append(neglected, "neglected");
    msg += "  Replay and gradients are exact for everything else (FD-verified: "
           "porosity/PV, transmissibility, permeability, end-point scaling, "
           "well and group control).";
    OpmLog::warning(msg);
}

} // namespace Opm

#endif // OPM_ADJOINT_DECK_WARNINGS_HPP
