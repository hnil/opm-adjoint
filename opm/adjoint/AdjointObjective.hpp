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
 * \brief First objective functions for the adjoint solver.
 *
 * Per-substep sum objectives J = sum_k J_k(x_k, xw_k) (the Jutul-style
 * interface), selected with --adjoint-objective:
 *
 *  - "pressure-average": J = (1/N) sum_i x_i[pressureIdx] at the final
 *    substep. Pure function of the final reservoir solution; no well
 *    coupling. Used to validate the reservoir part of the chain.
 *
 *  - "bhp:<WELL>": J = sum_k dt_k * BHP_<WELL>(x_k). The bottom-hole
 *    pressure is itself a well primary variable (last entry of the
 *    StandardWell primary variable block), so dJ_k/dx_w is exact and the
 *    full well-coupling machinery (B^T D^-T terms in the adjoint
 *    right-hand side) is exercised with no approximation. Every substep
 *    contributes, so the cross-term recursion is exercised as well.
 *
 * Well-curve matching against a reference summary (ESmry) follows; it
 * uses the same well-coupling path with dJ/dx_w obtained from the
 * rate <-> primary variable mapping.
 */
#ifndef OPM_ADJOINT_OBJECTIVE_HPP
#define OPM_ADJOINT_OBJECTIVE_HPP

#include <opm/common/ErrorMacros.hpp>

#include <opm/models/utils/propertysystem.hh>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace Opm {

template<class TypeTag>
class AdjointObjectiveFunction
{
public:
    using Simulator = GetPropType<TypeTag, Properties::Simulator>;
    using Scalar = GetPropType<TypeTag, Properties::Scalar>;
    using Indices = GetPropType<TypeTag, Properties::Indices>;
    using FluidSystem = GetPropType<TypeTag, Properties::FluidSystem>;

    enum class Kind { PressureAverage, WellBhp, WellRate, WellRateMatch };

    //! \param spec objective specification string (--adjoint-objective).
    explicit AdjointObjectiveFunction(const std::string& spec)
    {
        if (spec.empty() || spec == "pressure-average") {
            kind_ = Kind::PressureAverage;
        } else if (spec.rfind("bhp:", 0) == 0) {
            kind_ = Kind::WellBhp;
            wellName_ = spec.substr(4);
            if (wellName_.empty()) {
                OPM_THROW(std::runtime_error,
                          "--adjoint-objective=bhp:<WELL> needs a well name");
            }
        } else if (spec.rfind("rate:", 0) == 0) {
            // rate:<WELL>:<oil|water|gas> -- J = sum_k dt_k q_phase(k),
            // i.e. the cumulative produced/injected surface volume.
            kind_ = Kind::WellRate;
            const auto rest = spec.substr(5);
            const auto colon = rest.find(':');
            if (colon == std::string::npos) {
                OPM_THROW(std::runtime_error,
                          "--adjoint-objective=rate:<WELL>:<oil|water|gas>");
            }
            wellName_ = rest.substr(0, colon);
            parsePhase_(rest.substr(colon + 1));
        } else if (spec.rfind("match:", 0) == 0) {
            // match:<WELL>:<oil|water|gas>:<target sm3/day> --
            // J = sum_k dt_k (q_phase(k) - q_target)^2 : the quadratic
            // misfit form of well-curve matching with a constant target
            // (per-step observed curves via ESmry are pure I/O on top).
            kind_ = Kind::WellRateMatch;
            const auto rest = spec.substr(6);
            const auto c1 = rest.find(':');
            const auto c2 = (c1 == std::string::npos)
                ? std::string::npos : rest.find(':', c1 + 1);
            if (c2 == std::string::npos) {
                OPM_THROW(std::runtime_error,
                          "--adjoint-objective=match:<WELL>:<phase>:<target sm3/day>");
            }
            wellName_ = rest.substr(0, c1);
            parsePhase_(rest.substr(c1 + 1, c2 - c1 - 1));
            // surface rates are SI (m3/s) internally
            targetRate_ = std::stod(rest.substr(c2 + 1)) / 86400.0;
        } else {
            OPM_THROW(std::runtime_error,
                      "Unknown adjoint objective '" + spec +
                      "' (expected 'pressure-average', 'bhp:<WELL>', "
                      "'rate:<WELL>:<phase>' or 'match:<WELL>:<phase>:<target>')");
        }
    }

    Kind kind() const
    { return kind_; }

    const std::string& wellName() const
    { return wellName_; }

    //! \brief Value contribution of substep \p k; the simulator must hold
    //!        the (recorded) state of substep k.
    Scalar stepValue(const Simulator& simulator, int k, int numSubsteps,
                     double dt) const
    {
        switch (kind_) {
        case Kind::PressureAverage: {
            if (k != numSubsteps - 1) {
                return 0.0;
            }
            const auto& solution = simulator.model().solution(/*timeIdx=*/0);
            Scalar sum = 0.0;
            for (std::size_t i = 0; i < solution.size(); ++i) {
                sum += solution[i][Indices::pressureSwitchIdx];
            }
            return sum / solution.size();
        }
        case Kind::WellBhp: {
            const auto& wellState =
                simulator.problem().wellModel().wellState();
            return dt * wellState.well(wellName_).bhp;
        }
        case Kind::WellRate: {
            const auto& wellState =
                simulator.problem().wellModel().wellState();
            const int phasePos =
                FluidSystem::canonicalToActivePhaseIdx(canonicalPhaseIdx_);
            return dt * wellState.well(wellName_).surface_rates[phasePos];
        }
        case Kind::WellRateMatch: {
            const auto& wellState =
                simulator.problem().wellModel().wellState();
            const int phasePos =
                FluidSystem::canonicalToActivePhaseIdx(canonicalPhaseIdx_);
            const Scalar diff =
                wellState.well(wellName_).surface_rates[phasePos] - targetRate_;
            return dt * diff * diff;
        }
        }
        return 0.0;
    }

    //! \brief Add dJ_k/dx_r to \p gradient (reservoir part).
    template<class GradientVector>
    void addStepGradient(const Simulator& simulator,
                         int k,
                         int numSubsteps,
                         GradientVector& gradient) const
    {
        if (kind_ != Kind::PressureAverage || k != numSubsteps - 1) {
            return;
        }
        const std::size_t numCells = simulator.model().solution(0).size();
        for (std::size_t i = 0; i < numCells; ++i) {
            gradient[i][Indices::pressureSwitchIdx] += 1.0 / numCells;
        }
    }

    //! \brief dJ_k/d(bhp of \p well) — nonzero only for the bhp objective
    //!        and the selected well. The BHP is the last well primary
    //!        variable of a StandardWell.
    Scalar bhpGradient(const std::string& well, double dt) const
    {
        if (kind_ == Kind::WellBhp && well == wellName_) {
            return dt;
        }
        return 0.0;
    }

    //! \brief dJ_k/d(rate) weight: dt for the plain rate objective,
    //!        2 dt (q - q_target) for the quadratic match (q = current
    //!        rate of the selected phase, supplied by the caller).
    Scalar rateWeight(const std::string& well, double dt, Scalar q) const
    {
        if (well != wellName_) {
            return 0.0;
        }
        if (kind_ == Kind::WellRate) {
            return dt;
        }
        if (kind_ == Kind::WellRateMatch) {
            return 2.0 * dt * (q - targetRate_);
        }
        return 0.0;
    }

    //! \brief Active component index of the objective's rate phase.
    unsigned activeCompIdx() const
    { return FluidSystem::canonicalToActiveCompIdx(canonicalCompIdx_); }

private:
    void parsePhase_(const std::string& phase)
    {
        if (phase == "oil") {
            canonicalPhaseIdx_ = FluidSystem::oilPhaseIdx;
            canonicalCompIdx_ = FluidSystem::oilCompIdx;
        } else if (phase == "water") {
            canonicalPhaseIdx_ = FluidSystem::waterPhaseIdx;
            canonicalCompIdx_ = FluidSystem::waterCompIdx;
        } else if (phase == "gas") {
            canonicalPhaseIdx_ = FluidSystem::gasPhaseIdx;
            canonicalCompIdx_ = FluidSystem::gasCompIdx;
        } else {
            OPM_THROW(std::runtime_error,
                      "Unknown phase '" + phase + "' in adjoint objective");
        }
    }

    Kind kind_{Kind::PressureAverage};
    std::string wellName_{};
    unsigned canonicalPhaseIdx_{0};
    unsigned canonicalCompIdx_{0};
    Scalar targetRate_{0.0};
};

} // namespace Opm

#endif // OPM_ADJOINT_OBJECTIVE_HPP
