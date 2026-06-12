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

#include <opm/io/eclipse/ESmry.hpp>

#include <opm/models/utils/propertysystem.hh>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace Opm {

template<class TypeTag>
class AdjointObjectiveFunction
{
public:
    using Simulator = GetPropType<TypeTag, Properties::Simulator>;
    using Scalar = GetPropType<TypeTag, Properties::Scalar>;
    using Indices = GetPropType<TypeTag, Properties::Indices>;
    using FluidSystem = GetPropType<TypeTag, Properties::FluidSystem>;

    enum class Kind { PressureAverage, WellBhp, WellRate, WellRateMatch,
                      WellRateMatchRef };

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
        } else if (spec.rfind("matchref:", 0) == 0) {
            // matchref:<WELL>:<phase>:<reference case> --
            // J = sum_k dt_k (q(t_k) - q_obs(t_k))^2 with the observed
            // curve read from the reference run's summary (ESmry) and
            // interpolated linearly in time. Producers only in v1.
            kind_ = Kind::WellRateMatchRef;
            const auto rest = spec.substr(9);
            const auto c1 = rest.find(':');
            const auto c2 = (c1 == std::string::npos)
                ? std::string::npos : rest.find(':', c1 + 1);
            if (c2 == std::string::npos) {
                OPM_THROW(std::runtime_error,
                          "--adjoint-objective=matchref:<WELL>:<phase>:<refcase>");
            }
            wellName_ = rest.substr(0, c1);
            parsePhase_(rest.substr(c1 + 1, c2 - c1 - 1));
            referenceCase_ = rest.substr(c2 + 1);
        } else {
            OPM_THROW(std::runtime_error,
                      "Unknown adjoint objective '" + spec +
                      "' (expected 'pressure-average', 'bhp:<WELL>', "
                      "'rate:<WELL>:<phase>', 'match:<WELL>:<phase>:<target>' "
                      "or 'matchref:<WELL>:<phase>:<refcase>')");
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
        case Kind::WellRateMatchRef: {
            const auto& wellState =
                simulator.problem().wellModel().wellState();
            const int phasePos =
                FluidSystem::canonicalToActivePhaseIdx(canonicalPhaseIdx_);
            const Scalar tEnd = simulator.time() + dt;
            const Scalar diff =
                wellState.well(wellName_).surface_rates[phasePos] -
                observedRate(simulator, tEnd);
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
    //!        2 dt (q - q_target/q_obs(tEnd)) for the quadratic matches
    //!        (q = current rate of the selected phase).
    Scalar rateWeight(const Simulator& simulator, const std::string& well,
                      double dt, Scalar q) const
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
        if (kind_ == Kind::WellRateMatchRef) {
            const Scalar tEnd = simulator.time() + dt;
            return 2.0 * dt * (q - observedRate(simulator, tEnd));
        }
        return 0.0;
    }

    //! \brief Observed rate (internal sign and units) at time \p t [s],
    //!        interpolated linearly from the reference summary.
    Scalar observedRate(const Simulator& simulator, Scalar t) const
    {
        if (!observationsLoaded_) {
            loadObservations_(simulator);
        }
        const Scalar tDays = t / 86400.0;
        const auto& times = obsTimes_;
        if (times.empty()) {
            OPM_THROW(std::runtime_error, "Empty observed curve");
        }
        if (tDays <= times.front()) {
            return obsValues_.front();
        }
        if (tDays >= times.back()) {
            return obsValues_.back();
        }
        const auto it = std::upper_bound(times.begin(), times.end(), tDays);
        const std::size_t hi = it - times.begin();
        const Scalar w = (tDays - times[hi - 1]) / (times[hi] - times[hi - 1]);
        return (1.0 - w) * obsValues_[hi - 1] + w * obsValues_[hi];
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

    void loadObservations_(const Simulator& simulator) const
    {
        const auto& schedule = simulator.vanguard().schedule();
        const auto& well = schedule.getWell(wellName_, simulator.episodeIndex());
        if (!well.isProducer()) {
            OPM_THROW(std::runtime_error,
                      "matchref objectives support producers only (v1)");
        }
        std::string keyword;
        if (canonicalPhaseIdx_ == FluidSystem::oilPhaseIdx) {
            keyword = "WOPR:" + wellName_;
        } else if (canonicalPhaseIdx_ == FluidSystem::waterPhaseIdx) {
            keyword = "WWPR:" + wellName_;
        } else {
            keyword = "WGPR:" + wellName_;
        }

        EclIO::ESmry summary(referenceCase_);
        summary.loadData();
        const auto& time = summary.get("TIME");      // days
        const auto& rate = summary.get(keyword);     // in file units, positive

        // Convert the file's rate unit to internal SI (m3/s).
        const std::string unit = summary.get_unit(keyword);
        Scalar toSi;
        if (unit == "SM3/DAY") {
            toSi = 1.0 / 86400.0;
        } else if (unit == "MSCF/DAY") {
            toSi = 28.316846592 / 86400.0;   // 1 Mscf = 1000 ft^3
        } else if (unit == "STB/DAY") {
            toSi = 0.158987294928 / 86400.0;
        } else {
            OPM_THROW(std::runtime_error,
                      "Unsupported rate unit '" + unit +
                      "' in reference summary for " + keyword);
        }

        obsTimes_.assign(time.begin(), time.end());
        obsValues_.resize(rate.size());
        for (std::size_t i = 0; i < rate.size(); ++i) {
            // internal convention: producer rates are negative
            obsValues_[i] = -static_cast<Scalar>(rate[i]) * toSi;
        }
        observationsLoaded_ = true;
    }

    Kind kind_{Kind::PressureAverage};
    std::string wellName_{};
    unsigned canonicalPhaseIdx_{0};
    unsigned canonicalCompIdx_{0};
    Scalar targetRate_{0.0};
    std::string referenceCase_{};
    mutable bool observationsLoaded_{false};
    mutable std::vector<Scalar> obsTimes_;
    mutable std::vector<Scalar> obsValues_;
};

} // namespace Opm

#endif // OPM_ADJOINT_OBJECTIVE_HPP
