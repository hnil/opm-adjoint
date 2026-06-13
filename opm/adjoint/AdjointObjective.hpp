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
 * \brief Objective functions for the adjoint solver.
 *
 * Per-substep sum objectives J = sum_k J_k(x_k, xw_k) (the Jutul-style
 * interface), selected with --adjoint-objective:
 *
 *  - "pressure-average": J = (1/N) sum_i x_i[pressureIdx] at the final
 *    substep (reservoir-state validation objective).
 *  - "bhp:<WELL>": J = sum_k dt_k BHP(x_k) (well primary variable;
 *    exact dJ/dx_w, exercises the well coupling without approximation).
 *  - Rate-family objectives, all represented as a sum of rate terms
 *    (single- or multi-term):
 *      "rate:<WELL>:<phase>"                J += dt q
 *      "match:<WELL>:<phase>:<target>"      J += dt (q - q_t)^2
 *      "matchref:<WELL>:<phase>:<refcase>"  J += dt (q - q_obs(t))^2
 *      "matchsum:<refcase>:<W>.<p>[+<W>.<p>]..."
 *                                           J += sum_terms dt (q - q_obs)^2
 *    Observed curves are read from the reference run's summary via
 *    ESmry, interpolated linearly in time and converted from the file's
 *    rate unit to the internal SI/sign convention. Producers only (v1).
 *
 * Phase is one of oil|water|gas.
 */
#ifndef OPM_ADJOINT_OBJECTIVE_HPP
#define OPM_ADJOINT_OBJECTIVE_HPP

#include <opm/common/ErrorMacros.hpp>

#include <opm/io/eclipse/ESmry.hpp>

#include <opm/models/utils/propertysystem.hh>

#include <opm/adjoint/AdjointParallel.hpp>

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

    enum class Kind { PressureAverage, WellBhp, RateTerms };

    enum class TermType { Rate, MatchConst, MatchRef };

    //! One rate term of the objective.
    struct RateTerm
    {
        std::string well;
        unsigned canonicalPhaseIdx{0};
        unsigned canonicalCompIdx{0};
        TermType type{TermType::Rate};
        Scalar weight{1.0};
        Scalar targetRate{0.0};        //!< MatchConst (internal units)
        std::string referenceCase{};   //!< MatchRef
        // observed curve (lazily loaded)
        mutable bool loaded{false};
        mutable std::vector<Scalar> obsTimes;   //!< days
        mutable std::vector<Scalar> obsValues;  //!< internal units/sign
    };

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
            kind_ = Kind::RateTerms;
            const auto parts = split_(spec.substr(5), ':', 2);
            RateTerm term;
            term.well = parts[0];
            setPhase_(term, parts[1]);
            term.type = TermType::Rate;
            terms_.push_back(std::move(term));
        } else if (spec.rfind("match:", 0) == 0) {
            kind_ = Kind::RateTerms;
            const auto parts = split_(spec.substr(6), ':', 3);
            RateTerm term;
            term.well = parts[0];
            setPhase_(term, parts[1]);
            term.type = TermType::MatchConst;
            term.targetRate = std::stod(parts[2]) / 86400.0; // sm3/day -> SI
            terms_.push_back(std::move(term));
        } else if (spec.rfind("matchref:", 0) == 0) {
            kind_ = Kind::RateTerms;
            const auto parts = split_(spec.substr(9), ':', 3);
            RateTerm term;
            term.well = parts[0];
            setPhase_(term, parts[1]);
            term.type = TermType::MatchRef;
            term.referenceCase = parts[2];
            terms_.push_back(std::move(term));
        } else if (spec.rfind("matchsum:", 0) == 0) {
            // matchsum:<refcase>:<W>.<p>[+<W>.<p>]...
            kind_ = Kind::RateTerms;
            const auto rest = spec.substr(9);
            const auto lastColon = rest.rfind(':');
            if (lastColon == std::string::npos) {
                OPM_THROW(std::runtime_error,
                          "--adjoint-objective=matchsum:<refcase>:<W>.<p>[+<W>.<p>]...");
            }
            const std::string refcase = rest.substr(0, lastColon);
            const std::string termsSpec = rest.substr(lastColon + 1);
            std::size_t pos = 0;
            while (pos != std::string::npos && pos < termsSpec.size()) {
                const auto plus = termsSpec.find('+', pos);
                const std::string one = termsSpec.substr(
                    pos, plus == std::string::npos ? std::string::npos : plus - pos);
                const auto dot = one.find('.');
                if (dot == std::string::npos) {
                    OPM_THROW(std::runtime_error,
                              "matchsum term '" + one + "' must be <WELL>.<phase>");
                }
                RateTerm term;
                term.well = one.substr(0, dot);
                setPhase_(term, one.substr(dot + 1));
                term.type = TermType::MatchRef;
                term.referenceCase = refcase;
                terms_.push_back(std::move(term));
                pos = (plus == std::string::npos) ? std::string::npos : plus + 1;
            }
            if (terms_.empty()) {
                OPM_THROW(std::runtime_error,
                          "matchsum objective without terms");
            }
        } else {
            OPM_THROW(std::runtime_error,
                      "Unknown adjoint objective '" + spec + "'");
        }
    }

    Kind kind() const
    { return kind_; }

    const std::string& wellName() const
    { return wellName_; }

    const std::vector<RateTerm>& rateTerms() const
    { return terms_; }

    //! \brief True if the objective has well terms involving \p well.
    bool involvesWell(const std::string& well) const
    {
        if (kind_ == Kind::WellBhp) {
            return well == wellName_;
        }
        return std::any_of(terms_.begin(), terms_.end(),
                           [&well](const auto& t) { return t.well == well; });
    }

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
                if (par_ && !par_->interior(i)) {
                    continue;  // skip ghost cells (owned by another rank)
                }
                sum += solution[i][Indices::pressureSwitchIdx];
            }
            if (par_) {
                return par_->sum(sum) / globalNumCells_(simulator);
            }
            return sum / solution.size();
        }
        case Kind::WellBhp: {
            const auto& wellState =
                simulator.problem().wellModel().wellState();
            return dt * wellState.well(wellName_).bhp;
        }
        case Kind::RateTerms: {
            const auto& wellState =
                simulator.problem().wellModel().wellState();
            Scalar sum = 0.0;
            for (const auto& term : terms_) {
                const int phasePos =
                    FluidSystem::canonicalToActivePhaseIdx(term.canonicalPhaseIdx);
                const Scalar q =
                    wellState.well(term.well).surface_rates[phasePos];
                sum += termValue_(simulator, term, dt, q);
            }
            return sum;
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
        const Scalar denom = par_ ? globalNumCells_(simulator)
                                  : static_cast<Scalar>(numCells);
        for (std::size_t i = 0; i < numCells; ++i) {
            // ghost cells get the same per-cell weight; their rhs rows are
            // zeroed before the transposed solve, so this is harmless and
            // keeps the interior rows correct.
            gradient[i][Indices::pressureSwitchIdx] += 1.0 / denom;
        }
    }

    //! \brief Connect the parallel context (interior mask + reductions);
    //!        null in serial runs.
    void setParallel(const AdjointParallel<TypeTag>* par)
    { par_ = par; }

    //! \brief dJ_k/d(bhp of \p well) for the bhp objective.
    Scalar bhpGradient(const std::string& well, double dt) const
    {
        if (kind_ == Kind::WellBhp && well == wellName_) {
            return dt;
        }
        return 0.0;
    }

    //! \brief dJ_k/dq weight of one rate term given the current rate \p q.
    Scalar termWeight(const Simulator& simulator, const RateTerm& term,
                      double dt, Scalar q) const
    {
        switch (term.type) {
        case TermType::Rate:
            return term.weight * dt;
        case TermType::MatchConst:
            return term.weight * 2.0 * dt * (q - term.targetRate);
        case TermType::MatchRef: {
            const Scalar tEnd = simulator.time() + dt;
            return term.weight * 2.0 * dt *
                   (q - observedRate_(simulator, term, tEnd));
        }
        }
        return 0.0;
    }

private:
    Scalar termValue_(const Simulator& simulator, const RateTerm& term,
                      double dt, Scalar q) const
    {
        switch (term.type) {
        case TermType::Rate:
            return term.weight * dt * q;
        case TermType::MatchConst: {
            const Scalar diff = q - term.targetRate;
            return term.weight * dt * diff * diff;
        }
        case TermType::MatchRef: {
            const Scalar tEnd = simulator.time() + dt;
            const Scalar diff = q - observedRate_(simulator, term, tEnd);
            return term.weight * dt * diff * diff;
        }
        }
        return 0.0;
    }

    //! \brief Observed rate (internal sign/units) at time \p t [s].
    Scalar observedRate_(const Simulator& simulator, const RateTerm& term,
                         Scalar t) const
    {
        if (!term.loaded) {
            loadObservations_(simulator, term);
        }
        const Scalar tDays = t / 86400.0;
        const auto& times = term.obsTimes;
        if (times.empty()) {
            OPM_THROW(std::runtime_error, "Empty observed curve");
        }
        if (tDays <= times.front()) {
            return term.obsValues.front();
        }
        if (tDays >= times.back()) {
            return term.obsValues.back();
        }
        const auto it = std::upper_bound(times.begin(), times.end(), tDays);
        const std::size_t hi = it - times.begin();
        const Scalar w = (tDays - times[hi - 1]) / (times[hi] - times[hi - 1]);
        return (1.0 - w) * term.obsValues[hi - 1] + w * term.obsValues[hi];
    }

    void loadObservations_(const Simulator& simulator, const RateTerm& term) const
    {
        const auto& schedule = simulator.vanguard().schedule();
        const auto& well = schedule.getWell(term.well, simulator.episodeIndex());
        // Producers: W?PR keywords, internal rates negative.
        // Injectors:  W?IR keywords, internal rates positive.
        const bool producer = well.isProducer();
        const Scalar sign = producer ? -1.0 : 1.0;
        std::string keyword;
        if (term.canonicalPhaseIdx == FluidSystem::oilPhaseIdx) {
            keyword = producer ? "WOPR:" : "WOIR:";
        } else if (term.canonicalPhaseIdx == FluidSystem::waterPhaseIdx) {
            keyword = producer ? "WWPR:" : "WWIR:";
        } else {
            keyword = producer ? "WGPR:" : "WGIR:";
        }
        keyword += term.well;

        EclIO::ESmry summary(term.referenceCase);
        summary.loadData();
        const auto& time = summary.get("TIME");      // days
        const auto& rate = summary.get(keyword);     // file units, positive

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

        term.obsTimes.assign(time.begin(), time.end());
        term.obsValues.resize(rate.size());
        for (std::size_t i = 0; i < rate.size(); ++i) {
            term.obsValues[i] = sign * static_cast<Scalar>(rate[i]) * toSi;
        }
        term.loaded = true;
    }

    void setPhase_(RateTerm& term, const std::string& phase) const
    {
        if (phase == "oil") {
            term.canonicalPhaseIdx = FluidSystem::oilPhaseIdx;
            term.canonicalCompIdx = FluidSystem::oilCompIdx;
        } else if (phase == "water") {
            term.canonicalPhaseIdx = FluidSystem::waterPhaseIdx;
            term.canonicalCompIdx = FluidSystem::waterCompIdx;
        } else if (phase == "gas") {
            term.canonicalPhaseIdx = FluidSystem::gasPhaseIdx;
            term.canonicalCompIdx = FluidSystem::gasCompIdx;
        } else {
            OPM_THROW(std::runtime_error,
                      "Unknown phase '" + phase + "' in adjoint objective");
        }
    }

    static std::vector<std::string> split_(const std::string& text,
                                           char sep, std::size_t expected)
    {
        std::vector<std::string> parts;
        std::size_t pos = 0;
        while (parts.size() + 1 < expected) {
            const auto next = text.find(sep, pos);
            if (next == std::string::npos) {
                OPM_THROW(std::runtime_error,
                          "Malformed adjoint objective specification: " + text);
            }
            parts.push_back(text.substr(pos, next - pos));
            pos = next + 1;
        }
        parts.push_back(text.substr(pos));
        return parts;
    }

    //! \brief Global number of interior cells (cached), for the
    //!        pressure-average denominator.
    Scalar globalNumCells_(const Simulator& simulator) const
    {
        if (globalNumCells_cached_ < 0) {
            const auto& solution = simulator.model().solution(/*timeIdx=*/0);
            Scalar localInterior = 0.0;
            for (std::size_t i = 0; i < solution.size(); ++i) {
                if (!par_ || par_->interior(i)) {
                    localInterior += 1.0;
                }
            }
            globalNumCells_cached_ = par_ ? par_->sum(localInterior)
                                          : localInterior;
        }
        return globalNumCells_cached_;
    }

    Kind kind_{Kind::PressureAverage};
    std::string wellName_{};           //!< bhp objective
    std::vector<RateTerm> terms_;      //!< rate-family objectives
    const AdjointParallel<TypeTag>* par_{nullptr};
    mutable Scalar globalNumCells_cached_{-1};
};

} // namespace Opm

#endif // OPM_ADJOINT_OBJECTIVE_HPP
