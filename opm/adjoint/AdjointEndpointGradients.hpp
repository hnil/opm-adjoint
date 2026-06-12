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
 * \brief Saturation-function end-point scaling gradients for the adjoint
 *        solver, with a general registry mapping deck keywords to the
 *        scaled end points they control.
 *
 * Mechanism (cell-local parameter differences, the C.2 v1 route): for a
 * requested end-point parameter theta of cell i,
 *   1. copy the cell's EclEpsScalingPointsInfo, perturb the registered
 *      field by +-h, and re-derive the scaled points of the oil-water
 *      and gas-oil drainage curves in place
 *      (materialLawParams(i).{oilWater,gasOil}Params().drainageParams()
 *       .scaledPoints().init(info, config, system) - all public API);
 *   2. recompute the cell's intensive quantities at x_k and x_{k-1}
 *      through the public ElementContext-free
 *      IntensiveQuantities::update(problem, priVars, i, timeIdx) and
 *      re-evaluate the cell's storage and the fluxes of its faces with
 *      the public TPFA kernels;
 *   3. central-difference the residual contributions and contract with
 *      the adjoint:
 *        dJ/dtheta_i += lambda_i . dR_i + sum_faces lambda_J . dR_J
 *      where dR_i collects the storage difference and the cell's own
 *      flux evaluations, and dR_J = -dF (the neighbor sees the mirrored
 *      flux).
 *
 * No global solves, no opm-simulators changes; hysteresis-free scope
 * (drainage curves only). Exact parameter-AD would require
 * Evaluation-templated material-law parameter objects (long-term
 * upstream item).
 *
 * Note on validity: the gradient is taken at the recorded trajectory
 * with a fixed initial state. For equilibrated decks the initial state
 * itself depends on the end points (Pc scaling shifts the transition
 * zone) - that contribution is NOT included; use explicitly initialized
 * decks for FD verification.
 */
#ifndef OPM_ADJOINT_ENDPOINT_GRADIENTS_HPP
#define OPM_ADJOINT_ENDPOINT_GRADIENTS_HPP

#include <opm/common/ErrorMacros.hpp>

#include <opm/material/fluidmatrixinteractions/EclEpsScalingPoints.hpp>
#include <opm/material/fluidmatrixinteractions/EclEpsConfig.hpp>

#include <opm/models/utils/propertysystem.hh>

#include <string>
#include <vector>

namespace Opm {

//! \brief Registry entry: maps a deck end-point keyword to the field of
//!        EclEpsScalingPointsInfo it controls.
//!
//! This is the "general input -> derivative" mapping: adding support for
//! another scaled quantity is one table line, the perturbation /
//! re-derivation / contraction machinery is shared.
template<class Scalar>
struct EndpointParamDef
{
    std::string name;                                    //!< deck keyword
    Scalar EclEpsScalingPointsInfo<Scalar>::* field;     //!< controlled entry
};

template<class Scalar>
const std::vector<EndpointParamDef<Scalar>>& endpointParamRegistry()
{
    using Info = EclEpsScalingPointsInfo<Scalar>;
    static const std::vector<EndpointParamDef<Scalar>> registry = {
        {"SWL",   &Info::Swl},
        {"SGL",   &Info::Sgl},
        {"SWCR",  &Info::Swcr},
        {"SGCR",  &Info::Sgcr},
        {"SOWCR", &Info::Sowcr},
        {"SOGCR", &Info::Sogcr},
        {"SWU",   &Info::Swu},
        {"SGU",   &Info::Sgu},
        {"KRW",   &Info::maxKrw},
        {"KRG",   &Info::maxKrg},
        {"KRORW", &Info::maxKrow},
        {"KRORG", &Info::maxKrog},
        {"PCW",   &Info::maxPcow},
        {"PCG",   &Info::maxPcgo},
    };
    return registry;
}

template<class Scalar>
const EndpointParamDef<Scalar>& findEndpointParam(const std::string& name)
{
    for (const auto& def : endpointParamRegistry<Scalar>()) {
        if (def.name == name) {
            return def;
        }
    }
    OPM_THROW(std::runtime_error,
              "Unknown end-point parameter '" + name +
              "' (supported: SWL SGL SWCR SGCR SOWCR SOGCR SWU SGU "
              "KRW KRG KRORW KRORG PCW PCG)");
}

//! \brief Applies/reverts end-point perturbations on the live material
//!        law parameters of one cell (drainage curves; both two-phase
//!        systems are re-derived from the same modified info).
template<class TypeTag>
class EndpointPerturber
{
public:
    using Problem = GetPropType<TypeTag, Properties::Problem>;
    using Scalar = GetPropType<TypeTag, Properties::Scalar>;

    explicit EndpointPerturber(Problem& problem)
        : problem_(problem)
    {}

    //! \brief Set theta(cell) = original + delta (delta = 0 restores).
    void apply(unsigned cell, const EndpointParamDef<Scalar>& def, Scalar delta)
    {
        auto& manager = *problem_.materialLawManager();
        auto info = manager.oilWaterScaledEpsInfoDrainage(cell); // copy
        info.*def.field += delta;

        // materialLawParams(cell) is the multiplexer params; the scaled
        // points sit on the concrete 3-phase params underneath.
        auto& matParams = manager.materialLawParams(cell);
        switch (matParams.approach()) {
        case EclMultiplexerApproach::Default:
            reinit_(matParams.template getRealParams<EclMultiplexerApproach::Default>(),
                    manager, info);
            break;
        case EclMultiplexerApproach::Stone1:
            reinit_(matParams.template getRealParams<EclMultiplexerApproach::Stone1>(),
                    manager, info);
            break;
        case EclMultiplexerApproach::Stone2:
            reinit_(matParams.template getRealParams<EclMultiplexerApproach::Stone2>(),
                    manager, info);
            break;
        default:
            OPM_THROW(std::runtime_error,
                      "End-point gradients support 3-phase material laws only "
                      "(Default/Stone1/Stone2)");
        }
    }

    void restore(unsigned cell, const EndpointParamDef<Scalar>& def)
    {
        apply(cell, def, 0.0);
    }

private:
    template<class ConcreteParams, class Manager>
    void reinit_(ConcreteParams& params, Manager& manager,
                 const EclEpsScalingPointsInfo<Scalar>& info)
    {
        params.oilWaterParams().drainageParams().scaledPoints()
            .init(info, manager.oilWaterConfig(), EclTwoPhaseSystemType::OilWater);
        params.gasOilParams().drainageParams().scaledPoints()
            .init(info, manager.gasOilConfig(), EclTwoPhaseSystemType::GasOil);
        // the 3-phase params keep their own connate-water copy for the
        // oil-relperm saturation blending
        params.setSwl(info.Swl);
    }

    Problem& problem_;
};

} // namespace Opm

#endif // OPM_ADJOINT_ENDPOINT_GRADIENTS_HPP
