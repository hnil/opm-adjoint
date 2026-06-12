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
 * \brief Backward adjoint sweep with parameter-gradient accumulation.
 *
 * For each recorded substep k (walked backwards) the replay driver
 * recreates the converged linearization; this class then
 *  1. forms the Schur-reduced system matrix A_red = A - C^T D^-1 B by
 *     adding the well contributions (public BlackoilWellModel API),
 *  2. solves A_red^T lambda_k = -(dJ_k/dx_k)^T - Bdiag_{k+1}^T lambda_{k+1}
 *     with the direct transposed solver,
 *  3. accumulates the parameter gradients
 *       g_pv[i]   += lambda_k[i] . (V_i/dt_k) (S_i(x_k) - S_i(x_{k-1}))
 *       g_tmult[f] += (lambda_I - lambda_J) . flux_f
 *     (storage linear in the pore-volume multiplier; TPFA flux linear in
 *     the transmissibility multiplier; fluxes re-evaluated at the
 *     converged state with the public templated TPFA kernels).
 *
 * v1 restrictions: objective without explicit well-DOF dependence (the
 * well equations are eliminated exactly through the Schur complement),
 * isothermal, no diffusion/dispersion, serial.
 */
#ifndef OPM_ADJOINT_SOLVER_HPP
#define OPM_ADJOINT_SOLVER_HPP

#include <opm/common/ErrorMacros.hpp>
#include <opm/common/OpmLog/OpmLog.hpp>

#include <opm/models/utils/propertysystem.hh>

#include <opm/adjoint/AdjointLinearSolver.hpp>
#include <opm/adjoint/AdjointObjective.hpp>
#include <opm/adjoint/AdjointReplay.hpp>

#include <opm/input/eclipse/EclipseState/Grid/FaceDir.hpp>

#include <opm/simulators/wells/StandardWell.hpp>

#include <dune/common/fvector.hh>
#include <dune/istl/bvector.hh>

#include <fmt/format.h>

#include <array>
#include <fstream>
#include <map>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace Opm {

template<class TypeTag>
class AdjointSolver
{
public:
    using Simulator = GetPropType<TypeTag, Properties::Simulator>;
    using Scalar = GetPropType<TypeTag, Properties::Scalar>;
    using Evaluation = GetPropType<TypeTag, Properties::Evaluation>;
    using LocalResidual = GetPropType<TypeTag, Properties::LocalResidual>;
    using FluidSystem = GetPropType<TypeTag, Properties::FluidSystem>;
    using SparseMatrixAdapter = GetPropType<TypeTag, Properties::SparseMatrixAdapter>;

    static constexpr int numEq = getPropValue<TypeTag, Properties::NumEq>();

    using Replay = AdjointReplay<TypeTag>;
    using BdiagBlock = typename Replay::BdiagBlock;
    using Matrix = std::remove_cvref_t<
        decltype(std::declval<Simulator&>().model().linearizer().jacobian().istlMatrix())>;
    using Vector = Dune::BlockVector<Dune::FieldVector<Scalar, numEq>>;

    explicit AdjointSolver(Simulator& simulator)
        : simulator_(simulator)
        , replay_(simulator)
        , objective_(AdjointConfig::fromParameters().objective)
    {
        replay_.setComputeBdiag(true);
    }

    //! \brief Evaluate the objective on the recorded final state only.
    int evaluateObjectiveOnly()
    {
        const auto& meta = replay_.meta();
        const int numSubsteps = static_cast<int>(meta.substeps.size());
        if (numSubsteps == 0) {
            OpmLog::error("Adjoint archive contains no substeps");
            return EXIT_FAILURE;
        }
        Scalar value = 0.0;
        for (int k = 0; k < numSubsteps; ++k) {
            const auto& step = meta.substeps[k];
            replay_.loadSnapshot(k, step.reportStep);
            value += objective_.stepValue(simulator_, k, numSubsteps, step.dt);
        }
        OpmLog::info(fmt::format("Adjoint objective J = {:.16e}", value));
        return EXIT_SUCCESS;
    }

    //! \brief Backward sweep: solve all adjoint systems and accumulate
    //!        the parameter gradients.
    int run()
    {
        const auto& meta = replay_.meta();
        const int numSubsteps = static_cast<int>(meta.substeps.size());
        if (numSubsteps == 0) {
            OpmLog::error("Adjoint archive contains no substeps");
            return EXIT_FAILURE;
        }

        auto& model = simulator_.model();
        const std::size_t numCells = model.numGridDof();

        gradientPv_.assign(numCells, 0.0);

        Vector lambda(numCells);
        Vector lambdaPrev(numCells);
        lambdaPrev = 0.0;
        std::vector<BdiagBlock> bdiagPrev;

        Scalar objectiveValue = 0.0;
        int replayFailures = 0;

        for (int k = numSubsteps - 1; k >= 0; --k) {
            double worstResidual = 0.0;
            double worstJacobian = 0.0;
            if (!replay_.replayStep(k, worstResidual, worstJacobian)) {
                ++replayFailures;
            }
            const auto& step = meta.substeps[k];

            // Objective contribution of this substep (solution(0) = x_k).
            objectiveValue +=
                objective_.stepValue(simulator_, k, numSubsteps, step.dt);

            // Right-hand side:
            //   -(dJ_k/dx_r)^T + B^T D^-T (dJ_k/dx_w) - Bdiag_{k+1}^T lambda_{k+1}
            // (from stationarity of the Lagrangian in x_w and x_r; the well
            // equations are eliminated through the Schur complement).
            Vector rhs(numCells);
            rhs = 0.0;
            objective_.addStepGradient(simulator_, k, numSubsteps, rhs);
            rhs *= -1.0;
            addWellObjectiveRhs_(step.dt, rhs);
            if (k < numSubsteps - 1) {
                for (std::size_t i = 0; i < numCells; ++i) {
                    // rhs_i -= Bdiag_{k+1,i}^T lambda_{k+1,i}
                    bdiagPrev[i].usmtv(-1.0, lambdaPrev[i], rhs[i]);
                }
            }

            // Schur-reduced matrix: A -= C^T D^-1 B (in place on the
            // linearizer's matrix; it is re-assembled in the next
            // replayStep anyway).
            auto& jacobian = model.linearizer().jacobian();
            simulator_.problem().wellModel().addWellContributions(jacobian);

            // Solve A_red^T lambda_k = rhs.
            linearSolver_.solveTransposed(jacobian.istlMatrix(), rhs, lambda);

            computeWellAdjoints_(k, step.dt, lambda);
            accumulateGradients_(step.dt, lambda);

            bdiagPrev = replay_.bdiag();
            lambdaPrev = lambda;
        }

        OpmLog::info(fmt::format(
            "Adjoint sweep finished: J = {:.16e}, {} substeps, {} replay "
            "verification failures", objectiveValue, numSubsteps, replayFailures));

        writeResults_(objectiveValue);
        return replayFailures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }

private:
    //! \brief dJ_k/dx_w of the objective for one StandardWell
    //!        (empty vector if the well is not involved).
    Dune::DynamicVector<Scalar>
    wellObjectiveGradient_(const StandardWell<TypeTag>& stdWell, double dt)
    {
        using Kind = typename AdjointObjectiveFunction<TypeTag>::Kind;
        const auto& eqns = stdWell.linSys();
        const std::size_t numWellEq = eqns.Dmatrix()[0][0].N();
        Dune::DynamicVector<Scalar> dJdxw(numWellEq, 0.0);

        if (objective_.kind() == Kind::WellBhp) {
            // BHP is the last well primary variable.
            dJdxw[numWellEq - 1] = objective_.bhpGradient(stdWell.name(), dt);
        } else if (objective_.kind() == Kind::RateTerms) {
            // q_c is a function of the well primary variables; its
            // derivatives sit in the well slots of the EvalWell returned
            // by getQs (reservoir derivatives first, well derivatives at
            // numEq + j).
            for (const auto& term : objective_.rateTerms()) {
                if (term.well != stdWell.name()) {
                    continue;
                }
                const auto qs = stdWell.primaryVariables().getQs(
                    FluidSystem::canonicalToActiveCompIdx(term.canonicalCompIdx));
                const Scalar weight =
                    objective_.termWeight(simulator_, term, dt, qs.value());
                for (std::size_t j = 0; j < numWellEq; ++j) {
                    dJdxw[j] += weight * qs.derivative(numEq + j);
                }
            }
        }
        return dJdxw;
    }

    //! \brief Add the well-objective coupling B^T D^-T (dJ_k/dx_w) to the
    //!        adjoint right-hand side (StandardWells only).
    void addWellObjectiveRhs_(double dt, Vector& rhs)
    {
        const auto& wellModel = simulator_.problem().wellModel();
        for (const auto& wellPtr : wellModel.localNonshutWells()) {
            if (!objective_.involvesWell(wellPtr->name())) {
                continue;
            }
            const auto* stdWell =
                dynamic_cast<const StandardWell<TypeTag>*>(wellPtr.get());
            if (!stdWell) {
                OPM_THROW(std::runtime_error,
                          "Adjoint well objectives support StandardWells only");
            }
            const auto& eqns = stdWell->linSys();
            const auto& Dmat = eqns.Dmatrix()[0][0];
            const std::size_t numWellEq = Dmat.N();

            const auto dJdxw = wellObjectiveGradient_(*stdWell, dt);

            // y = D^-T dJdxw  (small dense transposed solve).
            Dune::DynamicMatrix<Scalar> Dt(numWellEq, numWellEq);
            for (std::size_t i = 0; i < numWellEq; ++i) {
                for (std::size_t j = 0; j < numWellEq; ++j) {
                    Dt[i][j] = Dmat[j][i];
                }
            }
            Dune::DynamicVector<Scalar> y(numWellEq, 0.0);
            Dt.solve(y, dJdxw);

            // rhs_cell += B[0][perf]^T y for all perforated cells.
            const auto& Bmat = eqns.Bmatrix();
            const auto& cells = stdWell->cells();
            for (auto colIt = Bmat[0].begin(); colIt != Bmat[0].end(); ++colIt) {
                const std::size_t perfIdx = colIt.index();
                const int cellIdx = cells[perfIdx];
                const auto& block = *colIt; // numWellEq x numEq
                for (int eq = 0; eq < numEq; ++eq) {
                    Scalar sum = 0.0;
                    for (std::size_t we = 0; we < numWellEq; ++we) {
                        sum += block[we][eq] * y[we];
                    }
                    rhs[cellIdx][eq] += sum;
                }
            }
        }
    }

    //! \brief Well adjoints of substep k:
    //!        lambda_w = -D^-T (dJ_k/dx_w + sum_perf C[0][perf] lambda_r[cell]),
    //!        from stationarity of the Lagrangian in x_w. Appended to the
    //!        lambda-well log (basis for later well-control gradients).
    void computeWellAdjoints_(int k, double dt, const Vector& lambda)
    {
        const auto& wellModel = simulator_.problem().wellModel();
        for (const auto& wellPtr : wellModel.localNonshutWells()) {
            const auto* stdWell =
                dynamic_cast<const StandardWell<TypeTag>*>(wellPtr.get());
            if (!stdWell) {
                continue;
            }
            const auto& eqns = stdWell->linSys();
            const auto& Dmat = eqns.Dmatrix()[0][0];
            const std::size_t numWellEq = Dmat.N();

            auto work = wellObjectiveGradient_(*stdWell, dt);

            const auto& Cmat = eqns.Cmatrix();
            const auto& cells = stdWell->cells();
            for (auto colIt = Cmat[0].begin(); colIt != Cmat[0].end(); ++colIt) {
                const std::size_t perfIdx = colIt.index();
                const int cellIdx = cells[perfIdx];
                const auto& block = *colIt; // numWellEq x numEq
                for (std::size_t we = 0; we < numWellEq; ++we) {
                    for (int eq = 0; eq < numEq; ++eq) {
                        work[we] += block[we][eq] * lambda[cellIdx][eq];
                    }
                }
            }

            Dune::DynamicMatrix<Scalar> Dt(numWellEq, numWellEq);
            for (std::size_t i = 0; i < numWellEq; ++i) {
                for (std::size_t j = 0; j < numWellEq; ++j) {
                    Dt[i][j] = Dmat[j][i];
                }
            }
            Dune::DynamicVector<Scalar> lambdaW(numWellEq, 0.0);
            Dt.solve(lambdaW, work);
            lambdaW *= -1.0;

            std::string line = std::to_string(k) + " " + stdWell->name();
            for (std::size_t j = 0; j < numWellEq; ++j) {
                line += " " + std::to_string(lambdaW[j]);
            }
            wellAdjointLog_.push_back(std::move(line));

            // Control-target gradient: the active control equation is the
            // last well-equation row with the convention
            // R_ctrl = (rate or bhp) - target, so dR_ctrl/d(target) = -1
            // and dJ/d(target) accumulates -lambda_w[last] over the
            // substeps where this control is active.
            controlGradient_[stdWell->name()] -= lambdaW[numWellEq - 1];
        }
    }

    void accumulateGradients_(double dt, const Vector& lambda)
    {
        auto& model = simulator_.model();
        const std::size_t numCells = model.numGridDof();

        // Pore-volume multiplier gradient from the storage term:
        // dR_i/dpvmult_i = (V_i/dt)(S_i(x_k) - S_i(x_{k-1})); after the
        // final replay linearization IQ(.,0) is at x_k and IQ(.,1) at
        // x_{k-1}.
        for (std::size_t i = 0; i < numCells; ++i) {
            Dune::FieldVector<Scalar, numEq> storageNew;
            Dune::FieldVector<Scalar, numEq> storageOld;
            LocalResidual::template computeStorage<Scalar>(
                storageNew, model.intensiveQuantities(i, /*timeIdx=*/0));
            LocalResidual::template computeStorage<Scalar>(
                storageOld, model.intensiveQuantities(i, /*timeIdx=*/1));
            const Scalar factor = model.dofTotalVolume(i) / dt;
            for (int eq = 0; eq < numEq; ++eq) {
                gradientPv_[i] += lambda[i][eq] * factor *
                                  (storageNew[eq] - storageOld[eq]);
            }
        }

        // Transmissibility-multiplier gradient from the flux term:
        // dR_I/dtmult_f = flux_f (TPFA flux linear in trans), so
        // g_tmult[f] += (lambda_I - lambda_J) . flux_f. Fluxes are
        // re-evaluated at the converged state with the public TPFA
        // kernels, mirroring linearize_cell.
        const auto& neighborInfo = model.linearizer().getNeighborInfo();
        if (gradientTrans_.empty()) {
            // Face list (I < J) built on first use.
            for (std::size_t globI = 0; globI < numCells; ++globI) {
                const auto& nbInfos = neighborInfo[globI];
                for (const auto& nbInfo : nbInfos) {
                    if (nbInfo.neighbor > globI) {
                        faces_.push_back({globI, nbInfo.neighbor});
                        faceDirs_.push_back(nbInfo.res_nbinfo.faceDir);
                    }
                }
            }
            gradientTrans_.assign(faces_.size(), 0.0);
        }

        std::size_t faceIdx = 0;
        for (std::size_t globI = 0; globI < numCells; ++globI) {
            const auto& nbInfos = neighborInfo[globI];
            const auto& intQuantsIn = model.intensiveQuantities(globI, /*timeIdx=*/0);
            for (const auto& nbInfo : nbInfos) {
                const unsigned globJ = nbInfo.neighbor;
                if (globJ <= globI) {
                    continue;
                }
                const auto& intQuantsEx = model.intensiveQuantities(globJ, /*timeIdx=*/0);
                Dune::FieldVector<Evaluation, numEq> flux(0.0);
                Dune::FieldVector<Evaluation, numEq> darcy(0.0);
                LocalResidual::computeFlux(flux, darcy, globI, globJ,
                                           intQuantsIn, intQuantsEx,
                                           nbInfo.res_nbinfo,
                                           simulator_.problem().moduleParams());
                for (int eq = 0; eq < numEq; ++eq) {
                    const Scalar fluxValue =
                        flux[eq].value() * nbInfo.res_nbinfo.faceArea;
                    gradientTrans_[faceIdx] +=
                        (lambda[globI][eq] - lambda[globJ][eq]) * fluxValue;
                }
                ++faceIdx;
            }
        }
    }

    void writeResults_(Scalar objectiveValue)
    {
        const auto& ioConfig = simulator_.vanguard().eclState().getIOConfig();
        const std::string base = ioConfig.getOutputDir() + "/" +
                                 simulator_.vanguard().caseName();

        {
            std::ofstream os(base + ".ADJOINT_GRADIENTS_PV.txt");
            os.precision(16);
            os << "# dJ/d(pvmult_i), one line per cell (J = " << objectiveValue
               << ")\n";
            for (const auto& value : gradientPv_) {
                os << value << "\n";
            }
        }
        {
            std::ofstream os(base + ".ADJOINT_GRADIENTS_TRANS.txt");
            os.precision(16);
            os << "# dJ/d(tmult_f): cellI cellJ gradient (J = " << objectiveValue
               << ")\n";
            for (std::size_t f = 0; f < faces_.size(); ++f) {
                os << faces_[f].first << " " << faces_[f].second << " "
                   << gradientTrans_[f] << "\n";
            }
        }
        writePermGradients_(base);
        {
            std::ofstream os(base + ".ADJOINT_GRADIENTS_WELLCTRL.txt");
            os.precision(16);
            os << "# dJ/d(active control target) per well, summed over all "
                  "substeps; SI units (rate targets: per m3/s, bhp targets: "
                  "per Pa)\n";
            for (const auto& [well, value] : controlGradient_) {
                os << well << " " << value << "\n";
            }
        }
        {
            std::ofstream os(base + ".ADJOINT_LAMBDA_WELLS.txt");
            os << "# substep well lambda_w (one line per well per substep, "
                  "backward order)\n";
            for (const auto& line : wellAdjointLog_) {
                os << line << "\n";
            }
        }
        OpmLog::info("Adjoint gradients written to " + base +
                     ".ADJOINT_GRADIENTS_{PV,TRANS}.txt");
    }

    //! \brief Permeability chain rule, post-processed from the
    //!        transmissibility-multiplier gradients:
    //!        dJ/dK_in^dir = sum_faces g_tmult[f] (h_out/(h_in+h_out)) / K_in^dir
    //!        (T proportional to the harmonic mean of the one-sided half
    //!        transmissibilities, which are linear in the directional
    //!        permeability; multipliers cancel since g_tmult is the
    //!        multiplier-based gradient).
    void writePermGradients_(const std::string& base)
    {
        const auto& trans = simulator_.problem().eclTransmissibilities();
        const std::size_t numCells = simulator_.model().numGridDof();
        // per cell: d/dPERMX, d/dPERMY, d/dPERMZ (SI: per m^2)
        std::vector<std::array<Scalar, 3>> gradient(numCells, {0.0, 0.0, 0.0});

        for (std::size_t f = 0; f < faces_.size(); ++f) {
            const auto [globI, globJ] = faces_[f];
            int dim;
            switch (faceDirs_[f]) {
            case FaceDir::XPlus:
            case FaceDir::XMinus: dim = 0; break;
            case FaceDir::YPlus:
            case FaceDir::YMinus: dim = 1; break;
            case FaceDir::ZPlus:
            case FaceDir::ZMinus: dim = 2; break;
            default:
                continue; // NNC: no single permeability direction
            }
            const Scalar hIn = trans.halfTransmissibility(globI, globJ);
            const Scalar hOut = trans.halfTransmissibility(globJ, globI);
            if (hIn + hOut <= 0.0) {
                continue;
            }
            const Scalar kIn = trans.permeability(globI)[dim][dim];
            const Scalar kOut = trans.permeability(globJ)[dim][dim];
            const Scalar g = gradientTrans_[f];
            if (kIn > 0.0) {
                gradient[globI][dim] += g * (hOut / (hIn + hOut)) / kIn;
            }
            if (kOut > 0.0) {
                gradient[globJ][dim] += g * (hIn / (hIn + hOut)) / kOut;
            }
        }

        std::ofstream os(base + ".ADJOINT_GRADIENTS_PERM.txt");
        os.precision(16);
        os << "# dJ/dPERMX dJ/dPERMY dJ/dPERMZ per cell, SI (per m^2); "
              "multiply by 9.869233e-16 for per-mD\n";
        for (const auto& g : gradient) {
            os << g[0] << " " << g[1] << " " << g[2] << "\n";
        }
    }

    Simulator& simulator_;
    Replay replay_;
    AdjointObjectiveFunction<TypeTag> objective_;
    AdjointLinearSolver<Matrix, Vector> linearSolver_;

    std::vector<Scalar> gradientPv_;
    std::vector<std::pair<std::size_t, unsigned>> faces_;
    std::vector<FaceDir::DirEnum> faceDirs_;
    std::vector<Scalar> gradientTrans_;
    std::vector<std::string> wellAdjointLog_;
    std::map<std::string, Scalar> controlGradient_;
};

} // namespace Opm

#endif // OPM_ADJOINT_SOLVER_HPP
