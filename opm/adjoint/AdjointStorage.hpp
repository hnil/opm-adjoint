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
 * \brief Archive used to store per-substep state for adjoint/replay runs.
 *
 * The archive stores (group, dataset) -> buffer mappings where the buffer is
 * produced by the generic serializeOp machinery (Serializer<MemPacker>), i.e.
 * the same encoding as the HDF5-based --save-step/--load-step restart files.
 *
 * Two interchangeable backends:
 *  - HDF5 (one .h5 file per run) when OPM is built with HDF5 — the preferred
 *    backend, browsable with h5dump and consistent with SimulatorSerializer;
 *  - a plain directory store (one file per dataset) with no external
 *    dependencies, so adjoint runs and their tests work on any build.
 *
 * The backend is chosen from the archive path: a .h5 or .hdf5 suffix
 * selects HDF5, anything else the directory store.
 */
#ifndef OPM_ADJOINT_STORAGE_HPP
#define OPM_ADJOINT_STORAGE_HPP

#include <opm/common/ErrorMacros.hpp>
#include <opm/common/utility/Serializer.hpp>
#include <opm/common/utility/MemPacker.hpp>

#include <opm/simulators/utils/SerializationPackers.hpp>

#include <opm/simulators/utils/ParallelCommunication.hpp>

#if HAVE_HDF5
#include <opm/simulators/utils/HDF5File.hpp>
#include <opm/simulators/utils/HDF5Serializer.hpp>
#endif

#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

namespace Opm {

//! \brief Serializer writing each dataset as a plain file in a directory tree.
//!
//! Mirrors the (group, dataset) interface of HDF5Serializer; group names map
//! to sub-directories, datasets to files with a .bin suffix. Serial only.
class AdjointRawDirSerializer : public Serializer<Serialization::MemPacker>
{
public:
    enum class Mode { Write, Read };

    AdjointRawDirSerializer(const std::string& dirName, Mode mode)
        : Serializer<Serialization::MemPacker>(m_packer_priv)
        , m_dir(dirName)
        , m_mode(mode)
    {
        if (m_mode == Mode::Write) {
            std::filesystem::create_directories(m_dir);
        } else if (!std::filesystem::is_directory(m_dir)) {
            OPM_THROW(std::runtime_error,
                      "Adjoint archive directory " + dirName + " does not exist");
        }
    }

    template<class T>
    void write(T& data, const std::string& group, const std::string& dset)
    {
        this->pack(data);
        const auto path = datasetPath(group, dset);
        std::filesystem::create_directories(path.parent_path());
        std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
        if (!ofs) {
            OPM_THROW(std::runtime_error,
                      "Failed to open " + path.string() + " for writing");
        }
        ofs.write(m_buffer.data(), static_cast<std::streamsize>(m_buffer.size()));
        if (!ofs) {
            OPM_THROW(std::runtime_error, "Failed to write " + path.string());
        }
    }

    template<class T>
    void read(T& data, const std::string& group, const std::string& dset)
    {
        const auto path = datasetPath(group, dset);
        std::ifstream ifs(path, std::ios::binary | std::ios::ate);
        if (!ifs) {
            OPM_THROW(std::runtime_error,
                      "Failed to open " + path.string() + " for reading");
        }
        const auto size = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        m_buffer.resize(static_cast<std::size_t>(size));
        ifs.read(m_buffer.data(), size);
        if (!ifs) {
            OPM_THROW(std::runtime_error, "Failed to read " + path.string());
        }
        this->unpack(data);
    }

    bool hasDataset(const std::string& group, const std::string& dset) const
    {
        return std::filesystem::exists(datasetPath(group, dset));
    }

    //! \brief Store free-form text (e.g. the parameter listing) next to the data.
    void writeText(const std::string& group,
                   const std::string& name,
                   const std::string& text)
    {
        const auto path = std::filesystem::path(m_dir) / relativeGroup(group) / name;
        std::filesystem::create_directories(path.parent_path());
        std::ofstream ofs(path, std::ios::trunc);
        ofs << text;
    }

private:
    static std::string relativeGroup(const std::string& group)
    {
        // Group names use absolute HDF5-style paths ("/adjoint/substep/3");
        // strip the leading slash for filesystem use.
        if (!group.empty() && group.front() == '/') {
            return group.substr(1);
        }
        return group;
    }

    std::filesystem::path datasetPath(const std::string& group,
                                      const std::string& dset) const
    {
        return std::filesystem::path(m_dir) / relativeGroup(group) / (dset + ".bin");
    }

    const Serialization::MemPacker m_packer_priv{};
    std::string m_dir;
    Mode m_mode;
};

//! \brief Backend-dispatching archive for adjoint state.
class AdjointArchive
{
public:
    enum class Mode { Write, Read };

    //! \param path Archive path; *.h5/*.hdf5 selects the HDF5 backend.
    //! \param mode Write (forward recording) or Read (replay/adjoint).
    //! \param comm Grid communicator; the HDF5 backend stores one
    //!             per-rank dataset (PROCESS_SPLIT), and the directory
    //!             store uses a per-rank sub-directory, so each rank
    //!             reads back exactly the local partition it wrote.
    explicit AdjointArchive(const std::string& path, Mode mode,
                            Parallel::Communication comm = Parallel::Communication{})
    {
        if (isHdf5Path(path)) {
#if HAVE_HDF5
            hdf5_ = std::make_unique<HDF5Serializer>(
                path,
                mode == Mode::Write ? HDF5File::OpenMode::OVERWRITE
                                    : HDF5File::OpenMode::READ,
                comm);
#else
            OPM_THROW(std::runtime_error,
                      "Adjoint archive path " + path + " requests the HDF5 "
                      "backend, but this build has no HDF5 support. Use a "
                      "path without .h5 suffix for the directory store.");
#endif
        } else {
            // The directory store has no per-rank dataset splitting, so
            // give each rank its own sub-directory in a parallel run.
            std::string dirPath = path;
            if (comm.size() > 1) {
                dirPath = (std::filesystem::path(path) /
                           ("rank_" + std::to_string(comm.rank()))).string();
            }
            raw_ = std::make_unique<AdjointRawDirSerializer>(
                dirPath,
                mode == Mode::Write ? AdjointRawDirSerializer::Mode::Write
                                    : AdjointRawDirSerializer::Mode::Read);
        }
    }

    static bool isHdf5Path(const std::string& path)
    {
        return path.size() >= 3 &&
               (path.rfind(".h5") == path.size() - 3 ||
                (path.size() >= 5 && path.rfind(".hdf5") == path.size() - 5));
    }

    //! \brief Default archive path for a case: HDF5 file if available,
    //!        directory store otherwise.
    static std::string defaultPath(const std::string& outputDir,
                                   const std::string& caseName)
    {
        auto base = (std::filesystem::path(outputDir) / (caseName + ".ADJOINT")).string();
#if HAVE_HDF5
        base += ".h5";
#endif
        return base;
    }

    template<class T>
    void write(T& data, const std::string& group, const std::string& dset)
    {
#if HAVE_HDF5
        if (hdf5_) {
            hdf5_->write(data, group, dset);
            return;
        }
#endif
        raw_->write(data, group, dset);
    }

    template<class T>
    void read(T& data, const std::string& group, const std::string& dset)
    {
#if HAVE_HDF5
        if (hdf5_) {
            hdf5_->read(data, group, dset);
            return;
        }
#endif
        raw_->read(data, group, dset);
    }

    //! \brief Store free-form text (parameter listing, warnings) for
    //!        provenance; ignored by the reader.
    void writeText(const std::string& group,
                   const std::string& name,
                   const std::string& text)
    {
#if HAVE_HDF5
        if (hdf5_) {
            // Stored as a regular (string) dataset in the HDF5 backend.
            std::string copy = text;
            hdf5_->write(copy, group, name);
            return;
        }
#endif
        raw_->writeText(group, name, text);
    }

    bool usesHdf5() const
    {
#if HAVE_HDF5
        return static_cast<bool>(hdf5_);
#else
        return false;
#endif
    }

private:
#if HAVE_HDF5
    std::unique_ptr<HDF5Serializer> hdf5_;
#endif
    std::unique_ptr<AdjointRawDirSerializer> raw_;
};

} // namespace Opm

#endif // OPM_ADJOINT_STORAGE_HPP
