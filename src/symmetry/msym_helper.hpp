#ifndef MSYMHELPER_H
#define MSYMHELPER_H

#include "../integrals/hfscf_shell.hpp"
#include <string>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#ifdef HAS_MESON_CONFIG
#include <libmsym/msym.h>
#else
#include "msym.h"
#endif
#include <vector>
#include <map>

template<typename T>
using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

using BASIS::Shell;
using ShellVector = std::vector<Shell>;

namespace MSYM
{
    struct atomdata
    {
        atomdata(const std::string at_id, int ctr, int n_, int l_, int idx_)
        : atom_id(at_id), center(ctr), n(n_), l(l_), idx(idx_) {}
        std::string atom_id;
        int center;
        int n;
        int l;
        int idx;
    };

    class Msym
    {
        public:
            Msym() = delete;
            explicit Msym(int natoms) : m_natoms(natoms){}
            Msym(const Msym&) = delete;
            Msym& operator=(const Msym& other) = delete;
            Msym(const Msym&&) = delete;
            Msym&& operator=(const Msym&& other) = delete;

            void do_symmetry_analysis(EigenMatrix<double>& geom, const std::vector<int>& zval);

            int build_salcs(const Eigen::Ref<const EigenMatrix<double>>& geom, const std::vector<int>& zval,
                             const ShellVector& shells, const std::vector<int>& centers);

            int get_sigma() const { return m_sigma;}
            double get_sym_deviation() const { return m_sym_deviation;}
            std::string get_point_group() const { return m_point_group;}
            std::string get_sub_group() const { return m_sub_group;}
            std::vector<std::string> get_sym_species() const { return sym_species;}
            std::vector<int> get_irrep_sizes() const { return irrep_size;}
            [[nodiscard]] std::vector<EigenMatrix<double>> get_sym_blocks() const {return  sblocks;}
        
        private:
            int m_natoms;
            int m_sigma{1}; // wil be updated if symmetry is used
            bool isaligned{false};
            double m_sym_deviation{0};
            std::string m_point_group{};
            std::string m_sub_group{};
            std::vector<atomdata> unique_atom_data;
            std::vector<atomdata> atom_data;
             std::unordered_map<std::string, int> bfs_to_ids;
            std::vector<std::string> sym_species;
            std::vector<int> irrep_size;
            std::vector<EigenMatrix<double>> sblocks;

            void set_shell_data(const ShellVector& shells, const std::vector<int>& centers, 
                                const std::vector<int>& zval);
            
            void build_symmetry_blocks(std::vector<EigenMatrix<double>>& symblock,
                                       const msym_salc_t& salc, const msym_element_t * melements, int bfsl);

    };
    // Just for debug
    void printSALC(const msym_salc_t& salc, const msym_element_t * melements);
}
#endif