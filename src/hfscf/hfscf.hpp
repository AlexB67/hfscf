#ifndef HFSCF_SCF_HF_H
#define HFSCF_SCF_HF_H

#include "../scfsolver/hfscf_hf_solver.hpp"
#include "../scfsolver/hfscf_uhf_solver.hpp"
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <string>

template <typename T>
using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
using Matrix2D = std::vector<std::vector<T> >;

template <typename T>
using Matrix1D = std::vector<T>;

template <typename T>
using EigenVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

namespace Mol
{
    class scf
    {
        public:
            explicit scf(const bool verbose, const std::string& geometry_file, const std::string& prefix);
            scf(const scf&) = delete;
            scf& operator=(const scf& other) = delete;
            scf(const scf&&) = delete;
            scf&& operator=(const scf&& other) = delete;     
            void hf_run();

         private:
            bool m_verbose;
            // TODO refactor the following 4 should come from the molecule class only
            std::unique_ptr<Mol::hf_solver> rhf_ptr;
            std::unique_ptr<Mol::uhf_solver> uhf_ptr;
            std::shared_ptr<MOLEC::Molecule> molecule;

            void post_rhf();
            void post_uhf();
            [[nodiscard]] int get_frozen_core();
            
             //UHF & RHF
            void geom_opt_numeric();
            void geom_opt();
            void calc_numeric_gradient(EigenMatrix<double>& gradient, 
                                       std::vector<bool>& coords, bool check_if_out_of_plane = true,
                                       bool print_gradient = true);
            
            void calc_numeric_hessian_from_analytic_gradient(const double rhf_mp2_energy);
            void calc_numeric_hessian_from_mp2_energy();

            void print_gradient(const Eigen::Ref<const EigenMatrix<double> >& gradient);

            void print_rhf_energies(const double mp2_energy, const double mp3_energy, const double ccsd_energy, 
                                    const double cc_triples_energy, const int ccsd_iterations, 
                                    const std::vector<double>& ccsd_energies, const std::vector<double>& ccsd_deltas, 
                                    const std::vector<double>& ccsd_rms, 
                                    const std::vector<std::pair<double, std::string>>& largest_t1,
                                    const std::vector<std::pair<double, std::string>>& largest_t2) const;
            
            void print_uhf_energies(const Eigen::Ref<const EigenVector<double>>& ump2_energies,
                                    const Eigen::Ref<const EigenVector<double>>& ump3_energies) const;
    }; 
}

#endif
