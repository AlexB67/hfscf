#ifndef HFSCF_SOLVER_HF_H
#define HFSCF_SOLVER_HF_H

#include "../molecule/hfscf_molecule.hpp"
#include "hfscf_diis.hpp"
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <tuple>
#include <algorithm>
#include <string>

template <typename T>
using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
using EigenVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

namespace Mol
{
    class hf_solver
    {
        public:
            explicit hf_solver(const std::shared_ptr<MOLEC::Molecule>& mol);
            virtual ~hf_solver(){}
            hf_solver(const hf_solver&) = delete;
            hf_solver& operator=(const hf_solver& other) = delete;
            hf_solver(const hf_solver&&) = delete;
            hf_solver&& operator=(const hf_solver&& other) = delete;

            virtual void get_data();
            virtual void init_data(bool print = true, bool use_guess_density = false);
            virtual bool scf_run(bool print = true);

            virtual void print_fock_matrix() const;
            virtual void print_mo_coefficients_matrix() const;
            virtual void print_density_matrix() const;
            virtual void print_mo_energies() const;
            virtual void print_scf_energies() const;

            int get_iterations() const { return iteration;}
            double get_scf_energy() const { return scf_energy_current;}
            double get_one_electron_energy() const;

            const Eigen::Ref<const EigenMatrix<double>> get_mo_coef() const {return mo_coff;}
            const Eigen::Ref<const EigenMatrix<double>> get_fock_matrix() const {return f_mat;}
            const Eigen::Ref<const EigenVector<double>> get_repulsion_vector() const { return e_rep_mat;}
            const Eigen::Ref<const EigenMatrix<double>> get_density_matrix() const {return d_mat;}
            const Eigen::Ref<const EigenMatrix<double>> get_overlap_matrix() const {return s_mat;}
            const Eigen::Ref<const EigenMatrix<double>> get_mo_energies() const {return mo_energies;}

        protected:
            Eigen::Index norbitals{0};
            Eigen::Index nelectrons{0};
            int iteration{0};
            int so_iterstart{std::numeric_limits<int>::max()};
            bool diis_log{true};
            bool use_sym{false};
            bool soscf{false};
            double scf_energy_previous{0};
            double scf_energy_current{std::numeric_limits<double>::max()};
            std::unique_ptr<DIIS_SOLVER::diis_solver> diis_ptr;
            std::shared_ptr<MOLEC::Molecule> molecule;
            std::vector<int> pop_index;

            std::vector<double> scf_energies;
            EigenMatrix<double> mo_energies;
            EigenMatrix<double> k_mat;          // Kintic energy
            EigenMatrix<double> v_mat;          // potential
            EigenMatrix<double> s_mat;          // overlap
            EigenMatrix<double> hcore_mat;      // Core hamiltonian
            EigenVector<double> e_rep_mat;      // 4D 2 electron repulsion integrals stored as a 1D array
            EigenMatrix<double> s_mat_sqrt;     // S^-1/2 
            EigenMatrix<double> f_mat;          // F
            EigenMatrix<double> fp_mat;         // F' Note:  F' = s_mat_sqrt.T * F * s_mat_sqrt
            EigenMatrix<double> mo_coff;        // MO coefficients
            EigenMatrix<double> d_mat;          // density matrix
            EigenMatrix<double> d_mat_previous; // density matrix store for the previous iteration
            EigenMatrix<double> e_rep_screen; // store <aa|bb> for screening in direct SCF

            // symmetry vector version

            //std::vector<std::vector<double>> vscf_energies;
            std::vector<EigenMatrix<double>> vmo_energies;
            std::vector<EigenMatrix<double>> vk_mat;          // Kintic energy
            std::vector<EigenMatrix<double>> vv_mat;          // potential
            std::vector<EigenMatrix<double>> vs_mat;          // overlap
            std::vector<EigenMatrix<double>> vhcore_mat;      // Core hamiltonian
            std::vector<EigenMatrix<double>> vs_mat_sqrt;     // S^-1/2 
            std::vector<EigenMatrix<double>> vf_mat;          // F
            std::vector<EigenMatrix<double>> vfp_mat;         // F' Note:  F' = s_mat_sqrt.T * F * s_mat_sqrt
            std::vector<EigenMatrix<double>> vmo_coff;        // MO coefficients
            std::vector<EigenMatrix<double>> vd_mat;          // density matrix
            std::vector<EigenMatrix<double>> vd_mat_previous; // density matrix store for the previous iteration
            std::vector<std::pair<double, std::string>> mo_energies_sym;
            

            void create_one_electron_hamiltonian();
            void allocate_memory();
            void create_repulsion_matrix();
            double calc_initial_fock_matrix();
            double calc_initial_fock_matrix_with_symmetry();
            virtual void update_fock_matrix();
            virtual void update_fock_matrix_with_symmetry();
            virtual void update_fock_matrix_scf_direct();
            virtual void calc_density_matrix();
            virtual void calc_density_matrix_with_symmetry();
            virtual void calc_scf_energy();
            virtual void calc_scf_energy_with_symmetry();
            virtual void get_sad_guess(bool print = true);
            virtual void so_scf();
            virtual void so_scf_symmetry();

            EigenMatrix<double> solve_H_rot(const Eigen::Ref<const EigenMatrix<double>>& rot, 
                                            const Eigen::Ref<const EigenMatrix<double>>& Fmo) const;

            std::vector<EigenMatrix<double>> solve_H_rot_sym(const std::vector<EigenMatrix<double>>& rot, 
                                                             const std::vector<EigenMatrix<double>>& Fmo) const;
    };
}

#endif
