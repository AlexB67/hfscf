#ifndef HFSCF_SAD_H
#define HFSCF_SAD_H

#include "../molecule/hfscf_molecule.hpp"
#include "hfscf_diis.hpp"
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <string>

template <typename T>
using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
using EigenVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

namespace Mol
{
    class sad_uhf_solver
    {
        public:
            explicit sad_uhf_solver(const std::shared_ptr<MOLEC::Molecule>& mol,
                                    const Eigen::Ref<const EigenVector<double>>& mol_eri,
                                    const Eigen::Ref<const EigenMatrix<double>>& mol_ovlap,
                                    const Eigen::Ref<const EigenMatrix<double>>& mol_kinetic,
                                    int atom_num)
                                    : molecule(mol), 
                                      moleri(mol_eri), 
                                      molovlap(mol_ovlap),
                                      molkin(mol_kinetic),
                                      atom(atom_num)
                                    {}

            virtual ~sad_uhf_solver(){}
            sad_uhf_solver(const sad_uhf_solver&) = delete;
            sad_uhf_solver& operator=(const sad_uhf_solver& other) = delete;
            sad_uhf_solver(const sad_uhf_solver&&) = delete;
            sad_uhf_solver&& operator=(const sad_uhf_solver&& other) = delete;

            void init_data(bool print = true);
            bool sad_run();
            const Eigen::Ref<const EigenMatrix<double>> get_atom_density() const { return 0.5 * (d_mat + d_mat_b);}
            double get_scf_energy() const { return scf_energy_current;}
            const Eigen::Ref<const EigenMatrix<double>> get_mo_coef() const {return mo_coff;}

        protected:
            std::shared_ptr<MOLEC::Molecule> molecule;
            EigenVector<double> moleri;
            EigenMatrix<double> molovlap;
            EigenMatrix<double> molkin;
            int atom;
            
            Eigen::Index norbitals{0};
            Eigen::Index nelectrons{0};
            Eigen::Index bf_offset{0};
            Eigen::Index occ{0};
            Eigen::Index nfrozen{0}; 
            Eigen::Index nactive{0};
            int iteration{0};
            bool diis_log{false};
            bool print_sad{false};
            double scf_energy_previous{0};
            double scf_energy_current{std::numeric_limits<double>::max()};
            std::unique_ptr<DIIS_SOLVER::diis_solver> diis_ptr;
            std::string atom_name{};

            std::vector<double> scf_energies;
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
            EigenMatrix<double> f_mat_b;            // F
            EigenMatrix<double> fp_mat_b;           // F' Note:  F' = s_mat_sqrt.T * F * s_mat_sqrt
            EigenMatrix<double> mo_coff_b;          // MO coefficients
            EigenMatrix<double> d_mat_b;            // beta density matrix
            EigenMatrix<double> d_mat_previous_b;   // density matrix store for the previous iteration
            EigenVector<double> pop_index;

            void create_one_electron_hamiltonian();
            void allocate_memory();
            void create_repulsion_matrix();
            void calc_initial_fock_matrix();
            void update_fock_matrix();
            void calc_density_matrix();
            void calc_scf_energy();
            Index get_pop_numbers();
            void calc_potential_matrix();
    };
}

#endif
