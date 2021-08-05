#ifndef HFSCF_SOLVER_UHF_H
#define HFSCF_SOLVER_UHF_H

#include "hfscf_hf_solver.hpp"

namespace Mol
{
    class uhf_solver : protected Mol::hf_solver
    {
        public:
            explicit uhf_solver(const std::shared_ptr<MOLEC::Molecule>& mol) : hf_solver(mol){}
            virtual ~uhf_solver(){}
            uhf_solver(const uhf_solver&) = delete;
            uhf_solver& operator=(const uhf_solver& other) = delete;
            uhf_solver(const uhf_solver&&) = delete;
            uhf_solver&& operator=(const uhf_solver&& other) = delete;

            void get_data() override;
            void init_data(bool print = true, bool use_previous_density = false) override;
            bool scf_run(bool print = true) override;
            
            double get_spin_contamination();
            int get_iterations() const { return iteration;}
            double get_scf_energy() const { return scf_energy_current;}

            void print_fock_matrix() const override;
            void print_mo_coefficients_matrix() const override;
            void print_density_matrix() const override;
            void print_mo_energies() const override;
            void print_scf_energies() const override { hf_solver::print_scf_energies();};
            double get_one_electron_energy() const;

            const Eigen::Ref<const EigenMatrix<double>> get_mo_coef_alpha() const {return mo_coff;}
            const Eigen::Ref<const EigenMatrix<double>> get_mo_coef_beta() const {return mo_coff_b;}
            const Eigen::Ref<const EigenMatrix<double>> get_fock_matrix_alpha() const {return f_mat;}
            const Eigen::Ref<const EigenMatrix<double>> get_fock_matrix_beta() const {return f_mat_b;}
            const Eigen::Ref<const EigenMatrix<double>> get_mo_energies_alpha() const {return mo_energies;}
            const Eigen::Ref<const EigenMatrix<double>> get_mo_energies_beta() const {return mo_energies_beta;}
            const Eigen::Ref<const EigenVector<double>> get_repulsion_vector() const { return e_rep_mat;}
            const Eigen::Ref<const EigenMatrix<double>> get_density_matrix_alpha() const {return d_mat;}
            const Eigen::Ref<const EigenMatrix<double>> get_density_matrix_beta() const {return d_mat_b;}
            const Eigen::Ref<const EigenMatrix<double>> get_overlap_matrix() const {return s_mat;}
        
        private:
            Eigen::Index nalpha{0};  // uhf alpha electrons size
            Eigen::Index nbeta{0};  // uhf beta  electrons size
            std::vector<int> pop_index_b;

            EigenMatrix<double> mo_energies_beta;
            EigenMatrix<double> f_mat_b;            // F
            EigenMatrix<double> fp_mat_b;           // F' Note:  F' = s_mat_sqrt.T * F * s_mat_sqrt
            EigenMatrix<double> mo_coff_b;          // MO coefficients
            EigenMatrix<double> d_mat_b;            // beta density matrix
            EigenMatrix<double> d_mat_previous_b;   // density matrix store for the previous iteration
            
            // symmetry blocks 
            std::vector<EigenMatrix<double>> vf_mat_b;            // F
            std::vector<EigenMatrix<double>> vfp_mat_b;           // F' Note:  F' = s_mat_sqrt.T * F * s_mat_sqrt
            std::vector<EigenMatrix<double>> vmo_coff_b;          // MO coefficients
            std::vector<EigenMatrix<double>> vd_mat_b;            // beta density matrix
            std::vector<EigenMatrix<double>> vd_mat_previous_b;   // density matrix store for the previous iteration
            std::vector<EigenMatrix<double>> vmo_energies_beta;
            std::vector<std::pair<double, std::string>> mo_energies_sym_b;
            
            void update_fock_matrix() override;
            void update_fock_matrix_with_symmetry() override;
            void update_fock_matrix_scf_direct() override;
            void calc_scf_energy() override;
            void calc_scf_energy_with_symmetry() override;
            void calc_density_matrix() override;
            void calc_density_matrix_with_symmetry() override;
            void get_sad_guess(bool print = true) override;
            void so_scf() override;
            void so_scf_symmetry() override;

            void solve_H_rot(const Eigen::Ref<const EigenMatrix<double>>& rot_a, 
                             const Eigen::Ref<const EigenMatrix<double>>& rot_b, 
                             const Eigen::Ref<const EigenMatrix<double>>& Fmo_a,
                             const Eigen::Ref<const EigenMatrix<double>>& Fmo_b, 
                             EigenMatrix<double>& Fa, EigenMatrix<double>& Fb) const;
            
            void solve_H_rot_sym(const std::vector<EigenMatrix<double>>& rot_a, 
                                 const std::vector<EigenMatrix<double>>& rot_b, 
                                 const std::vector<EigenMatrix<double>>& Fmo_a,
                                 const std::vector<EigenMatrix<double>>& Fmo_b, 
                                 std::vector<EigenMatrix<double>>& Fa, 
                                 std::vector<EigenMatrix<double>>& Fb) const;
    };
}

#endif
