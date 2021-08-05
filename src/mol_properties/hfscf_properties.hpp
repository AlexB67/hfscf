#ifndef HFSCF_MOLPROPS_H
#define HFSCF_MOLPROPS_H

#include "../molecule/hfscf_molecule.hpp"
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>

template <typename T>
using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
using EigenVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

using Eigen::Index;

namespace MolProps
{
    class Molprops
    {
        public:
            explicit Molprops(const std::shared_ptr<MOLEC::Molecule>& mol)
            : m_mol(mol)
            {
                num_orbitals = m_mol->get_num_orbitals();
                occ = m_mol->get_num_electrons() / 2;
                virt = m_mol->get_num_orbitals() - occ;
            }
            
            Molprops(const Molprops&) = delete;
            Molprops& operator=(const Molprops& other) = delete;
            Molprops(const Molprops&&) = delete;
            Molprops&& operator=(const Molprops&& other) = delete;
            
            //! Calculate dipole matrices, X, Y, Z.
            //! include_nuc_contrib: Whether to include the nuclear contribution. Default is true
            void create_dipole_matrix(bool include_nuc_contrib = true);
            
            //! Calculate dipole vectors, X, Y, Z. RHF only.
            //! d_mat: A const reference to the density matrix.
            void create_dipole_vectors_rhf(const Eigen::Ref<const EigenMatrix<double> >& d_mat);

            //! Calculate dipole vectors, X, Y, Z. UHF only.
            //! d_mat_a: A const reference to the alpha density matrix.
            //! d_mat_b: A const reference to beta alpha density matrix.
            void create_dipole_vectors_uhf(const Eigen::Ref<const EigenMatrix<double> >& d_mat_a,
                                           const Eigen::Ref<const EigenMatrix<double> >& d_mat_b);

            //! Calculate the electronic Hessian 
            //! G_aibj = kron_del_ij * kron_del_ab * eps_ij * eps_ab + 4 <ij|ab> - <ij|ba> - <ia|jb>

            //! C  : A const reference to MO coefficients.
            //! eps: A const reference to MO energies as a diagonal 2D matrix.
            //! eri: A const reference to the 2 electron integrals.
            //! G  : A reference to the Matrix, will be popluted with tHe electronic Hessian.  
            void create_hessian_matrix(const Eigen::Ref<const EigenMatrix<double> >& C,
                                       const Eigen::Ref<const EigenMatrix<double> >& eps,
                                       const Eigen::Ref<const EigenVector<double> >& eri, 
                                       EigenMatrix<double>& G);
            
            //! Calculate static polarizabilties via CPHF using direct inversion - RHF only
            //! C  : A const reference to MO coefficients.
            //! eps: A const reference to MO energies as a diagonal 2D matrix.
            //! eri: A const reference to the 2 electron integrals.
            void calc_static_polarizabilities(const Eigen::Ref<const EigenMatrix<double> >& C,
                                              const Eigen::Ref<const EigenMatrix<double> >& eps,
                                              const Eigen::Ref<const EigenVector<double> >& eri);
            
            //! Calculate static polarizabilties using iterative CPHF - RHF only
            //! C  : A const reference to MO coefficients.
            //! eps: A const reference to MO energies as a diagonal 2D matrix.
            //! eri: A const reference to the 2 electron integrals.
            void calc_static_polarizabilities_iterative(const Eigen::Ref<const EigenMatrix<double> >& C,
                                                        const Eigen::Ref<const EigenMatrix<double> >& eps,
                                                        const Eigen::Ref<const EigenVector<double> >& eri);

            //! Population analysis - RHF only
            //! s_mat: A const reference to the overlap matrix.
            //! d_mat: A const reference to the density matrix.
            void population_analysis_rhf(const Eigen::Ref<const EigenMatrix<double> >& s_mat,
                                         const Eigen::Ref<const EigenMatrix<double> >& d_mat);

            //! Population analysis - UHF only
            //! s_mat: A const reference to the overlap matrix.
            //! d_mat_a: A const reference to the alpha density matrix.
            //! d_mat_b: A const reference to the beta density matrix.
            void population_analysis_uhf(const Eigen::Ref<const EigenMatrix<double> >& s_mat,
                                         const Eigen::Ref<const EigenMatrix<double> >& d_mat_a,
                                         const Eigen::Ref<const EigenMatrix<double> >& d_mat_b);
            
            //! Compute and print quadrupole moments - RHF only
            //! d_mat: A const reference to the density matrix.
            //! print: Wheher to print out quadrupole information.
            void create_quadrupole_tensors_rhf(const Eigen::Ref<const EigenMatrix<double> >& d_mat, bool print = true);

            //! Compute and print quadrupole moments - RHF only
            //! d_mat_a: A const reference to the alpha density matrix.
            //! d_mat_b: A const reference to the beta density matrix.
            //! print: Wheher to print out quadrupole information.
            void create_quadrupole_tensors_uhf(const Eigen::Ref<const EigenMatrix<double> >& d_mat_a,
                                               const Eigen::Ref<const EigenMatrix<double> >& d_mat_b,
                                               bool print = true);
            
            //! Print dipole information - RHF and UHF
            void print_dipoles() const;

            //! Print quadrupole information - RHF and UHF
            void print_quadrupoles() const;

            //! Print population analysis information - RHF and UHF
            void print_population_analysis() const;

            //! get a reference to the X component dipole matrix
            const Eigen::Ref<const EigenMatrix<double>> get_dipole_x() const {return mu_x;}

            //! get a reference to the Y component dipole matrix
            const Eigen::Ref<const EigenMatrix<double>> get_dipole_y() const {return mu_y;}

            //! get a reference to the Z component dipole matrix
            const Eigen::Ref<const EigenMatrix<double>> get_dipole_z() const {return mu_z;}

         private:
            std::shared_ptr<MOLEC::Molecule> m_mol;
            Index occ{0};
            Index virt{0};
            Index num_orbitals{0};
            // DIIS
            std::vector<EigenMatrix<double>> U1_list;
            std::vector<EigenMatrix<double>> U2_list;
            std::vector<EigenMatrix<double>> U3_list;
            std::vector<EigenMatrix<double>> del_U1_list;
            std::vector<EigenMatrix<double>> del_U2_list;
            std::vector<EigenMatrix<double>> del_U3_list;
            //
            EigenMatrix<double> mu_x;
            EigenMatrix<double> mu_y;
            EigenMatrix<double> mu_z;
            EigenVector<double> mul_charge;     // Mulliken charges
            EigenVector<double> low_charge;     // Lowdin charges
            Vec3D mu_cart;                      // Holds the nuclear cartesian dipole components
            EigenVector<double> quadp_moments;  // Quadrupole components

            void cphf_diis(EigenMatrix<double>& U_1, EigenMatrix<double>& U_2, EigenMatrix<double>& U_3,
                           const Eigen::Ref<const EigenMatrix<double> >& del_U1,
                           const Eigen::Ref<const EigenMatrix<double> >& del_U2,
                           const Eigen::Ref<const EigenMatrix<double> >& del_U3);
            
            EigenVector<double> get_nuclear_quadrupole_contribution() const;
    }; 
}

#endif
