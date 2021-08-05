#ifndef SCF_GRADIENT_H
#define SCF_GRADIENT_H

#include "../molecule/hfscf_molecule.hpp"
#include "../math/hfscf_math.hpp"
#include "../math/hfscf_tensor4d.hpp"
#include "../math/hfscf_tensors.hpp"
#include <memory>

template <typename T>
using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

using hfscfmath::Cart;
using tensor4dmath::symm4dTensor;
using tensor4dmath::tensor4d;
using tensormath::tensor4d1234;
using tensor4dmath::tensor4d1122;
using tensormath::tensor5d;
using tensormath::tensor3d;

namespace Mol
{
    class scf_gradient
    {
       public:
            explicit scf_gradient(const std::shared_ptr<MOLEC::Molecule>& molecule)
            : m_mol(molecule){}
            ~scf_gradient() = default;
            scf_gradient(const scf_gradient&) = delete;
            scf_gradient& operator=(const scf_gradient& other) = delete;
            scf_gradient(const scf_gradient&&) = delete;
            scf_gradient&& operator=(const scf_gradient&& other) = delete;

            bool calc_scf_gradient_rhf(const Eigen::Ref<const EigenMatrix<double> >& d_mat,
                                       const Eigen::Ref<const EigenMatrix<double> >& f_mat,
                                       std::vector<bool>& coords, bool check_if_out_of_plane = true,
                                       bool geom_opt = false, bool print_gradient = true);
            
            bool calc_mp2_gradient_rhf(const Eigen::Ref<const EigenMatrix<double> >& mo_energies,
                                       const Eigen::Ref<const EigenMatrix<double> >& s_mat,
                                       const Eigen::Ref<const EigenMatrix<double> >& d_mat,
                                       const Eigen::Ref<const EigenMatrix<double> >& mo_coff,
                                       const Eigen::Ref<const EigenVector<double> >& eri,
                                       std::vector<bool>& coords, bool check_if_out_of_plane = true,
                                       bool geom_opt = false, bool print_gradient = true);
            
            bool calc_scf_gradient_uhf(const Eigen::Ref<const EigenMatrix<double> >& d_mat_a,
                                       const Eigen::Ref<const EigenMatrix<double> >& d_mat_b,
                                       const Eigen::Ref<const EigenMatrix<double> >& f_mat_a,
                                       const Eigen::Ref<const EigenMatrix<double> >& f_mat_b,
                                       std::vector<bool>& out_of_plane, bool check_out_of_plane = true,
                                       bool geom_opt = false, bool print_gradient = true);
            
            void calc_scf_hessian_response_rhf(const Eigen::Ref<const EigenMatrix<double> >& mo_coff,
                                               const Eigen::Ref<const EigenMatrix<double> >& mo_energies,                                             
                                               const Eigen::Ref<const EigenVector<double> >& eri);
            
            //! Calculates the MP2 response hessian.
            void calc_scf_hessian_response_rhf_mp2(const Eigen::Ref<const EigenMatrix<double> >& mo_energies);

            //! Calculates dipole derivatives.
            //! U_occ_start is 0 if U is a virt x occ matrix, or occ if U is orbitals x orbitals matrix.
            void dip_derivs(const Eigen::Ref<const EigenMatrix<double> >& mo_coff, Index U_occ_start);
            
            //! Pint CPHF coefficients from the MP2/RHF response hessian.
            void print_cphf_coefficients(const Index dim1, const Index dim2);

            //! Pint dipole derivatives and gradient
            void print_dip_derivs();

            //! Get a copy of the gradient for all centers
            EigenMatrix<double> get_gradient() const {return gradient;}
            
            //! Get a reference to the gradient for all centers
            const Eigen::Ref<const EigenMatrix<double> > get_gradient_ref() const {return gradient;}

            //! Get a reference to the Ipq matrix.
            const Eigen::Ref<const EigenMatrix<double> > get_I_ref() const {return I;}
            
            //! Get a reference to the Ppq matrix.
            const Eigen::Ref<const EigenMatrix<double> > get_Ppq_ref() const {return Ppq;}
            
            const tensor4d<double>& get_Ppqrs_ref() const {return Ppqrs_;}
            
            //! Get a reference to the response hessian
            const Eigen::Ref<const EigenMatrix<double> > get_response_rhessian_ref() const {return hessianResp;}

            //! Get a reference to the first derivative dipole matrix (dmu/da)
            const Eigen::Ref<const EigenMatrix<double> > get_dipderiv_ref() const {return dipderiv;}

            //! Get a reference to the dipole gradient matrix
            const Eigen::Ref<const EigenMatrix<double> > get_dipgrad_ref() const {return dipgrad;}

        private:
            std::shared_ptr<MOLEC::Molecule> m_mol;
            EigenMatrix<double> gradient_previous;
            EigenMatrix<double> gradient;
            EigenVector<double> irc_grad_previous;
            EigenVector<double> irc_grad;
            EigenVector<double> irc_coords;
            EigenMatrix<double> hessianResp;
            EigenMatrix<double> irc_hessian;
            EigenVector<double> e_rep_mo;
            EigenMatrix<double> I;
            EigenMatrix<double> Ppq;
            tensor3d<double> Bpq;
            tensor3d<double> U;
            tensor3d<double> dSmo;
            tensor3d<double> Fpq;
            tensor3d<double> dPpq;
            tensor4d<double> Ppqrs_;
            tensor4d1122<double> t2;
            tensor4d1122<double> t2_tilde;
            std::vector<symm4dTensor<double>> dVdXmo_list;
            std::vector<tensor4d<double>> dPpqrs_list;
            EigenMatrix<double> dipderiv;
            EigenMatrix<double> dipgrad;
            
            bool check_geom_opt(const std::vector<int>& zval, int natoms);

            void calc_overlap_kinetic_gradient(EigenMatrix<double>& dSdX, EigenMatrix<double>& dTdX,
                                               const tensor3d<double>& dSdq,
                                               const tensor3d<double>& dTdq,
                                               const Cart coord, const EigenVector<bool>& mask) const;
            
            void calc_overlap_centers(tensor3d<double>& dSdq) const;
            void calc_kinetic_centers(tensor3d<double>& dTdq) const;
            
            void calc_coulomb_exchange_integrals(symm4dTensor<double>& dVdX,
                                                 symm4dTensor<double>& dVdXa,
                                                 symm4dTensor<double>& dVdXb,
                                                 symm4dTensor<double>& dVdXc,
                                                 const EigenVector<bool>& mask);
            
            void calc_coulomb_exchange_integral_centers(symm4dTensor<double>& dVdXa,
                                                        symm4dTensor<double>& dVdXb,
                                                        symm4dTensor<double>& dVdXc,
                                                        symm4dTensor<double>& dVdYa,
                                                        symm4dTensor<double>& dVdYb,
                                                        symm4dTensor<double>& dVdYc,
                                                        symm4dTensor<double>& dVdZa,
                                                        symm4dTensor<double>& dVdZb,
                                                        symm4dTensor<double>& dVdZc,
                                                        const std::vector<bool>& out_of_plane);
            
            void calc_nuclear_potential_centers(tensor3d<double>& dVdqa,
                                                tensor3d<double>& dVdqb, 
                                                const std::vector<bool>& out_of_plane) const;
            
            void calc_potential_gradient(const tensor3d<double>& dVdqa,
                                         const tensor3d<double>& dVdqb,
                                         EigenMatrix<double>& dVndX, const EigenVector<bool>& mask,
                                         const int atom, const int cart_dir) const;
        
            void print_gradient_info(const Eigen::Ref<EigenMatrix<double> >& gradNuc, 
                                     const Eigen::Ref<EigenMatrix<double> >& gradCoreHamil,
                                     const Eigen::Ref<EigenMatrix<double> >& gradOvlap,
                                     const Eigen::Ref<EigenMatrix<double> >& gradCoulomb,
                                     const Eigen::Ref<EigenMatrix<double> >& gradExchange,
                                     const Eigen::Ref<EigenMatrix<double> >& gradient) const;
            
            void print_gradient_info_mp2(const Eigen::Ref<EigenMatrix<double> >& gradNuc, 
                                         const Eigen::Ref<EigenMatrix<double> >& gradCoreHamil,
                                         const Eigen::Ref<EigenMatrix<double> >& gradOvlap,
                                         const Eigen::Ref<EigenMatrix<double> >& gradTEI,
                                         const Eigen::Ref<EigenMatrix<double> >& gradient) const;
            
            void hessian_resp_terms(const Eigen::Ref<const EigenMatrix<double> >& mo_coff,
                                    const Eigen::Ref<const EigenMatrix<double> >& dTdX,
                                    const Eigen::Ref<const EigenMatrix<double> >& dVndX,
                                    const Eigen::Ref<const EigenMatrix<double> >& dSdX,
                                    const Eigen::Ref<const EigenMatrix<double> >& mo_energies,
                                    const Eigen::Ref<const EigenMatrix<double> >& Ginv,
                                    const Eigen::Ref<const EigenVector<double> >& dVdXmo,
                                    int atom, int cart_dir);
            
            EigenVector<bool> get_atom_mask(const int atom) const;
    };

    constexpr double geom_opt_tol_very_high = 1.0E-07;
    constexpr double geom_opt_tol_high = 1.0E-06;
    constexpr double geom_opt_tol_med =  0.5E-04;
    constexpr double geom_opt_tol_low =  1.0E-03;

}

#endif // 