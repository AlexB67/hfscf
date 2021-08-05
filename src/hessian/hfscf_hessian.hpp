#ifndef SCF_HESSIAN_H
#define SCF_HESSIAN_H

#include "../molecule/hfscf_molecule.hpp"
#include "../math/hfscf_tensor4d.hpp"
#include <memory>

template <typename T>
using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

using tensor4dmath::tensor4d;

namespace Mol
{
    class scf_hessian
    {
       public:
            explicit scf_hessian(const std::shared_ptr<MOLEC::Molecule>& molecule, 
                                 const Eigen::Ref<const EigenMatrix<double> >& dens_mat,
                                 const Eigen::Ref<const EigenMatrix<double> >& mo_coeff,
                                 const Eigen::Ref<const EigenMatrix<double> >& mo_eps,
                                 const Eigen::Ref<const EigenMatrix<double> >& ovlap,
                                 const Eigen::Ref<const EigenVector<double> >& eri_vec,
                                 const Eigen::Ref<const EigenMatrix<double> >& fock_mat)
           : m_mol(molecule),
             d_mat(dens_mat),
             mo_coff(mo_coeff),
             mo_energies(mo_eps),
             s_mat(ovlap),
             eri(eri_vec) 
            { // weighted Q matrix same as eps * C * C.T sum over elec in Ostlund
                Q_mat =  d_mat * fock_mat * d_mat;
            }
            
            ~scf_hessian() = default;
            scf_hessian(const scf_hessian&) = delete;
            scf_hessian& operator=(const scf_hessian& other) = delete;
            scf_hessian(const scf_hessian&&) = delete;
            scf_hessian&& operator=(const scf_hessian&& other) = delete;

            void calc_scf_hessian_rhf(const double E_electronic);
            void calc_scf_hessian_rhf_mp2(const double E_electronic);
            
            const Eigen::Ref<const EigenMatrix<double> > get_hessian_ref() const {return m_hessian;}

        private:
            std::shared_ptr<MOLEC::Molecule> m_mol;
            const Eigen::Ref<const EigenMatrix<double> > d_mat;
            const Eigen::Ref<const EigenMatrix<double> > mo_coff;
            const Eigen::Ref<const EigenMatrix<double> > mo_energies;
            const Eigen::Ref<const EigenMatrix<double> > s_mat;
            const Eigen::Ref<const EigenVector<double> > eri;
            EigenMatrix<double> Q_mat;
            EigenMatrix<double> m_hessian;
            EigenMatrix<double> m_hessianNuc;
            EigenMatrix<double> m_hessianOvlap;
            EigenMatrix<double> m_hessianKin;
            EigenMatrix<double> m_hessianVn;
            EigenMatrix<double> m_hessianC;
            EigenMatrix<double> m_hessianEx;
            EigenMatrix<double> m_hessianResp;
            EigenMatrix<double> I;
            EigenMatrix<double> Ppq;
            tensor4d<double> Ppqrs;
            std::vector<std::vector<bool>> mask; // cartesian mask
            std::vector<std::vector<bool>> smask; // spherical mask

            void calc_scf_hessian_overlap();      
            void calc_scf_hessian_kinetic();
            void calc_scf_hessian_pot();
            void calc_scf_hessian_tei();
            void calc_scf_hessian_tei_mp2();      
            void calc_scf_hessian_nuclear();
    };
}

#endif // 