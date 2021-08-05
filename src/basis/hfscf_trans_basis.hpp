#ifndef TRANSFORM_BASIS_H
#define TRANSFORM_BASIS_H

#include "../math/hfscf_math.hpp"
#include "../math/hfscf_tensor4d.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>

template <typename T>
using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
using EigenVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

using tensor4dmath::tensor4d;
using tensor4dmath::symm4dTensor;

namespace BTRANS
{
    class basis_transform
    {
        public:
            explicit basis_transform(const Eigen::Index num_orbitals) : m_num_orbitals(num_orbitals)
            {
                spin_mos = 2 * m_num_orbitals;
            }

            ~basis_transform() = default;
            basis_transform(const basis_transform&) = delete;
            basis_transform& operator=(const basis_transform& other) = delete;
            basis_transform(const basis_transform&&) = delete;
            basis_transform&& operator=(const basis_transform&& other) = delete;

            void ao_to_mo_transform_2e(const Eigen::Ref<const EigenMatrix<double> >& mo_coff,
                                       const Eigen::Ref<const EigenVector<double> >& e_rep_mat, 
                                       EigenVector<double>& e_rep_mo);
            
            void ao_to_mo_transform_2e(const Eigen::Ref<const EigenMatrix<double> >& mo_coff_a,
                                       const Eigen::Ref<const EigenMatrix<double> >& mo_coff_b,
                                       const Eigen::Ref<const EigenVector<double> >& e_rep_mat, 
                                       tensor4d<double>& e_rep_mo);
            
            void ao_to_mo_transform_2e(const Eigen::Ref<const EigenMatrix<double> >& mo_coff_a,
                                       const Eigen::Ref<const EigenMatrix<double> >& mo_coff_b,
                                       const tensor4d<double>& e_rep_mat, tensor4d<double>& e_rep_mo);

            void mo_to_spin_transform_2e(const Eigen::Ref<const EigenVector<double> >& e_rep_mo, 
                                         tensor4d<double>& e_rep_spin_mo);
            
            void mo_to_spin_transform_2e(const Eigen::Ref<const EigenVector<double> >& e_rep_mo, 
                                         symm4dTensor<double>& e_rep_spin_mo);
            
        private:
            Eigen::Index m_num_orbitals;
            Eigen::Index spin_mos;
    };
}

#endif
