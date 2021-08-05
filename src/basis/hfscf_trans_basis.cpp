#include "hfscf_trans_basis.hpp"

using tensor4dmath::index2;
using tensor4dmath::index4;
using Eigen::Index;

void BTRANS::basis_transform::ao_to_mo_transform_2e(const Eigen::Ref<const EigenMatrix<double> >& mo_coff,
                                                    const Eigen::Ref<const EigenVector<double> >& e_rep_mat,
                                                    EigenVector<double>& e_rep_mo) 
{
    // Reduced to O^5 from the O^8 Noddy algorithm using temp arrays
    EigenMatrix<double> x_mat = EigenMatrix<double>::Zero(m_num_orbitals, m_num_orbitals);
    EigenMatrix<double> z_mat = EigenMatrix<double>::Zero((m_num_orbitals * (m_num_orbitals + 1) / 2),
                                                           m_num_orbitals * (m_num_orbitals + 1) / 2);

    for (Index i = 0, ij = 0; i < m_num_orbitals; ++i)
        for (Index j = 0; j <= i; ++j, ++ij) 
        {
            for (Index k = 0, kl = 0; k < m_num_orbitals; k++)
                for (Index l = 0; l <= k; ++l, ++kl) 
                {
                    Index ijkl = index2(ij, kl);
                    x_mat(k, l) = x_mat(l, k) = e_rep_mat(ijkl);
                }

            x_mat = mo_coff.transpose() * x_mat.eval() * mo_coff;

            for (Index k = 0, kl = 0; k < m_num_orbitals; ++k)
                for (Index l = 0; l <= k; ++l, ++kl) z_mat(kl, ij) = x_mat(k, l);
        }

    for (Index k = 0, kl = 0; k < m_num_orbitals; ++k)
        for (Index l = 0; l <= k; ++l, ++kl) 
        {
            x_mat.setZero();

            for (Index i = 0, ij = 0; i < m_num_orbitals; i++)
                for (Index j = 0; j <= i; ++j, ++ij) x_mat(i, j) = x_mat(j, i) = z_mat(kl, ij);

            x_mat = mo_coff.transpose() * x_mat.eval() * mo_coff;

            for (Index i = 0, ij = 0; i < m_num_orbitals; i++)
                for (Index j = 0; j <= i; ++j, ++ij) 
                {
                    Index klij = index4(k, l, i, j);
                    e_rep_mo(klij) = x_mat(i, j);
                }
        }
}

void BTRANS::basis_transform::ao_to_mo_transform_2e(const Eigen::Ref<const EigenMatrix<double> >& mo_coff_a,
                                                    const Eigen::Ref<const EigenMatrix<double> >& mo_coff_b,
                                                    const Eigen::Ref<const EigenVector<double> >& e_rep_mat, 
                                                    tensor4d<double>& e_rep_mo)
{
    // Reduced to O^5 from the O^8 Noddy algorithm using temp tensors

    e_rep_mo.setDim(m_num_orbitals);

    tensor4d<double> tmp1 = tensor4d<double>(m_num_orbitals);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index i = 0; i < m_num_orbitals; ++i) 
        for (Index j = 0; j < m_num_orbitals; ++j)
            for (Index k = 0; k < m_num_orbitals; ++k) 
                for (Index l = k; l < m_num_orbitals; ++l)
                    for (Index p = 0; p < m_num_orbitals; ++p)
                    {
                        const Index pjkl = index4(p, j, k, l);
                        tmp1(i, j, k, l) += e_rep_mat[pjkl] * mo_coff_a(p, i);
                        tmp1(i, j, l, k) = tmp1(i, j, k, l);
                    }

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
     for (Index i = 0; i < m_num_orbitals; ++i) 
        for (Index j = 0; j < m_num_orbitals; ++j)
            for (Index k = 0; k < m_num_orbitals; ++k) 
                for (Index l = k; l < m_num_orbitals; ++l) 
                    for (Index q = 0; q < m_num_orbitals; ++q) 
                    {
                        e_rep_mo(i, j, k, l) += tmp1(i, q, k, l) * mo_coff_a(q, j);
                        e_rep_mo(i, j, l, k) = e_rep_mo(i, j, k, l);
                    }
    
    tmp1.setZero();

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
     for (Index i = 0; i < m_num_orbitals; ++i) 
        for (Index j = 0; j < m_num_orbitals; ++j)
            for (Index k = 0; k < m_num_orbitals; ++k) 
                for (Index l = 0; l < m_num_orbitals; ++l) 
                    for (Index r = 0; r < m_num_orbitals; ++r) 
                        tmp1(i, j, k, l) += e_rep_mo(i, j, r, l) * mo_coff_b(r, k);

    e_rep_mo.setZero();

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
     for (Index i = 0; i < m_num_orbitals; ++i) 
        for (Index j = 0; j < m_num_orbitals; ++j)
            for (Index k = 0; k < m_num_orbitals; ++k) 
                for (Index l = 0; l < m_num_orbitals; ++l) 
                    for (Index s = 0; s < m_num_orbitals; ++s) 
                        e_rep_mo(i, j, k, l) += tmp1(i, j, k, s) * mo_coff_b(s, l);
}

void BTRANS::basis_transform::ao_to_mo_transform_2e(const Eigen::Ref<const EigenMatrix<double> >& mo_coff_a,
                                                    const Eigen::Ref<const EigenMatrix<double> >& mo_coff_b,
                                                    const tensor4d<double>& e_rep_mat, tensor4d<double>& e_rep_mo)
{
    // Reduced to O^5 from the O^8 Noddy algorithm using temp tensors

    e_rep_mo = tensor4d<double>(m_num_orbitals);

    tensor4d<double> tmp1 = tensor4d<double>(m_num_orbitals);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index i = 0; i < m_num_orbitals; ++i) 
        for (Index j = 0; j < m_num_orbitals; ++j)
            for (Index k = 0; k < m_num_orbitals; ++k) 
                for (Index l = 0; l < m_num_orbitals; ++l)
                    for (Index p = 0; p < m_num_orbitals; ++p) 
                        tmp1(i, j, k, l) += e_rep_mat(p, j, k, l) * mo_coff_a(p, i);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
     for (Index i = 0; i < m_num_orbitals; ++i) 
        for (Index j = 0; j < m_num_orbitals; ++j)
            for (Index k = 0; k < m_num_orbitals; ++k) 
                for (Index l = 0; l < m_num_orbitals; ++l) 
                    for (Index q = 0; q < m_num_orbitals; ++q) 
                        e_rep_mo(i, j, k, l) += tmp1(i, q, k, l) * mo_coff_a(q, j);
    
    tmp1.setZero();

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
     for (Index i = 0; i < m_num_orbitals; ++i) 
        for (Index j = 0; j < m_num_orbitals; ++j)
            for (Index k = 0; k < m_num_orbitals; ++k) 
                for (Index l = 0; l < m_num_orbitals; ++l) 
                    for (Index r = 0; r < m_num_orbitals; ++r) 
                        tmp1(i, j, k, l) += e_rep_mo(i, j, r, l) * mo_coff_b(r, k);

    e_rep_mo.setZero();

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
     for (Index i = 0; i < m_num_orbitals; ++i) 
        for (Index j = 0; j < m_num_orbitals; ++j)
            for (Index k = 0; k < m_num_orbitals; ++k) 
                for (Index l = 0; l < m_num_orbitals; ++l) 
                    for (Index s = 0; s < m_num_orbitals; ++s) 
                        e_rep_mo(i, j, k, l) += tmp1(i, j, k, s) * mo_coff_b(s, l);
}
                                
void BTRANS::basis_transform::mo_to_spin_transform_2e(const Eigen::Ref<const EigenVector<double> >& e_rep_mo, 
                                                      tensor4d<double>& e_rep_spin_mo)
{
   // <ij||kl> in physicist format
   e_rep_spin_mo = tensor4d<double>(spin_mos);

   #ifdef _OPENMP
   #pragma omp parallel for schedule(dynamic)
   #endif
   for(Index p = 0; p < spin_mos; ++p)
      for(Index q = 0; q <= p; ++q)
         for(Index r = 0; r < spin_mos; ++r)
            for(Index s = 0; s <= r; ++s) 
            {
               double J = 0.0;
               double K = 0.0;

               if ((p % 2  == r % 2) && (q % 2 == s % 2))
               {
                  Index prqs = index4( p / 2, r / 2, q / 2, s / 2);
                  J = e_rep_mo[prqs];
               }

               if ((p % 2 == s % 2) && (q % 2 == r % 2))
               {
                  Index psqr = index4(p / 2, s / 2, q / 2, r / 2);
                  K = e_rep_mo[psqr];
               }

               e_rep_spin_mo(p, q, r, s) = J - K;
               e_rep_spin_mo(q, p, s, r) = J - K;
               e_rep_spin_mo(p, q, s, r) = K - J;
               e_rep_spin_mo(q, p, r, s) = K - J;
            }
}

// Method 2 using symmetry of the 2 electron spin integrals to save a lot of space but slower runtime accessing elements
// used in ccsd to save memory where cost is small due to dominance of other factors
void BTRANS::basis_transform::mo_to_spin_transform_2e(const Eigen::Ref<const EigenVector<double> >& e_rep_mo, 
                                                      symm4dTensor<double>& e_rep_spin_mo)
{
   // <ij||kl> in physicist format 
   // just like the 2e electron integrals. We access elements via the asymm call to satisfy the assymetric property
   // of the integrals. See symmTensor4d for impl
   
   e_rep_spin_mo = symm4dTensor<double>(spin_mos); 

   #ifdef _OPENMP
   #pragma omp parallel for schedule(dynamic)
   #endif
   for(Index p = 0; p < spin_mos; ++p)
      for(Index q = 0; q <= p; ++q)
         for(Index r = 0; r < spin_mos; ++r)
            for(Index s = 0; s <= r; ++s) 
            {
               double J = 0.0;
               double K = 0.0;

               if ((p % 2  == r % 2) && (q % 2 == s % 2))
               {
                  Index prqs = index4( p / 2, r / 2, q / 2, s / 2);
                  J = e_rep_mo[prqs];
               }

               if ((p % 2 == s % 2) && (q % 2 == r % 2))
               {
                  Index psqr = index4(p / 2, s / 2, q / 2, r / 2);
                  K = e_rep_mo[psqr];
               }

               Index pq = index2(p, q);
               Index rs = index2(r, s);

               if(pq <= rs )
               {
                  e_rep_spin_mo(p, q, r, s) = J - K;
               }
            }
}

// Note: 
// 1st 4 sym terms

// e_rep_spin_mo(p, q, r, s) = J - K;
// e_rep_spin_mo(q, p, s, r) = J - K;
// e_rep_spin_mo(r, s, p, q) = J - K;
// e_rep_spin_mo(s, r, q, p) = J - K;

// 2nd 4 symm terms

// e_rep_spin_mo(p, q, s, r) = K - J;
// e_rep_spin_mo(q, p, r, s) = K - J;
// e_rep_spin_mo(s, r, p, q) = K - J;
// e_rep_spin_mo(r, s, q, p) = K - J;

// see symmTensor4d for impl
