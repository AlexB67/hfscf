#include "../settings/hfscf_settings.hpp"
#include "../basis/hfscf_trans_basis.hpp"
#include "hfscf_post_mp.hpp"
#include <memory>
#include <algorithm>

POSTSCF::post_scf_mp::post_scf_mp(Eigen::Index num_orbitals, Eigen::Index nelectrons, Eigen::Index fcore)
: m_num_orbitals(num_orbitals),
  m_nelectrons(nelectrons),
  m_fcore(fcore)
{
    m_spin_mos = 2 * m_num_orbitals;

    if (!m_fcore % 2 && m_fcore > 0) 
        m_fcore = 0;
   //  in case an incorrect odd number is passed set to zero .. must be even

    if(!HF_SETTINGS::hf_settings::get_freeze_core()) m_fcore = 0;
}
using tensor4dmath::index4;

double POSTSCF::post_scf_mp::calc_mp2_energy(const Eigen::Ref<const EigenMatrix<double> >& mo_energies,
                                             const Eigen::Ref<const EigenMatrix<double> >& mo_coff,
                                             const Eigen::Ref<const EigenVector<double> >& e_rep_mat) 
{
    std::unique_ptr<BTRANS::basis_transform> ao_to_mo_ptr = std::make_unique<BTRANS::basis_transform>(m_num_orbitals);

    // translate AOs to MOs
    e_rep_mo = EigenVector<double>::Zero((m_num_orbitals * (m_num_orbitals + 1) / 2) *
                                        ((m_num_orbitals * (m_num_orbitals + 1) / 2) + 1) / 2);
    ao_to_mo_ptr->ao_to_mo_transform_2e(mo_coff, e_rep_mat, e_rep_mo);

    double energy_mp2 = 0.0;
    #ifdef _OPENMP
    #pragma omp parallel for reduction ( + : energy_mp2)
    #endif
    for (Index i = m_fcore / 2; i < m_nelectrons / 2; ++i)
        for (Index a = m_nelectrons / 2; a < m_num_orbitals; ++a)
            for (Index j = m_fcore / 2; j < m_nelectrons / 2; ++j)
                for (Index b = m_nelectrons / 2; b < m_num_orbitals; ++b) 
                {
                    Index iajb = index4(i, a, j, b);
                    Index ibja = index4(i, b, j, a);
                    energy_mp2 += e_rep_mo(iajb) * (2 * e_rep_mo(iajb) - e_rep_mo(ibja)) /
                                  (mo_energies(i, i) + mo_energies(j, j) - mo_energies(a, a) - mo_energies(b, b));
                }
    
    return energy_mp2;
}

// Unrestricted post HF
Vec3D POSTSCF::post_scf_mp::calc_ump2_energy(const Eigen::Ref<const EigenMatrix<double> >& mo_energies_a,
                                             const Eigen::Ref<const EigenMatrix<double> >& mo_energies_b,
                                             const Eigen::Ref<const EigenMatrix<double> >& mo_coff_a,
                                             const Eigen::Ref<const EigenMatrix<double> >& mo_coff_b,   
                                             const Eigen::Ref<const EigenVector<double> >& e_rep_mat,
                                             const Index spin)
{

    // change to physics indexing
    const auto e_rep_spin = [this](Index p, Index q, Index r, Index s) -> double
    {
        Index prqs = index4(p, r, q, s);
        Index psqr = index4(p, s, q, r); 
        return e_rep_mo[prqs] - e_rep_mo[psqr];   
    };

    Index a_size =  static_cast<Index>(std::floor(0.5 * static_cast<double>(m_nelectrons + spin))); // spin 2S
	Index b_size = m_nelectrons - a_size;

    if(b_size > a_size) 
        std::swap(a_size, b_size);
    
    std::unique_ptr<BTRANS::basis_transform> ao_to_mo_ptr = std::make_unique<BTRANS::basis_transform>(m_num_orbitals);

    // translate AOs to MOs
    e_rep_mo = EigenVector<double>::Zero((m_num_orbitals * (m_num_orbitals + 1) / 2) *
                                        ((m_num_orbitals * (m_num_orbitals + 1) / 2) + 1) / 2);
    ao_to_mo_ptr->ao_to_mo_transform_2e(mo_coff_a, e_rep_mat, e_rep_mo);


    double uu_mp2 = 0;
    #ifdef _OPENMP
    #pragma omp parallel for reduction ( + : uu_mp2)
    #endif
    for (Index i = m_fcore / 2; i < a_size; ++i)
        for (Index j = m_fcore / 2; j < a_size; ++j)
            for (Index a = a_size; a < m_num_orbitals; ++a)
                for (Index b = a_size; b < m_num_orbitals; ++b)
                {
                    uu_mp2 += e_rep_spin(i, j, a, b) * e_rep_spin(i, j, a, b) 
                            / (mo_energies_a(i, i) + mo_energies_a(j, j) - mo_energies_a(a, a) - mo_energies_a(b, b));
                }

    uu_mp2 *= 0.25;

     // translate AOs to MOs
    ao_to_mo_ptr->ao_to_mo_transform_2e(mo_coff_b, e_rep_mat, e_rep_mo);

    double dd_mp2 = 0;
    #ifdef _OPENMP
    #pragma omp parallel for reduction ( + : dd_mp2)
    #endif
    for (Index i = m_fcore / 2; i < b_size; ++i)
        for (Index j = m_fcore / 2; j < b_size; ++j)
            for (Index a = b_size; a < m_num_orbitals; ++a)
                for (Index b = b_size; b < m_num_orbitals; ++b)
                {
                    dd_mp2 += e_rep_spin(i, j, a, b) * e_rep_spin(i, j, a, b) 
                            / (mo_energies_b(i, i) + mo_energies_b(j, j) - mo_energies_b(a, a) - mo_energies_b(b, b));
                }
    
    dd_mp2 *= 0.25;

    e_rep_mo.resize(0);

     //  translate AOs to MOs
    tensor4d<double> e_rep_mo2; // Full 4d tensor for this step due to lower symmetry 

    ao_to_mo_ptr->ao_to_mo_transform_2e(mo_coff_a, mo_coff_b, e_rep_mat, e_rep_mo2);

    double ud_mp2 = 0;
    #ifdef _OPENMP
    #pragma omp parallel for reduction ( + : ud_mp2)
    #endif
    for (Index i = m_fcore / 2; i < a_size; ++i)
        for (Index j = m_fcore / 2; j < b_size; ++j)
            for (Index a = a_size; a < m_num_orbitals; ++a)
                for (Index b = b_size; b < m_num_orbitals; ++b)
                {
                    ud_mp2 += e_rep_mo2(i, a, j, b) * e_rep_mo2(i, a, j, b) // Note chemist indexing
                            / (mo_energies_a(i, i) + mo_energies_b(j, j) - mo_energies_a(a, a) - mo_energies_b(b, b));
                }

    
    return Vec3D(uu_mp2, dd_mp2, ud_mp2);
}

EigenVector<double> 
POSTSCF::post_scf_mp::calc_ump3_energy(const Eigen::Ref<const EigenMatrix<double> >& mo_energies_a,
                                       const Eigen::Ref<const EigenMatrix<double> >& mo_energies_b,
                                       const Eigen::Ref<const EigenMatrix<double> >& mo_coff_a,
                                       const Eigen::Ref<const EigenMatrix<double> >& mo_coff_b,   
                                       const Eigen::Ref<const EigenVector<double> >& e_rep_mat,
                                       const Index spin)
{
    EigenVector<double> e_rep_mo2;
    tensor4d<double> e_rep_mo3; // Full 4d tensor due to lower symmetry for alpha beta terms

    // change to physics indexing throughout
    const auto e_rep_uu = [this](Index p, Index q, Index r, Index s) -> double
    {
        Index prqs = index4(p, r, q, s);
        Index psqr = index4(p, s, q, r); 
        return e_rep_mo[prqs] - e_rep_mo[psqr];   
    };

    const auto e_rep_dd = [&](Index p, Index q, Index r, Index s) -> double
    {
        Index prqs = index4(p, r, q, s);
        Index psqr = index4(p, s, q, r); 
        return e_rep_mo2[prqs] - e_rep_mo2[psqr];   
    };

    const auto e_rep_ud = [&](Index p, Index q, Index r, Index s) -> double
    {
        return e_rep_mo3(p, r, q, s);
    };

    Index a_size =  static_cast<Index>(std::floor(0.5 * static_cast<double>(m_nelectrons + spin)));
	Index b_size = m_nelectrons - a_size;

    if(b_size > a_size) 
        std::swap(a_size, b_size);

    EigenVector<double> eps_a = EigenVector<double>(m_num_orbitals);

    for(Index i = 0; i < m_num_orbitals; ++i)
        eps_a[i] = mo_energies_a(i, i);

    EigenVector<double> eps_b = EigenVector<double>(m_num_orbitals);

     for(Index i = 0; i < m_num_orbitals; ++i)
        eps_b[i] = mo_energies_b(i, i);

    
    std::unique_ptr<BTRANS::basis_transform> ao_to_mo_ptr = std::make_unique<BTRANS::basis_transform>(m_num_orbitals);

    // translate AOs to MOs
    e_rep_mo = EigenVector<double>::Zero((m_num_orbitals * (m_num_orbitals + 1) / 2) *
                                        ((m_num_orbitals * (m_num_orbitals + 1) / 2) + 1) / 2);
    ao_to_mo_ptr->ao_to_mo_transform_2e(mo_coff_a, e_rep_mat, e_rep_mo); // UU
    e_rep_mo2 = EigenVector<double>::Zero((m_num_orbitals * (m_num_orbitals + 1) / 2) *
                                         ((m_num_orbitals * (m_num_orbitals + 1) / 2) + 1) / 2);
    ao_to_mo_ptr->ao_to_mo_transform_2e(mo_coff_b, e_rep_mat, e_rep_mo2); // DD
    ao_to_mo_ptr->ao_to_mo_transform_2e(mo_coff_a, mo_coff_b, e_rep_mat, e_rep_mo3); // UD

    const Index fc = m_fcore / 2;

    // MP2
    double uu_mp2 = 0;
    #ifdef _OPENMP
    #pragma omp parallel for reduction ( + : uu_mp2)
    #endif
    for (Index i = fc; i < a_size; ++i)
        for (Index j = fc; j < a_size; ++j)
            for (Index a = a_size; a < m_num_orbitals; ++a)
                for (Index b = a_size; b < m_num_orbitals; ++b)
                {
                    uu_mp2 += e_rep_uu(i, j, a, b) * e_rep_uu(i, j, a, b) 
                            / (eps_a(i) + eps_a(j) - eps_a(a) - eps_a(b));
                }

    uu_mp2 *= 0.25;

    double dd_mp2 = 0;
    #ifdef _OPENMP
    #pragma omp parallel for reduction ( + : dd_mp2)
    #endif
    for (Index i = fc; i < b_size; ++i)
        for (Index j = fc; j < b_size; ++j)
            for (Index a = b_size; a < m_num_orbitals; ++a)
                for (Index b = b_size; b < m_num_orbitals; ++b)
                {
                    dd_mp2 += e_rep_dd(i, j, a, b) * e_rep_dd(i, j, a, b) 
                            / (eps_b(i) + eps_b(j) - eps_b(a) - eps_b(b));
                }
    
    dd_mp2 *= 0.25;

    double ud_mp2 = 0;
    #ifdef _OPENMP
    #pragma omp parallel for reduction ( + : ud_mp2)
    #endif
    for (Index i = fc; i < a_size; ++i)
        for (Index j = fc; j < b_size; ++j)
            for (Index a = a_size; a < m_num_orbitals; ++a)
                for (Index b = b_size; b < m_num_orbitals; ++b)
                {
                    ud_mp2 += e_rep_ud(i, j, a, b) * e_rep_ud(i, j, a, b)
                            / (mo_energies_a(i, i) + mo_energies_b(j, j) - mo_energies_a(a, a) - mo_energies_b(b, b));
                }
    
    double uu_mp3 = 0;
    double dd_mp3 = 0;
    double ud_mp3 = 0;
    #ifdef _OPENMP
    #pragma omp parallel for reduction ( + : uu_mp3) schedule(dynamic)
    #endif
    for (Index i = fc; i < a_size; ++i)
        for (Index j = fc; j < a_size; ++j)
            for (Index k = fc; k < a_size; ++k)
                for (Index a = a_size; a < m_num_orbitals; ++a)
                    for (Index b = a_size; b < m_num_orbitals; ++b)
                        for (Index c = a_size; c < m_num_orbitals; ++c)
                            uu_mp3 -= e_rep_uu(i, j, a, b) * e_rep_uu(k, b, i, c) *  e_rep_uu(a, c, k, j)
                            / ((eps_a(i) + eps_a(j) - eps_a(a) - eps_a(b)) 
                            * (eps_a(k) + eps_a(j) - eps_a(a) - eps_a(c)));

    #ifdef _OPENMP
    #pragma omp parallel for reduction ( + : ud_mp3) schedule(dynamic)
    #endif
    for (Index i = fc; i < a_size; ++i)
        for (Index j = fc; j < b_size; ++j)
            for (Index k = fc; k < a_size; ++k)
                for (Index a = b_size; a < m_num_orbitals; ++a)
                    for (Index b = a_size; b < m_num_orbitals; ++b)
                        for (Index c = a_size; c < m_num_orbitals; ++c)
                            ud_mp3 -= e_rep_ud(i, j, b, a) * e_rep_uu(k, b, i, c) *  e_rep_ud(c, a, k, j)
                            / ((eps_a(i) + eps_b(j) - eps_b(a) - eps_a(b)) 
                            * (eps_a(k) + eps_b(j) - eps_b(a) - eps_a(c)));
    
    #ifdef _OPENMP
    #pragma omp parallel for reduction ( + : ud_mp3) schedule(dynamic)
    #endif
    for (Index i = fc; i < b_size; ++i)
        for (Index j = fc; j < a_size; ++j)
            for (Index k = fc; k < a_size; ++k)
                for (Index a = a_size; a < m_num_orbitals; ++a)
                    for (Index b = b_size; b < m_num_orbitals; ++b)
                        for (Index c = a_size; c < m_num_orbitals; ++c)
                            ud_mp3 -= e_rep_ud(j, i, a, b) * e_rep_ud(k, b, c, i) *  e_rep_uu(a, c, k, j)
                            / ((eps_b(i) + eps_a(j) - eps_a(a) - eps_b(b)) 
                            * (eps_a(k) + eps_a(j) - eps_a(a) - eps_a(c)));

    #ifdef _OPENMP
    #pragma omp parallel for reduction ( + : ud_mp3) schedule(dynamic)
    #endif
    for (Index i = fc; i < b_size; ++i)
        for (Index j = fc; j < b_size; ++j)
            for (Index k = fc; k < a_size; ++k)
                for (Index a = b_size; a < m_num_orbitals; ++a)
                    for (Index b = b_size; b < m_num_orbitals; ++b)
                        for (Index c = a_size; c < m_num_orbitals; ++c)
                            ud_mp3 -= e_rep_dd(i, j, a, b) * e_rep_ud(k, b, c, i) *  e_rep_ud(c, a, k, j)
                            / ((eps_b(i) + eps_b(j) - eps_b(a) - eps_b(b)) 
                            * (eps_a(k) + eps_b(j) - eps_b(a) - eps_a(c)));
    
    #ifdef _OPENMP
    #pragma omp parallel for reduction ( + : ud_mp3) schedule(dynamic)
    #endif
    for (Index i = fc; i < a_size; ++i)
        for (Index j = fc; j < b_size; ++j)
            for (Index k = fc; k < a_size; ++k)
                for (Index a = a_size; a < m_num_orbitals; ++a)
                    for (Index b = b_size; b < m_num_orbitals; ++b)
                        for (Index c = b_size; c < m_num_orbitals; ++c)
                            ud_mp3 -= e_rep_ud(i, j, a, b) * e_rep_ud(k, b, i, c) *  e_rep_ud(a, c, k, j)
                            / ((eps_a(i) + eps_b(j) - eps_a(a) - eps_b(b)) 
                            * (eps_a(k) + eps_b(j) - eps_a(a) - eps_b(c)));
    
      
    #ifdef _OPENMP
    #pragma omp parallel for reduction ( + : dd_mp3) schedule(dynamic)
    #endif
    for (Index i = fc; i < b_size; ++i)
        for (Index j = fc; j < b_size; ++j)
            for (Index k = fc; k < b_size; ++k)
                for (Index a = b_size; a < m_num_orbitals; ++a)
                    for (Index b = b_size; b < m_num_orbitals; ++b)
                        for (Index c = b_size; c < m_num_orbitals; ++c)
                            dd_mp3 -= e_rep_dd(i, j, a, b) * e_rep_dd(k, b, i, c) *  e_rep_dd(a, c, k, j)
                            / ((eps_b(i) + eps_b(j) - eps_b(a) - eps_b(b)) 
                            * (eps_b(k) + eps_b(j) - eps_b(a) - eps_b(c)));
    
    #ifdef _OPENMP
    #pragma omp parallel for reduction ( + : ud_mp3) schedule(dynamic)
    #endif
    for (Index i = fc; i < b_size; ++i)
        for (Index j = fc; j < a_size; ++j)
            for (Index k = fc; k < b_size; ++k)
                for (Index a = a_size; a < m_num_orbitals; ++a)
                    for (Index b = b_size; b < m_num_orbitals; ++b)
                        for (Index c = b_size; c < m_num_orbitals; ++c)
                            ud_mp3 -= e_rep_ud(j, i, a, b) * e_rep_dd(k, b, i, c) *  e_rep_ud(a, c, j, k)
                            / ((eps_b(i) + eps_a(j) - eps_a(a) - eps_b(b)) 
                            * (eps_b(k) + eps_a(j) - eps_a(a) - eps_b(c)));
    
    #ifdef _OPENMP
    #pragma omp parallel for reduction ( + : ud_mp3) schedule(dynamic)
    #endif
    for (Index i = fc; i < a_size; ++i)
        for (Index j = fc; j < b_size; ++j)
            for (Index k = fc; k < b_size; ++k)
                for (Index a = b_size; a < m_num_orbitals; ++a)
                    for (Index b = a_size; b < m_num_orbitals; ++b)
                        for (Index c = b_size; c < m_num_orbitals; ++c)
                            ud_mp3 -= e_rep_ud(i, j, b, a) * e_rep_ud(b, k, i, c) *  e_rep_dd(a, c, k, j)
                            / ((eps_a(i) + eps_b(j) - eps_b(a) - eps_a(b)) 
                            * (eps_b(k) + eps_b(j) - eps_b(a) - eps_b(c)));
    
    #ifdef _OPENMP
    #pragma omp parallel for reduction ( + : ud_mp3) schedule(dynamic)
    #endif
    for (Index i = fc; i < a_size; ++i)
        for (Index j = fc; j < a_size; ++j)
            for (Index k = fc; k < b_size; ++k)
                for (Index a = a_size; a < m_num_orbitals; ++a)
                    for (Index b = a_size; b < m_num_orbitals; ++b)
                        for (Index c = b_size; c < m_num_orbitals; ++c)
                            ud_mp3 -= e_rep_uu(i, j, a, b) * e_rep_ud(b, k, i, c) *  e_rep_ud(a, c, j, k)
                            / ((eps_a(i) + eps_a(j) - eps_a(a) - eps_a(b)) 
                            * (eps_b(k) + eps_a(j) - eps_a(a) - eps_b(c)));
    
    #ifdef _OPENMP
    #pragma omp parallel for reduction ( + : ud_mp3) schedule(dynamic)
    #endif
    for (Index i = fc; i < b_size; ++i)
        for (Index j = fc; j < a_size; ++j)
            for (Index k = fc; k < b_size; ++k)
                for (Index a = b_size; a < m_num_orbitals; ++a)
                    for (Index b = a_size; b < m_num_orbitals; ++b)
                        for (Index c = a_size; c < m_num_orbitals; ++c)
                            ud_mp3 -= e_rep_ud(j, i, b, a) * e_rep_ud(b, k, c, i) *  e_rep_ud(c, a, j, k)
                            / ((eps_b(i) + eps_a(j) - eps_b(a) - eps_a(b)) 
                            * (eps_b(k) + eps_a(j) - eps_b(a) - eps_a(c)));

    #ifdef _OPENMP
    #pragma omp parallel for reduction ( + : uu_mp3) schedule(dynamic)
    #endif
    for (Index i = fc; i < a_size; ++i)
        for (Index j = fc; j < a_size; ++j)
            for (Index a = a_size; a < m_num_orbitals; ++a)
                for (Index b = a_size; b < m_num_orbitals; ++b)
                    for (Index c = a_size; c < m_num_orbitals; ++c)
                        for (Index d = a_size; d < m_num_orbitals; ++d)
                            uu_mp3 += e_rep_uu(i, j, a, b) * e_rep_uu(a, b, c, d) *  e_rep_uu(c, d, i, j)
                            / (8.0 * (eps_a(i) + eps_a(j) - eps_a(a) - eps_a(b)) 
                            * (eps_a(i) + eps_a(j) - eps_a(c) - eps_a(d)));
    
    #ifdef _OPENMP
    #pragma omp parallel for reduction ( + : dd_mp3) schedule(dynamic)
    #endif
    for (Index i = fc; i < b_size; ++i)
        for (Index j = fc; j < b_size; ++j)
            for (Index a = b_size; a < m_num_orbitals; ++a)
                for (Index b = b_size; b < m_num_orbitals; ++b)
                    for (Index c = b_size; c < m_num_orbitals; ++c)
                        for (Index d = b_size; d < m_num_orbitals; ++d)
                            dd_mp3 += e_rep_dd(i, j, a, b) * e_rep_dd(a, b, c, d) *  e_rep_dd(c, d, i, j)
                            / (8.0 * (eps_b(i) + eps_b(j) - eps_b(a) - eps_b(b)) 
                            * (eps_b(i) + eps_b(j) - eps_b(c) - eps_b(d)));
    
    #ifdef _OPENMP
    #pragma omp parallel for reduction ( + : ud_mp3) schedule(dynamic)
    #endif
    for (Index i = fc; i < a_size; ++i)
        for (Index j = fc; j < b_size; ++j)
            for (Index a = a_size; a < m_num_orbitals; ++a)
                for (Index b = b_size; b < m_num_orbitals; ++b)
                    for (Index c = a_size; c < m_num_orbitals; ++c)
                        for (Index d = b_size; d < m_num_orbitals; ++d)
                            ud_mp3 += e_rep_ud(i, j, a, b) * e_rep_ud(a, b, c, d) * e_rep_ud(c, d, i, j)
                            / ((eps_a(i) + eps_b(j) - eps_a(a) - eps_b(b)) 
                            * (eps_a(i) + eps_b(j) - eps_a(c) - eps_b(d)));
    
    #ifdef _OPENMP
    #pragma omp parallel for reduction ( + : uu_mp3) schedule(dynamic)
    #endif
    for (Index i = fc; i < a_size; ++i)
        for (Index j = fc; j < a_size; ++j)
            for (Index a = a_size; a < m_num_orbitals; ++a)
                for (Index b = a_size; b < m_num_orbitals; ++b)
                    for (Index k = fc; k < a_size; ++k)
                        for (Index l = fc; l < a_size; ++l)
                            uu_mp3 += e_rep_uu(i, j, a, b) * e_rep_uu(a, b, k, l) * e_rep_uu(k, l, i, j)
                            / (8.0 * (eps_a(i) + eps_a(j) - eps_a(a) - eps_a(b)) 
                            * (eps_a(k) + eps_a(l) - eps_a(a) - eps_a(b)));
    
      
    #ifdef _OPENMP
    #pragma omp parallel for reduction ( + : dd_mp3) schedule(dynamic)
    #endif
    for (Index i = fc; i < b_size; ++i)
        for (Index j = fc; j < b_size; ++j)
            for (Index a = b_size; a < m_num_orbitals; ++a)
                for (Index b = b_size; b < m_num_orbitals; ++b)
                    for (Index k = fc; k < b_size; ++k)
                        for (Index l = fc; l < b_size; ++l)
                            dd_mp3 += e_rep_dd(i, j, a, b) * e_rep_dd(a, b, k, l) * e_rep_dd(k, l, i, j)
                            / (8.0 * (eps_b(i) + eps_b(j) - eps_b(a) - eps_b(b)) 
                            * (eps_b(k) + eps_b(l) - eps_b(a) - eps_b(b)));
    
    #ifdef _OPENMP
    #pragma omp parallel for reduction ( + : ud_mp3) schedule(dynamic)
    #endif
    for (Index i = fc; i < a_size; ++i)
        for (Index j = fc; j < b_size; ++j)
            for (Index a = a_size; a < m_num_orbitals; ++a)
                for (Index b = b_size; b < m_num_orbitals; ++b)
                    for (Index k = fc; k < a_size; ++k)
                        for (Index l = fc; l < b_size; ++l)
                            ud_mp3 += e_rep_ud(i, j, a, b) * e_rep_ud(a, b, k, l) * e_rep_ud(k, l, i, j)
                            / ((eps_a(i) + eps_b(j) - eps_a(a) - eps_b(b)) 
                            * (eps_a(k) + eps_b(l) - eps_a(a) - eps_b(b)));
    
    EigenVector<double> mp3 = EigenVector<double>(6);
    mp3(0) = uu_mp2; 
    mp3(1) = dd_mp2;
    mp3(2) = ud_mp2;
    mp3(3) = uu_mp3;
    mp3(4) = dd_mp3;
    mp3(5) = ud_mp3;

    return mp3;
}

// Close shell versions of MP3 in MO basis, much faster than SO version
std::pair<double, double> POSTSCF::post_scf_mp::calc_mp3_energy(const Eigen::Ref<const EigenMatrix<double> >& mo_energies,
                                                                const Eigen::Ref<const EigenMatrix<double> >& mo_coff,
                                                                const Eigen::Ref<const EigenVector<double> >& e_rep_mat) 
{
  
    // change to physics indexing
    const auto e_rep2 = [this](Index i, Index j, Index k, Index l) -> double
    {
        Index ijkl = index4(i, k, j, l);
        Index ilkj = index4(i, l, k, j);
        return 2.0 * e_rep_mo(ijkl) - e_rep_mo(ilkj); 
    };

    const auto e_rep1 = [this](Index i, Index j, Index k, Index l) -> double
    {
        Index ijkl = index4(i, k, j, l);
        return e_rep_mo[ijkl];
    };

    std::unique_ptr<BTRANS::basis_transform> ao_to_mo_ptr = std::make_unique<BTRANS::basis_transform>(m_num_orbitals);

    // translate AOs to MOs
    e_rep_mo = EigenVector<double>::Zero((m_num_orbitals * (m_num_orbitals + 1) / 2) *
                                        ((m_num_orbitals * (m_num_orbitals + 1) / 2) + 1) / 2);
    ao_to_mo_ptr->ao_to_mo_transform_2e(mo_coff, e_rep_mat, e_rep_mo);

    EigenVector<double> eps = EigenVector<double>(m_num_orbitals);

    for (Index i = 0; i < m_num_orbitals; ++i) eps(i) = mo_energies(i, i);

    const Index fc  = m_fcore / 2;
    const Index occ = m_nelectrons / 2;

    double sum_mp2 = 0.0;
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) reduction ( + : sum_mp2)
    #endif
    for (Index i = fc; i < occ; ++i)
        for (Index a = occ; a < m_num_orbitals; ++a)
            for (Index j = fc; j < occ; ++j)
                for (Index b = occ; b < m_num_orbitals; ++b)
                { 
                    sum_mp2 += e_rep1(i, j, a, b) * e_rep2(i, j, a, b) 
                             / (eps(i) + eps(j) - eps(a) - eps(b));
                }

    double sum1_mp3 = 0.0;
    #ifdef _OPENMP
    #pragma omp parallel for  reduction(+ : sum1_mp3)
    #endif
    for (Index a = fc; a < occ; ++a)
        for (Index b = fc; b < occ; ++b)
            for (Index c = fc; c < occ; ++c)
                for (Index d = fc; d < occ; ++d)
                    for (Index r = occ; r < m_num_orbitals; ++r)
                        for (Index s = occ; s < m_num_orbitals; ++s) 
                            sum1_mp3 += e_rep1(a, b, r, s) * e_rep2(c, d, a, b) * e_rep1(c, d, r, s)
                                      / ((eps(a) + eps(b) - eps(r) - eps(s)) * (eps(c) + eps(d) - eps(r) - eps(s)));

    double sum2_mp3 = 0;
    #ifdef _OPENMP
    #pragma omp parallel for reduction ( + : sum2_mp3)
    #endif
    for (Index a = fc; a < occ; ++a)
        for (Index b = fc; b < occ; ++b)
            for (Index r = occ; r < m_num_orbitals; ++r)
                for (Index s = occ; s < m_num_orbitals; ++s)
                    for (Index t = occ; t < m_num_orbitals; ++t)
                        for (Index u = occ; u < m_num_orbitals; ++u)
                            sum2_mp3 += e_rep1(a, b, r, s) * e_rep2(t, u, a, b) * e_rep1(r, s, t, u)
                                      / ((eps(a) + eps(b) - eps(r) - eps(s)) * (eps(a) + eps(b) - eps(t) - eps(u)));

    double sum3_mp3 = 0.0;
    #ifdef _OPENMP
    #pragma omp parallel for reduction ( + : sum3_mp3)
    #endif
    for (Index a = fc; a < occ; ++a)
        for (Index b = fc; b < occ; ++b)
            for (Index c = fc; c < occ; ++c)
                for (Index r = occ; r < m_num_orbitals; ++r)
                    for (Index s = occ; s < m_num_orbitals; ++s)
                        for (Index t = occ; t < m_num_orbitals; ++t)
                            sum3_mp3 +=(
                                          e_rep1(a, b, r, s) * e_rep1(c, s, a, t) * e_rep2(r, t, c, b)
                                        + e_rep1(a, b, r, s) * e_rep1(c, s, t, a) * e_rep2(r, t, b, c)
                                        + e_rep1(a, b, s, r) * e_rep1(c, s, a, t) * e_rep2(r, t, b, c)
                                        + e_rep1(a, b, s, r) * e_rep1(c, s, t, a)
                                        * (2.0 * e_rep1(r, t, c, b) - 4.0 *  e_rep1(r, t, b, c))
                                       ) / ((eps(a) + eps(b) - eps(r) - eps(s)) * (eps(b) + eps(c) - eps(r) - eps(t)));
    sum3_mp3 *= -2.0;

    return std::pair<double, double>(sum_mp2, sum1_mp3 + sum2_mp3 + sum3_mp3);
}

// Close shell versions of MP3 in SMO basis, no longer used
std::pair<double, double> POSTSCF::post_scf_mp::calc_mp3_energy_so(const Eigen::Ref<const EigenMatrix<double> >& mo_energies,
                                                                   const Eigen::Ref<const EigenMatrix<double> >& mo_coff,
                                                                   const Eigen::Ref<const EigenVector<double> >& e_rep_mat) 
{
    // Currently rhf only
    // -0.142119825 h2o_2 test;

    std::unique_ptr<BTRANS::basis_transform> ao_to_mo_ptr = std::make_unique<BTRANS::basis_transform>(m_num_orbitals);

    // translate AOs to MOs
    e_rep_mo = EigenVector<double>::Zero((m_num_orbitals * (m_num_orbitals + 1) / 2) *
                                        ((m_num_orbitals * (m_num_orbitals + 1) / 2) + 1) / 2);
    ao_to_mo_ptr->ao_to_mo_transform_2e(mo_coff, e_rep_mat, e_rep_mo);

    // Note: 1D vector to mimick 4D tensor
    // translate MOs to spin MOs

    ao_to_mo_ptr->mo_to_spin_transform_2e(e_rep_mo, e_rep_smo);

    energy_smo = EigenVector<double>(m_spin_mos);

    for (Index i = 0; i < m_spin_mos; ++i)
        energy_smo(i) = mo_energies(i / 2, i / 2);
    

    double sum_mp2 = 0.0;
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) reduction ( + : sum_mp2)
    #endif
    for (Index i = m_fcore; i < m_nelectrons; ++i)
        for (Index j = m_fcore; j < m_nelectrons; ++j)
            for (Index a = m_nelectrons; a < m_spin_mos; ++a)
                for (Index b = m_nelectrons; b < m_spin_mos; ++b)
                {
                    sum_mp2 += e_rep_smo(i, j, a, b) * e_rep_smo(i, j, a, b) 
                            / (energy_smo(i) + energy_smo(j) - energy_smo(a) - energy_smo(b));
                }

    sum_mp2 *= 0.25;

    // Formulae from Szabo & Ostlund Modern MQC page 35
    double sum1_mp3 = 0.0;
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) reduction(+ : sum1_mp3) schedule(dynamic) default(none)
    #endif
    for (Index a = m_fcore; a < m_nelectrons; ++a)
        for (Index b = m_fcore; b < m_nelectrons; ++b)
            for (Index c = m_fcore; c < m_nelectrons; ++c)
                for (Index d = m_fcore; d < m_nelectrons; ++d)
                    for (Index r = m_nelectrons; r < m_spin_mos; ++r)
                        for (Index s = m_nelectrons; s < m_spin_mos; ++s) 
                        {
                            sum1_mp3 += e_rep_smo(a, b, r, s) * e_rep_smo(c, d, a, b) * e_rep_smo(r, s, c, d)
                                      / ((energy_smo(a) + energy_smo(b) - energy_smo(r) - energy_smo(s)) 
                                      *  (energy_smo(c) + energy_smo(d) - energy_smo(r) - energy_smo(s)));
                        }

    sum1_mp3 *= 0.125;

    double sum2_mp3 = 0;
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) reduction ( + : sum2_mp3) schedule(dynamic) default(none)
    #endif
    for (Index a = m_fcore; a < m_nelectrons; ++a)
        for (Index b = m_fcore; b < m_nelectrons; ++b)
            for (Index r = m_nelectrons; r < m_spin_mos; ++r)
                for (Index s = m_nelectrons; s < m_spin_mos; ++s)
                    for (Index t = m_nelectrons; t < m_spin_mos; ++t)
                        for (Index u = m_nelectrons; u < m_spin_mos; ++u)
                        {
                            sum2_mp3 += e_rep_smo(a, b, r, s) * e_rep_smo(r, s, t, u) * e_rep_smo(t, u, a, b)
                                      / ((energy_smo(a) + energy_smo(b) - energy_smo(r) - energy_smo(s)) 
                                      *  (energy_smo(a) + energy_smo(b) - energy_smo(t) - energy_smo(u)));
                        }
    
    sum2_mp3 *= 0.125;

    double sum3_mp3 = 0.0;
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) reduction ( + : sum3_mp3) schedule(dynamic) default(none)
    #endif
    for (Index a = m_fcore; a < m_nelectrons; ++a)
        for (Index b = m_fcore; b < m_nelectrons; ++b)
            for (Index c = m_fcore; c < m_nelectrons; ++c)
                for (Index r = m_nelectrons; r < m_spin_mos; ++r)
                    for (Index s = m_nelectrons; s < m_spin_mos; ++s)
                        for (Index t = m_nelectrons; t < m_spin_mos; ++t)
                        {
                            sum3_mp3 += e_rep_smo(a, b, r, s) * e_rep_smo(c, s, t, b) * e_rep_smo(r, t, a, c)
                                      / ((energy_smo(a) + energy_smo(b) - energy_smo(r) - energy_smo(s)) 
                                      *  (energy_smo(a) + energy_smo(c) - energy_smo(r) - energy_smo(t)));
                        }
    
    return std::pair<double, double>(sum_mp2, sum1_mp3 + sum2_mp3 + sum3_mp3);
}

