#include "../settings/hfscf_settings.hpp"
#include "../basis/hfscf_trans_basis.hpp"
#include "hfscf_post_cc_rhf.hpp"
#include "../math/hfscf_tensors.hpp"
#include <memory>
#include <algorithm>
#include <iomanip>
#include <sstream>

using Eigen::Index;
// This is the MO spatial only basis

POSTSCF::post_rhf_ccsd::post_rhf_ccsd(Index num_orbitals, Index nelectrons, Index fcore)
: m_num_orbitals(num_orbitals), m_fcore(fcore) 
{
    m_occ = nelectrons / 2;
    m_virt = m_num_orbitals - m_occ;
    m_fcore /= 2;

    if (!m_fcore % 2 && m_fcore > 0) 
        m_fcore = 0;
   //  incase an incorrect odd number is passed set to zero .. must be even

    if(!HF_SETTINGS::hf_settings::get_freeze_core()) m_fcore = 0;
}

using tensor4dmath::index4;

std::pair<double, double> 
POSTSCF::post_rhf_ccsd::calc_ccsd(const Eigen::Ref<const EigenMatrix<double> >& mo_energies,
                                  const Eigen::Ref<const EigenMatrix<double> >& mo_coff,
                                  const Eigen::Ref<const EigenVector<double> >& e_rep_mat)
{
    std::unique_ptr<BTRANS::basis_transform> ao_to_smo_ptr = std::make_unique<BTRANS::basis_transform>(m_num_orbitals);

    // translate AOs to MOs
    e_rep_mo = EigenVector<double>::Zero((m_num_orbitals * (m_num_orbitals + 1) / 2) *
                                        ((m_num_orbitals * (m_num_orbitals + 1) / 2) + 1) / 2);
    ao_to_smo_ptr->ao_to_mo_transform_2e(mo_coff, e_rep_mat, e_rep_mo);
    eps = mo_energies;

    f_ki = EigenMatrix<double>::Zero(m_occ, m_occ);
    f_ac = EigenMatrix<double>::Zero(m_virt, m_virt);
    f_kc = EigenMatrix<double>::Zero(m_occ, m_virt);

    L_ki = EigenMatrix<double>::Zero(m_occ, m_occ);
    L_ac = EigenMatrix<double>::Zero(m_virt, m_virt);

    d_ai = EigenMatrix<double>::Zero(m_virt, m_occ);
    d_abij = tensor4d1122<double>(m_virt, m_occ);
    
    w_klij = tensor4d<double>(m_occ);
    w_abcd = tensor4d<double>(m_virt);
    w_akic = tensor4d1221<double>(m_virt, m_occ);
    w_akci = tensor4d1234<double>(m_virt, m_occ, m_virt, m_occ);

    ts = EigenMatrix<double>::Zero(m_occ, m_virt);
    ts_new = EigenMatrix<double>::Zero(m_occ, m_virt);
    td = tensor4d1122<double>(m_occ, m_virt);
    td_new = tensor4d1122<double>(m_occ, m_virt);

    const auto v_rep = [this](Index p, Index q, Index r, Index s) noexcept -> double
    {
        Index prqs = index4(p, r, q, s);
        return e_rep_mo[prqs];
    };

    double sum_mp2 = 0.0;
    #ifdef _OPENMP
    #pragma omp parallel for reduction ( + : sum_mp2)
    #endif
    for (Index i = m_fcore; i < m_occ; ++i)
        for (Index a = m_occ; a < m_num_orbitals; ++a)
            for (Index j = m_fcore; j < m_occ; ++j)
                for (Index b = m_occ; b < m_num_orbitals; ++b) 
                {
                    Index iajb = hfscfmath::index_ijkl(i, a, j, b);
                    Index ibja = hfscfmath::index_ijkl(i, b, j, a);
                    sum_mp2 += e_rep_mo(iajb) * (2 * e_rep_mo(iajb) - e_rep_mo(ibja)) /
                    (eps(i, i) + eps(j, j) - eps(a, a) - eps(b, b));
                }
    
    const Index offset = m_occ;
    for (Index a = m_occ; a < m_num_orbitals; ++a)
        for (Index i = 0; i < m_occ; ++i)
            d_ai(a - offset, i) = eps(i, i) - eps(a, a);

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (Index a = m_occ; a < m_num_orbitals; ++a)
        for (Index b = m_occ; b < m_num_orbitals; ++b)
            for (Index i = 0; i < m_occ; ++i)
                for (Index j = 0; j < m_occ; ++j)
                {
                    d_abij(a - offset, b - offset, i, j) = eps(i, i) + eps(j, j) - eps(a, a) - eps(b, b);
                    td(i, j, a - offset, b - offset) = v_rep(i, j, a, b) 
                                                     / d_abij(a - offset, b - offset, i, j);
                    td_new(i, j, a - offset, b - offset) = td(i, j, a - offset, b - offset);
                }

    double e_ccsd = 0.0;
    double previous_e_ccsd = 0.0;
    double delta = std::numeric_limits<double>::max();

    for(;;)
    {
        previous_e_ccsd = e_ccsd;
        update();
        T1();
        T2();
        ccsd_diis();
        ts = ts_new;
        td = td_new;
        e_ccsd = calc_ccsd_energy();
        ccsd_energies.emplace_back(e_ccsd);
        delta = e_ccsd - previous_e_ccsd;
        delta_energies.emplace_back(delta);

        if( std::fabs<double>(delta) < HF_SETTINGS::hf_settings::get_ccsd_energy_tol() && 
           ccsd_rms[iteration] < HF_SETTINGS::hf_settings::get_ccsd_rms_tol() )
           break;
        else if(HF_SETTINGS::hf_settings::get_max_ccsd_iterations() == iteration)
        {
            std::cout << "  Warning: Maximum iterations reached during CCSD DIIS convergence, aborting.\n";
            exit(EXIT_FAILURE);
        }

        ++iteration;
    }

    calc_largest_T1T2();
    return std::pair(sum_mp2, e_ccsd);
}

double POSTSCF::post_rhf_ccsd::calc_ccsd_energy() const noexcept
{
    Index offset = m_occ;

    const auto w_rep = [this](Index p, Index q, Index r, Index s) noexcept -> double
    {
        Index prqs = index4(p, r, q, s);
        Index psqr = index4(p, s, q, r); 
        return 2.0 * e_rep_mo[prqs] - e_rep_mo[psqr];
    };

    double e_ccsd  = 0.0;
    #ifdef _OPENMP
    #pragma omp parallel for reduction ( + :e_ccsd)
    #endif
    for(Index i = m_fcore; i < m_occ; ++i)
        for(Index j = m_fcore; j < m_occ; ++j)
            for (Index a = m_occ; a < m_num_orbitals; ++a)
                for (Index b = m_occ; b < m_num_orbitals; ++b)
                    e_ccsd += w_rep(i, j, a, b) * td(i, j, a - offset, b - offset) 
                            + w_rep(i, j, a, b) * ts(i, a - offset) * ts(j, b - offset);

    return e_ccsd;
}

void POSTSCF::post_rhf_ccsd::update() noexcept
{
    // Hirata eqns https://aip.scitation.org/doi/10.1063/1.1637577

    const auto w_rep = [this](Index p, Index q, Index r, Index s) noexcept -> double
    {
        Index prqs = index4(p, r, q, s);
        Index psqr = index4(p, s, q, r); 
        return 2.0 * e_rep_mo[prqs] - e_rep_mo[psqr];
    };

    const auto v_rep = [this](Index p, Index q, Index r, Index s) noexcept -> double
    {
        Index prqs = index4(p, r, q, s);
        return e_rep_mo[prqs];
    };

    f_ki.setZero();
    const Index offset = m_occ;
    #ifdef _OPENMP
    #pragma omp parallel for schedule (dynamic)
    #endif
    for (Index k = m_fcore; k < m_occ; ++k)
    {
        f_ki(k, k) = eps(k, k);
        for (Index i = m_fcore; i < m_occ; ++i)
            for (Index l = m_fcore; l < m_occ; ++l)
                for(Index c = m_occ; c < m_num_orbitals; ++c)
                    for (Index d = m_occ; d < m_num_orbitals; ++d)
                        f_ki(k,  i) += w_rep(k, l, c, d) 
                      * (td(i, l, c - offset, d - offset) + ts(i, c - offset) * ts(l, d - offset));
    }

    L_ki.setZero();
    #ifdef _OPENMP
    #pragma omp parallel for schedule (dynamic)
    #endif
    for (Index k = m_fcore; k < m_occ; ++k)
        for (Index i = m_fcore; i < m_occ; ++i)
        {
            L_ki(k, i) = f_ki(k, i);
            for (Index l = m_fcore; l < m_occ; ++l)
                for(Index c = m_occ; c < m_num_orbitals; ++c)
                    L_ki(k, i) += w_rep(l, k, c, i) * ts(l, c - offset);
        }

    //std::cout << std::setprecision(6) << "\n L_ki \n" << L_ki << "\n\n";
 
    f_ac.setZero();
    #ifdef _OPENMP
    #pragma omp parallel for schedule (dynamic)
    #endif
    for (Index a = m_occ; a < m_num_orbitals; ++a)
    {
        f_ac(a - offset, a - offset) = eps(a, a);
        for (Index c = m_occ; c < m_num_orbitals; ++c)
            for (Index k = m_fcore; k < m_occ; ++k)
                for(Index l = m_fcore; l < m_occ; ++l)
                    for (Index d = m_occ; d < m_num_orbitals; ++d)
                        f_ac(a - offset,  c - offset) -= w_rep(k, l, c, d) 
                      * (td(k, l, a - offset, d - offset) + ts(k, a - offset) * ts(l, d - offset));
    }

    L_ac.setZero();
    #ifdef _OPENMP
    #pragma omp parallel for schedule (dynamic)
    #endif
    for (Index a = m_occ; a < m_num_orbitals; ++a)
        for (Index c = m_occ; c < m_num_orbitals; ++c)
        {
            L_ac(a - offset, c - offset) = f_ac(a - offset, c - offset);
            for (Index k = m_fcore; k < m_occ; ++k)
                for (Index d = m_occ; d < m_num_orbitals; ++d)
                    L_ac(a - offset, c - offset) += w_rep(k, a, d, c) * ts(k, d - offset);
        }
  
    f_kc.setZero();
    #ifdef _OPENMP
    #pragma omp parallel for schedule (dynamic)
    #endif
    for (Index k = m_fcore; k < m_occ; ++k)
        for (Index c = m_occ; c < m_num_orbitals; ++c)
            for(Index l = m_fcore; l < m_occ; ++l)
                for (Index d = m_occ; d < m_num_orbitals; ++d)
                    f_kc(k, c - offset) += w_rep(k, l, c, d) * ts(l, d - offset);
    
    w_klij.setZero(); // oooo
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index k = m_fcore; k < m_occ; ++k)
        for (Index l = m_fcore; l < m_occ; ++l)
            for (Index i = m_fcore; i < m_occ; ++i)
                for (Index j = m_fcore; j < m_occ; ++j)
                {
                    if(fabs(v_rep(k, l, i, j)) < HF_SETTINGS::hf_settings::get_integral_tol()) continue;

                    w_klij(k, l, i, j) = v_rep(k, l, i, j);
                    for (Index c = m_occ; c < m_num_orbitals; ++c)
                    {
                        w_klij(k, l, i, j) += v_rep(l, k, c, i) * ts(j, c - offset)
                                            + v_rep(k, l, c, j) * ts(i, c - offset);

                        for (Index d = m_occ; d < m_num_orbitals; ++d)
                            w_klij(k, l, i, j) += v_rep(k, l, c, d) 
                            * (td(i, j, c - offset, d - offset) + ts(i, c - offset) * ts(j, d - offset));

                    }
                }
    
    w_abcd.setZero(); // vvvv
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index a = m_occ; a < m_num_orbitals; ++a)
        for (Index b = m_occ; b < m_num_orbitals; ++b)
            for (Index c = m_occ; c < m_num_orbitals; ++c)
                for (Index d = m_occ; d < m_num_orbitals; ++d)
                {
                    if(fabs(v_rep(a, b, c, d)) < HF_SETTINGS::hf_settings::get_integral_tol()) continue;

                    w_abcd(a - offset, b - offset, c - offset, d - offset) = v_rep(a, b, c, d);
                    for (Index k = m_fcore; k < m_occ; ++k)
                        w_abcd(a - offset, b - offset, c - offset, d - offset) -=
                        v_rep(k, a, d, c) * ts(k, b - offset) + v_rep(k, b, c, d) * ts(k, a - offset);
                }

    w_akic.setZero(); // voov
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index a = m_occ; a < m_num_orbitals ; ++a)
        for (Index k = m_fcore; k < m_occ; ++k)
            for (Index i = m_fcore; i < m_occ; ++i)
                for (Index c = m_occ; c < m_num_orbitals ; ++c)
                {
                    if(fabs(v_rep(a, k, i, c)) < HF_SETTINGS::hf_settings::get_integral_tol()) continue; 

                    w_akic(a - offset, k, i, c - offset) = v_rep(a, k, i, c);

                    for (Index d = m_occ; d < m_num_orbitals ; ++d)
                        w_akic(a - offset, k, i, c - offset) += v_rep(k, a, c, d) * ts(i, d - offset);

                    for (Index l = m_fcore; l < m_occ; ++l)
                    {
                        w_akic(a - offset, k, i, c - offset) -= v_rep(k, l, c, i) * ts(l, a - offset);
                        for (Index d = m_occ; d < m_num_orbitals ; ++d)
                            w_akic(a - offset, k, i, c - offset) +=  v_rep(l, k, d, c) *
                            (- 0.5 * td(i, l, d - offset, a - offset) - ts(i, d - offset) * ts(l, a - offset))
                             + 0.5 * w_rep(l, k, d, c) * td(i, l, a - offset, d - offset);
                    }
                }

    w_akci.setZero(); // vovo
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index a = m_occ; a < m_num_orbitals ; ++a)
        for (Index k = m_fcore; k < m_occ; ++k)
            for (Index c = m_occ; c < m_num_orbitals ; ++c)
                for (Index i = m_fcore; i < m_occ; ++i)
                {
                    if(fabs(v_rep(a, k, c, i)) < HF_SETTINGS::hf_settings::get_integral_tol()) continue;
                    
                    w_akci(a - offset, k, c - offset, i) = v_rep(a, k, c, i);

                    for (Index d = m_occ; d < m_num_orbitals ; ++d)
                        w_akci(a - offset, k, c - offset, i) += v_rep(k, a, d, c) * ts(i, d - offset);

                    for (Index l = m_fcore; l < m_occ; ++l)
                    {
                        w_akci(a - offset, k, c - offset, i) -= v_rep(l, k, c, i) * ts(l, a - offset);
                        for (Index d = m_occ; d < m_num_orbitals ; ++d)
                            w_akci(a - offset, k, c - offset, i) +=  v_rep(l, k, c, d) *
                            (- 0.5 * td(i, l, d - offset, a - offset) - ts(i, d - offset) * ts(l, a - offset));
                    }
                }
}

void POSTSCF::post_rhf_ccsd::T1() noexcept
{
    ts_new.setZero();
    Index offset = m_occ;

    const auto w_rep = [this](Index p, Index q, Index r, Index s) noexcept -> double
    {
        Index prqs = index4(p, r, q, s);
        Index psqr = index4(p, s, q, r); 
        return 2.0 * e_rep_mo[prqs] - e_rep_mo[psqr];
    };

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index i = m_fcore; i < m_occ; ++i) 
        for (Index a = m_occ; a < m_num_orbitals; ++a)
        {
            for (Index c = m_occ; c < m_num_orbitals; ++c)
                ts_new(i, a - offset) += (f_ac(a - offset, c - offset) - eps(a, c)) * ts(i, c - offset);
            
            for (Index k = m_fcore; k < m_occ; ++k)
            {
                ts_new(i, a - offset) -= (f_ki(k, i) - eps(k, i)) * ts(k, a - offset);
                
                for (Index c = m_occ; c < m_num_orbitals; ++c)
                {
                    ts_new(i, a - offset) += f_kc(k, c - offset)
                    * (2.0 * td(k, i, c - offset, a - offset) - td(i, k, c - offset, a - offset) 
                    + ts(i, c - offset) * ts(k, a - offset)) + w_rep(a, k, i, c) * ts(k, c - offset);

                    for (Index d = m_occ; d < m_num_orbitals; ++d)
                        ts_new(i, a - offset) += w_rep(a, k, c, d)
                       * (td(i, k, c - offset, d - offset) + ts(i, c - offset) * ts(k, d - offset));
                }

                for (Index l = m_fcore; l < m_occ; ++l)
                    for (Index c = m_occ; c < m_num_orbitals; ++c)
                        ts_new(i, a - offset) -= w_rep(k, l, i, c)
                        *(td(k, l, a - offset, c - offset) + ts(k, a - offset) * ts(l, c - offset));
            }

            ts_new(i, a - offset) /= d_ai(a - offset, i);
        }
}

void POSTSCF::post_rhf_ccsd::T2() noexcept
{
    td_new.setZero();
    const Index offset = m_occ;

    const auto v_rep = [this](Index p, Index q, Index r, Index s) noexcept -> double
    {
        Index prqs = index4(p, r, q, s);
        return e_rep_mo[prqs];
    };

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif 
    for (Index i = m_fcore; i < m_occ; ++i)
        for (Index j = m_fcore; j < m_occ; ++j)
            for (Index a = m_occ; a < m_num_orbitals; ++a)
                for (Index b = m_occ; b < m_num_orbitals; ++b)
                {
                    td_new(i, j, a - offset, b - offset) += 0.5 * v_rep(i, j, a, b);

                    for (Index c = m_occ; c < m_num_orbitals; ++c)
                        for (Index d = m_occ; d < m_num_orbitals; ++d)
                            td_new(i, j, a - offset, b - offset) +=
                            0.5 * w_abcd(a - offset, b - offset, c - offset, d - offset)
                            * (td(i, j, c - offset, d - offset) + ts(i, c - offset) * ts(j, d - offset));
                    
                    for (Index k = m_fcore; k < m_occ; ++k)
                        for (Index l = m_fcore; l < m_occ; ++l)
                            td_new(i, j, a - offset, b - offset) +=
                            0.5 * w_klij(k, l, i, j) * (td(k, l, a - offset, b - offset) 
                            + ts(k, a  - offset) * ts(l, b - offset));
                }
    // break loop into 2 parts gives better OMP performance
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif 
    for (Index i = m_fcore; i < m_occ; ++i)
        for (Index j = m_fcore; j < m_occ; ++j)
            for (Index a = m_occ; a < m_num_orbitals; ++a)
                for (Index b = m_occ; b < m_num_orbitals; ++b)
                {
                    for (Index c = m_occ; c < m_num_orbitals; ++c)
                    {
                        td_new(i, j, a - offset, b - offset) += td(i, j, c - offset, b - offset)
                        * (L_ac(a - offset, c - offset) - eps(a, c)) + v_rep(a, b, i, c) * ts(j, c - offset);
                    }

                    for (Index k = m_fcore; k < m_occ; ++k)
                    {
                        td_new(i, j, a - offset, b - offset) -= td(k, j, a - offset, b - offset)
                        *(L_ki(k, i) - eps(k, i)) + v_rep(a, k, i, j) * ts(k, b - offset);

                        for (Index c = m_occ; c < m_num_orbitals; ++c)
                        {
                            td_new(i, j, a - offset, b - offset) += 
                            - v_rep(k, b, i, c) * ts(k, a - offset) * ts(j, c - offset)
                            - v_rep(a, k, i, c) * ts(j, c - offset) * ts(k, b - offset);

                            td_new(i, j, a - offset, b - offset) += td(k, j, c - offset, b - offset)
                            * (2.0 * w_akic(a - offset, k, i, c - offset) 
                              - w_akci(a - offset, k, c - offset, i))
                            - w_akic(a - offset, k, i, c - offset) * td(k, j, b - offset, c - offset)
                            - w_akci(b - offset, k, c - offset, i) * td(k, j, a - offset, c - offset);
                        }
                    }
                    
                    td_new(i, j, a - offset, b - offset) /= d_abij(a - offset, b - offset, i, j);
                }
    
    tensor4d1122<double> tmp = td_new; // P(ija) on td(ijab) permutation operator
    // Permutation operator
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index i = m_fcore; i < m_occ; ++i)
        for (Index j = m_fcore; j < m_occ; ++j)
            for (Index a = m_occ; a < m_num_orbitals; ++a)
                for (Index b = m_occ; b < m_num_orbitals; ++b)
                {
                    td_new(i, j, a - offset, b - offset) = 
                    tmp(j, i, b - offset, a - offset) + tmp(i, j, a - offset, b - offset);
                }
}

void POSTSCF::post_rhf_ccsd::ccsd_diis()
{
    int diis_range = HF_SETTINGS::hf_settings::get_ccsd_diis_size();

    // // T1, + T2 size
    EigenVector<double> tmp = EigenVector<double>(m_virt * m_virt * m_occ * m_occ   
                                                + m_virt * m_occ);

    Index index = 0;
    for (Index r = 0; r < m_occ; ++r)
        for (Index s = 0; s < m_occ; ++s)
            for (Index p = 0; p < m_virt; ++p)
                for (Index q = 0; q < m_virt; ++q) 
                {
                    tmp(index) = td_new(r, s, p, q) - td(r, s, p, q);
                    ++index;
                }

    index = 0;
    const Index offset = m_virt * m_virt * m_occ * m_occ;

    for (Index j = 0; j < m_occ; ++j)
        for(Index a = 0; a < m_virt; ++a)
        {
            tmp(offset + index) = ts_new(j, a) - ts(j, a);
            ++index;
        }
                                       
    double rms = tmp.cwiseAbs2().mean();
    ccsd_rms.emplace_back(std::sqrt(rms));

    if(0 == diis_range) // Only compute rms when not using DIIS 
        return;

    if(iteration >= diis_range) 
    {
        ccsd_errors.erase(ccsd_errors.begin()); ccsd_errors.shrink_to_fit();
        t2_list.erase(t2_list.begin()); t2_list.shrink_to_fit();
        t1_list.erase(t1_list.begin()); t1_list.shrink_to_fit();
    }

    ccsd_errors.emplace_back(tmp);
    t2_list.emplace_back(td_new);
    t1_list.emplace_back(ts_new);


    if(iteration < 2) return; // start DIIS at 2 (note we start at 0)

    Index size = static_cast<Index>(ccsd_errors.size());

    EigenMatrix<double> L_mat = EigenMatrix<double>(size + 1, size + 1);

    for (Index i = 0; i < size; ++i)
        for (Index j = 0; j < size; ++j)
        {
            L_mat(i, j) = 0;
            for (Index k = 0; k < tmp.size(); ++k)
                    L_mat(i, j) += ccsd_errors[i](k) * ccsd_errors[j](k);
        }

    for (Index i = 0; i < size + 1; ++i) 
    {
        L_mat(size, i) = -1.0;
        L_mat(i, size) = -1.0;
    }

    L_mat(size, size) = 0;

    EigenVector<double> b = EigenVector<double>::Zero(size + 1);
    b(size) = -1.0;

    EigenVector<double> coffs = L_mat.householderQr().solve(b);  // fairly stable and reasonably fast

    for (Index p = 0; p < m_occ; ++p)
        for (Index q = 0; q < m_virt; ++q) 
        {
            ts_new(p, q) = 0.0;
            for (Index k = 0; k < size; ++k) 
                ts_new(p, q) += coffs(k) * t1_list[k](p, q);
        }
    
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index p = 0; p < m_occ; ++p)
        for (Index q = 0; q < m_occ; ++q)
            for (Index r = 0; r < m_virt; ++r)
                for (Index s = 0; s < m_virt; ++s) 
                {
                    td_new(p, q, r, s) = 0.0;
                    for (int k = 0; k < size; ++k) 
                        td_new(p, q, r, s) += coffs(k) * t2_list[k](p, q, r, s);
                }
}

using p = std::pair<double, std::string>;
void POSTSCF::post_rhf_ccsd::calc_largest_T1T2()
{
    std::vector<p> t1_amps = 
    std::vector<p>(m_occ * m_virt);

    size_t pos = 0;
    for (Index i = m_fcore; i < m_occ; ++i)
        for (Index a = 0; a < m_virt; ++a)
        {
            if(fabs(ts_new(i, a)) > 1.0e-10)
            {
                std::ostringstream osstr;
                osstr << std::right << std::setfill(' ') << std::setw(5) << a
                      << std::right << std::setfill(' ') << std::setw(5) << i;
                t1_amps[pos] = std::make_pair(ts_new(i, a), osstr.str());
                ++pos;
            }
        }
    
    t1_amps.resize(pos);
    
    std::sort(t1_amps.data(), t1_amps.data() + t1_amps.size(), [](const p& t1a, const p& t1b) -> bool
    {
        return fabs(t1a.first) >= fabs(t1b.first);
    });

    for (size_t i = 0; i < 10; ++i)  // every second biggest 10
    {                                   // or less if list is smaller
        if (i >= t1_amps.size()) break;
        largest_t1.emplace_back(t1_amps[i]);
    }

    std::vector<p> t2_amps = std::vector<p>(m_occ * m_virt * m_occ * m_virt);

    pos = 0;
    for (Index i = m_fcore; i < m_occ; ++i)
        for (Index j = m_fcore; j < m_occ; ++j)
            for (Index a = 0; a < m_virt; ++a)
                for (Index b = 0; b < m_virt; ++b)
                {
                    if(fabs(td_new(i, j, a, b)) > 1.0E-10)
                    {
                        std::ostringstream osstr;
                        osstr << std::right << std::setfill(' ') << std::setw(5) << a
                              << std::right << std::setfill(' ') << std::setw(5) << i
                              << std::right << std::setfill(' ') << std::setw(5) << b
                              << std::right << std::setfill(' ') << std::setw(5) << j;
                        t2_amps[pos] = std::make_pair(td_new(i, j, a, b), osstr.str());
                        ++pos;
                    }
                }

    t2_amps.resize(pos);
    std::sort(t2_amps.data(), t2_amps.data() + t2_amps.size(), [](const p& t2a, const p& t2b) -> bool 
    {
        return fabs(t2a.first) >= fabs(t2b.first);
    });

    for (size_t i = 0; i < 10; ++i)  
    {                                   
        if (i >= t2_amps.size()) break;
        largest_t2.emplace_back(t2_amps[i]);
    }
}