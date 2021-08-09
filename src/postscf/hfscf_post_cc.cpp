#include "../settings/hfscf_settings.hpp"
#include "../basis/hfscf_trans_basis.hpp"
#include "hfscf_post_cc.hpp"
#include "../math/hfscf_tensors.hpp"
#include <memory>
#include <algorithm>
#include <iomanip>
#include <sstream>

using Eigen::Index;
// TODO do an MO basis impl. Spin orbs are expesive and redundant for RHF.
// This is the spin basis version
// https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2305

POSTSCF::post_scf_ccsd::post_scf_ccsd(Index num_orbitals, Index nelectrons, Index fcore)
: m_num_orbitals(num_orbitals),
  m_nelectrons(nelectrons),
  m_fcore(fcore)
{
    m_spin_mos = 2 * m_num_orbitals;

    if (!m_fcore % 2 && m_fcore > 0) 
        m_fcore = 0;
   //  incase an incorrect odd number is passed set to zero .. must be even

    if(!HF_SETTINGS::hf_settings::get_freeze_core()) m_fcore = 0;
}

std::pair<double, double> POSTSCF::post_scf_ccsd::calc_ccsd(const Eigen::Ref<const EigenMatrix<double> >& mo_energies,
                                                            const Eigen::Ref<const EigenMatrix<double> >& mo_coff,
                                                            const Eigen::Ref<const EigenVector<double> >& e_rep_mat)
{
    std::unique_ptr<BTRANS::basis_transform> ao_to_smo_ptr = std::make_unique<BTRANS::basis_transform>(m_num_orbitals);

    // translate AOs to MOs
    e_rep_mo = EigenVector<double>::Zero((m_num_orbitals * (m_num_orbitals + 1) / 2) *
                                        ((m_num_orbitals * (m_num_orbitals + 1) / 2) + 1) / 2);
    ao_to_smo_ptr->ao_to_mo_transform_2e(mo_coff, e_rep_mat, e_rep_mo);

    //translate MOs to spim MOs using the compact version
    ao_to_smo_ptr->mo_to_spin_transform_2e(e_rep_mo, e_rep_smo);
    
    e_s = EigenMatrix<double>::Zero(m_spin_mos, m_spin_mos);

    f_ae = EigenMatrix<double>::Zero(m_spin_mos - m_nelectrons, m_spin_mos - m_nelectrons);
    f_mi = EigenMatrix<double>::Zero(m_nelectrons, m_nelectrons);
    f_me = EigenMatrix<double>::Zero(m_nelectrons, m_spin_mos - m_nelectrons);
    
    d_ai = EigenMatrix<double>::Zero(m_spin_mos - m_nelectrons, m_nelectrons);
    d_abij = tensor4d1122<double>(m_spin_mos - m_nelectrons, m_nelectrons);
    w_mnij = tensor4d<double>(m_nelectrons);
    w_abef = tensor4d<double>(m_spin_mos - m_nelectrons);
    w_mbej = tensor4d1221<double>(m_nelectrons, m_spin_mos - m_nelectrons);

    ts = EigenMatrix<double>::Zero(m_spin_mos - m_nelectrons, m_nelectrons);
    ts_new = EigenMatrix<double>::Zero(m_spin_mos - m_nelectrons, m_nelectrons);
    td = tensor4d1122<double>(m_spin_mos - m_nelectrons, m_nelectrons);
    td_new = tensor4d1122<double>(m_spin_mos - m_nelectrons, m_nelectrons);

    for (Index i = 0; i < m_spin_mos; ++i)
        e_s(i, i) = mo_energies(i / 2, i / 2);

    // mp2 contribution in mo basis
    double sum_mp2 = 0.0;
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) reduction ( + : sum_mp2)
    #endif
    for (Index i = m_fcore / 2; i < m_nelectrons / 2; ++i)
        for (Index a = m_nelectrons / 2; a < m_num_orbitals; ++a)
            for (Index j = m_fcore / 2; j < m_nelectrons / 2; ++j)
                for (Index b = m_nelectrons / 2; b < m_num_orbitals; ++b) 
                {
                    Index iajb = hfscfmath::index_ijkl(i, a, j, b);
                    Index ibja = hfscfmath::index_ijkl(i, b, j, a);
                    sum_mp2 += e_rep_mo(iajb) * (2 * e_rep_mo(iajb) - e_rep_mo(ibja)) /
                    (mo_energies(i, i) + mo_energies(j, j) - mo_energies(a, a) - mo_energies(b, b));
                }
    
    const Index offset = m_nelectrons;
    for (Index a = m_nelectrons; a < m_spin_mos; ++a)
        for (Index i = 0; i < m_nelectrons; ++i)
            d_ai(a - offset, i) = e_s(i, i) - e_s(a, a);

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (Index a = m_nelectrons; a < m_spin_mos; ++a)
        for (Index b = m_nelectrons; b < m_spin_mos; ++b)
            for (Index i = 0; i < m_nelectrons; ++i)
                for (Index j = 0; j < m_nelectrons; ++j)
                {
                    d_abij(a - offset, b - offset, i, j) = e_s(i, i) + e_s(j, j) - e_s(a, a) - e_s(b, b);
                    //td_new(a - offset, b - offset, i, j) = e_rep_smo.asymm(i, j, a, b) 
                    //                                      / d_abij(a - offset, b - offset, i, j); 
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

double POSTSCF::post_scf_ccsd::calc_ccsd_energy() const noexcept
{
    Index offset = m_nelectrons;

    double e_ccsd  = 0.0;
    #ifdef _OPENMP
    #pragma omp parallel for reduction ( + :e_ccsd)
    #endif
    for(Index i = m_fcore; i < m_nelectrons; ++i)
        for (Index a = m_nelectrons; a < m_spin_mos; ++a)
        {
            e_ccsd += e_s(i,  a) * ts(a - offset, i);
            for(Index j = m_fcore; j < m_nelectrons; ++j)
                for (Index b = m_nelectrons; b < m_spin_mos; ++b)
                    e_ccsd += 0.50 * e_rep_smo.asymm(i, j, a, b) * td(a - offset, b - offset, i, j) 
                            + e_rep_smo.asymm(i, j, a, b) * ts(a - offset, i) * ts(b - offset, j);
        }

    return 0.5 * e_ccsd;
}

void POSTSCF::post_scf_ccsd::update() noexcept
{
    const auto tau_t = [this](Index a, Index b, Index i, Index j) noexcept -> double // tau tilde
    {
        return (td(a, b, i, j) + 0.5 * (ts(a, i) * ts(b, j) - ts(b, i) * ts(a, j)));
    };

    const auto tau = [this](Index a, Index b, Index i, Index j) noexcept -> double // tau
    { 
        return (td(a, b, i, j) + ts(a, i) * ts(b, j) - ts(b, i) * ts(a, j)); 
    };

    // Stanton eqns
    f_ae.setZero();
    Index offset = m_nelectrons;

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (Index a = m_nelectrons; a < m_spin_mos; ++a)
        for (Index e = m_nelectrons; e < m_spin_mos; ++e)
            for (Index m = m_fcore; m < m_nelectrons; ++m)
            {
                f_ae(a - offset, e - offset) += -0.5 * e_s(m, e) * ts(a - offset, m);
                for(Index f = m_nelectrons; f < m_spin_mos; ++f)
                {  
                    f_ae(a - offset, e - offset) += ts(f - offset, m) * e_rep_smo.asymm(m, a, f, e); 
                    for (Index n = m_fcore; n < m_nelectrons; ++n)
                        f_ae(a - offset, e - offset) 
                        -= 0.5 * tau_t(a - offset, f - offset, m, n) * e_rep_smo.asymm(m, n, e, f);
                }
            }
    
    f_mi.setZero();

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for(Index m = m_fcore; m < m_nelectrons; ++m)
        for(Index i = m_fcore; i < m_nelectrons; ++i)
            for(Index e = m_nelectrons; e < m_spin_mos; ++e)
            {
                f_mi(m, i) += 0.5 * ts(e - offset, i) * e_s(m , e);
                for(Index n  = m_fcore; n < m_nelectrons; ++n)
                {
                    f_mi(m, i) += ts(e - offset, n) * e_rep_smo.asymm(m, n, i, e);
                    for(Index f = m_nelectrons; f < m_spin_mos; ++f)
                        f_mi(m, i) 
                        += 0.5 * tau_t(e - offset, f - offset, i, n) * e_rep_smo.asymm(m, n, e, f);
                }
            }

    f_me.setZero();

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (Index m = m_fcore; m < m_nelectrons; ++m)
        for (Index e = m_nelectrons; e < m_spin_mos; ++e)
        {
            f_me(m, e - offset) = e_s(m, e);
            for(Index n = m_fcore; n < m_nelectrons; ++n)
                for (Index f = m_nelectrons; f < m_spin_mos; ++f)
                    f_me(m, e - offset) += ts(f - offset, n) * e_rep_smo.asymm(m, n, e, f);
        }
    
    w_mnij.setZero();

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index m = m_fcore; m < m_nelectrons; ++m) // take advantage of symmetry note a, a, b, b terms are 0
        for (Index n = m_fcore; n < m; ++n)
            for (Index i = m_fcore; i < m_nelectrons; ++i)
                for (Index j = m_fcore; j < i; ++j)
                {   
                    if(fabs(e_rep_smo.asymm(m, n, i, j)) < HF_SETTINGS::hf_settings::get_integral_tol()) 
                        continue;

                    w_mnij(m, n, i, j) = e_rep_smo.asymm(m, n, i, j);
                    for (Index e = m_nelectrons; e < m_spin_mos; ++e)
                    {
                        w_mnij(m, n, i, j) 
                        += ts(e - offset, j) * e_rep_smo.asymm(m, n, i, e) - ts(e - offset, i) 
                                             * e_rep_smo.asymm(m, n, j, e);

                        for (Index f = m_nelectrons; f < m_spin_mos; ++f)
                            w_mnij(m, n, i, j) += 0.25 * tau(e - offset, f - offset, i, j) 
                                                * e_rep_smo.asymm(m, n, e, f);
                    }
                    // symmetry terms
                    w_mnij(n, m, i, j) = -w_mnij(m, n, i, j);
                    w_mnij(n, m, j, i) =  w_mnij(m, n, i, j);
                    w_mnij(m, n, j, i) = -w_mnij(m, n, i, j);
                }
    
    w_abef.setZero();
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index a = m_nelectrons; a < m_spin_mos; ++a) // take advantage of symmetry note a, a, b, b terms are 0
        for (Index b = m_nelectrons; b < a; ++b)
            for (Index e = m_nelectrons; e < m_spin_mos; ++e)
                for (Index f = m_nelectrons; f < e; ++f)
                {
                    if(fabs(e_rep_smo.asymm(a, b, e, f)) < HF_SETTINGS::hf_settings::get_integral_tol()) 
                        continue;

                    w_abef(a - offset, b - offset, e - offset, f - offset) = e_rep_smo.asymm(a, b, e, f);

                    for (Index m = 0; m < m_nelectrons; ++m)
                    {
                        w_abef(a - offset, b - offset, e - offset, f - offset) 
                        += -ts(b - offset, m) * e_rep_smo.asymm(a, m, e, f) 
                           +ts(a - offset, m) * e_rep_smo.asymm(b, m, e, f);
                        
                        for (Index n = 0; n < m_nelectrons; ++n)
                            w_abef(a - offset, b - offset, e - offset, f - offset) 
                            += 0.25 * tau(a - offset, b - offset, m, n) 
                                    * e_rep_smo.asymm(m, n, e, f);
                    }
                    // symmetry terms
                    w_abef(a - offset, b - offset, f - offset, e - offset) 
                    = -w_abef(a - offset, b - offset, e - offset, f - offset);
                    w_abef(b - offset, a - offset, f - offset, e - offset) 
                    =  w_abef(b - offset, a - offset, e - offset, f - offset);
                    w_abef(b - offset, a - offset, e - offset, f - offset) 
                    = -w_abef(a - offset, b - offset, e - offset, f - offset);
                }

    w_mbej.setZero();
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index m = m_fcore; m < m_nelectrons; ++m) // skip symmetry due to dimension
        for (Index b = m_nelectrons; b < m_spin_mos; ++b)
            for (Index e = m_nelectrons; e < m_spin_mos; ++e)
                for (Index j = m_fcore; j < m_nelectrons; ++j)
                {
                    if(fabs(e_rep_smo.asymm(m, b, e, j)) < HF_SETTINGS::hf_settings::get_integral_tol()) 
                        continue;
                    
                    w_mbej(m, b - offset, e - offset, j) = e_rep_smo.asymm(m, b, e, j);
                    for (Index f = m_nelectrons; f < m_spin_mos; ++f)
                        w_mbej(m, b - offset, e - offset, j) += ts(f - offset, j) * e_rep_smo.asymm(m, b, e, f);
                    
                    for (Index n = m_fcore; n < m_nelectrons; ++n)
                    {
                        w_mbej(m, b - offset, e - offset, j) += -ts(b - offset, n) * e_rep_smo.asymm(m, n, e, j);
                        for (Index f = m_nelectrons; f < m_spin_mos; ++f)
                            w_mbej(m, b - offset, e - offset, j) 
                            += -(0.5 * td(f - offset, b - offset, j, n) 
                               + ts(f - offset, j) * ts(b - offset, n)) * e_rep_smo.asymm(m, n, e, f);
                    }
                }
}

void POSTSCF::post_scf_ccsd::T1() noexcept
{
    ts_new.setZero();
    Index offset = m_nelectrons;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index a = m_nelectrons; a < m_spin_mos; ++a)
        for (Index i = m_fcore; i < m_nelectrons; ++i)
        {
            ts_new(a - offset, i) = e_s(i, a);
            for (Index e = m_nelectrons; e < m_spin_mos; ++e)
                ts_new(a - offset, i) += ts(e - offset, i) * f_ae(a - offset, e - offset);

            for (Index m = m_fcore; m < m_nelectrons; ++m) 
            {
                ts_new(a - offset, i) += -ts(a - offset, m) * f_mi(m, i);
                for (Index e = m_nelectrons; e < m_spin_mos; ++e) 
                {
                    ts_new(a - offset, i) += td(a - offset, e - offset, i, m) * f_me(m, e - offset);
                    for (Index f = m_nelectrons; f < m_spin_mos; ++f)
                        ts_new(a - offset, i) -= 0.5 * td(e - offset, f - offset, i, m) * e_rep_smo.asymm(m, a, e, f);
                    
                    for (Index n = m_fcore; n < m_nelectrons; ++n)
                        ts_new(a - offset, i) -= 0.5 * td(a - offset, e - offset, m, n) * e_rep_smo.asymm(n, m, e, i);
                }
            }

            for (Index n = m_fcore; n < m_nelectrons; ++n)
                for (Index f = m_nelectrons; f < m_spin_mos; ++f)
                    ts_new(a - offset, i) -= ts(f - offset, n) * e_rep_smo.asymm(n, a, i, f);

            ts_new(a - offset, i) /= d_ai(a - offset, i);
        }
}

void POSTSCF::post_scf_ccsd::T2() noexcept
{
    td_new.setZero();
    const Index offset = m_nelectrons;

    const auto tau = [this](Index a, Index b, Index i, Index j) noexcept -> double // tau
    { 
        return (td(a, b, i, j) + ts(a, i) * ts(b, j) - ts(b, i) * ts(a, j)); 
    };

    // note terms of te form aa, bb are zero so we can skip these in the sum too
    // whilst taking advantage of permutation symmetry
    // was for (Index b = m_nelectrons; b < m_spin_mos; ++b)
    // was for (Index j = m_fcore; j < m_nelectrons; ++j) use symmetry
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index a = m_nelectrons; a < m_spin_mos; ++a)
        for (Index b = m_nelectrons; b < a; ++b) 
            for (Index i = m_fcore; i < m_nelectrons; ++i)
                for (Index j = m_fcore; j < i; ++j)
                {
                    if(fabs(e_rep_smo.asymm(i, j, a, b)) < HF_SETTINGS::hf_settings::get_integral_tol()) 
                        continue;

                    td_new(a - offset, b - offset, i, j) += e_rep_smo.asymm(i, j, a, b);
                    for (Index e = m_nelectrons; e < m_spin_mos; ++e) 
                    {
                        td_new(a - offset, b - offset, i, j) += 
                                              td(a - offset, e - offset, i, j) * f_ae(b - offset, e - offset) 
                                            - td(b - offset, e - offset, i, j) * f_ae(a - offset, e - offset);

                        for (Index m = m_fcore; m < m_nelectrons; ++m)
                            td_new(a - offset, b - offset, i, j) += 
                            -0.5 * td(a - offset, e - offset, i, j) * ts(b - offset, m) * f_me(m, e - offset)
                            +0.5 * td(b - offset, e - offset, i, j) * ts(a - offset, m) * f_me(m, e - offset);
                    }

                    for (Index m = m_fcore; m < m_nelectrons; ++m) 
                    {
                        td_new(a - offset, b - offset, i, j) += - td(a - offset, b - offset, i, m) * f_mi(m, j) 
                                                                + td(a - offset, b - offset, j, m) * f_mi(m, i);
                        for (Index e = m_nelectrons; e < m_spin_mos; ++e) 
                            td_new(a - offset, b - offset, i, j) += 
                            -0.5 * td(a - offset, b - offset, i, m) * ts(e - offset, j) * f_me(m, e - offset)
                            +0.5 * td(a - offset, b - offset, j, m) * ts(e - offset, i) * f_me(m, e - offset);
                    }

                    for (Index e = m_nelectrons; e < m_spin_mos; ++e) 
                    {
                        td_new(a - offset, b - offset, i, j) 
                        += ts(e - offset, i) * e_rep_smo.asymm(a, b, e, j) - ts(e - offset, j) 
                                             * e_rep_smo.asymm(a, b, e, i);

                        for (Index f = m_nelectrons; f < m_spin_mos; ++f)
                            td_new(a - offset, b - offset, i, j) 
                            += 0.5 * tau(e - offset, f - offset, i, j) 
                                   * w_abef(a - offset, b - offset, e - offset, f - offset);
                    }

                    for (Index m = m_fcore; m < m_nelectrons; ++m) 
                    {
                        td_new(a - offset, b - offset, i, j) 
                        += -ts(a - offset, m) * e_rep_smo.asymm(m, b, i, j) + ts(b - offset, m) 
                                              * e_rep_smo.asymm(m, a, i, j);

                        for (Index e = m_nelectrons; e < m_spin_mos; ++e) 
                        {
                            td_new(a - offset, b - offset, i, j) +=  
                            td(a - offset, e - offset, i, m) * w_mbej(m, b - offset, e - offset, j) 
                           -ts(e - offset, i) * ts(a - offset, m) * e_rep_smo.asymm(m, b, e, j);
                            
                            td_new(a - offset, b - offset, i, j) += 
                            -td(a - offset, e - offset, j, m) * w_mbej(m, b - offset, e - offset, i) 
                            +ts(e - offset, j) * ts(a - offset, m) * e_rep_smo.asymm(m, b, e, i);
                            
                            td_new(a - offset, b - offset, i, j) += 
                            -td(b - offset, e - offset, i, m) * w_mbej(m, a - offset, e - offset, j) 
                            +ts(e - offset, i) * ts(b - offset, m) * e_rep_smo.asymm(m, a, e, j);
                            
                            td_new(a - offset, b - offset, i, j) +=  
                            td(b - offset, e - offset, j, m) * w_mbej(m, a - offset, e - offset, i) 
                           -ts(e - offset, j) * ts(b - offset, m) * e_rep_smo.asymm(m, a, e, i);
                        }

                        for (Index n = m_fcore; n < m_nelectrons; ++n) 
                            td_new(a - offset, b - offset, i, j) += 
                            0.5 * tau(a - offset, b - offset, m, n) * w_mnij(m, n, i, j);
                    }

                    td_new(a - offset, b - offset, i, j) /= d_abij(a - offset, b - offset, i, j);

                    // Symmetry terms see revised loop structure at start ... gives decent speed up
                    td_new(a - offset, b - offset, j, i) =  -td_new(a - offset, b - offset, i, j);
                    td_new(b - offset, a - offset, j, i) =   td_new(a - offset, b - offset, i, j);
                    td_new(b - offset, a - offset, i, j) =  -td_new(a - offset, b - offset, i, j);
                }
}

void POSTSCF::post_scf_ccsd::ccsd_diis()
{
    int diis_range = HF_SETTINGS::hf_settings::get_ccsd_diis_size();

    // T1, + T2 size
    EigenVector<double> tmp = EigenVector<double>((m_spin_mos - m_nelectrons) * 
                                                  (m_spin_mos - m_nelectrons) * 
                                                   m_nelectrons * m_nelectrons     
                                                + (m_spin_mos  - m_nelectrons) * m_nelectrons);

    Index index = 0;
    for (Index p = 0; p < m_spin_mos - m_nelectrons; ++p)
        for (Index q = 0; q < m_spin_mos - m_nelectrons; ++q)
            for (Index r = 0; r < m_nelectrons; ++r)
                for (Index s = 0; s < m_nelectrons; ++s)
                {
                        tmp(index) = td_new(p, q, r, s) - td(p, q, r, s);
                        ++index;
                }

    index = 0;
    const Index offset = (m_spin_mos - m_nelectrons) * (m_spin_mos - m_nelectrons) * m_nelectrons * m_nelectrons;

    for(Index i = 0; i < m_spin_mos - m_nelectrons; ++i)
        for(Index j = 0; j < m_nelectrons; ++j)
        {
            tmp(offset + index) = ts_new(i, j) - ts(i, j);
            ++index;
        }
                                       
    double rms = tmp.cwiseAbs2().mean();
    ccsd_rms.emplace_back(std::sqrt(rms));

    if(0 == diis_range) // Only compute rms when not using DIIS 
        return;

    if(iteration >= diis_range) 
    {
        ccsd_errors.erase(ccsd_errors.begin());
        t2_list.erase(t2_list.begin());
        t1_list.erase(t1_list.begin());
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

    for (Index p = 0; p < m_spin_mos - m_nelectrons; ++p)
        for (Index q = 0; q < m_nelectrons; ++q) 
        {
            ts_new(p, q) = 0.0;
            for (Index k = 0; k < size; ++k) 
                ts_new(p, q) += coffs(k) * t1_list[k](p, q);
        }
    
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index p = 0; p < m_spin_mos - m_nelectrons; ++p)
        for (Index q = 0; q < m_spin_mos - m_nelectrons; ++q)
            for (Index r = 0; r < m_nelectrons; ++r)
                for (Index s = 0; s < m_nelectrons; ++s) 
                {
                    td_new(p, q, r, s) = 0.0;
                    for (int k = 0; k < size; ++k) 
                        td_new(p, q, r, s) += coffs(k) * t2_list[k](p, q, r, s);
                }
}

double POSTSCF::post_scf_ccsd::calc_perturbation_triples() const
{
    // Faster than the default one at a time approach.  
    // Use intermediate tensors to reduce multiplications, 
    // and exluding i = j j = k i = k zero terms, still expensive.
    // about 3x faster
    // https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2306

    const Index virt = m_spin_mos - m_nelectrons;
    tensor3d<double> tdis = tensor3d<double>(virt, virt, virt);
    tensor3d<double> tcon = tensor3d<double>(virt, virt, virt);

    EigenVector<double> eps = EigenVector<double>(m_spin_mos);
    for(Index i = 0; i < m_spin_mos; ++i)
        eps(i) = e_s(i, i);
    
    const Index offset = m_nelectrons;
    double td_; double tc_;
    double energy_triples = 0.0;
    //parallelisation of outerloop less efficient. tdis, tcon1, tcon2 already parallelised
    for(Index i = m_fcore; i < m_nelectrons; ++i)
        for(Index j = m_fcore; j < m_nelectrons; ++j)
            for(Index k = m_fcore; k < m_nelectrons; ++k)
            {
                if(i == j || j == k || i == k) continue;

                discon(tdis, tcon, i, j, k);
                #ifdef _OPENMP
                #pragma omp parallel for private(tc_, td_) reduction(+:energy_triples) schedule(dynamic)
                #endif
                for(Index a = m_nelectrons; a < m_spin_mos; ++a)
                    for(Index b = m_nelectrons; b < m_spin_mos; ++b)
                        for(Index c = m_nelectrons; c < m_spin_mos; ++c)
                        {
                            // disconnected contribution
                            td_ = +tdis(a - offset, b - offset, c - offset)
                                  -tdis(b - offset, a - offset, c - offset)
                                  -tdis(c - offset, b - offset, a - offset);
                            // type 1 & 2 connected contribution
                            tc_ = +tcon(a - offset, b - offset, c - offset)
                                  -tcon(b - offset, a - offset, c - offset)
                                  -tcon(c - offset, b - offset, a - offset);
                            
                            energy_triples += tc_ * (tc_ + td_) 
                                            / (eps(i) + eps(j) + eps(k) - eps(a) - eps(b) - eps(c));
                        }
            }

    return energy_triples / 36.0;
}

void POSTSCF::post_scf_ccsd::discon(tensor3d<double>& tdis, tensor3d<double>& tcon, 
                                    Index i, Index j, Index k) const noexcept //triplecon type 2
{
    const Index offset = m_nelectrons;
    tcon.setZero();
    // set all abc for ijk, jik, kji
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for(Index a = m_nelectrons; a < m_spin_mos; ++a)
        for(Index b = m_nelectrons; b < m_spin_mos; ++b)
            for(Index c = m_nelectrons; c < m_spin_mos; ++c)
            {                                            // omp note:
                if(a == b || b == c || a == c) continue; // more efficient check than doing this in main loop

                tdis(a - offset, b - offset, c - offset) // disconnected
                = ts(a - offset, i) * e_rep_smo.asymm(j, k, b, c)
                - ts(a - offset, j) * e_rep_smo.asymm(i, k, b, c) 
                - ts(a - offset, k) * e_rep_smo.asymm(j, i, b, c);

                for (Index n = m_nelectrons; n < m_spin_mos; ++n) // connected 1
                {
                    tcon(a - offset, b - offset, c - offset) 
                    += td(a - offset, n - offset, j, k) * e_rep_smo.asymm(n, i, b, c)
                     - td(a - offset, n - offset, i, k) * e_rep_smo.asymm(n, j, b, c)
                     - td(a - offset, n - offset, j, i) * e_rep_smo.asymm(n, k, b, c);
                }

                for (Index e = m_fcore; e < m_nelectrons; ++e) // connected 2
                {
                    tcon(a - offset, b - offset, c - offset) 
                    += -td(b - offset, c - offset, i, e) * e_rep_smo.asymm(e, a, j, k)
                       + td(b - offset, c - offset, j, e) * e_rep_smo.asymm(e, a, i, k)
                       + td(b - offset, c - offset, k, e) * e_rep_smo.asymm(e, a, j, i);    
                }
            }
}

using p = std::pair<double, std::string>;
void POSTSCF::post_scf_ccsd::calc_largest_T1T2()
{
    std::vector<p> t1_amps = 
    std::vector<p>(m_nelectrons * (m_spin_mos - m_nelectrons));

    size_t pos = 0;
    for (Index a = 0; a < m_spin_mos - m_nelectrons; ++a)
        for (Index i = m_fcore; i < m_nelectrons; ++i)
        {
            if(fabs(ts_new(a, i)) > 1.0e-10)
            {
                std::ostringstream osstr;
                osstr << std::right << std::setfill(' ') << std::setw(5) << (a / 2) * 2 // round to lowest
                      << std::right << std::setfill(' ') << std::setw(5) << (i / 2) * 2;
                t1_amps[pos] = std::make_pair(ts_new(a, i), osstr.str());
                ++pos;
            }
        }
    
    t1_amps.resize(pos);
    
    std::sort(t1_amps.data(), t1_amps.data() + t1_amps.size(), [](const p& t1a, const p& t1b) -> bool
    {
        return fabs(t1a.first) >= fabs(t1b.first);
    });

    for (size_t i = 0; i < 20; i += 2)  // every second biggest 10
    {                                   // or less if list is smaller
        if (i >= t1_amps.size() - 1) break;
        largest_t1.emplace_back(t1_amps[i]);
    }

    std::vector<p> t2_amps = std::vector<p>(m_nelectrons * (m_spin_mos - m_nelectrons) * 
                                            m_nelectrons * (m_spin_mos - m_nelectrons));

    pos = 0;
    for (Index a = 0; a < m_spin_mos - m_nelectrons; ++a)
        for (Index b = 0; b < m_spin_mos - m_nelectrons; ++b)
            for (Index i = m_fcore; i < m_nelectrons; ++i)
                for (Index j = m_fcore; j < m_nelectrons; ++j)
                {
                    if(fabs(td_new(a, b, i, j)) > 1.0E-10)
                    {
                        std::ostringstream osstr;
                        osstr << std::right << std::setfill(' ') << std::setw(5) << (a / 2) * 2 // round to lowest
                              << std::right << std::setfill(' ') << std::setw(5) << (i / 2) * 2
                              << std::right << std::setfill(' ') << std::setw(5) << (b / 2) * 2
                              << std::right << std::setfill(' ') << std::setw(5) << (j / 2) * 2;
                        t2_amps[pos] = std::make_pair(td_new(a, b, i, j), osstr.str());
                        ++pos;
                    }
                }

    t2_amps.resize(pos);
    std::sort(t2_amps.data(), t2_amps.data() + t2_amps.size(), [](const p& t2a, const p& t2b) -> bool 
    {
        return fabs(t2a.first) >= fabs(t2b.first);
    });

    for (size_t i = 0; i < 40; i += 4)  // every 4th biggest 10
    {                                   // or less if list is smaller
        if (i >= t2_amps.size() - 1) break;
        largest_t2.emplace_back(t2_amps[i]);
    }
}