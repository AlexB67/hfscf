#include "../basis/hfscf_trans_basis.hpp"
#include "../math/hfscf_tensor4d.hpp"
#include "../molecule/hfscf_constants.hpp"
#include "../pretty_print/hfscf_pretty_print.hpp"
#include "../settings/hfscf_settings.hpp"
#include "../mol_properties/hfscf_properties.hpp"
#include "hfscf_cis.hpp"
#include <Eigen/Eigenvalues>
#include <memory>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <tuple>
#include <sstream>


using tensor4dmath::symm4dTensor;
using tensor4dmath::index4;
using Eigen::Index;
using MOLEC_CONSTANTS::hartree_to_eV;
using HFCOUT::pretty_print_matrix;
using HF_SETTINGS::hf_settings;


void POSTSCF::post_scf_cis::calc_cis_energies(const std::shared_ptr<MOLEC::Molecule>& mol,
                                              const Eigen::Ref<const EigenMatrix<double> >& mo_energies,
                                              const Eigen::Ref<const EigenMatrix<double> >& mo_coff,
                                              const Eigen::Ref<const EigenVector<double> >& e_rep_mat)
{
    std::cout << "\n  ___Configuration_Interaction___\n";

    std::unique_ptr<BTRANS::basis_transform> ao_to_mo_ptr = std::make_unique<BTRANS::basis_transform>(m_num_orbitals);

    EigenVector<double> e_rep_mo;
    
    // translate AOs to MOs
    e_rep_mo = EigenVector<double>::Zero((m_num_orbitals * (m_num_orbitals + 1) / 2) *
                                        ((m_num_orbitals * (m_num_orbitals + 1) / 2) + 1) / 2);
    ao_to_mo_ptr->ao_to_mo_transform_2e(mo_coff, e_rep_mat, e_rep_mo);

    // In the MO basis
    // https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2312

    Index occ = m_nelectrons / 2;
    Index virt = m_num_orbitals - occ;
    EigenMatrix<double> Hs = EigenMatrix<double>::Zero(occ * virt, occ * virt);
    EigenMatrix<double> Ht = EigenMatrix<double>::Zero(occ * virt, occ * virt);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index i = 0; i < occ; ++i)
        for (Index a = occ; a < m_num_orbitals; ++a)
        {   // Note ia jb are  OMP private, since loop counters are private by default
            Index ia = a - occ + i * virt;
            for (Index j = 0; j < occ; ++j)
                for (Index b = occ; b < m_num_orbitals; ++b)
                {
                    Index jb = b - occ + j * virt;

                    if (ia > jb) continue; // symmetric, calculate triangular block only

                    Index ajib = index4(a, i, j, b); // Note: chemist notation.
                    Index ajbi = index4(a, b, j, i);

                    Hs(ia, jb)   =  2.0 * e_rep_mo(ajib) - e_rep_mo(ajbi);
                    Ht(ia, jb)   =  - e_rep_mo(ajbi);

                    if(i == j)
                    {
                        Hs(ia, jb) += mo_energies(a, b);
                        Ht(ia, jb) += mo_energies(a, b);
                    }

                    if(a == b)
                    {
                        Hs(ia, jb) -= mo_energies(i, j);
                        Ht(ia, jb) -= mo_energies(i, j);
                    }

                    Hs(jb, ia) = Hs(ia, jb);
                    Ht(jb, ia) = Ht(ia, jb);
                }
        }
    
    if (hf_settings::get_verbosity() > 1 && Ht.outerSize() < 100)
    {
        std::cout << '\n';
        std::cout << "  ****************************************\n";
        std::cout << "  *      CI Hamiltonian (singlets)       *\n";
        std::cout << "  *                                      *\n";
        std::cout << "  *                E / Eh                *\n";
        std::cout << "  ****************************************\n";
        pretty_print_matrix<double>(Hs);
        
        std::cout << '\n';
        std::cout << "  ****************************************\n";
        std::cout << "  *      CI Hamiltonian (triplets)       *\n";
        std::cout << "  *                                      *\n";
        std::cout << "  *                E / Eh                *\n";
        std::cout << "  ****************************************\n";
        pretty_print_matrix<double>(Ht);
    }

    Eigen::SelfAdjointEigenSolver<EigenMatrix<double> > solvers(Hs);
    EigenVector<double> s_energies = solvers.eigenvalues();
    EigenMatrix<double> s_eigenvecs = solvers.eigenvectors();
    //std::cout << "\n\n" << std::setprecision(6) << s_eigenvecs  <<  "\n\n";

    Eigen::SelfAdjointEigenSolver<EigenMatrix<double> > solvert(Ht, Eigen::EigenvaluesOnly);
    EigenVector<double> t_energies = solvert.eigenvalues();

    Index size = m_nelectrons * (m_spin_mos - m_nelectrons);
    EigenVector<std::tuple<double, double, std::string, std::string, std::string>> exc_data
    = EigenVector<std::tuple<double, double, std::string, std::string, std::string>>(size);

    // Oscillator strengths
    std::unique_ptr<MolProps::Molprops> molprops = std::make_unique<MolProps::Molprops>(mol);
    
    molprops->create_dipole_matrix(false);
    const Eigen::Ref<const EigenMatrix<double>> mux = molprops->get_dipole_x();
    const Eigen::Ref<const EigenMatrix<double>> muy = molprops->get_dipole_y();
    const Eigen::Ref<const EigenMatrix<double>> muz = molprops->get_dipole_z();

    std::vector<EigenMatrix<double>> dip_mo = std::vector<EigenMatrix<double>>(3);
    dip_mo[0] = (mo_coff.transpose() * mux * mo_coff).block(0, occ, occ, virt);
    dip_mo[1] = (mo_coff.transpose() * muy * mo_coff).block(0, occ, occ, virt);
    dip_mo[2] = (mo_coff.transpose() * muz * mo_coff).block(0, occ, occ, virt);

    EigenVector<double> oscillator_strength = EigenVector<double>::Zero(s_energies.size());
    EigenVector<std::tuple<double, Index, Index>> assignment_s = 
    EigenVector<std::tuple<double, Index, Index>>(s_energies.size());

    for (Index q = 0; q < s_energies.size(); ++q)
    {
        double trans_dipx = 0;
        double trans_dipy = 0;
        double trans_dipz = 0;
        Index idx = 0;
        for (Index i = 0; i < occ; ++i)
            for (Index a = 0; a < virt; ++a)
            {
                trans_dipx += dip_mo[0](i, a) * s_eigenvecs(a + virt * i, q);
                trans_dipy += dip_mo[1](i, a) * s_eigenvecs(a + virt * i, q);
                trans_dipz += dip_mo[2](i, a) * s_eigenvecs(a + virt * i, q);

                /// convert to spin basis
                std::get<0>(assignment_s[idx]) = 
                s_eigenvecs(a + virt * i, q) * s_eigenvecs(a + virt * i, q)  * 50.0;
                std::get<1>(assignment_s[idx]) = 2 * i;
                std::get<2>(assignment_s[idx]) = 2 * (a + occ);

                ++idx;
            }
        
        std::sort(assignment_s.data(), assignment_s.data() + assignment_s.size(), std::greater<>());

        std::get<3>(exc_data[q]) = "";

        std::stringstream data;
        data << std::right << std::setprecision(1) << std::fixed << std::setw(4) 
             << std::get<0>(assignment_s[0]) << "% " << std::setw(3) << std::right
             << std::get<1>(assignment_s[0]) << " -> " << std::setw(3) 
             << std::left << std::get<2>(assignment_s[0]);
        
        std::get<3>(exc_data[q]) = data.str();

        std::stringstream data2;
        data2 << std::right << std::setprecision(1) << std::fixed << std::setw(4) 
              << std::get<0>(assignment_s[0]) << "% " << std::setw(3) << std::right
              << std::get<1>(assignment_s[0]) + 1 << " -> " << std::setw(3) 
              << std::left << std::get<2>(assignment_s[0]) + 1;
        
        std::get<4>(exc_data[q]) = data2.str();

        Index cutoff = (assignment_s.size() > 3) ? 3 : assignment_s.size(); // could happen h2 sto-3g
        for(Index k = 1; k < cutoff; ++k)
        {
            if (std::get<0>(assignment_s[k]) > 10.0)
            {
                std::stringstream data3;
                data3 << std::right << std::setprecision(1) << std::fixed << std::setw(4) 
                    << std::get<0>(assignment_s[k]) 
                    << "% " << std::setw(3) << std::right
                    << std::get<1>(assignment_s[k]) << " -> " << std::setw(3) 
                    << std::left << std::get<2>(assignment_s[k]);
                
                std::get<3>(exc_data[q]) += data3.str();

                std::stringstream data4;
                data4 << std::right << std::setprecision(1) << std::fixed << std::setw(4) 
                    << std::get<0>(assignment_s[k]) 
                    << "% " << std::setw(3) << std::right
                    << std::get<1>(assignment_s[k]) + 1 << " -> " << std::setw(3) 
                    << std::left << std::get<2>(assignment_s[k]) + 1;
                
                std::get<4>(exc_data[q]) += data4.str();
            }
        }
        
        const double sum_of_sq = trans_dipx * trans_dipx + trans_dipy * trans_dipy + trans_dipz * trans_dipz;
        const double osc_strength = 2.0 * sum_of_sq * s_energies[q] / 3.0;
        // I think we need a factor of 2 here, because of the spatial - spin basis factoring
        oscillator_strength[q] = 2.0 * osc_strength;
    }

    // all singlets + x3 triplets, oscillator strengths singlets only

    Index offset = size / 4;
    for(Index q = 0; q < size / 4; ++q)
    {
        std::get<0>(exc_data[q]) = s_energies[q];
        std::get<1>(exc_data[q]) = oscillator_strength[q];
        std::get<2>(exc_data[q]) = "S";
        std::get<0>(exc_data[offset + 3 * q]) = t_energies[q];
        std::get<0>(exc_data[offset + 3 * q + 1]) = t_energies[q];
        std::get<0>(exc_data[offset + 3 * q + 2]) = t_energies[q];
        std::get<1>(exc_data[offset + 3 * q]) = 0;
        std::get<1>(exc_data[offset + 3 * q + 1]) = 0;
        std::get<1>(exc_data[offset + 3 * q + 2]) = 0;
        std::get<2>(exc_data[offset + 3 * q]) = "T";
        std::get<2>(exc_data[offset + 3 * q + 1]) = "T";
        std::get<2>(exc_data[offset + 3 * q + 2]) = "T";
    }

    std::sort(exc_data.data(), exc_data.data() + exc_data.size());

    std::cout << "\n**************************************************************************************\n";
    std::cout << "   CIS excitation energies and singlet oscillator strengths\n";
    std::cout << "   # Determinants = " << size << "\n";
    std::cout << "   Spatial representation Hamiltonian dimensions: ";
    std::cout << " " << occ * virt << "x" << occ * virt << "S, " << occ * virt << "x" << occ * virt << "T\n"; 
    std::cout << "   (Approx. total memory use " << (2 * e_rep_mo.size() * sizeof(double) / 1048576) + 
              // Include eigenvecs for singlets, mo eri tensor + eri ao basis eri tensor
              3 * s_energies.size() * s_energies.size() * sizeof(double) / 1048576 << "MB)\n\n";
    std::cout << "                                                       (> 10% max 8) \n";
    std::cout << "   #     E / Eh           E / eV        f       state  Singlet assignments\n";
    std::cout << "***************************************************************************************\n";

    for(Index i = 0; i < exc_data.size(); ++i)
    {
        std::cout << std::right  << std::setw(4) << i + 1;
        std::cout << std::right  << std::setprecision(10) << std::setw(17)
                  << std::get<0>(exc_data[i]);
        std::cout << std::right  << std::setprecision(10) << std::setw(17) 
                  << std::get<0>(exc_data[i]) * hartree_to_eV;
        std::cout << std::right  << std::setprecision(6) << std::setw(10) 
                  << std::get<1>(exc_data[i]);
        std::cout << std::right  << std::setw(5) << std::get<2>(exc_data[i]);

        if (std::get<2>(exc_data[i]) == "S")
        {
            std::cout << "  " << std::right << std::get<3>(exc_data[i]) << "\n";
            std::cout << std::setw(55) << std::left << " " << std::get<4>(exc_data[i]);
        }

        std::cout << "\n";

        if (199 == i)
        {
            std::cout << "\n Ouput truncated after a 200 items.\n";
            break; 
        }
    }   
}

void POSTSCF::post_scf_cis::calc_rpa_energies(const Eigen::Ref<const EigenMatrix<double> >& mo_energies,
                                              const Eigen::Ref<const EigenMatrix<double> >& mo_coff,
                                              const Eigen::Ref<const EigenVector<double> >& e_rep_mat)
{
    std::cout << "\n  ___Configuration_Interaction___\n";
    
    std::unique_ptr<BTRANS::basis_transform> ao_to_smo_ptr = std::make_unique<BTRANS::basis_transform>(m_num_orbitals);

    EigenVector<double> e_rep_mo;
    symm4dTensor<double> e_rep_smo;
    
    // translate AOs to MOs
    e_rep_mo = EigenVector<double>::Zero((m_num_orbitals * (m_num_orbitals + 1) / 2) *
                                        ((m_num_orbitals * (m_num_orbitals + 1) / 2) + 1) / 2);
    ao_to_smo_ptr->ao_to_mo_transform_2e(mo_coff, e_rep_mat, e_rep_mo);
    // translate MO to spin basis
    ao_to_smo_ptr->mo_to_spin_transform_2e(e_rep_mo, e_rep_smo);

    EigenMatrix<double> e_s = EigenMatrix<double>::Zero(m_spin_mos, m_spin_mos);

    for (Index i = 0; i < m_spin_mos; ++i)
        e_s(i, i) = mo_energies(i / 2, i / 2);

    EigenMatrix<double> HA = EigenMatrix<double>::Zero(m_nelectrons * (m_spin_mos - m_nelectrons),
                                                       m_nelectrons * (m_spin_mos - m_nelectrons));
    
    EigenMatrix<double> HB = EigenMatrix<double>::Zero(m_nelectrons * (m_spin_mos - m_nelectrons),
                                                       m_nelectrons * (m_spin_mos - m_nelectrons));

    Index virt = m_spin_mos - m_nelectrons;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index i = 0; i < m_nelectrons; ++i)
        for (Index a = m_nelectrons; a < m_spin_mos; ++a)
        {   // Note ia jb are  OMP private, since loop counters are private by default
            Index ia = a - m_nelectrons + i * virt;
            for (Index j = 0; j < m_nelectrons; ++j)
                for (Index b = m_nelectrons; b < m_spin_mos; ++b)
                {
                    Index jb = b - m_nelectrons + j * virt;
                    
                    if (ia > jb) continue; // symmetric calculate triangular block only

                    HA(ia, jb)   = e_rep_smo.asymm(a, j, i, b); // Use the assymetric propety of this tensor
                    HA(ia, jb)  += (i == j) * e_s(a, b);
                    HA(ia, jb)  -= (a == b) * e_s(i, j);
                    HB(ia, jb)   = e_rep_smo.asymm(a, b, i, j);

                    HA(jb, ia) = HA(ia, jb);
                    HB(jb, ia) = HB(ia, jb);
                }
        }

    const EigenMatrix<double> HAB = (HA + HB) * (HA - HB);

    if (hf_settings::get_verbosity() > 1 && HAB.outerSize() < 100)
    {
        std::cout << '\n';
        std::cout << "  ****************************************\n";
        std::cout << "  *         TDHF/RPA Hamiltonian         *\n";
        std::cout << "  *                                      *\n";
        std::cout << "  *                E / Eh                *\n";
        std::cout << "  ****************************************\n";
        pretty_print_matrix<double>(HAB);
    }

    Eigen::EigenSolver<EigenMatrix<double> > solver(HAB);
    Eigen::VectorXcd energies = solver.eigenvalues();

    EigenVector<double> exc_energies = energies.real();
    std::sort(exc_energies.data(), exc_energies.data() + exc_energies.size());

    std::cout << "\n********************************************\n";
    std::cout << "  RPA excitation energies\n";
    std::cout << "   #        E / Eh              E / eV\n";
    std::cout << "********************************************\n";

    for(Index i = 0; i < exc_energies.size(); ++i)
    {
        std::cout << std::right  << std::setw(4) << i + 1;
        std::cout << std::right  << std::setprecision(10) << std::setw(20) 
                  << std::sqrt(exc_energies[i]);
        std::cout << std::right  << std::setprecision(10) << std::setw(20) 
                  << std::sqrt(exc_energies[i]) * hartree_to_eV << '\n';
        
        if (199 == i)
        {
            std::cout << "\n Ouput truncated after a 200 items.\n";
            break;
        }
    }
    // TO DO oscillator strengths
}