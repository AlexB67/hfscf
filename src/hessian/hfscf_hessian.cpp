#include "../settings/hfscf_settings.hpp"
#include "../gradient/hfscf_gradient.hpp"
#include "../molecule/hfscf_elements.hpp"
#include "../pretty_print/hfscf_pretty_print.hpp"
#include "hfscf_hessian.hpp"
#include "hfscf_freq.hpp"
#include <iostream>

using HFCOUT::pretty_print_matrix;
using HF_SETTINGS::hf_settings;
using Eigen::Index;

void Mol::scf_hessian::calc_scf_hessian_rhf(const double E_electronic)
{
    std::cout << "\n  ___Hessian___\n";

    const Index num_cartorbs = m_mol->get_num_cart_orbitals(); // cartesian basis
    const Index num_orbitals = m_mol->get_num_orbitals(); // spherical basis
    const int natoms = static_cast<int>(m_mol->get_atoms().size());
    const auto& atom_mask = m_mol->get_atom_mask();
    const auto& atom_smask = m_mol->get_atom_spherical_mask();

    mask = std::vector<std::vector<bool>>(natoms, std::vector<bool>(num_cartorbs));

    for(int atom = 0; atom < natoms; ++atom)
        for(int m = 0; m < num_cartorbs; ++m)
            (m >= atom_mask[atom].mask_start && m <= atom_mask[atom].mask_end) 
                ? mask[atom][m] = true : mask[atom][m] = false;
    
    smask = std::vector<std::vector<bool>>(natoms, std::vector<bool>(num_orbitals));

    for(int atom = 0; atom < natoms; ++atom)
        for(int m = 0; m < num_orbitals; ++m)
            (m >= atom_smask[atom].mask_start && m <= atom_smask[atom].mask_end) 
                ? smask[atom][m] = true : smask[atom][m] = false;
    
    if (hf_settings::get_verbosity() > 2)
        std::cout << "\n\n  Computing 1 electron 2nd dervative ints.\n";

    calc_scf_hessian_overlap();
    calc_scf_hessian_kinetic();
    calc_scf_hessian_nuclear();
    calc_scf_hessian_pot();

    if (hf_settings::get_verbosity() > 2)
        std::cout << "  Computing 2 electron 2nd dervative eri ints.\n";

    calc_scf_hessian_tei();

    std::unique_ptr<Mol::scf_gradient> grad_ptr = std::make_unique<Mol::scf_gradient>(m_mol);

    if (hf_settings::get_verbosity() > 2)
        std::cout << "  Computing gradient/response terms.\n";

    grad_ptr->calc_scf_hessian_response_rhf(mo_coff, mo_energies, eri);
    m_hessianResp = grad_ptr->get_response_rhessian_ref();

    grad_ptr->dip_derivs(mo_coff, 0);
    const Eigen::Ref<const EigenMatrix<double>> dipderiv = grad_ptr->get_dipderiv_ref();

    if (hf_settings::get_verbosity() > 1)
    {
        if (hf_settings::get_verbosity() > 4)
        {
            grad_ptr->print_cphf_coefficients(m_mol->get_num_electrons() / 2 , 
                                              num_orbitals - m_mol->get_num_electrons() / 2);
            grad_ptr->print_dip_derivs();
        }

        std::cout << '\n';
        std::cout << "  ****************************************\n";
        std::cout << "  *      Analytic Response Hessian:      *\n";
        std::cout << "  *                                      *\n";
        std::cout << "  *      d2E/dX1dX2 / (Eh a0^-2)         *\n";
        std::cout << "  ****************************************\n";
        pretty_print_matrix<double>(m_hessianResp);
    }

    m_hessian = m_hessianKin + m_hessianOvlap + m_hessianVn + m_hessianNuc + m_hessianC + m_hessianEx + m_hessianResp;

    FREQ::calc_frequencies(m_mol, m_hessian, dipderiv, E_electronic);
}

void Mol::scf_hessian::calc_scf_hessian_rhf_mp2(const double  E_electronic)
{
    if (hf_settings::get_verbosity() > 2)
        std::cout << "\n\n  Computing MP2 gradient and response terms.\n";

    std::unique_ptr<Mol::scf_gradient> grad_ptr = std::make_unique<Mol::scf_gradient>(m_mol);
    std::vector<bool> coords = std::vector<bool>(3, true);
    grad_ptr->calc_mp2_gradient_rhf(mo_energies, s_mat, d_mat, mo_coff, eri, coords, false, false, true);
    I = grad_ptr->get_I_ref();
    Ppq = grad_ptr->get_Ppq_ref();
    Ppqrs = grad_ptr->get_Ppqrs_ref();

    const Index num_cartorbs = m_mol->get_num_cart_orbitals(); // cartesian basis
    const Index num_orbitals = m_mol->get_num_orbitals(); // spherical basis
    const Index natoms = m_mol->get_atoms().size();
    const auto& atom_mask = m_mol->get_atom_mask();
    const auto& atom_smask = m_mol->get_atom_spherical_mask();
    
    mask = std::vector<std::vector<bool>>(natoms, std::vector<bool>(num_cartorbs));

    for(int atom = 0; atom < natoms; ++atom)
        for(int m = 0; m < num_cartorbs; ++m)
            (m >= atom_mask[atom].mask_start && m <= atom_mask[atom].mask_end) 
                ? mask[atom][m] = true : mask[atom][m] = false;
    
    smask = std::vector<std::vector<bool>>(natoms, std::vector<bool>(num_orbitals));

    for(int atom = 0; atom < natoms; ++atom)
        for(int m = 0; m < num_orbitals; ++m)
            (m >= atom_smask[atom].mask_start && m <= atom_smask[atom].mask_end) 
                ? smask[atom][m] = true : smask[atom][m] = false;
    
    std::cout << "\n  ___Hessian___\n";
    
    if (hf_settings::get_verbosity() > 2)
        std::cout << "\n  Computing 1 electron 2nd dervative ints.\n";

    calc_scf_hessian_overlap();
    calc_scf_hessian_kinetic();
    calc_scf_hessian_nuclear();
    calc_scf_hessian_pot();

    if (hf_settings::get_verbosity() > 2)
        std::cout << "  Computing 2 electron 2nd dervative eri ints.\n";

    calc_scf_hessian_tei_mp2();

    if (hf_settings::get_verbosity() > 2)
        std::cout << "  Computing response hessian.\n";
        
    grad_ptr->calc_scf_hessian_response_rhf_mp2(mo_energies);
    m_hessianResp = grad_ptr->get_response_rhessian_ref();

    grad_ptr->dip_derivs(mo_coff, m_mol->get_num_electrons() / 2); // SCF level only
    const Eigen::Ref<const EigenMatrix<double>> dipderiv = grad_ptr->get_dipderiv_ref();

    if (hf_settings::get_verbosity() > 1)
    {
        if (hf_settings::get_verbosity() > 4)
        {
            grad_ptr->print_cphf_coefficients(num_orbitals, num_orbitals);
            grad_ptr->print_dip_derivs();
        }
            
        std::cout << '\n';
        std::cout << "  ****************************************\n";
        std::cout << "  *   MP2 Analytic Response Hessian:     *\n";
        std::cout << "  *                                      *\n";
        std::cout << "  *      d2E/dX1dX2 / (Eh a0^-2)         *\n";
        std::cout << "  ****************************************\n";
        pretty_print_matrix<double>(m_hessianResp);
    }

    m_hessian = m_hessianKin + m_hessianOvlap + m_hessianVn + m_hessianNuc + m_hessianC + m_hessianResp;
 
    FREQ::calc_frequencies(m_mol, m_hessian, dipderiv, E_electronic);
}
