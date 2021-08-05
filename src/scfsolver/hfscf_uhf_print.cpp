#include "hfscf_uhf_solver.hpp"
#include "../settings/hfscf_settings.hpp"
#include "../pretty_print/hfscf_pretty_print.hpp"
#include "../molecule/hfscf_constants.hpp"

using HF_SETTINGS::hf_settings;
using HFCOUT::pretty_print_matrix;
using MOLEC_CONSTANTS::hartree_to_eV;

void Mol::uhf_solver::print_mo_coefficients_matrix() const
{
    //if(hf_settings::get_guess_type() == "SAD" && 0 == iteration) return;

        std::cout << "\n  ********************************\n";
        std::cout << "  *  MO coefficients             *\n";
        std::cout << "  *                              *\n";
        std::cout << "  *  C_alpha(u, v)               *\n";
        std::cout << "  ********************************\n";

        if (use_sym)
            pretty_print_matrix<double>(vmo_coff);
        else
            pretty_print_matrix<double>(mo_coff); // norbitals, nalpha

        std::cout << "\n  ********************************\n";
        std::cout << "  *  MO coefficients             *\n";
        std::cout << "  *                              *\n";
        std::cout << "  *  C_beta(u, v)                *\n";
        std::cout << "  ********************************\n";

        if (use_sym)
            pretty_print_matrix<double>(vmo_coff_b);
        else
            pretty_print_matrix<double>(mo_coff_b); // norbitals, nalpha
}

void Mol::uhf_solver::print_fock_matrix() const
{
    hf_solver::print_fock_matrix(); // alpha

    std::cout << "\n  ********************************\n";
	std::cout << "  *  Fock matrix                 *\n";
    std::cout << "  *                              *\n";
    std::cout << "  *  F_beta(u, v) / Eh          *\n";
    std::cout << "  ********************************\n";

    if (use_sym)
        pretty_print_matrix<double>(vf_mat_b);
    else
        pretty_print_matrix<double>(f_mat_b);
}

void Mol::uhf_solver::print_density_matrix() const
{
    hf_solver::print_density_matrix(); // alpha

    std::cout << "\n  ********************************\n";
    std::cout << "  *  Density  matrix             *\n";
    std::cout << "  *                              *\n";
    std::cout << "  *  D_beta(u, v)                *\n";
    std::cout << "  ********************************\n";

    if (use_sym)
    {
        pretty_print_matrix<double>(vd_mat_b);

        std::cout << "\n  *******************************************";
        std::cout << "\n  *  Occupation numbers:                    *\n";
        std::cout << "  *******************************************\n";

        int line = 0;
        auto iter = molecule->get_symmetry_species().begin();

        for (const auto& p : pop_index_b)
        {
            ++line;
            std::cout << "  " << std::setw(9) << std::to_string(p) + "(" + *iter + ")";

            if ((line % 4) == 0) std::cout << "\n";
            ++iter;
        }

        std::cout << "\n";
    }
    else
        pretty_print_matrix<double>(d_mat_b);
}

void Mol::uhf_solver::print_mo_energies() const
{
    if (use_sym)
    {
        std::cout << "\n  *******************************************";
        std::cout << "\n  *  Final alpha Occupation numbers:        *\n";
        std::cout << "  *******************************************\n";

        int line = 0;
        auto iter = molecule->get_symmetry_species().begin();

        for (const auto& p : pop_index) 
        {
            ++line;
            std::cout << "  " << std::setw(9) << std::to_string(p) + "(" + *iter + ")";

            if ((line % 4) == 0) std::cout << "\n";
            ++iter;
        }

        std::cout << "\n";
        std::cout << "\n  *******************************************";
        std::cout << "\n  *  Final beta Occupation numbers:         *\n";
        std::cout << "  *******************************************\n";

        line = 0;
        iter = molecule->get_symmetry_species().begin();

        for (const auto& p : pop_index_b) 
        {
            ++line;
            std::cout << "  " << std::setw(9) << std::to_string(p) + "(" + *iter + ")";

            if ((line % 4) == 0) std::cout << "\n";
            ++iter;
        }

    std::cout << "\n\n";
    std::cout << "  ******************************************************************************************\n";
    std::cout << "  *  Mo energies                                                                           *\n"; 
    std::cout << "  *                                                                                        *\n";
    std::cout << "  * #     E(\u03B1) / Eh     E(\u03B1) / eV      Type  sym      E(\u03B2) / Eh     E(\u03B2) / eV    Type  sym *\n";
    std::cout << "  ******************************************************************************************\n";

        for (int j = 0; j < norbitals; ++j) 
        {
            std::cout << "  ";
            std::cout << std::right << std::setw(3) << j + 1;
            std::cout << std::right << std::fixed << std::setprecision(6) 
                    << std::setw(14) << mo_energies_sym[j].first;
            std::cout << std::right << std::fixed << std::setprecision(5) 
                    << std::setw(14) << mo_energies_sym[j].first * hartree_to_eV;
            
            std::string occ_alpha;
            (j < nalpha) ? occ_alpha = "O" : occ_alpha = "V";
            std::cout << std::setw(8) << occ_alpha << "    ";
            std::cout << std::left << std::setw(4) << mo_energies_sym[j].second;

            std::cout << std::right << std::fixed << std::setprecision(6) 
                    << std::setw(14) << mo_energies_sym_b[j].first;
            std::cout << std::right << std::fixed << std::setprecision(5) 
                    << std::setw(14) << mo_energies_sym_b[j].first * hartree_to_eV;
            std::string occ_beta;
            (j < nbeta) ? occ_beta = "O" : occ_beta = "V";
            std::cout << std::setw(6) << occ_beta << "    ";
            std::cout << std::left << std::setw(4) << mo_energies_sym_b[j].second << "\n";
        }
    }
    else
    {
    
    std::cout << "\n\n";
    std::cout << "  *******************************************************************************\n";
    std::cout << "  *  Mo energies                                                                *\n"; 
    std::cout << "  *                                                                             *\n";
    std::cout << "  * #     E(\u03B1) / Eh        E(\u03B1) / eV    Type   E(\u03B2) / Eh     E(\u03B2) / eV    Type  *\n";
    std::cout << "  *******************************************************************************\n";

        for (int j = 0; j < norbitals; ++j) 
        {
            std::cout << "  ";
            std::cout << std::right << std::setw(3) << j + 1;
            std::cout << std::right << std::fixed << std::setprecision(6) << std::setw(14) << mo_energies(j, j);
            std::cout << std::right << std::setw(3) << j + 1;
            std::cout << std::right << std::fixed << std::setprecision(5) << std::setw(14) 
                      << mo_energies(j, j) * hartree_to_eV;
            std::string occ_alpha;
            (j < nalpha) ? occ_alpha = "O" : occ_alpha = "V";
            std::cout << std::setw(6) << occ_alpha;
            std::cout << std::right << std::fixed << std::setprecision(6) << std::setw(14) << mo_energies_beta(j, j);
            std::cout << std::right << std::fixed << std::setprecision(5) << std::setw(14) 
                      << mo_energies_beta(j, j) * hartree_to_eV;
            std::string occ_beta;
            (j < nbeta) ? occ_beta = "O" : occ_beta = "V";
            std::cout << std::setw(6) << occ_beta << '\n';
        }
    }
}