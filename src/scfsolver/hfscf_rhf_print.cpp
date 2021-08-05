#include "hfscf_hf_solver.hpp"
#include "../settings/hfscf_settings.hpp"
#include "../pretty_print/hfscf_pretty_print.hpp"
#include "../molecule/hfscf_constants.hpp"

using HF_SETTINGS::hf_settings;
using HFCOUT::pretty_print_matrix;
using MOLEC_CONSTANTS::hartree_to_eV;

void Mol::hf_solver::print_mo_coefficients_matrix() const
{
    if(hf_settings::get_guess_type() == "SAD" && 0 == iteration) return;

    std::cout << "\n  ********************************\n";
    std::cout << "  *  MO coefficients             *\n";
    std::cout << "  *                              *\n";
    std::cout << "  *  C(u, v)                     *\n";
    std::cout << "  ********************************\n";

    if (use_sym)
        pretty_print_matrix<double>(vmo_coff);
    else
        pretty_print_matrix<double>(mo_coff);
}

void Mol::hf_solver::print_fock_matrix() const
{
    std::cout << "\n  ********************************\n";
	std::cout << "  *  Fock matrix                 *\n";
    std::cout << "  *                              *\n";
    
    if("RHF" == hf_settings::get_hf_type())
        std::cout << "  *  F(u, v) / Eh               *\n";
    else 
        std::cout << "  *  F_alpha(u, v) / Eh         *\n";
	
    std::cout << "  ********************************\n";

    if (use_sym)
        pretty_print_matrix<double>(vf_mat);
    else
        pretty_print_matrix<double>(f_mat);
}

void Mol::hf_solver::print_density_matrix() const
{
    std::cout << "\n  ********************************\n";
    std::cout << "  *  Density  matrix             *\n";
    std::cout << "  *                              *\n";
    
    if("RHF" == hf_settings::get_hf_type())
        std::cout << "  *  D(u, v)                     *\n";
    else 
        std::cout << "  *  D_alpha(u, v)               *\n";

    std::cout << "  ********************************\n";

    if (use_sym)
    {
        pretty_print_matrix<double>(vd_mat);

        if (pop_index.size())
        {
            std::cout << "\n  *******************************************";
            std::cout << "\n  *  Occupation numbers:                    *\n";
            std::cout << "  *******************************************\n";
        }

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
    }
    else
        pretty_print_matrix<double>(d_mat);
}

void Mol::hf_solver::print_mo_energies() const
{

    if (use_sym)
    {
        std::cout << "\n  *******************************************";
        std::cout << "\n  *  Final Occupation numbers:              *\n";
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

        std::cout << "\n  *********************************************************************************************\n";
        std::cout << "  *  Mo energies summary                                                                      *\n";
        std::cout << "  *                                                                                           *\n";
        std::cout << "  * #     E(i) / Eh     E(i) / eV   Type  sym    #     E(i) / Eh     E(i) / eV   Type  sym    *\n";
        std::cout << "  *********************************************************************************************\n";

        for (int j = 0; j < norbitals; j += 2)
        {
                std::cout << "  ";
                std::cout << std::right << std::setw(3) << j + 1;
                std::cout << std::right << std::fixed << std::setprecision(6) << std::setw(14) 
                        << mo_energies_sym[j].first;
                std::cout << std::right << std::fixed << std::setprecision(5) << std::setw(14) 
                        << mo_energies_sym[j].first * hartree_to_eV;
                std::string occ;
                (j < nelectrons / 2) ? occ = "O" : occ = "V";
                std::cout << std::setw(5) << occ << "    ";
                std::cout << std::left << std::setw(4) << mo_energies_sym[j].second;

                if (j + 1 == norbitals) 
                {
                    std::cout << "\n";
                    break;
                }
                
                std::cout << std::right << std::setw(4) << j + 2;
                std::cout << std::right << std::fixed << std::setprecision(6) << std::setw(14) 
                        << mo_energies_sym[j + 1].first;
                std::cout << std::right << std::fixed << std::setprecision(5) << std::setw(14) 
                        << mo_energies_sym[j + 1].first * hartree_to_eV;
                (j + 1 < nelectrons / 2) ? occ = "O" : occ = "V";
                std::cout << std::setw(5) << occ << "    ";
                std::cout << std::left << std::setw(7) << mo_energies_sym[j + 1].second << "\n";
        }
    }
    else
    {
        std::cout << "\n  *********************************************************************************\n";
        std::cout << "  *  Mo energies summary                                                          *\n";
        std::cout << "  *                                                                               *\n";
        std::cout << "  * #     E(i) / Eh     E(i) / eV   Type  #     E(i) / Eh     E(i) / eV   Type    *\n";
        std::cout << "  *********************************************************************************\n";

        for (int j = 0; j < norbitals; j += 2)
        {
            std::cout << "  ";
            std::cout << std::right << std::setw(3) << j + 1;
            std::cout << std::right << std::fixed << std::setprecision(6) << std::setw(14) 
                      << mo_energies(j, j);
            std::cout << std::right << std::fixed << std::setprecision(5) << std::setw(14) 
                      << mo_energies(j, j) * hartree_to_eV;
            std::string occ;
            (j < nelectrons / 2) ? occ = "O" : occ = "V";
            std::cout << std::setw(5) << occ;

            if (j + 1 == norbitals) 
            {
                std::cout << "\n";
                break;
            }

            std::cout << std::right << std::setw(5) << j + 2;
            std::cout << std::right << std::fixed << std::setprecision(6) << std::setw(14) 
                      << mo_energies(j + 1, j + 1);
            std::cout << std::right << std::fixed << std::setprecision(5) << std::setw(14) 
                      << mo_energies(j + 1, j + 1) * hartree_to_eV;
            (j + 1 < nelectrons / 2) ? occ = "O" : occ = "V";
            std::cout << std::setw(5) << occ << "\n";
        }
    }

    std::cout << "\n*********************************\n";
    std::cout << "  Ionisation potential data";
    std::cout << "\n*********************************\n";
    std::cout << "  Koopmans' 1st IP / eV = " << std::setw(7) << std::setprecision(4)
              << - mo_energies(nelectrons / 2 - 1, nelectrons / 2 - 1) * 27.2114;
    std::cout << "\n  Koopmans' 1st IP / Eh = " << std::setw(7) << std::setprecision(5)
              << - mo_energies(nelectrons / 2 - 1, nelectrons / 2 - 1);
    std::cout << "\n*********************************\n";
}

void Mol::hf_solver::print_scf_energies() const
{
    std::cout << "\n\n*****************************************************************************************************\n";
    std::cout << "  Iteration   E(e) / Eh             Etot / Eh             dE(e) / Eh           dRMS             DIIS\n";
    std::cout << "*****************************************************************************************************\n";

    size_t start = 0; // Reduce printing when optimising geometry
    if(hf_settings::get_geom_opt().length() && scf_energies.size() > 3) start = scf_energies.size() - 3;

    for(size_t i = start; i < scf_energies.size(); ++i)
    {
        if (0 == i)
        {
            std::cout << "  ";
            std::cout << std::setw(5) << std::left << i << std::right << std::fixed 
            << std::setprecision(12) << std::setw(22) << scf_energies[0];
            std::cout << std::right << std::fixed << std::setprecision(12) << std::setw(22) 
            << scf_energies[0] + molecule->get_enuc() << '\n';
        }
        else
        {
            // rms values offset by one (a bit clumsy) since they start at iteration 1 ..
            std::cout << "  ";
            std::cout << std::setw(5) << std::left << i << std::right << 
            std::fixed << std::setprecision(12) << std::setw(22) << scf_energies[i] << 
            std::fixed << std::setprecision(12) << std::setw(22) << scf_energies[i] + molecule->get_enuc() << 
            std::right << std::fixed << std::setprecision(12) << std::setw(21) << 
            scf_energies[i] - scf_energies[i - 1] << 
            std::right << std::fixed << std::setprecision(12)
            << std::setw(21) << diis_ptr->get_rms_values_ref()[i - 1];

            if (i <= 1)
                std::cout << "   N";
            else if( i > 1 && 0 == hf_settings::get_diis_size())
                std::cout << "   N";
            else
            {
                if (i > static_cast<size_t>(so_iterstart))
                    std::cout << "   SO";
                else
                    std::cout << "   Y";
            }
                 
            std::cout << '\n';
        }  
    }
}