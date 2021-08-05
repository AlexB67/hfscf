#include "hfscf.hpp"
#include "../settings/hfscf_settings.hpp"
#include "../molecule/hfscf_elements.hpp"
#include "../pretty_print/hfscf_pretty_print.hpp"
#include <iomanip>
#include <iostream>

using HF_SETTINGS::hf_settings;
using HFCOUT::pretty_print_matrix;

void Mol::scf::print_gradient(const Eigen::Ref<const EigenMatrix<double> >& gradient)
{
    // Used by numerical gradients only
    const auto& zval = molecule->get_z_values();
    const int natoms = static_cast<int>(molecule->get_atoms().size());
    
    std::cout << '\n';
    std::cout << "*****************************************************************************\n";
    std::cout << "                             Numerical MP2 gradients:\n\n";
    std::cout << "  #  atom      dE/dX / (Eh a0^-1)    dE/dY / (Eh a0^-1)    dE/dZ / (Eh a0^-1)\n";
    std::cout << "*****************************************************************************\n";

    for(int j = 0; j < natoms; ++j)
    {
        std::cout << " " << std::setw(2) << j + 1 << "  "    << ELEMENTDATA::atom_names[zval[j] - 1];
        std::cout << std::setprecision(12) << std::setw(22) << gradient(j , 0);
        std::cout << std::setprecision(12) << std::setw(22) << gradient(j , 1);
        std::cout << std::setprecision(12) << std::setw(22) << gradient(j , 2) << '\n';
    }
}

void Mol::scf::print_rhf_energies(const double mp2_energy, const double mp3_energy, const double ccsd_energy, 
                                  const double cc_triples_energy, const int ccsd_iterations, 
                                  const std::vector<double>& ccsd_energies, const std::vector<double>& ccsd_deltas, 
                                  const std::vector<double>& ccsd_rms, 
                                  const std::vector<std::pair<double, std::string>>& largest_t1,
                                  const std::vector<std::pair<double, std::string>>& largest_t2) const
{
    rhf_ptr->print_scf_energies();
    std::string post_scf = hf_settings::get_post_scf_type();

    if(post_scf.substr(0, 4) == "CCSD" && !hf_settings::get_geom_opt().length())
    {
        std::cout << "\n  ___CCSD Solver___\n";
        std::cout << "\n*********************************************************************************\n";
        std::cout << "  Iteration    E(ccsd) / Eh          dE(e) / Eh            dRMS             DIIS\n";
        std::cout << "*********************************************************************************\n";

        for (size_t i = 0; i < ccsd_energies.size(); ++i)
        {
            std::cout << "  ";
            std::cout << std::setw(5) << std::left << i << std::right << 
            std::fixed << std::setprecision(12) << std::setw(22) << ccsd_energies[i];
            if(i > 0)
            { 
                std::cout << std::fixed << std::setprecision(12) << std::setw(22) << ccsd_deltas[i];
                std::cout << std::fixed << std::setprecision(12) << std::setw(22) << std::setw(22) << ccsd_rms[i];
            }

            if (0 != hf_settings::get_ccsd_diis_size() && i >= 3)
                std::cout << "   Y";
            else if(i > 0)
                std::cout << "   N";

            std::cout << '\n';
        }

        std::cout << "\n";
        std::cout << "  *******************************************************************\n";
        std::cout << "  *   Largest CCSD amplitudes.                                      *\n";
        std::cout << "  *   a    i    T1(a, i)        a     i    b    j    T2(a, i, b, j) *\n";
        std::cout << "  *******************************************************************\n";

        for(size_t i = 0; i < largest_t2.size(); ++i)
        {
            if(i < largest_t1.size())
            {
                std::cout << std::setw(12) << std::right << largest_t1[i].second
                        << std::setw(15) << std::right << std::setprecision(9) << largest_t1[i].first
                        << std::setw(22) << std::right << largest_t2[i].second
                        << std::setw(15) << std::right << std::setprecision(9) << largest_t2[i].first << "\n";
            }
            else
            {
                std::cout << std::setw(49) << std::right << largest_t2[i].second
                        << std::setw(15) << std::right << std::setprecision(9) << largest_t2[i].first << "\n";
            }
        }

        std::cout << "\n";
    }

    const double energy1e = rhf_ptr->get_one_electron_energy();
    const double energy2e = rhf_ptr->get_scf_energy() - energy1e;
    const double scf_energy = rhf_ptr->get_scf_energy();
    const int iteration = rhf_ptr->get_iterations();

    std::cout << '\n';
    std::cout << "********************************\n";
    std::cout << "  Energetics summary.\n";
    std::cout << "********************************\n";
    std::cout << "  E(NUC)  / Eh = ";
    std::cout << std::right << std::fixed << std::setprecision(8) << std::setw(15) << molecule->get_enuc() << '\n';
    std::cout << "  E(1E )  / Eh = ";
    std::cout << std::right << std::fixed << std::setprecision(8) << std::setw(15) << energy1e << '\n';
    std::cout << "  E(2E )  / Eh = ";
    std::cout << std::right << std::fixed << std::setprecision(8) << std::setw(15) << energy2e << '\n';
    std::cout << "  E(SCF)  / Eh = "; 
    std::cout << std::right << std::fixed << std::setprecision(8) << std::setw(15) << scf_energy 
                                                                                    + molecule->get_enuc() << '\n';
    if(hf_settings::get_geom_opt().length()) return;

    if (post_scf == "MP2")
    {
        std::cout << "  E(MP2)  / Eh = ";
        std::cout << std::right << std::fixed << std::setprecision(9) << std::setw(15) << mp2_energy << '\n';
        std::cout << "  E(TOTAL)/ Eh = ";
        std::cout << std::right << std::fixed << std::setprecision(8) 
                      << std::setw(15) << scf_energy + molecule->get_enuc() + mp2_energy << '\n';
    }
    else if (post_scf == "MP3")
    {
        std::cout << "  E(MP2)  / Eh = ";
        std::cout << std::right << std::fixed << std::setprecision(9) << std::setw(15) << mp2_energy << '\n';
        std::cout << "  E(MP3)  / Eh = ";
        std::cout << std::right << std::fixed << std::setprecision(9) << std::setw(15) 
                  << mp2_energy + mp3_energy << '\n';
        std::cout << "  E(TOTAL)/ Eh = ";
        std::cout << std::right << std::fixed << std::setprecision(8) << std::setw(15) << scf_energy 
                   + molecule->get_enuc() + mp2_energy + mp3_energy << '\n';
    }
    else if (post_scf.substr(0, 4) == "CCSD")
    {
        std::cout << "  E(MP2)  / Eh = ";
        std::cout << std::right << std::fixed << std::setprecision(9) 
                  << std::setw(15) << mp2_energy << '\n';
        std::cout << "  E(CCSD) / Eh = ";
        std::cout << std::right << std::fixed << std::setprecision(9) 
                  << std::setw(15) << ccsd_energy << '\n';
        
        if (post_scf == "CCSD(T)")
        {
            std::cout << "  E(PT)   / Eh = ";
            std::cout << std::right << std::fixed << std::setprecision(9) 
                      << std::setw(15) << cc_triples_energy << '\n';
        }

        std::cout << "  E(TOTAL)/ Eh = ";
        std::cout << std::right << std::fixed << std::setprecision(8) 
                  << std::setw(15) << scf_energy + molecule->get_enuc() 
                                   + ccsd_energy + cc_triples_energy << '\n'; // ccsd includes mp2 contribution
    }

    std::cout << "\n  SCF iterations = "; 
    (hf_settings::get_guess_type() == "SAD") ? std::cout << iteration - 1 : std::cout << iteration;
    std::cout << '\n';

    if(post_scf.substr(0, 4) == "CCSD" && hf_settings::get_hf_type() == "RHF")
        std::cout << " CCSD iterations = " << ccsd_iterations << '\n';
}

void Mol::scf::print_uhf_energies(const Eigen::Ref<const EigenVector<double>>& ump2_energies,
                                  const Eigen::Ref<const EigenVector<double>>& ump3_energies) const
{
    std::cout << "\n***********************************\n";
    std::cout << "  Spin contamination analysis\n";
    std::cout << "***********************************\n";
    std::cout << "  Pure ideal value = " << std::right << std::fixed << std::setprecision(9) << std::setw(13)
              << static_cast<double>(molecule->get_spin()) << '\n';
    std::cout << "  Calculated value = " << std::right << std::fixed << std::setprecision(9) << std::setw(13)
              << uhf_ptr->get_spin_contamination() << '\n';

    uhf_ptr->print_scf_energies();

    std::string post_scf = hf_settings::get_post_scf_type();

    const double energy1e = uhf_ptr->get_one_electron_energy();
    const double energy2e = uhf_ptr->get_scf_energy() - energy1e;
    const double scf_energy = uhf_ptr->get_scf_energy();
    const int iteration = uhf_ptr->get_iterations();

    std::cout << '\n';
    std::cout << "********************************\n";
    std::cout << "  Energetics summary.\n";
    std::cout << "********************************\n";
    std::cout << "  E(NUC)  / Eh = ";
    std::cout << std::right << std::fixed << std::setprecision(8) << std::setw(15) << molecule->get_enuc() << '\n';
    std::cout << "  E(1E )  / Eh = ";
    std::cout << std::right << std::fixed << std::setprecision(8) << std::setw(15) << energy1e << '\n';
    std::cout << "  E(2E )  / Eh = ";
    std::cout << std::right << std::fixed << std::setprecision(8) << std::setw(15) << energy2e << '\n';
    std::cout << "  E(SCF)  / Eh = "; 
    std::cout << std::right << std::fixed << std::setprecision(8) << std::setw(15) << scf_energy 
                                                                                    + molecule->get_enuc() << '\n';
    if(hf_settings::get_geom_opt().length()) return;

    if (post_scf == "MP2")
    {
        std::cout << "  E(MP2\u03B1\u03B1)/ Eh = ";
        std::cout << std::right << std::fixed << std::setprecision(9) << std::setw(15)
                  << ump2_energies[0] << '\n';
        std::cout << "  E(MP2\u03B2\u03B2)/ Eh = ";
        std::cout << std::right << std::fixed << std::setprecision(9) << std::setw(15)
                  << ump2_energies[1] << '\n';
        std::cout << "  E(MP2\u03B1\u03B2)/ Eh = ";
        std::cout << std::right << std::fixed << std::setprecision(9) << std::setw(15) << ump2_energies[2] << '\n';
        std::cout << "  E(MP2CE)/ Eh = ";
        std::cout << std::right << std::fixed << std::setprecision(9) << std::setw(15) << ump2_energies.sum() << '\n';
        std::cout << "  E(TOTAL)/ Eh = ";
        std::cout << std::right << std::fixed << std::setprecision(8) << std::setw(15)
                  << scf_energy + molecule->get_enuc() + ump2_energies.sum() << '\n';
    }
    else if (post_scf == "MP3")
    {   // MP2 part
        std::cout << "  E(MP2\u03B1\u03B1)/ Eh = ";
        std::cout << std::right << std::fixed << std::setprecision(9) << std::setw(15) << ump3_energies[0] << '\n';
        std::cout << "  E(MP2\u03B2\u03B2)/ Eh = ";
        std::cout << std::right << std::fixed << std::setprecision(9) << std::setw(15) << ump3_energies[1] << '\n';
        std::cout << "  E(MP2\u03B1\u03B2)/ Eh = ";
        std::cout << std::right << std::fixed << std::setprecision(9) << std::setw(15) << ump3_energies[2] << '\n';
        std::cout << "  E(MP2CE)/ Eh = ";
        std::cout << std::right << std::fixed << std::setprecision(9) << std::setw(15)
                  << ump3_energies[0] + ump3_energies[1] + ump3_energies[2] <<'\n';
        std::cout << "  E(MP2)  / Eh = ";
        std::cout << std::right << std::fixed << std::setprecision(8) << std::setw(15)
                  << ump3_energies[0] + ump3_energies[1] + + ump3_energies[2] 
                                      + scf_energy + molecule->get_enuc() << '\n';
        // MP3 part
        std::cout << "  E(MP3\u03B1\u03B1)/ Eh = ";
        std::cout << std::right << std::fixed << std::setprecision(9) << std::setw(15) 
                  << ump3_energies[3] << '\n';
        std::cout << "  E(MP3\u03B2\u03B2)/ Eh = ";
        std::cout << std::right << std::fixed << std::setprecision(9) << std::setw(15) 
                  << ump3_energies[4] << '\n';
        std::cout << "  E(MP3\u03B1\u03B2)/ Eh = ";
        std::cout << std::right << std::fixed << std::setprecision(9) << std::setw(15) 
                  << ump3_energies[5] << '\n';
        std::cout << "  E(MP3)  / Eh = ";
        std::cout << std::right << std::fixed << std::setprecision(9) << std::setw(15) 
                  << ump3_energies[3] + ump3_energies[4] + ump3_energies[5] << '\n';
        std::cout << "  E(MP3CE)/ Eh = ";
        std::cout << std::right << std::fixed << std::setprecision(9) << std::setw(15) << ump3_energies.sum() << '\n';
        std::cout << "  E(TOTAL)/ Eh = ";
        std::cout << std::right << std::fixed << std::setprecision(8) << std::setw(15)
                  << scf_energy + molecule->get_enuc() + ump3_energies.sum() << '\n';
    }

    std::cout << "\n  SCF iterations = "; 
    (hf_settings::get_guess_type() == "SAD") ? std::cout << iteration - 1 : std::cout << iteration;
    std::cout << '\n';
}
