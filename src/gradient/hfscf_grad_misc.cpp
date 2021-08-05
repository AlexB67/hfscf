#include "../molecule/hfscf_elements.hpp"
#include "../settings/hfscf_settings.hpp"
#include "../irc/cart_to_int.hpp"
#include "hfscf_gradient.hpp"
#include <iomanip>

using HF_SETTINGS::hf_settings;

void Mol::scf_gradient::print_gradient_info(const Eigen::Ref<EigenMatrix<double> >& gradNuc, 
                                            const Eigen::Ref<EigenMatrix<double> >& gradCoreHamil,
                                            const Eigen::Ref<EigenMatrix<double> >& gradOvlap,
                                            const Eigen::Ref<EigenMatrix<double> >& gradCoulomb,
                                            const Eigen::Ref<EigenMatrix<double> >& gradExchange,
                                            const Eigen::Ref<EigenMatrix<double> >& gradtotal) const
{
    const auto print_grad = [&](const Eigen::Ref<const EigenMatrix<double> >& grad_mat)
    {
        const std::vector<int>& charge = m_mol->get_z_values();
        const int natoms = static_cast<int>(m_mol->get_atoms().size());

        for(int j = 0; j < natoms; ++j)
        {
            std::cout << " " << std::setw(2) << j + 1 << "  "    << ELEMENTDATA::atom_names[charge[j] - 1];
            std::cout << std::setprecision(12) << std::setw(22) << grad_mat(j , 0);
            std::cout << std::setprecision(12) << std::setw(22) << grad_mat(j , 1);
            std::cout << std::setprecision(12) << std::setw(22) << grad_mat(j , 2) << '\n';
        }
    };
    std::cout << "\n  ___Gradient___\n\n";
    std::cout << "*****************************************************************************\n";
    std::cout << "                            Nuclear gradients:\n\n";
    std::cout << "  #  atom      dE/dX / (Eh a0^-1)    dE/dY / (Eh a0^-1)    dE/dZ / (Eh a0^-1)\n";
    std::cout << "*****************************************************************************\n";
    print_grad(gradNuc);
        
    std::cout << '\n';
    std::cout << "*****************************************************************************\n";
    std::cout << "                       Core Hamiltonian SCF gradients:\n\n";
    std::cout << "  #  atom      dE/dX / (Eh a0^-1)    dE/dY / (Eh a0^-1)    dE/dZ / (Eh a0^-1)\n";
    std::cout << "*****************************************************************************\n";
    print_grad(gradCoreHamil);

    std::cout << '\n';
    std::cout << "*****************************************************************************\n";
    std::cout << "                           Overlap SCF gradients:\n\n";
    std::cout << "  #  atom      dE/dX / (Eh a0^-1)    dE/dY / (Eh a0^-1)    dE/dZ / (Eh a0^-1)\n";
    std::cout << "*****************************************************************************\n";
    print_grad(gradOvlap);

    std::cout << '\n';
    std::cout << "*****************************************************************************\n";
    std::cout << "                           Coulomb SCF gradients:\n\n";
    std::cout << "  #  atom      dE/dX / (Eh a0^-1)    dE/dY / (Eh a0^-1)    dE/dZ / (Eh a0^-1)\n";
    std::cout << "*****************************************************************************\n";
    print_grad(gradCoulomb);

    std::cout << '\n';
    std::cout << "*****************************************************************************\n";
    std::cout << "                           Exchange SCF gradients:\n\n";
    std::cout << "  #  atom      dE/dX / (Eh a0^-1)    dE/dY / (Eh a0^-1)    dE/dZ / (Eh a0^-1)\n";
    std::cout << "*****************************************************************************\n";
    print_grad(gradExchange);

    std::cout << '\n';
    std::cout << "*****************************************************************************\n";
    std::cout << "                              Analytic SCF gradients:\n\n";
    std::cout << "  #  atom      dE/dX / (Eh a0^-1)    dE/dY / (Eh a0^-1)    dE/dZ / (Eh a0^-1)\n";
    std::cout << "*****************************************************************************\n";
    print_grad(gradtotal);
}

void Mol::scf_gradient::print_gradient_info_mp2(const Eigen::Ref<EigenMatrix<double> >& gradNuc, 
                                                const Eigen::Ref<EigenMatrix<double> >& gradCoreHamil,
                                                const Eigen::Ref<EigenMatrix<double> >& gradOvlap,
                                                const Eigen::Ref<EigenMatrix<double> >& gradTEI,
                                                const Eigen::Ref<EigenMatrix<double> >& gradtotal) const
{
    const auto print_grad = [&](const Eigen::Ref<const EigenMatrix<double> >& grad_mat)
    {
        const std::vector<int>& charge = m_mol->get_z_values();
        const int natoms = static_cast<int>(m_mol->get_atoms().size());

        for(int j = 0; j < natoms; ++j)
        {
            std::cout << " " << std::setw(2) << j + 1 << "  "    << ELEMENTDATA::atom_names[charge[j] - 1];
            std::cout << std::setprecision(12) << std::setw(22) << grad_mat(j , 0);
            std::cout << std::setprecision(12) << std::setw(22) << grad_mat(j , 1);
            std::cout << std::setprecision(12) << std::setw(22) << grad_mat(j , 2) << '\n';
        }
    };

    std::cout << "\n  ___Gradient___\n\n";
    std::cout << "*****************************************************************************\n";
    std::cout << "                            Nuclear gradients:\n";
    std::cout << "  #  atom      dE/dX / (Eh a0^-1)    dE/dY / (Eh a0^-1)    dE/dZ / (Eh a0^-1)\n";
    std::cout << "*****************************************************************************\n";
    print_grad(gradNuc);
        
    std::cout << '\n';
    std::cout << "*****************************************************************************\n";
    std::cout << "                       Core Hamiltonian MP2 gradients:\n";
    std::cout << "  #  atom      dE/dX / (Eh a0^-1)    dE/dY / (Eh a0^-1)    dE/dZ / (Eh a0^-1)\n";
    std::cout << "*****************************************************************************\n";
    print_grad(gradCoreHamil);

    std::cout << '\n';
    std::cout << "*****************************************************************************\n";
    std::cout << "                           Overlap MP2 gradients:\n";
    std::cout << "  #  atom      dE/dX / (Eh a0^-1)    dE/dY / (Eh a0^-1)    dE/dZ / (Eh a0^-1)\n";
    std::cout << "*****************************************************************************\n";
    print_grad(gradOvlap);

    std::cout << '\n';
    std::cout << "*****************************************************************************\n";
    std::cout << "                           TEI MP2 gradients:\n";
    std::cout << "  #  atom      dE/dX / (Eh a0^-1)    dE/dY / (Eh a0^-1)    dE/dZ / (Eh a0^-1)\n";
    std::cout << "*****************************************************************************\n";
    print_grad(gradTEI);

    std::cout << '\n';
    std::cout << "*****************************************************************************\n";
    std::cout << "                              Analytic MP2 gradients:\n\n";
    std::cout << "  #  atom      dE/dX / (Eh a0^-1)    dE/dY / (Eh a0^-1)    dE/dZ / (Eh a0^-1)\n";
    std::cout << "*****************************************************************************\n";
    print_grad(gradtotal);
}

bool Mol::scf_gradient::check_geom_opt(const std::vector<int>& zval, int natoms)
{
    // sanity check should never happen

    if (hf_settings::get_geom_opt_algorithm() != "CGSD" &&
        hf_settings::get_geom_opt_algorithm() != "RFO")
    {
        std::cout << "\n\n  Error: Invalid geometry optimization method encountered.\n";
         std::cout << "  Value: " << hf_settings::get_geom_opt_algorithm() << "\n\n";
        exit(EXIT_FAILURE);
    }

    // const double tol = std::fabs<double>(gradient.maxCoeff());
    double geom_opt_tol = geom_opt_tol_low;

    if("VERYHIGH" == hf_settings::get_geom_opt_tol())
        geom_opt_tol = geom_opt_tol_very_high;
    else if("HIGH" == hf_settings::get_geom_opt_tol())
        geom_opt_tol = geom_opt_tol_high;
    else if ("MEDIUM" == hf_settings::get_geom_opt_tol())
        geom_opt_tol = geom_opt_tol_med;

    double tol = gradient.maxCoeff();
    if(fabs(gradient.minCoeff()) > tol) tol = fabs(gradient.minCoeff());

    if(tol > geom_opt_tol)
    {   
        if(hf_settings::get_geom_opt_algorithm() == "RFO")
        {
            std::unique_ptr<CART_INT::Cart_int> intco = std::make_unique<CART_INT::Cart_int>(m_mol);
            intco->do_internal_coord_analysis();
            intco->print_bonding_info(false, "\n  Current internal\n (redundant) coordinates: ");
            intco->rfo_step(gradient, irc_grad, irc_coords, irc_hessian);
            intco->do_internal_coord_analysis();
            intco->print_bonding_info(false, "\n  New internal\n  (redundant) coordinates: ");
            
        }
        else if (hf_settings::get_geom_opt_algorithm() == "CGSD")
        {
            const double stepsize = hf_settings::get_geom_opt_stepsize();
            std::unique_ptr<CART_INT::Cart_int> intco = std::make_unique<CART_INT::Cart_int>(m_mol);
            intco->do_internal_coord_analysis();
            intco->print_bonding_info(false, "\n  Current internal coordinates\n");
            irc_grad = intco->get_irc_gradient(gradient);
            EigenVector<double> ics = intco->get_cartesian_to_irc();

            std::cout << "\n**************************\n";
            std::cout << "  New IRC Gradients\n";
            std::cout << "**************************\n";

            for(int i = 0; i < irc_grad.size(); ++i)
                std::cout << std::setw(18) << std::right << std::setprecision(12) << irc_grad(i) << '\n';

            std::cout << "\n**************************\n";
            std::cout << "  New Internal coordinates\n";
            std::cout << "**************************\n";

            for(int i = 0; i < ics.size(); ++i)
                std::cout << std::setw(18) << std::right << std::setprecision(12) << ics(i) << '\n';
            
            intco->do_internal_coord_analysis();
            intco->print_bonding_info(false, "\n  New internal coordinates\n");

            EigenVector<double> del_ics;
            if(!gradient_previous.outerSize())
            {
                del_ics = -stepsize * irc_grad;
            }
            else
            {
                const double gamma = irc_grad.squaredNorm() / irc_grad_previous.squaredNorm();	
                del_ics = -stepsize * (irc_grad + gamma * irc_grad_previous);
            }
        
            intco->irc_to_cartesian(ics, del_ics);

            // if(!gradient_previous.outerSize()) // old cartesian .. perhaps reinstate as an option
            //      m_mol->update_geom(gradient, stepsize);
            // else
            //     m_mol->update_geom(gradient, gradient_previous, stepsize);

            //gradient_previous = gradient;
            irc_grad_previous = irc_grad;
        }
    }
    else 
    {
        m_mol->update_geom(gradient, 0.0, hf_settings::get_symmetrize_geom());
        
        std::cout << "\n  Geometry optimisation complete. Final geometry\n";
        
        if (hf_settings::get_symmetrize_geom()) 
            std::cout << "  Symmetrized: Yes\n";
        else 
            std::cout << "  Symmetrized: No\n";

        std::cout << "  COM Cartesian coordinates, aligned.\n";
        std::cout << "***********************************************************************\n";
        std::cout << "  Atom  X / bohr          Y / bohr          Z / bohr          Mass\n";
	    std::cout << "***********************************************************************\n";
    
        for(int i = 0; i < natoms; ++i)
        {
            std::cout << "  " << ELEMENTDATA::atom_names[zval[i] - 1];
            std::cout << std::right << std::fixed << std::setprecision(12) 
                      << std::setw(18) << m_mol->get_geom()(i, 0)
                      << std::right << std::fixed << std::setw(18) << m_mol->get_geom()(i, 1)
                      << std::right << std::fixed << std::setw(18) << m_mol->get_geom()(i, 2)
                      << std::right << std::fixed << std::setw(13) << std::setprecision(7) 
                      << ELEMENTDATA::masses[zval[i] - 1] << '\n';
        }

        return true;
    }

    return false;
}

EigenVector<bool> Mol::scf_gradient::get_atom_mask(const int atom) const
{
    bool is_pure = m_mol->use_pure_am();
    EigenVector<bool> mask;

    if (is_pure)
    {
        const std::vector<MOLEC::mask>& atom_smask = m_mol->get_atom_spherical_mask();
        const Index num_orbitals = m_mol->get_num_orbitals();
        mask = EigenVector<bool>(num_orbitals);

        for (int m = 0; m < num_orbitals; ++m)
            (m >= atom_smask[atom].mask_start && m <= atom_smask[atom].mask_end) 
            ? mask[m] = true : mask[m] = false;
    }
    else
    {
        const std::vector<MOLEC::mask>& atom_mask = m_mol->get_atom_mask();
        const Index num_cartorbs = m_mol->get_num_cart_orbitals(); // Cartesian basis
        mask = EigenVector<bool>(num_cartorbs);

        for (int m = 0; m < num_cartorbs; ++m)
            (m >= atom_mask[atom].mask_start && m <= atom_mask[atom].mask_end) 
            ? mask[m] = true : mask[m] = false;
    }
    
    return mask;
}
