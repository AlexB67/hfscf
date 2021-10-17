
#include "hfscf.hpp"
#include "../settings/hfscf_settings.hpp"
#include "../gradient/hfscf_gradient.hpp"
#include "../molecule/hfscf_elements.hpp"
#include "../irc/cart_to_int.hpp"
#include <iomanip>
#include <iostream>
#include <fstream>

using HF_SETTINGS::hf_settings;

void Mol::scf::geom_opt()
{
    const std::string& trajectory_file = hf_settings::get_geom_opt_trajectory_file();
    std::ifstream file(trajectory_file);
    // delete trajectory file if it exists
    if(file.good())
    {
        file.close();
        std::remove(trajectory_file.c_str());
    }

    bool is_rhf = true;
    if("UHF" == hf_settings::get_hf_type()) is_rhf = false;
    std::unique_ptr<Mol::scf_gradient> grad_ptr = std::make_unique<Mol::scf_gradient>(molecule);
    
    int geom_opt_iter = 0;
    bool use_previous_density = false;
    for(;;)
    {
        std::cout << "\n  Geometry optimization iteration: " << geom_opt_iter;
        if(use_previous_density) std::cout << "\n  Using the previous iteration density as SCF inital guess.";
        if(is_rhf)
        {
            rhf_ptr->init_data(false, use_previous_density);
            rhf_ptr->scf_run(false);
            rhf_ptr->print_scf_energies();
        }
        else
        {
            uhf_ptr->init_data(false, use_previous_density);
            uhf_ptr->scf_run(false);
            uhf_ptr->print_scf_energies();
        }

        bool converged = false;
        std::vector<bool> coords;
        if(is_rhf)
        {
            if ("MP2" == hf_settings::get_geom_opt())
                converged = grad_ptr->calc_mp2_gradient_rhf(rhf_ptr->get_mo_energies(), 
                rhf_ptr->get_overlap_matrix(), rhf_ptr->get_density_matrix(), rhf_ptr->get_mo_coef(), 
                rhf_ptr->get_repulsion_vector(), coords, true, true, false);
            else
                converged = grad_ptr->calc_scf_gradient_rhf(rhf_ptr->get_density_matrix(), 
                rhf_ptr->get_fock_matrix(), coords, true, true, false);
        }
        else
            converged = grad_ptr->calc_scf_gradient_uhf(uhf_ptr->get_density_matrix_alpha(),
            uhf_ptr->get_density_matrix_beta(), uhf_ptr->get_fock_matrix_alpha(), 
            uhf_ptr->get_fock_matrix_beta(), coords, true, true, false);
        
        if(converged)
        {
            hf_settings::set_geom_opt("");
            std::cout << "\n  Geometry optimization completed in "  << geom_opt_iter << " iterations.\n";
            std::cout << "  Entering SCF mode.\n\n";
            molecule->print_info(true);
            goto done;
        }
        else if(geom_opt_iter == hf_settings::get_max_geomopt_iterations())
        {
            std::clog << "\n!!! Reached a " << geom_opt_iter 
                      << " geometry optimisation iterations. Failed to converge !!!\n";
            exit(EXIT_FAILURE);
            
        }

        ++geom_opt_iter;
        use_previous_density = true;
    }

done:
    return;
}

void Mol::scf::geom_opt_numeric()
{
    if (hf_settings::get_use_symmetry())
    {
        std::clog << "\n  Error: Symmetry is unavailable for numeric geometry optimization.\n";
        std::clog << "         This mode is available with symmetry disabled only.\n";
        exit(EXIT_FAILURE);
    }

    const int natoms = static_cast<int>(molecule->get_atoms().size());
    int geom_opt_iter = 0;
    const double stepsize = hf_settings::get_geom_opt_stepsize();
    const auto& zval = molecule->get_z_values();
    
    double geom_opt_tol = geom_opt_tol_low;

    if("HIGH" == hf_settings::get_geom_opt_tol() || "VERYHIGH" == hf_settings::get_geom_opt_tol())
    {
        hf_settings::set_geom_opt_tol("MEDIUM"); // Numeric not useful for anything higher
        geom_opt_tol = geom_opt_tol_med;
    }
    else if ("MEDIUM" == hf_settings::get_geom_opt_tol())
        geom_opt_tol =  2.0 * geom_opt_tol_med;

    EigenMatrix<double> gradient = EigenMatrix<double>::Zero(natoms, 3);
    EigenMatrix<double> gradient_previous;
    std::vector<bool> coords;
    EigenVector<double> irc_grad_previous;
    EigenVector<double> irc_grad;
    EigenVector<double> irc_coords;
    EigenMatrix<double> irc_hessian;

    for(;;)
    {
        std::cout << "\n  Geometry optimization iteration: " << geom_opt_iter; 
        calc_numeric_gradient(gradient, coords, true, false);
        uhf_ptr->print_scf_energies();

        const double tol = std::fabs<double>(gradient.maxCoeff());

        if(geom_opt_iter == hf_settings::get_max_geomopt_iterations())
        {
            std::clog << "\n!!! Reached a " << geom_opt_iter 
                      << " geometry optimisation iterations. Failed to converge !!!\n";
            exit(1);
            
        }

        if(tol > geom_opt_tol && hf_settings::get_geom_opt_algorithm() == "CGSD")
        {
            std::unique_ptr<CART_INT::Cart_int> intco = std::make_unique<CART_INT::Cart_int>(molecule);
            intco->do_internal_coord_analysis();
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
            
            intco->print_bonding_info();

            EigenVector<double> del_ics;
            if(0 == geom_opt_iter) 
            {
                del_ics = -stepsize * irc_grad;
            }
            else
            {
               const double gamma = irc_grad.squaredNorm() / irc_grad_previous.squaredNorm();	
               del_ics = -stepsize * (irc_grad + gamma * irc_grad_previous);
            }
            
            intco->irc_to_cartesian(ics, del_ics, true); // true will force the molecule to retain symmetry
            irc_grad_previous = irc_grad;                // things may get out of control in numerical mode otherwise
   
            // if(0 == geom_opt_iter) // in cartesians, to be reinstated as an option perhaps
            //     molecule->update_geom(gradient, stepsize);
            // else 
            //     molecule->update_geom(gradient, gradient_previous, stepsize);
            
            // gradient_previous = gradient;
        }
        else if (tol > geom_opt_tol && hf_settings::get_geom_opt_algorithm() == "RFO")
        {
            std::unique_ptr<CART_INT::Cart_int> intco = std::make_unique<CART_INT::Cart_int>(molecule);
            intco->do_internal_coord_analysis();
            intco->print_bonding_info(false, "\n  Current internal\n  (redundant) coordinates: ");
            intco->rfo_step(gradient, irc_grad, irc_coords, irc_hessian);
            intco->do_internal_coord_analysis();
            intco->print_bonding_info(false, "\n  New internal\n  (redundant) coordinates: ");
        }
        else
        {
            molecule->update_geom(gradient, 0.0, true);
        
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
                        << std::setw(18) << molecule->get_geom()(i, 0)
                        << std::right << std::fixed << std::setw(18) << molecule->get_geom()(i, 1)
                        << std::right << std::fixed << std::setw(18) << molecule->get_geom()(i, 2)
                        << std::right << std::fixed << std::setw(13) << std::setprecision(7) 
                        << ELEMENTDATA::masses[zval[i] - 1] << '\n';
            }

            hf_settings::set_geom_opt("");
            std::cout << "\n  Geometry optimization completed in "  << geom_opt_iter << " iterations.\n";
            std::cout << "  Entering SCF mode.\n\n";
            molecule->print_info(true);
            goto done;
        }

        ++geom_opt_iter;
    }

done:
    return;
}
