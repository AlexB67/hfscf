#include "hfscf.hpp"
#include "../settings/hfscf_settings.hpp"
#include "../postscf/hfscf_post_cc.hpp"
#include "../postscf/hfscf_post_cc_rhf.hpp"
#include "../postscf/hfscf_post_mp.hpp"
#include "../config_interaction/hfscf_cis.hpp"
#include "../integrals/hfscf_oseri.hpp"
#include "../gradient/hfscf_gradient.hpp"
#include "../hessian/hfscf_hessian.hpp"
#include "../mol_properties/hfscf_properties.hpp"
#include "../irc/cart_to_int.hpp"

using HF_SETTINGS::hf_settings;

Mol::scf::scf(const bool verbose, const std::string& geometry_file, const std::string& prefix) 
: m_verbose(verbose)
{   
    molecule = std::make_shared<MOLEC::Molecule>(prefix);
    molecule->init_molecule(geometry_file);
}

void Mol::scf::hf_run()
{
    bool do_geom_opt = false;
    if(hf_settings::get_geom_opt().length()) do_geom_opt = true;
    
    if("UHF" == hf_settings::get_hf_type())
    {
        uhf_ptr = std::make_unique<Mol::uhf_solver>(molecule);
        uhf_ptr->get_data();

        if (do_geom_opt && !hf_settings::get_scf_direct())
        {
            if ("MP2" == hf_settings::get_geom_opt()) geom_opt_numeric();
            else geom_opt();
        }
        
        uhf_ptr->init_data(true);
        bool converged = uhf_ptr->scf_run(true);
        if(converged) post_uhf();
    }
    else
    {
        rhf_ptr = std::make_unique<Mol::hf_solver>(molecule);
        rhf_ptr->get_data();

        if (do_geom_opt && !hf_settings::get_scf_direct()) geom_opt();

        rhf_ptr->init_data(true);
        bool converged = rhf_ptr->scf_run(true);
        if(converged) post_rhf();
    }
}

void Mol::scf::post_rhf()
{
    if(hf_settings::get_verbosity() > 1)
    {
        std::cout << "\n******************\n";
        std::cout << "Iteration: " << rhf_ptr->get_iterations();
        std::cout << "\n******************\n";

        rhf_ptr->print_mo_coefficients_matrix();
        rhf_ptr->print_density_matrix();
        rhf_ptr->print_fock_matrix();
    }

    const Eigen::Ref<const EigenMatrix<double>> C_mat = rhf_ptr->get_mo_coef();
    const Eigen::Ref<const EigenMatrix<double>> f_mat = rhf_ptr->get_fock_matrix();
    const Eigen::Ref<const EigenMatrix<double>> d_mat = rhf_ptr->get_density_matrix();
    const Eigen::Ref<const EigenMatrix<double>> s_mat = rhf_ptr->get_overlap_matrix();
    const Eigen::Ref<const EigenMatrix<double>> mo_energies = rhf_ptr->get_mo_energies();

    Index norbitals = molecule->get_num_orbitals();
    Index nelectrons = molecule->get_num_electrons();
    const Eigen::Ref<const EigenVector<double>> eri_vec = rhf_ptr->get_repulsion_vector();

    const std::string& post_scf = hf_settings::get_post_scf_type();

    int ccsd_iterations = 0;
    double mp2_energy = 0;  double mp3_energy = 0; 
    double ccsd_energy = 0; double cc_triples_energy = 0;
    std::vector<double> ccsd_energies, ccsd_rms, ccsd_deltas;
    std::vector<std::pair<double, std::string>> largest_t1, largest_t2;

    int frozencore = get_frozen_core();

    if (post_scf.length() && !hf_settings::get_scf_direct()) 
    {
        if (post_scf == "MP2") 
        {
            std::shared_ptr<POSTSCF::post_scf_mp> post_scf_ptr =
                std::make_shared<POSTSCF::post_scf_mp>(norbitals, nelectrons, frozencore);
            mp2_energy = post_scf_ptr->calc_mp2_energy(mo_energies, C_mat, eri_vec);
        } 
        else if (post_scf == "MP3") 
        {
            std::shared_ptr<POSTSCF::post_scf_mp> post_scf_ptr =
                std::make_shared<POSTSCF::post_scf_mp>(norbitals, nelectrons, frozencore);
            auto mpn = post_scf_ptr->calc_mp3_energy(mo_energies, C_mat, eri_vec);
            mp2_energy = mpn.first;
            mp3_energy = mpn.second;
        } 
        else if (post_scf == "CCSD(T)") // In spin basis
        {
            std::shared_ptr<POSTSCF::post_scf_ccsd> post_scf_ptr =
            std::make_shared<POSTSCF::post_scf_ccsd>(norbitals, nelectrons, frozencore);
            auto ccsd = post_scf_ptr->calc_ccsd(mo_energies, C_mat, eri_vec);
            mp2_energy = ccsd.first;
            ccsd_energy = ccsd.second;
            ccsd_energies = post_scf_ptr->get_ccsd_energies();
            ccsd_deltas = post_scf_ptr->get_ccsd_deltas();
            ccsd_rms = post_scf_ptr->get_ccsd_rms();
            ccsd_iterations = post_scf_ptr->get_iterations();
            largest_t1 = post_scf_ptr->get_largest_t1();
            largest_t2 = post_scf_ptr->get_largest_t2();
            cc_triples_energy = post_scf_ptr->calc_perturbation_triples();
        }
        else if (post_scf == "CCSD") // In MO spatial basis
        {
            std::shared_ptr<POSTSCF::post_rhf_ccsd> post_scf_ptr =
            std::make_shared<POSTSCF::post_rhf_ccsd>(norbitals, nelectrons, frozencore);
            auto ccsd = post_scf_ptr->calc_ccsd(mo_energies, C_mat, eri_vec);
            mp2_energy = ccsd.first;
            ccsd_energy = ccsd.second;
            ccsd_energies = post_scf_ptr->get_ccsd_energies();
            ccsd_deltas = post_scf_ptr->get_ccsd_deltas();
            ccsd_rms = post_scf_ptr->get_ccsd_rms();
            ccsd_iterations = post_scf_ptr->get_iterations();
            largest_t1 = post_scf_ptr->get_largest_t1();
            largest_t2 = post_scf_ptr->get_largest_t2();
        }
    }

    rhf_ptr->print_mo_energies();
    print_rhf_energies(mp2_energy, mp3_energy, ccsd_energy, cc_triples_energy, ccsd_iterations,
                       ccsd_energies, ccsd_deltas, ccsd_rms, largest_t1, largest_t2);

    if(hf_settings::get_do_mol_props())
    {
        std::unique_ptr<MolProps::Molprops> molprops = std::make_unique<MolProps::Molprops>(molecule);
        molprops->create_dipole_vectors_rhf(d_mat);
        molprops->create_quadrupole_tensors_rhf(d_mat);
        molprops->population_analysis_rhf(s_mat, d_mat);
        molprops->mayer_indices_rhf(s_mat, d_mat);
        molprops->print_dipoles();
        molprops->print_quadrupoles();

        if (hf_settings::get_molprops_cphf_iter())
            molprops->calc_static_polarizabilities_iterative(C_mat, mo_energies, eri_vec);
        else
            molprops->calc_static_polarizabilities(C_mat, mo_energies, eri_vec);

        molprops->print_population_analysis();
        molprops->print_mayer_indices();
    }

    if(hf_settings::get_gradient_type().length() && !hf_settings::get_scf_direct()
                                                 && "MP2" != hf_settings::get_frequencies_type())
    {
        if("MP2" == hf_settings::get_gradient_type()) 
        { //mp2 freq already includes gradient
            std::unique_ptr<Mol::scf_gradient> grad_ptr = std::make_unique<Mol::scf_gradient>(molecule);
            std::vector<bool> coords;
            grad_ptr->calc_mp2_gradient_rhf(mo_energies, s_mat, d_mat, C_mat, eri_vec, coords, true, false, true);         
        }
        else
        {
            std::unique_ptr<Mol::scf_gradient> grad_ptr = std::make_unique<Mol::scf_gradient>(molecule);
            std::vector<bool> coords;
            grad_ptr->calc_scf_gradient_rhf(d_mat, f_mat, coords, true, false, true);
        } 
    }

    if (hf_settings::get_frequencies_type().length())
    {
         std::unique_ptr<Mol::scf_hessian> hes_ptr
           = std::make_unique<Mol::scf_hessian>(molecule, d_mat, C_mat, mo_energies, s_mat, eri_vec, f_mat);
        if("MP2" == hf_settings::get_frequencies_type())
        {
           // Analytic
            hes_ptr->calc_scf_hessian_rhf_mp2(rhf_ptr->get_scf_energy() + molecule->get_enuc() + mp2_energy);
          // calc_numeric_hessian_from_analytic_gradient(mp2_energy);
        }
        else
        {
           // Analytic
           hes_ptr->calc_scf_hessian_rhf(rhf_ptr->get_scf_energy() + molecule->get_enuc());
           // calc_numeric_hessian_from_analytic_gradient();
        }
    }

    const std::string& CI = hf_settings::get_ci_type();

    if (CI.length() && !hf_settings::get_scf_direct()) 
    {
        std::shared_ptr<POSTSCF::post_scf_cis> post_scf_ptr =
            std::make_shared<POSTSCF::post_scf_cis>(norbitals, nelectrons);

        if (CI == "CIS")
            post_scf_ptr->calc_cis_energies(molecule, mo_energies, C_mat, eri_vec);
        else if (CI == "RPA")
            post_scf_ptr->calc_rpa_energies(mo_energies, C_mat, eri_vec);
        else
            std::cout << "  Error: Invalid configuration request\"" << CI << "\" request ignored.\n";
    }
}

void Mol::scf::post_uhf()
{
    if (hf_settings::get_verbosity() > 1)
    {
        std::cout << "\n******************\n";
        std::cout << "Iteration: " << uhf_ptr->get_iterations();
        std::cout << "\n******************\n";
        uhf_ptr->print_mo_coefficients_matrix();
        uhf_ptr->print_density_matrix();
        uhf_ptr->print_fock_matrix();
    }

    const Eigen::Ref<const EigenMatrix<double>> mo_energies_alpha = uhf_ptr->get_mo_energies_alpha();
    const Eigen::Ref<const EigenMatrix<double>> mo_energies_beta = uhf_ptr->get_mo_energies_beta();
    const Eigen::Ref<const EigenMatrix<double>> C_alpha = uhf_ptr->get_mo_coef_alpha();
    const Eigen::Ref<const EigenMatrix<double>> C_beta = uhf_ptr->get_mo_coef_beta();
    const Eigen::Ref<const EigenMatrix<double>> f_alpha = uhf_ptr->get_fock_matrix_alpha();
    const Eigen::Ref<const EigenMatrix<double>> f_beta = uhf_ptr->get_fock_matrix_beta();
    const Eigen::Ref<const EigenMatrix<double>> d_alpha = uhf_ptr->get_density_matrix_alpha();
    const Eigen::Ref<const EigenMatrix<double>> d_beta = uhf_ptr->get_density_matrix_beta();
    const Eigen::Ref<const EigenMatrix<double>> s_mat = uhf_ptr->get_overlap_matrix();
    const Eigen::Ref<const EigenVector<double>> eri_vec = uhf_ptr->get_repulsion_vector();

    Index norbitals = molecule->get_num_orbitals();
    Index nelectrons = molecule->get_num_electrons();

    EigenVector<double> ump2_energies, ump3_energies;
    if (hf_settings::get_post_scf_type() == "MP2")
    {
        int frozencore = get_frozen_core();
        std::shared_ptr<POSTSCF::post_scf_mp> post_scf_ptr =
        std::make_shared<POSTSCF::post_scf_mp>(norbitals, nelectrons, frozencore);
        
        ump2_energies = post_scf_ptr->calc_ump2_energy(mo_energies_alpha, mo_energies_beta, 
                                                       C_alpha, C_beta, eri_vec, molecule->get_spin());
    }
    else if (hf_settings::get_post_scf_type() == "MP3") 
    {
        int frozencore = get_frozen_core();
        std::shared_ptr<POSTSCF::post_scf_mp> post_scf_ptr =
        std::make_shared<POSTSCF::post_scf_mp>(norbitals, nelectrons, frozencore);

        ump3_energies = post_scf_ptr->calc_ump3_energy(mo_energies_alpha, mo_energies_beta, 
                                                       C_alpha, C_beta, eri_vec, molecule->get_spin());
    } 
    else if (hf_settings::get_post_scf_type().length()) 
    {
        std::cout << "  Warning: Post SCF request ignored. MP2/3 only.\n";
    } 
    
    if (hf_settings::get_ci_type().length()) 
    {
        std::cout << "  Warning: CI request ignored. RHF only.\n";
    }

    uhf_ptr->print_mo_energies();
    print_uhf_energies(ump2_energies, ump3_energies);

    if(hf_settings::get_do_mol_props())
    {
        std::unique_ptr<MolProps::Molprops> molprops = std::make_unique<MolProps::Molprops>(molecule);
        molprops->create_dipole_vectors_uhf(d_alpha, d_beta);
        molprops->create_quadrupole_tensors_uhf(d_alpha, d_beta);
        molprops->population_analysis_uhf(s_mat, d_alpha, d_beta);
        molprops->mayer_indices_uhf(s_mat, d_alpha, d_beta);
        molprops->print_dipoles();
        molprops->print_quadrupoles();
        molprops->print_population_analysis();
        molprops->print_mayer_indices();
    }

    if(hf_settings::get_gradient_type().length() && !hf_settings::get_scf_direct())
    {
        if("MP2" == hf_settings::get_gradient_type())
        {
            int natoms = (int)molecule->get_atoms().size();
            EigenMatrix<double> grad = EigenMatrix<double>::Zero(natoms, 3);
            std::vector<bool> coords;
            calc_numeric_gradient(grad, coords, true, true);
        }
        else
        {
            std::unique_ptr<Mol::scf_gradient> grad_ptr = std::make_unique<Mol::scf_gradient>(molecule);
            std::vector<bool> coords;
            grad_ptr->calc_scf_gradient_uhf(d_alpha, d_beta, f_alpha, f_beta, coords, true, false, true);
        }
    }

    const std::string& freq = hf_settings::get_frequencies_type();

    if (hf_settings::get_use_symmetry() && freq.length()) 
    {
        std::clog << "\n  Error: Symmetry is unavailable for numeric frequencies.\n";
        std::clog << "         This mode is available with symmetry disabled only.\n";
        exit(EXIT_FAILURE);
    }

    if ("SCF" == freq) calc_numeric_hessian_from_analytic_gradient(0);
    else if ("MP2" == freq) calc_numeric_hessian_from_mp2_energy();
}
