#include "hfscf.hpp"
#include "../settings/hfscf_settings.hpp"
#include "../postscf/hfscf_post_mp.hpp"
#include "../molecule/hfscf_elements.hpp"
#include "../hessian/hfscf_freq.hpp"
#include "../gradient/hfscf_gradient.hpp"
#include <Eigen/Eigenvalues>
#include <iomanip>

using hfscfmath::index_ijkl;
using hfscfmath::index_ij;
using HF_SETTINGS::hf_settings;
using Eigen::Index;

int Mol::scf::get_frozen_core()
{
    if(!hf_settings::get_freeze_core()) return 0;
    
    const auto& zvals = molecule->get_z_values();

    int frozencore = 0;

    for(const auto& Z: zvals)
    {
        if (Z > 2 && Z <= 10) frozencore += 2;
        else if ( Z > 10 && Z <= 18) frozencore +=10;
        else if (Z > 18 ) frozencore += 18;
        // higher elements no supported anyway
    }

    return frozencore; // electron count, not orbital count
}

void  Mol::scf::calc_numeric_gradient(EigenMatrix<double>& gradient, 
                                      std::vector<bool>& out_of_plane, bool check_if_out_of_plane,
                                      bool print)
{
    // Only used by UHF now
    int natoms = static_cast<int>(molecule->get_atoms().size());
    const double delta = hf_settings::get_grad_step_from_energy();
    constexpr double cutoff = 1.0E-08;
    int frozencore = get_frozen_core();
    Index spin = molecule->get_spin();
    gradient.setZero();

    Index norbitals = molecule->get_num_orbitals();
    Index nelectrons = molecule->get_num_electrons();    

    bool is_rhf = true;
    if("UHF" == hf_settings::get_hf_type()) is_rhf = false;
    
    EigenMatrix<double> start_geom = molecule->get_geom_copy();

    const auto calc_e = 
    [this, is_rhf, norbitals, nelectrons, frozencore, spin]
    (double del, int atom, int cart_dir) -> double 
    {   
        molecule->update_geom(+del, atom, cart_dir);

        if(is_rhf)
        {
            rhf_ptr->init_data(false);
            rhf_ptr->scf_run(false);

            const Eigen::Ref<const EigenMatrix<double>> C_mo = rhf_ptr->get_mo_coef();
            const Eigen::Ref<const EigenVector<double>> eri_vec = rhf_ptr->get_repulsion_vector();
            const Eigen::Ref<const EigenMatrix<double>> mo_energies =  rhf_ptr->get_mo_energies();
            
            std::shared_ptr<POSTSCF::post_scf_mp> post_scf_ptr = 
            std::make_shared<POSTSCF::post_scf_mp>(norbitals, nelectrons, frozencore);
            const double e_mp2 = post_scf_ptr->calc_mp2_energy(mo_energies, C_mo, eri_vec);
            return rhf_ptr->get_scf_energy() + molecule->get_enuc() + e_mp2;
        }
        else
        {
            uhf_ptr->init_data(false);
            uhf_ptr->scf_run(false);

            const Eigen::Ref<const EigenVector<double>> eri_vec = uhf_ptr->get_repulsion_vector();
            const Eigen::Ref<const EigenMatrix<double>> mo_e_alpha = uhf_ptr->get_mo_energies_alpha();
            const Eigen::Ref<const EigenMatrix<double>> mo_e_beta = uhf_ptr->get_mo_energies_beta();
            const Eigen::Ref<const EigenMatrix<double>> C_alpha = uhf_ptr->get_mo_coef_alpha();
            const Eigen::Ref<const EigenMatrix<double>> C_beta = uhf_ptr->get_mo_coef_beta();

            std::shared_ptr<POSTSCF::post_scf_mp> post_scf_ptr = 
            std::make_shared<POSTSCF::post_scf_mp>(norbitals, nelectrons, frozencore);
            double e_mp2 = post_scf_ptr->calc_ump2_energy(mo_e_alpha, mo_e_beta,
                                                          C_alpha, C_beta, eri_vec, spin).sum();
            return  uhf_ptr->get_scf_energy() + molecule->get_enuc() + e_mp2;
        }
        
        return 0;
    };

    if(check_if_out_of_plane)
    {
        out_of_plane = std::vector<bool>(3, false);

        for(int cart_dir = 0; cart_dir < 3; ++cart_dir)
            for(int i = 0; i < natoms; ++i)
            {
                if(std::fabs<double>(start_geom(i , cart_dir)) > cutoff)
                    out_of_plane[cart_dir] = true;
            }
    }

    for (int atom = 0; atom < natoms; ++atom) 
    {
        if (out_of_plane[0]) 
        {
            const double emax_x = calc_e(delta, atom, 0);
            const double emin_x = calc_e(2.0 * delta, atom, 0);
            gradient(atom, 0) = -(emax_x - emin_x) / (2.0 * delta);
            molecule->update_geom(start_geom);  // restore geometry
        }

        if (out_of_plane[1]) 
        {
            const double emax_y = calc_e(delta, atom, 1);
            const double emin_y = calc_e(2.0 * delta, atom, 1);
            gradient(atom, 1) = -(emax_y - emin_y) / (2.0 * delta);
            molecule->update_geom(start_geom);  // restore geometry
        }

        if (out_of_plane[2]) 
        {
            const double emax_z = calc_e(delta, atom, 2);
            const double emin_z = calc_e(2.0 * delta, atom, 2);
            gradient(atom, 2) = -(emax_z - emin_z) / (2.0 * delta);
            molecule->update_geom(start_geom);  // restore geometry
        }
    }

    molecule->update_geom(start_geom, true);  // restore geometry
    if(print) print_gradient(gradient);
}

void Mol::scf::calc_numeric_hessian_from_mp2_energy()
{
    int natoms = static_cast<int>(molecule->get_atoms().size());
    const double delta = hf_settings::get_hessian_step_from_energy();
    EigenMatrix<double> start_geom = molecule->get_geom_copy();
    EigenMatrix<double> hessian = EigenMatrix<double>::Zero(natoms * 3, natoms * 3);
    EigenMatrix<double> e_min =  EigenMatrix<double>::Zero(natoms, 3);
    EigenMatrix<double> e_max =  EigenMatrix<double>::Zero(natoms, 3);
    int frozencore = get_frozen_core();
    Index spin = molecule->get_spin();
    Index nelectrons = molecule->get_num_electrons();
    Index norbitals = molecule->get_num_orbitals();
    bool is_rhf = true;
    if("UHF" == hf_settings::get_hf_type()) is_rhf = false;
    // Note only used by UHF now
    // Last eqn in https://en.wikipedia.org/wiki/Finite_difference
    // https://v8doc.sas.com/sashtml/ormp/chap5/sect28.htm

    const auto calc_e2 = [this, frozencore, spin, nelectrons, norbitals, is_rhf] 
    (double del, int atom1, int atom2, int cart_dir1, int cart_dir2) -> double 
    {   
        molecule->update_geom(+del, atom1, cart_dir1);
        molecule->update_geom(+del, atom2, cart_dir2);

        if(is_rhf)
        {
            rhf_ptr->init_data(false);
            rhf_ptr->scf_run(false);

            const Eigen::Ref<const EigenMatrix<double>> C_mo =  rhf_ptr->get_mo_coef();
            const Eigen::Ref<const EigenMatrix<double>> mo_energies =  rhf_ptr->get_mo_energies();
            const Eigen::Ref<const EigenVector<double>> eri_vec = rhf_ptr->get_repulsion_vector();
            
            std::shared_ptr<POSTSCF::post_scf_mp> post_scf_ptr = 
            std::make_shared<POSTSCF::post_scf_mp>(norbitals, nelectrons, frozencore);
            const double e_mp2 = post_scf_ptr->calc_mp2_energy(mo_energies, C_mo, eri_vec);
            return rhf_ptr->get_scf_energy() + molecule->get_enuc() + e_mp2;
        }
        else
        {
            uhf_ptr->init_data(false);
            uhf_ptr->scf_run(false);

            const Eigen::Ref<const EigenMatrix<double>> eps_alpha = uhf_ptr->get_mo_energies_alpha();
            const Eigen::Ref<const EigenMatrix<double>> eps_beta = uhf_ptr->get_mo_energies_beta();
            const Eigen::Ref<const EigenMatrix<double>> C_alpha = uhf_ptr->get_mo_coef_alpha();
            const Eigen::Ref<const EigenMatrix<double>> C_beta = uhf_ptr->get_mo_coef_beta();
            const Eigen::Ref<const EigenVector<double>> eri_vec = uhf_ptr->get_repulsion_vector();

            std::shared_ptr<POSTSCF::post_scf_mp> post_scf_ptr = 
            std::make_shared<POSTSCF::post_scf_mp>(norbitals, nelectrons, frozencore);
            double e_mp2 = post_scf_ptr->calc_ump2_energy(eps_alpha, eps_beta, C_alpha, C_beta, 
                                                          eri_vec, spin).sum();
            return  uhf_ptr->get_scf_energy() + molecule->get_enuc() + e_mp2;
        }
        
        return 0;
    }; 

    const auto calc_e1 = [this, frozencore, spin, nelectrons, norbitals, is_rhf] 
    (double del, int atom, int cart_dir) -> double 
    {   
        molecule->update_geom(+del, atom, cart_dir);

        if(is_rhf)
        {
            rhf_ptr->init_data(false);
            rhf_ptr->scf_run(false);

            const Eigen::Ref<const EigenMatrix<double>> C_mo =  rhf_ptr->get_mo_coef();
            const Eigen::Ref<const EigenMatrix<double>> mo_energies =  rhf_ptr->get_mo_energies();
            const Eigen::Ref<const EigenVector<double>> eri_vec = rhf_ptr->get_repulsion_vector();
            
            std::shared_ptr<POSTSCF::post_scf_mp> post_scf_ptr = 
            std::make_shared<POSTSCF::post_scf_mp>(norbitals, nelectrons, frozencore);
            const double e_mp2 = post_scf_ptr->calc_mp2_energy(mo_energies, C_mo, eri_vec);
            return rhf_ptr->get_scf_energy() + molecule->get_enuc() + e_mp2;
        }
        else
        {
            uhf_ptr->init_data(false);
            uhf_ptr->scf_run(false);

            const Eigen::Ref<const EigenMatrix<double>> eps_alpha = uhf_ptr->get_mo_energies_alpha();
            const Eigen::Ref<const EigenMatrix<double>> eps_beta = uhf_ptr->get_mo_energies_beta();
            const Eigen::Ref<const EigenMatrix<double>> C_alpha = uhf_ptr->get_mo_coef_alpha();
            const Eigen::Ref<const EigenMatrix<double>> C_beta = uhf_ptr->get_mo_coef_beta();
            const Eigen::Ref<const EigenVector<double>> eri_vec = uhf_ptr->get_repulsion_vector();

            std::shared_ptr<POSTSCF::post_scf_mp> post_scf_ptr = 
            std::make_shared<POSTSCF::post_scf_mp>(norbitals, nelectrons, frozencore);
            double e_mp2 = post_scf_ptr->calc_ump2_energy(eps_alpha, eps_beta, C_alpha, C_beta, 
                                                          eri_vec, spin).sum();
            return  uhf_ptr->get_scf_energy() + molecule->get_enuc() + e_mp2;
        }
        
        return 0;
    };

    // center point
    double e_c = 0.0;
    if(is_rhf)
    {
        rhf_ptr->init_data(false);
        rhf_ptr->scf_run(false);
        e_c = rhf_ptr->get_scf_energy() + molecule->get_enuc();
        
        const Eigen::Ref<const EigenMatrix<double>> C_mo = rhf_ptr->get_mo_coef();
        const Eigen::Ref<const EigenMatrix<double>> mo_energies =  rhf_ptr->get_mo_energies();
        const Eigen::Ref<const EigenVector<double>> eri_vec = rhf_ptr->get_repulsion_vector();
        
        std::shared_ptr<POSTSCF::post_scf_mp> post_scf_ptr =
        std::make_shared<POSTSCF::post_scf_mp>(norbitals, nelectrons, frozencore);
        
        const double e_mp2 = post_scf_ptr->calc_mp2_energy(mo_energies, C_mo, eri_vec);
        e_c += e_mp2;
    }
    else
    {
        uhf_ptr->init_data(false);
        uhf_ptr->scf_run(false);
        e_c = uhf_ptr->get_scf_energy() + molecule->get_enuc();

        const Eigen::Ref<const EigenMatrix<double>> eps_alpha = uhf_ptr->get_mo_energies_alpha();
        const Eigen::Ref<const EigenMatrix<double>> eps_beta = uhf_ptr->get_mo_energies_beta();
        const Eigen::Ref<const EigenMatrix<double>> C_alpha = uhf_ptr->get_mo_coef_alpha();
        const Eigen::Ref<const EigenMatrix<double>> C_beta = uhf_ptr->get_mo_coef_beta();
        const Eigen::Ref<const EigenVector<double>> eri_vec = uhf_ptr->get_repulsion_vector();
        
        std::shared_ptr<POSTSCF::post_scf_mp> post_scf_ptr = 
        std::make_shared<POSTSCF::post_scf_mp>(norbitals, nelectrons, frozencore);
        double e_mp2 = post_scf_ptr->calc_ump2_energy(eps_alpha, eps_beta, 
                                                    C_alpha, C_beta, eri_vec, spin).sum();
        e_c += e_mp2;
    }

    // diagonal terms
    for(int atom = 0; atom < natoms; ++atom)
    {
        for(int cart_dir = 0; cart_dir < 3; ++cart_dir)
        {
            e_max(atom, cart_dir) = calc_e1(delta, atom, cart_dir);
            e_min(atom, cart_dir) = calc_e1(-2.0 * delta, atom, cart_dir);

            hessian(3 * atom + cart_dir, 3 * atom + cart_dir) =
               (e_max(atom, cart_dir) -2.0 * e_c + e_min(atom, cart_dir) ) / (delta * delta);

            molecule->update_geom(start_geom);  // restore geometry
        }
    }

    for(int atom = 0; atom < natoms; ++atom)
    {
       for(int atom2 = 0; atom2 < natoms; ++atom2)
       { 
            if(atom2 != atom)
            {
                // xx yy zz different atoms
                for(int cart_dir = 0; cart_dir < 3; ++cart_dir)
                {
                    const double xx_max = calc_e2(delta, atom, atom2, cart_dir, cart_dir);
                    const double xx_min = calc_e2(-2.0 * delta, atom, atom2, cart_dir, cart_dir);
                    molecule->update_geom(start_geom);  // restore geometry

                    hessian(3 * atom + cart_dir, 3 * atom2 + cart_dir)
                    = (xx_max  + 2.0 * e_c + xx_min - e_max(atom2, cart_dir) - e_min(atom2, cart_dir) 
                    - e_max(atom, cart_dir) - e_min(atom, cart_dir))
                    / (2.0 * delta * delta);

                    hessian(3 * atom2 + cart_dir, 3 * atom + cart_dir) = 
                        hessian(3 * atom + cart_dir, 3 * atom2 + cart_dir);
                }
            }

            // xy
            const double xy_max = calc_e2(delta, atom, atom2, 0, 1);
            const double xy_min = calc_e2(-2.0 * delta, atom, atom2, 0, 1);
            molecule->update_geom(start_geom);  // restore geometry

            hessian(3 * atom, 3 * atom2 + 1) 
            = (xy_max  + 2.0 * e_c + xy_min - e_max(atom2, 1) - e_min(atom2, 1) - e_max(atom, 0) - e_min(atom, 0))
            / (2.0 * delta * delta);
            if (std::fabs(hessian(3 * atom, 3 * atom2 + 1)) < 1.0E-05) hessian(3 * atom, 3 * atom2 + 1) = 0.0;
            hessian(3 * atom2 + 1, 3 * atom) =  hessian(3 * atom, 3 * atom2 + 1);

            //xz
            const double xz_max = calc_e2(delta, atom, atom2, 0, 2);
            const double xz_min = calc_e2(-2.0 * delta, atom, atom2, 0, 2);
            molecule->update_geom(start_geom);  // restore geometry

            hessian(3 * atom, 3 * atom2 + 2)
            = (xz_max  + 2.0 * e_c + xz_min - e_max(atom2, 2) - e_min(atom2, 2) - e_max(atom, 0) - e_min(atom, 0))
            / (2.0 * delta * delta);
            if (std::fabs(hessian(3 * atom, 3 * atom2 + 2)) < 1.0E-05) hessian(3 * atom, 3 * atom2 + 2) = 0.0;
            hessian(3 * atom2 + 2, 3 * atom) = hessian(3 * atom, 3 * atom2 + 2);

            //yz
            const double yz_max = calc_e2(delta, atom, atom2, 1, 2);
            const double yz_min = calc_e2(-2.0 * delta, atom, atom2, 1, 2);
            molecule->update_geom(start_geom);  // restore geometry

            hessian(3 * atom + 1, 3 * atom2 + 2)
            = (yz_max  + 2.0 * e_c + yz_min - e_max(atom2, 2) - e_min(atom2, 2) - e_max(atom, 1) - e_min(atom, 1))
            / (2.0 * delta * delta);
            if (std::fabs(hessian(3 * atom + 1, 3 * atom2 + 2)) < 1.0E-05) hessian(3 * atom + 1, 3 * atom2 + 2) = 0.0;
            hessian(3 * atom2 + 2, 3 * atom + 1) =  hessian(3 * atom + 1, 3 * atom2 + 2);
        }
    }
    
    molecule->update_geom(start_geom, true);
    
    EigenMatrix<double> dipderiv;
    FREQ::calc_frequencies(molecule, hessian, dipderiv, e_c);
    
}

void Mol::scf::calc_numeric_hessian_from_analytic_gradient(const double rhf_mp2_energy)
{   // UHF only now.
    const int natoms = static_cast<int>(molecule->get_atoms().size());
    const double delta = hf_settings::get_hessian_step_from_grad();
    EigenMatrix<double> start_geom = molecule->get_geom_copy();
    EigenMatrix<double> hessian = EigenMatrix<double>::Zero(natoms * 3, natoms * 3);
    std::vector<bool> coords = std::vector<bool>(3, true);

    std::unique_ptr<Mol::scf_gradient> grad_ptr = std::make_unique<Mol::scf_gradient>(molecule);

    std::vector<bool> out_of_plane = std::vector<bool>(3, false);
    bool is_rhf = true;
    if( "UHF" == hf_settings::get_hf_type()) is_rhf = false;
    bool mp2 = false;
    if("MP2" == hf_settings::get_frequencies_type()) mp2 = true;

    const auto calc_grad = [this, &coords, &grad_ptr, mp2](EigenMatrix<double>& q, bool isrhf) -> void
    {
        if(isrhf)
        {
            const Eigen::Ref<const EigenMatrix<double>> d_mat = rhf_ptr->get_density_matrix();

            if(mp2)
            {
                const Eigen::Ref<const EigenMatrix<double>> mo_energies =  rhf_ptr->get_mo_energies();
                const Eigen::Ref<const EigenMatrix<double>> C_mo =  rhf_ptr->get_mo_coef();
                const Eigen::Ref<const EigenMatrix<double>> s_mat =  rhf_ptr->get_overlap_matrix();
                const Eigen::Ref<const EigenVector<double>> eri_vec = rhf_ptr->get_repulsion_vector();
                grad_ptr->calc_mp2_gradient_rhf(mo_energies, s_mat, d_mat, C_mo, eri_vec, coords, 
                                                false, false, false);
            }
            else
            {
                const Eigen::Ref<const EigenMatrix<double>> f_mat =  rhf_ptr->get_fock_matrix();
                grad_ptr->calc_scf_gradient_rhf(d_mat, f_mat, coords, false, false, false);
            }
        }
        else
        {
            const Eigen::Ref<const EigenMatrix<double>> d_alpha = uhf_ptr->get_density_matrix_alpha();
            const Eigen::Ref<const EigenMatrix<double>> d_beta = uhf_ptr->get_density_matrix_beta();
            const Eigen::Ref<const EigenMatrix<double>> f_alpha = uhf_ptr->get_fock_matrix_alpha();
            const Eigen::Ref<const EigenMatrix<double>> f_beta = uhf_ptr->get_fock_matrix_beta();
            grad_ptr->calc_scf_gradient_uhf(d_alpha, d_beta, f_alpha, f_beta, coords, false, false, false);
        }
            
        q = grad_ptr->get_gradient_ref();
    };
    
    for(int atom = 0; atom < natoms; ++atom)
    {
        for(int cart_dir = 0; cart_dir < 3; ++cart_dir)
        {
            if (0 == cart_dir)
            {
                coords[0] = true; coords[1] = true; coords[2] = true;
            }
            else if (1 == cart_dir)
            { 
                coords[0] = false; coords[1] = true; coords[2] = true;
            }
            else if (2 == cart_dir)
            {
                coords[0] = false; coords[1] = false; coords[2] = true;
            }
            
             // + displaced
            molecule->update_geom(start_geom);
            molecule->update_geom(delta, atom, cart_dir);
            if (is_rhf)
            {
                rhf_ptr->init_data(false);
                rhf_ptr->scf_run(false);
            }
            else
            {
                uhf_ptr->init_data(false);
                uhf_ptr->scf_run(false);
            }
            
            EigenMatrix<double> qmax; calc_grad(qmax, is_rhf);

            for (int atom2 = 0; atom2 < natoms; ++atom2)
            { 
                // XX YY ZZ 
                hessian(3 * atom + cart_dir, 3 * atom2 + cart_dir) = qmax(atom2, cart_dir) / (2.0 * delta);

               if(0 == cart_dir) // XY XZ
               {
                    hessian(3 * atom, 3 * atom2 + 1) = qmax(atom2, 1) / (2.0 * delta);
                    hessian(3 * atom, 3 * atom2 + 2) = qmax(atom2, 2) / (2.0 * delta);
                }
                else if(1 == cart_dir) // YZ
                {
                    hessian(3 * atom + 1, 3 * atom2 + 2) = qmax(atom2, 2) / (2.0 * delta);
                }
            }
            // - displaced
            molecule->update_geom(start_geom);
            molecule->update_geom(-delta, atom, cart_dir);
            if (is_rhf)
            {
                rhf_ptr->init_data(false);
                rhf_ptr->scf_run(false);
            }
            else
            {
                uhf_ptr->init_data(false);
                uhf_ptr->scf_run(false);
            }

            EigenMatrix<double> qmin; calc_grad(qmin, is_rhf);
            
            for (int atom2 = 0; atom2 < natoms; ++atom2)
            {   
                hessian(3 * atom + cart_dir, 3 * atom2 + cart_dir) -= qmin(atom2, cart_dir) / (2.0 * delta);
                hessian(3 * atom2 + cart_dir, 3 * atom + cart_dir) = hessian(3 * atom + cart_dir, 3 * atom2 + cart_dir); 

                if(0 == cart_dir) // XY XZ
                {
                    hessian(3 * atom, 3 * atom2 + 1) -= qmin(atom2, 1) / (2.0 * delta);
                    hessian(3 * atom, 3 * atom2 + 2) -= qmin(atom2, 2) / (2.0 * delta);
                    hessian(3 * atom2 + 1, 3 * atom) = hessian(3 * atom, 3 * atom2 + 1);
                    hessian(3 * atom2 + 2, 3 * atom) = hessian(3 * atom, 3 * atom2 + 2);
                }
                else if(1 == cart_dir) // YZ
                {
                    hessian(3 * atom + 1, 3 * atom2 + 2) -= qmin(atom2, 2) / (2.0 * delta); 
                    hessian(3 * atom2 + 2, 3 * atom + 1) = hessian(3 * atom + 1, 3 * atom2 + 2);
                }
            }
        }
    }

    molecule->update_geom(start_geom, true);
    //. mp2_energy will be zero if not doing mp2
    EigenMatrix<double> dipderiv;

    if(is_rhf) // no longer used for RHF (testing only)
        FREQ::calc_frequencies(molecule, hessian, dipderiv, rhf_ptr->get_scf_energy() + molecule->get_enuc() + rhf_mp2_energy);
    else 
        FREQ::calc_frequencies(molecule, hessian, dipderiv, uhf_ptr->get_scf_energy() + molecule->get_enuc());
}
