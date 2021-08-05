#include "../integrals/hfscf_oseri.hpp"
#include "../settings/hfscf_settings.hpp"
#include "../irc/cart_to_int.hpp"
#include "hfscf_hf_print.hpp"
#include "hfscf_hf_solver.hpp"
#include "hfscf_sad.hpp"
#include <Eigen/Eigenvalues>

using HF_SETTINGS::hf_settings;
using Eigen::Index;
using hfscfmath::index_ij;
using hfscfmath::index_ijkl;

Mol::hf_solver::hf_solver(const std::shared_ptr<MOLEC::Molecule>& mol)
: molecule(mol)
{
}

void Mol::hf_solver::get_data()
{
    nelectrons = molecule->get_num_electrons();

    if(nelectrons % 2 != 0 && hf_settings::get_hf_type() == "RHF")
    {
        std::cout << "\n  Error: Molecule cannot have unpaired electrons for RHF method. Aborting.\n";
        exit(EXIT_FAILURE);
    }
    else if (molecule->get_spin() != 0 && hf_settings::get_hf_type() == "RHF")
    {
        std::cout << "\n  Error: Invalid spin for RHF method. Aborting.\n";
        exit(EXIT_FAILURE);
    }

    bool is_spherical = molecule->use_pure_am();
    if (is_spherical && hf_settings::get_use_symmetry()) use_sym = true;

    norbitals = molecule->get_num_orbitals();

    allocate_memory();

    nelectrons = molecule->get_num_electrons();
    molecule->print_info(false);
}

void Mol::hf_solver::allocate_memory()
{   // No need to zero out matrices
    s_mat = EigenMatrix<double>(norbitals, norbitals);
    k_mat = EigenMatrix<double>(norbitals, norbitals);
    v_mat = EigenMatrix<double>(norbitals, norbitals);
    f_mat = EigenMatrix<double>(norbitals, norbitals);
    fp_mat = EigenMatrix<double>(norbitals, norbitals);
    s_mat_sqrt = EigenMatrix<double>(norbitals, norbitals);
    mo_coff = EigenMatrix<double>(norbitals, norbitals);
    d_mat = EigenMatrix<double>(norbitals, norbitals);

    if (false == hf_settings::get_scf_direct())
        e_rep_mat = EigenVector<double>(norbitals * (norbitals + 1)
                  * (norbitals * norbitals + norbitals + 2) / 8);
    
    if (use_sym)
    {
        const std::vector<int> irreps = molecule->get_irreps();
        vs_mat_sqrt = std::vector<EigenMatrix<double>>(irreps.size());
        vfp_mat = std::vector<EigenMatrix<double>>(irreps.size());
        vd_mat = std::vector<EigenMatrix<double>>(irreps.size());
        vd_mat_previous = std::vector<EigenMatrix<double>>(irreps.size());
        vmo_coff = std::vector<EigenMatrix<double>>(irreps.size());
        vf_mat = std::vector<EigenMatrix<double>>(irreps.size());

        for (size_t k = 0; k < irreps.size(); ++k)
        {
            vs_mat_sqrt[k] = vfp_mat[k] = vd_mat[k] = vd_mat_previous[k] 
            = vmo_coff[k] = vf_mat[k] = EigenMatrix<double>(irreps[k], irreps[k]);
        }

    }
}

void Mol::hf_solver::init_data(bool print, bool use_previous_density)
{
    //reset for geomopt or other multiple calls
    //if we are still alive
    
    iteration = 0;
    scf_energies.clear();
    diis_log = print;
    soscf = false;
    
    create_one_electron_hamiltonian();
    create_repulsion_matrix();

    if(print)
    {
        if(molecule->get_atoms().size() > 1)
        {
            std::cout << "\n  ___Internal coordinate analysis___\n";
            std::unique_ptr<CART_INT::Cart_int> intco = std::make_unique<CART_INT::Cart_int>(molecule);
            intco->do_internal_coord_analysis();
            intco->print_bonding_info();
        }
        
        std::cout << "\n  ___SCF Solver___\n";
        HFCOUT::print_e_rep_matrix<Index>(norbitals);
    }

    double min_eval;
    
    if(use_sym)
        min_eval = calc_initial_fock_matrix_with_symmetry();
    else
        min_eval = calc_initial_fock_matrix();

    if(use_previous_density) goto use_previous_density;

    if (hf_settings::get_guess_type() == "SAD")
    {
       get_sad_guess(print);
    }
    else
    {
        if(use_sym) calc_density_matrix_with_symmetry();
        else calc_density_matrix();
    }

use_previous_density:

    //TODO canonical orth. alternative.
    if (print)
        std::cout << "\n  Using Symmetric orthogonalisation.\n  Overlap matrix minimum eigenvalue: " 
                << std::scientific << min_eval << "\n";

    if (use_sym && print) 
    {
        const std::vector<std::string>& species = molecule->get_symmetry_species();

        std::cout << "\n  *************************************************\n";
        std::cout << "  *   NSO counts and symmetry:                    *";
        std::cout << "\n  *************************************************\n";

        auto iter = molecule->get_irreps().begin();
        std::cout << "   ";
        int line = 0;
        int count = 0;
        for (const auto& sp : species) {
            ++line;
            std::cout << std::left << std::setw(8) << std::to_string(*iter) + "x" + sp;
            count += *iter;
            if ((line % 6) == 0) std::cout << "\n   ";
            ++iter;
        }

        std::cout << "\n";
        std::cout << "\n   Total: " << count << "\n\n";
    }

    if (hf_settings::get_verbosity() > 2 && print)
    {
        std::cout << "\n******************\n";
        std::cout << "Iteration: " << iteration;
        std::cout << "\n******************\n";

        if (use_sym)
        {
            HFCOUT::print_overlap_matrix<double>(vs_mat);
            HFCOUT::print_kinetic_matrix<double>(vk_mat);
            HFCOUT::print_npot_matrix<double>(vv_mat);
            HFCOUT::print_core_hamiltonian<double>(vhcore_mat);
            HFCOUT::print_ortho_overlap_matrix<double>(vs_mat_sqrt);
        }
        else
        {
            HFCOUT::print_overlap_matrix<double>(s_mat);
            HFCOUT::print_kinetic_matrix<double>(k_mat);
            HFCOUT::print_npot_matrix<double>(v_mat);
            HFCOUT::print_core_hamiltonian<double>(hcore_mat);
            HFCOUT::print_ortho_overlap_matrix<double>(s_mat_sqrt);
        }

        print_mo_coefficients_matrix();
        print_density_matrix();
    }
}

double Mol::hf_solver::calc_initial_fock_matrix()
{
    Eigen::SelfAdjointEigenSolver<EigenMatrix<double> > solver(s_mat);
    const EigenMatrix<double>& s_evecs = solver.eigenvectors();
    const EigenMatrix<double>& s_evals = solver.eigenvalues();

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index i = 0; i < norbitals; ++i)
         for (Index j = 0; j < norbitals; ++j)
         {
             s_mat_sqrt(i, j) = 0;
             for (Index k = 0; k < norbitals; ++k)
                s_mat_sqrt(i, j) += (1.0 / sqrt(s_evals(k))) * s_evecs(i, k) * s_evecs(j, k);
         }
    
    fp_mat.noalias() = s_mat_sqrt.transpose() * hcore_mat * s_mat_sqrt;

    return s_evals.minCoeff();
}

void Mol::hf_solver::calc_density_matrix()
{
    Eigen::SelfAdjointEigenSolver<EigenMatrix<double> > solver(fp_mat, Eigen::ComputeEigenvectors);
    
    if(iteration > 0) d_mat_previous = d_mat;
    
    if (!soscf)
    {
        const EigenMatrix<double>& f_mat_evecs = solver.eigenvectors();
        mo_coff.noalias() = s_mat_sqrt * f_mat_evecs; // only if soscf is not active
    }

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2)
    #endif
    for (Index i = 0; i < norbitals; ++i)
    {
        for (Index j = 0; j < norbitals; ++j)
        {
            d_mat(i, j) = 0.0;
            for (Index k = 0; k < nelectrons / 2; ++k)
            {
                d_mat(i, j) += mo_coff(i, k) * mo_coff(j, k);
            }
        }
    }

    // 1) We'll guard againt damping smaller than 0.2 or we'll probably never converge in a 100 cycles
    // 2) Lets not waste time recalculating density if damping is close to one becuase that's effectively undamped;

    double scf_damp = std::fabs(hf_settings::get_scf_damping_factor());
    
    if (iteration > 0 && std::fabs(scf_damp) > 0.2 && std::fabs(scf_damp) < 0.9)
        d_mat = (1.0 - scf_damp) * d_mat_previous + scf_damp * d_mat.eval();
    
    calc_scf_energy();
}

void Mol::hf_solver::calc_scf_energy()
{
    scf_energy_previous = scf_energy_current;
    // f_mat = hcore_mat at iteration 0
    (iteration > 0) ? scf_energy_current = d_mat.cwiseProduct(hcore_mat + f_mat).sum()
                    : scf_energy_current = d_mat.cwiseProduct(hcore_mat + hcore_mat).sum();

    scf_energies.emplace_back(scf_energy_current);
}

void Mol::hf_solver::update_fock_matrix()
{
    ++iteration;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index i = 0; i < norbitals; ++i)
    {
        for (Index j = i; j < norbitals; ++j)
        {
            f_mat(i, j) = hcore_mat(i, j);
            for (Index k = 0; k < norbitals; ++k)
            {
                for (Index l = 0; l < norbitals; ++l)
                {
                    Index ijkl = index_ijkl(i, j, k, l);
                    Index ikjl = index_ijkl(i, k, j, l);
                    f_mat(i, j) += d_mat(k, l) * (2.0 * e_rep_mat(ijkl) - e_rep_mat(ikjl));
                }
            }
            
            f_mat(j, i) = f_mat(i, j);
        }
    }

    diis_ptr->diis_extrapolate(f_mat, s_mat, d_mat, s_mat_sqrt, diis_log);
    fp_mat.noalias() = s_mat_sqrt.transpose() * f_mat * s_mat_sqrt;
    
    if (iteration > 1 && hf_settings::get_soscf()) so_scf(); // SAD needs the previous cycle
                                                             // In any case we'll not be sufficiently converged
    calc_density_matrix();
}

void Mol::hf_solver::update_fock_matrix_scf_direct()
{
    std::cout << "SCF Direct is under maintenance - reinstated soon.  \n";
    exit(EXIT_FAILURE);

    bool is_spherical = molecule->use_pure_am();
    std::unique_ptr<ERIOS::Erios> os_ptr  = std::make_unique<ERIOS::Erios>(is_spherical);

   // const auto& orbitals = molecule->get_orbitals();
    ++iteration;

    EigenMatrix<double> g_mat = EigenMatrix<double>::Zero(norbitals, norbitals);
    
    for (Index i = 0; i < norbitals; ++i)
    {
        for (Index j = 0; j <= i; ++j)
        {
            f_mat(i, j) = hcore_mat(i, j);
            f_mat(j, i) = f_mat(i, j);
            const Index ij = hfscfmath::index_ij(i, j);
            
            for (Index k = 0; k < norbitals; ++k)
            {
                for (Index l = 0; l <= k; ++l)
                {
                    const Index kl = hfscfmath::index_ij(k, l);

                    if(ij < kl) continue;

                    EigenVector<double> delDmax = EigenVector<double>(6);
                    delDmax << 4.0 * std::fabs(d_mat(i, j)), 4.0 * std::fabs(d_mat(k, l)), std::fabs(d_mat(i, k)),
                    std::fabs(d_mat(i, l)), std::fabs(d_mat(j, k)), std::fabs(d_mat(j, l));

                    const double dmax = delDmax.maxCoeff();
                    const double erilimit = std::sqrt(e_rep_screen(i, j)) * std::sqrt(e_rep_screen(k, l)) * dmax;

                    if(erilimit < hf_settings::get_integral_tol()) continue;

                    double d12 = 2.0;
                    double d34 = 2.0;
                    
                    if(i == j) d12 = 1.0;
                    if(k == l) d34 = 1.0;
                    
                    double d1234 = 2.0;
                    
                    if (i == k)
                    {
                        if (j == l) d1234 = 1.0;
                    }
                    
                    const double degen = d12 * d34 * d1234;
                    double e_rep = 0.0;

                    if(i == k && j == l) 
                    {
                        e_rep = degen * e_rep_screen(i, j);
                    }
                    else
                    {
                        // const auto& cgf_orb1 = orbitals[static_cast<size_t>(i)];
                        // const auto& cgf_orb2 = orbitals[static_cast<size_t>(j)];
                        // const auto& cgf_orb3 = orbitals[static_cast<size_t>(k)];
                        // const auto& cgf_orb4 = orbitals[static_cast<size_t>(l)];

                        e_rep = degen;  // TODO * os_ptr->contracted_obararecur_rep(cgf_orb1, cgf_orb2, 
                                                                          //   cgf_orb3, cgf_orb4);
                    }

                    g_mat(i, j) += d_mat(k, l) * e_rep;
                    g_mat(k, l) += d_mat(i, j) * e_rep;
                    g_mat(i, k) += -0.25 * d_mat(j, l) * e_rep;
                    g_mat(j, l) += -0.25 * d_mat(i, k) * e_rep;
                    g_mat(i, l) += -0.25 * d_mat(j, k) * e_rep;
                    g_mat(k, j) += -0.25 * d_mat(i, l) * e_rep;
                }
            }
        }
    }

    f_mat += 0.5 * (g_mat + g_mat.transpose());

    diis_ptr->diis_extrapolate(f_mat, s_mat, d_mat, s_mat_sqrt, diis_log);
    fp_mat.noalias() = s_mat_sqrt.transpose() * f_mat * s_mat_sqrt;
    calc_density_matrix();
}

double Mol::hf_solver::get_one_electron_energy() const 
{
    if (use_sym)
    {
        double e_1e = 0;
        for (size_t i = 0; i < vd_mat.size(); ++i)
            e_1e += vd_mat[i].cwiseProduct(vhcore_mat[i] + vhcore_mat[i]).sum();
        
        return e_1e;
    }
    else
        return d_mat.cwiseProduct(hcore_mat + hcore_mat).sum();
}

bool Mol::hf_solver::scf_run(bool print)
{
    bool converged = true;
    double rms_tol = hf_settings::get_rms_tol();
    double e_tol = hf_settings::get_energy_tol();
    int diis_steps = hf_settings::get_diis_size();
    const bool scf_direct = hf_settings::get_scf_direct();
    const int maxiter = hf_settings::get_max_scf_iterations();

    diis_ptr.reset();
    diis_ptr = std::make_unique<DIIS_SOLVER::diis_solver>(diis_steps);

    for(;;)
    {
        if (scf_direct)
            update_fock_matrix_scf_direct(); 
        else
        {
            if (use_sym)
                update_fock_matrix_with_symmetry();
            else
                update_fock_matrix();
        }
        
        double rms_diff = diis_ptr->get_current_rms();
        
        if(rms_diff < rms_tol && std::fabs(scf_energy_current - scf_energy_previous) < e_tol)
            break;
        else if (iteration == maxiter)
        {
            print_scf_energies();
            std::clog << "\n!!! Reached a " << maxiter << " iterations. Failed to converge !!!\n";
            converged = false;
            exit(EXIT_FAILURE);
        }

        if (hf_settings::get_verbosity() > 3 && print)
        {
            std::cout << "\n******************\n";
            std::cout << "Iteration: " << iteration;
            std::cout << "\n******************\n";
            print_fock_matrix();
            print_mo_coefficients_matrix();
            print_density_matrix();
        }

        if(!converged) break;
    }

    if (use_sym) // For now we project everything back to C1 for post tasks
    {
        const auto& irreps = molecule->get_irreps();
        const auto& species = molecule->get_symmetry_species();
        vmo_energies = std::vector<EigenMatrix<double>>(irreps.size());
        mo_energies_sym = std::vector<std::pair<double, std::string>>(norbitals);

        Index offset = 0;
        for (size_t m = 0; m < irreps.size(); ++m)
        {
            vmo_energies[m] = vmo_coff[m].transpose() * vf_mat[m] * vmo_coff[m];
            
            for (Index i = 0; i < vmo_energies[m].outerSize(); ++i)
                mo_energies_sym[i + offset] = std::make_pair(vmo_energies[m](i, i), species[m]);
            
            offset += irreps[m];
        }

        std::sort(mo_energies_sym.begin(), mo_energies_sym.end());

        const std::vector<EigenMatrix<double>>& symblocks = molecule->get_sym_blocks();

        d_mat.setZero(); f_mat.setZero(); fp_mat.setZero();
        for (size_t d = 0; d < irreps.size(); ++d)
        {
            d_mat += symblocks[d] * vd_mat[d] * symblocks[d].transpose();
            f_mat += symblocks[d] * vf_mat[d] * symblocks[d].transpose();
            fp_mat += symblocks[d] * vfp_mat[d] * symblocks[d].transpose();
        }
    }
    // Final coefficients
    Eigen::SelfAdjointEigenSolver<EigenMatrix<double> > solver(fp_mat);
    const EigenMatrix<double>& f_mat_evecs = solver.eigenvectors();
    mo_coff.noalias() = s_mat_sqrt * f_mat_evecs;
    
    mo_energies.noalias() = mo_coff.transpose() * f_mat * mo_coff;

    return converged;
}

void Mol::hf_solver::get_sad_guess(bool print)
{
    const auto get_offset = [this](int at_Z) -> std::vector<Index>
    {
        const auto& mask = (molecule->use_pure_am()) ? molecule->get_atom_spherical_mask()
                                                     : molecule->get_atom_mask();
        const auto& zvals = molecule->get_z_values();

        std::vector<Index> offsets;

        size_t at = 0;
        for (const auto& Z : zvals)
        {
            if (Z == at_Z) offsets.emplace_back(mask[at].mask_start);
            ++at;
        }

        return offsets;
    };

    if (print)
    {
        if (hf_settings::get_verbosity() > 1)
        {
            std::cout << "\n  SAD guess for unique atoms start.";
            std::cout << "\n  *********************************\n";
        }
        else
            std::cout << "\n  Estimating a guess density using SAD\n";
    }

    d_mat.setZero();
    const auto& zval = molecule->get_z_values();
    for (size_t atom = 0; atom < molecule->get_atoms().size(); ++atom) 
    {
        // skip duplicate atoms
		repeat:
		for (size_t at = 0; at < atom; ++at)
		{
			if (at == static_cast<size_t>(molecule->get_atoms().size()) - 1) goto done;
			else if (zval[at] == zval[atom]){++atom; goto repeat;} // note: unsorted atoms so we repeat
		}

        std::unique_ptr<Mol::sad_uhf_solver> sad_ptr =
            std::make_unique<Mol::sad_uhf_solver>(molecule, e_rep_mat, s_mat, k_mat, atom);

        sad_ptr->init_data(print);
        sad_ptr->sad_run();
        const Eigen::Ref<const EigenMatrix<double>> dens_block = sad_ptr->get_atom_density();

        const auto offsets = get_offset(zval[atom]);
        // For each unique atom loop over the same atoms, stitch together atomic densities
        // to build the guess moleculular density matrix
        for (const auto &offset : offsets)
            for (Index i = offset; i < dens_block.outerSize() + offset; ++i)
                for (Index j = offset; j < dens_block.outerSize() + offset; ++j)
                    d_mat(i, j) = dens_block(i - offset, j - offset);
    }

    done:
    
    if (use_sym)
    {
        const std::vector<EigenMatrix<double>>& sblocks = molecule->get_sym_blocks();

        size_t k = 0;
        for(const auto &sm : sblocks)
        {
            vd_mat[k] = sm.transpose() * d_mat * sm;
            ++k;
        }
    }
    
    if (hf_settings::get_verbosity() > 1 && print)
    {
        std::cout << "\n  SAD guess for unique atoms end.";
        std::cout << "\n  *******************************\n";
    }
}
