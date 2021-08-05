#include "../settings/hfscf_settings.hpp"
#include "../irc/cart_to_int.hpp"
#include "hfscf_hf_print.hpp"
#include "hfscf_uhf_solver.hpp"
#include <Eigen/Eigenvalues>

using HF_SETTINGS::hf_settings;
using Eigen::Index;
using hfscfmath::index_ij;
using hfscfmath::index_ijkl;

void Mol::uhf_solver::get_data()
{
    nelectrons = molecule->get_num_electrons();
    norbitals = molecule->get_num_orbitals();

    if (nelectrons % 2 != 0 && molecule->get_multiplicity() == 1)
    {
        std::cout << "\n\n  Error: Error in spin value for given number of electrons. Aborting.\n";
        exit(EXIT_FAILURE);
    }

    nalpha = molecule->get_nalpha();
    nbeta = molecule->get_nbeta();

    if (nalpha != nbeta && molecule->get_multiplicity() == 1) 
    {
        std::cout << "\n\n  Error: Error in spin value for given number of electrons. Aborting.\n";
        exit(EXIT_FAILURE);
    }

    f_mat_b = EigenMatrix<double>(norbitals, norbitals);
    fp_mat_b = EigenMatrix<double>(norbitals, norbitals);
    mo_coff_b = EigenMatrix<double>(norbitals, norbitals);
    d_mat_b = EigenMatrix<double>(norbitals, norbitals);
    d_mat_previous_b = EigenMatrix<double>(norbitals, norbitals);

    bool is_spherical = molecule->use_pure_am();
    if (is_spherical && hf_settings::get_use_symmetry()) use_sym = true;

    if (use_sym)
    {
        const auto &irreps = molecule->get_irreps();
        vfp_mat_b = std::vector<EigenMatrix<double>>(irreps.size());
        vd_mat_b = std::vector<EigenMatrix<double>>(irreps.size());
        vd_mat_previous_b = std::vector<EigenMatrix<double>>(irreps.size());
        vmo_coff_b = std::vector<EigenMatrix<double>>(irreps.size());
        vf_mat_b = std::vector<EigenMatrix<double>>(irreps.size());

        for (size_t i = 0; i < irreps.size(); ++i)
        {
            vfp_mat_b[i] = EigenMatrix<double>(irreps[i], irreps[i]);
            vd_mat_b[i] = EigenMatrix<double>(irreps[i], irreps[i]);
            vd_mat_previous_b[i] = EigenMatrix<double>(irreps[i], irreps[i]);
            vmo_coff_b[i] = EigenMatrix<double>(irreps[i], irreps[i]);
            vf_mat_b[i] = EigenMatrix<double>(irreps[i], irreps[i]);
        }
    }

    hf_solver::get_data(); // init base object members
}

void Mol::uhf_solver::init_data(bool print, bool use_previous_density)
{   // Use base class
    hf_solver::init_data(print, use_previous_density);
}

void Mol::uhf_solver::calc_density_matrix()
{
    if(0 == iteration)
    {
        Eigen::SelfAdjointEigenSolver<EigenMatrix<double> > solver(fp_mat, Eigen::ComputeEigenvectors);
        EigenMatrix<double> f_mat_evecs = solver.eigenvectors();
        mo_coff = s_mat_sqrt * f_mat_evecs;
        mo_coff_b = mo_coff;

        if(nalpha == nbeta && hf_settings::get_uhf_guess_mix())
        {   
            const double k = 0.15;
            const double kfactor =  (1.0 / std::sqrt(1.0 + std::pow(k, 2)));
            for(Index i = 0; i < norbitals; ++i)
            {
                double homo = mo_coff(i, nalpha - 1);
                double lumo = mo_coff(i, nalpha);
                mo_coff(i, nalpha - 1) = kfactor * (homo + k * lumo);
                mo_coff(i, nalpha) = kfactor * (lumo - k * homo);
            }
        }
    }
    else 
    {
        d_mat_previous = d_mat;
        d_mat_previous_b = d_mat_b;

        if (!soscf)
        {
            Eigen::SelfAdjointEigenSolver<EigenMatrix<double> > solver_alpha(fp_mat, Eigen::ComputeEigenvectors);
            EigenMatrix<double> f_mat_evecs_alpha = solver_alpha.eigenvectors();
            mo_coff = s_mat_sqrt * f_mat_evecs_alpha;
            
            Eigen::SelfAdjointEigenSolver<EigenMatrix<double> > solver_beta(fp_mat_b, Eigen::ComputeEigenvectors);
            EigenMatrix<double> f_mat_evecs_beta = solver_beta.eigenvectors();
            mo_coff_b = s_mat_sqrt * f_mat_evecs_beta;
        }
    }
    
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2)
    #endif
    for (Index i = 0; i < norbitals; ++i) 
        for (Index j = 0; j < norbitals; ++j) 
        {
            d_mat(i, j) = 0;
            for (Index k = 0; k < nalpha; ++k)
                d_mat(i, j) += mo_coff(i, k) * mo_coff(j, k);
        }

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2)
    #endif
    for (Index i = 0; i < norbitals; ++i) 
        for (Index j = 0; j < norbitals; ++j) 
        {
            d_mat_b(i, j) = 0;
            for (int k = 0; k < nbeta; ++k) 
                d_mat_b(i, j) += mo_coff_b(i, k) * mo_coff_b(j, k);
        }
    
    // 1) We'll guard againt damping smaller than 0.2 or we'll probably never converge in a 100 cycles
    // 2) Lets not waste time recalculating density if damping is close to one becuase that's effectively undamped;

    double scf_damp = std::fabs(hf_settings::get_scf_damping_factor());
    
    if (iteration > 0 && std::fabs(scf_damp) > 0.2 && std::fabs(scf_damp) < 0.9)
    {
        d_mat   = (1.0 - scf_damp) * d_mat_previous + scf_damp * d_mat.eval();
        d_mat_b = (1.0 - scf_damp) * d_mat_previous_b + scf_damp * d_mat_b.eval();
    }

    calc_scf_energy();
}

void Mol::uhf_solver::calc_scf_energy()
{
    scf_energy_previous = scf_energy_current;

    (iteration > 0) ? scf_energy_current = 0.5 * (d_mat.cwiseProduct(hcore_mat + f_mat).sum()
                                         + d_mat_b.cwiseProduct(hcore_mat + f_mat_b).sum())
                    : scf_energy_current = d_mat.cwiseProduct(hcore_mat + hcore_mat).sum();
                                        // f_mat = hcore_mat at iteration 0

    scf_energies.emplace_back(scf_energy_current);
}

void Mol::uhf_solver::update_fock_matrix()
{
   ++iteration;
    #ifdef _OPENMP
    #pragma omp parallel for schedule (dynamic)
    #endif
    for (Index i = 0; i < norbitals; ++i) 
    {
        for (Index j = i; j < norbitals; ++j) 
        {
            f_mat(i, j) = hcore_mat(i, j);
            f_mat_b(i, j) = hcore_mat(i, j);
            for (Index k = 0; k < norbitals; ++k) 
            {
                for (Index l = 0; l < norbitals; ++l) 
                {
                    Index ijkl = index_ijkl(i, j, k, l);
                    Index ikjl = index_ijkl(i, k, j, l);
                    f_mat(i, j)   += d_mat(k, l)   * (e_rep_mat(ijkl) - e_rep_mat(ikjl)) 
                                   + d_mat_b(k, l) * e_rep_mat(ijkl);
                    f_mat_b(i, j) += d_mat_b(k, l) * (e_rep_mat(ijkl) - e_rep_mat(ikjl)) 
                                   + d_mat(k, l)   * e_rep_mat(ijkl);
                }
            }
            f_mat(j, i) = f_mat(i, j);
            f_mat_b(j, i) = f_mat_b(i, j);
        }
    }

    diis_ptr->diis_extrapolate(f_mat, f_mat_b, s_mat, d_mat, d_mat_b, s_mat_sqrt, diis_log);
    fp_mat.noalias() = s_mat_sqrt.transpose() * f_mat * s_mat_sqrt;
    fp_mat_b.noalias() = s_mat_sqrt.transpose() * f_mat_b * s_mat_sqrt;

    if (iteration > 1 && hf_settings::get_soscf()) so_scf();

    calc_density_matrix();
}

void Mol::uhf_solver::update_fock_matrix_scf_direct()
{
    std::cout << "\n\n  Error: SCF direct not implemented for UHF.\n\n";
    exit(EXIT_FAILURE);
}

bool Mol::uhf_solver::scf_run(bool print)
{
    bool converged = true;
    double rms_tol = hf_settings::get_rms_tol();
    double e_tol = hf_settings::get_energy_tol();
    int diis_steps = hf_settings::get_diis_size();
    const int maxiter = hf_settings::get_max_scf_iterations();

    diis_ptr.reset();
    diis_ptr = std::make_unique<DIIS_SOLVER::diis_solver>(diis_steps);
    
    for(;;)
    {
        if (use_sym)
            update_fock_matrix_with_symmetry();
        else
            update_fock_matrix();

        double rms_diff_a = diis_ptr->get_current_rms_a();
        double rms_diff_b = diis_ptr->get_current_rms_b();

        if (rms_diff_a < rms_tol && rms_diff_b < rms_tol 
                                 && std::fabs(scf_energy_current - scf_energy_previous) < e_tol)
                break;
        else if (iteration == maxiter)
        {
            print_scf_energies();
            std::cout << "!!! Reached a " << maxiter << " iterations. Failed to converge !!!\n";
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
        vmo_energies_beta = std::vector<EigenMatrix<double>>(irreps.size());
        mo_energies_sym = std::vector<std::pair<double, std::string>>(norbitals);
        mo_energies_sym_b = std::vector<std::pair<double, std::string>>(norbitals);


        Index offset = 0;
        for (size_t m = 0; m < irreps.size(); ++m)
        {
            vmo_energies[m] = vmo_coff[m].transpose() * vf_mat[m] * vmo_coff[m];
            vmo_energies_beta[m] = vmo_coff_b[m].transpose() * vf_mat_b[m] * vmo_coff_b[m];
            
            for (Index i = 0; i < vmo_energies[m].outerSize(); ++i)
            {
                mo_energies_sym[i + offset] = std::make_pair(vmo_energies[m](i, i), species[m]);
                mo_energies_sym_b[i + offset] = std::make_pair(vmo_energies_beta[m](i, i), species[m]);
            }
            
            offset += irreps[m];
        }

        std::sort(mo_energies_sym.begin(), mo_energies_sym.end());
        std::sort(mo_energies_sym_b.begin(), mo_energies_sym_b.end());

        const std::vector<EigenMatrix<double>>& symblocks = molecule->get_sym_blocks();

        d_mat.setZero(); f_mat.setZero(); fp_mat.setZero();
        d_mat_b.setZero(); f_mat_b.setZero(); fp_mat_b.setZero();

        for (size_t d = 0; d < irreps.size(); ++d)
        {
            d_mat += symblocks[d] * vd_mat[d] * symblocks[d].transpose();
            d_mat_b += symblocks[d] * vd_mat_b[d] * symblocks[d].transpose();
            f_mat += symblocks[d] * vf_mat[d] * symblocks[d].transpose();
            f_mat_b += symblocks[d] * vf_mat_b[d] * symblocks[d].transpose();
            fp_mat += symblocks[d] * vfp_mat[d] * symblocks[d].transpose();
            fp_mat_b += symblocks[d] * vfp_mat_b[d] * symblocks[d].transpose();
        }

    }
    // final coefficients
    Eigen::SelfAdjointEigenSolver<EigenMatrix<double>> solver_a(fp_mat);
    const EigenMatrix<double>& f_mat_evecs_a = solver_a.eigenvectors();
    mo_coff.noalias() = s_mat_sqrt * f_mat_evecs_a;

    Eigen::SelfAdjointEigenSolver<EigenMatrix<double>> solver_b(fp_mat_b);
    const EigenMatrix<double>& f_mat_evecs_b = solver_b.eigenvectors();
    mo_coff_b.noalias() = s_mat_sqrt * f_mat_evecs_b;

    mo_energies.noalias() = mo_coff.transpose() * f_mat * mo_coff;
    mo_energies_beta.noalias() = mo_coff_b.transpose() * f_mat_b * mo_coff_b;

    return converged;
}

double Mol::uhf_solver::get_spin_contamination()
{
    EigenMatrix<double>ov = EigenMatrix<double>(norbitals, norbitals);

    EigenMatrix<double> cof_a = mo_coff; EigenMatrix<double> cof_b = mo_coff_b; 
    cof_a.conservativeResize(Eigen::NoChange, nalpha);
    cof_b.conservativeResize(Eigen::NoChange, nbeta);

    ov = cof_b.transpose() * s_mat * cof_a;
    double sum = ov.cwiseProduct(ov).sum();
    sum -= static_cast<double>(nbeta);
    return -sum + static_cast<double>(molecule->get_spin());
}

double Mol::uhf_solver::get_one_electron_energy() const 
{
    if (use_sym)
    {
        double e_1 = 0;
        for (size_t i = 0; i < vhcore_mat.size(); ++i)
            e_1 += (vd_mat[i] + vd_mat_b[i]).cwiseProduct(vhcore_mat[i]).sum();
        
        return e_1;
    }
    else
        return (d_mat + d_mat_b).cwiseProduct(hcore_mat).sum();
}

void Mol::uhf_solver::get_sad_guess(bool print)
{
    hf_solver::get_sad_guess(print); // base

    d_mat_b = d_mat;

    if (use_sym)
    {
        const std::vector<EigenMatrix<double>>& sblocks = molecule->get_sym_blocks();

        size_t k = 0;
        for(const auto &sm : sblocks)
        {
            vd_mat_b[k] = sm.transpose() * d_mat_b * sm;
            ++k;
        }
    }
}