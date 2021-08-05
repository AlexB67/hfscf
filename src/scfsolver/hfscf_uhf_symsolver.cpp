#include "../settings/hfscf_settings.hpp"
#include "../irc/cart_to_int.hpp"
#include "hfscf_hf_print.hpp"
#include "hfscf_uhf_solver.hpp"
#include <Eigen/Eigenvalues>

using HF_SETTINGS::hf_settings;
using Eigen::Index;
using hfscfmath::index_ij;
using hfscfmath::index_ijkl;

void Mol::uhf_solver::calc_scf_energy_with_symmetry()
{
    scf_energy_previous = scf_energy_current;
    scf_energy_current = 0;

    if (iteration > 0)
    {
        for (size_t i = 0; i < molecule->get_irreps().size(); ++i)
            scf_energy_current += 0.5 * (vd_mat[i].cwiseProduct(vhcore_mat[i] + vf_mat[i]).sum()
                               + vd_mat_b[i].cwiseProduct(vhcore_mat[i] + vf_mat_b[i]).sum());
    }
    else
    {
        for (size_t i = 0; i < molecule->get_irreps().size(); ++i)
            scf_energy_current += vd_mat[i].cwiseProduct(vhcore_mat[i] + vhcore_mat[i]).sum();
                                        // f_mat = hcore_mat at iteration 0
    }

    scf_energies.emplace_back(scf_energy_current);
}

void Mol::uhf_solver::calc_density_matrix_with_symmetry()
{
    std::vector<std::pair<double, double>> eps_a = std::vector<std::pair<double, double>>(norbitals);
    std::vector<std::pair<double, double>> eps_b = std::vector<std::pair<double, double>>(norbitals);

    if(0 == iteration)
    {
        Index offset = 0;
        for (size_t i = 0; i < vfp_mat.size(); ++i)
        {
            Eigen::SelfAdjointEigenSolver<EigenMatrix<double> > 
            solver(vfp_mat[i], Eigen::ComputeEigenvectors);
            
            const EigenMatrix<double>& f_mat_evecs = solver.eigenvectors();
            vmo_coff[i] = vs_mat_sqrt[i] * f_mat_evecs;
            vmo_coff_b[i] = vmo_coff[i];

            const EigenVector<double>& f_mat_evals = solver.eigenvalues();

            for (int l = 0; l < f_mat_evals.size(); ++l)
                eps_a[l + offset] = std::make_pair(f_mat_evals(l), i);

            offset += f_mat_evals.size();
        }

        eps_b = eps_a;
    }
    else 
    {
        Index offset = 0;
        for (size_t i = 0; i < vfp_mat.size(); ++i)
        {
            vd_mat_previous[i] = vd_mat[i];
            vd_mat_previous_b[i] = vd_mat_b[i];

            Eigen::SelfAdjointEigenSolver<EigenMatrix<double> > 
            solver_alpha(vfp_mat[i]);
            
            if(!soscf)
            {
                const EigenMatrix<double>& f_mat_evecs_alpha = solver_alpha.eigenvectors();
                vmo_coff[i] = vs_mat_sqrt[i] * f_mat_evecs_alpha;
            }
            
            Eigen::SelfAdjointEigenSolver<EigenMatrix<double> > 
            solver_beta(vfp_mat_b[i]);
            
            if(!soscf)
            {
                const EigenMatrix<double>& f_mat_evecs_beta = solver_beta.eigenvectors();
                vmo_coff_b[i] = vs_mat_sqrt[i] * f_mat_evecs_beta;
            }

            const EigenVector<double>& f_mat_evals_a = solver_alpha.eigenvalues();
            const EigenVector<double>& f_mat_evals_b = solver_beta.eigenvalues();

            for (int l = 0; l < f_mat_evals_a.size(); ++l)
            {
                eps_a[l + offset] = std::make_pair(f_mat_evals_a(l), i);
                eps_b[l + offset] = std::make_pair(f_mat_evals_b(l), i);
            }

            offset += f_mat_evals_a.size();
        }
    }
    
    std::sort(eps_a.begin(), eps_a.end());
    eps_a.resize(nalpha);
    std::sort(eps_b.begin(), eps_b.end());
    eps_b.resize(nbeta);

    if (!pop_index.size())
    {
        pop_index = std::vector<int>(molecule->get_irreps().size());
        pop_index_b = std::vector<int>(molecule->get_irreps().size());
    }

    for (size_t occ = 0; occ < molecule->get_irreps().size(); ++occ) 
    {
        int tot = 0;
        for (const auto& s : eps_a)
            if ((size_t)s.second == occ) ++tot;

        pop_index[occ] = tot;

        tot = 0;
        for (const auto& s : eps_b)
            if ((size_t)s.second == occ) ++tot;

        pop_index_b[occ] = tot;
    }

    for (size_t k = 0; k < vd_mat.size(); ++k)
    {
        const Index dim = vd_mat[k].outerSize();

        #ifdef _OPENMP
        #pragma omp parallel for collapse(2)
        #endif
        for (Index i = 0; i < dim; ++i)
        {
            for (Index j = 0; j < dim; ++j)
            {
                vd_mat[k](i, j) = 0.0;
                for (Index l = 0; l < pop_index[k]; ++l)
                {
                    vd_mat[k](i, j) += vmo_coff[k](i, l) * vmo_coff[k](j, l);
                }
            }
        }

        #ifdef _OPENMP
        #pragma omp parallel for collapse(2)
        #endif
        for (Index i = 0; i < dim; ++i)
        {
            for (Index j = 0; j < dim; ++j)
            {
                vd_mat_b[k](i, j) = 0.0;
                for (Index l = 0; l < pop_index_b[k]; ++l)
                {
                    vd_mat_b[k](i, j) += vmo_coff_b[k](i, l) * vmo_coff_b[k](j, l);
                }
            }
        }

        // 1) We'll guard againt damping smaller than 0.2 or we'll probably never converge in a 100 cycles
        // 2) Lets not waste time recalculating density if damping is close to one becuase that's effectively undamped;

        double scf_damp = std::fabs(hf_settings::get_scf_damping_factor());
        
        if (iteration > 0 && std::fabs(scf_damp) > 0.2 && std::fabs(scf_damp) < 0.9)
        {
            vd_mat[k] = (1.0 - scf_damp) * vd_mat_previous[k] + scf_damp * vd_mat[k].eval();
            vd_mat_b[k] = (1.0 - scf_damp) * vd_mat_previous_b[k] + scf_damp * vd_mat_b[k].eval();
        }
    }

    calc_scf_energy_with_symmetry();
}

void Mol::uhf_solver::update_fock_matrix_with_symmetry()
{
    ++iteration;
    const std::vector<int>& irreps = molecule->get_irreps();
    const auto& symblocks = molecule->get_sym_blocks();

    d_mat.setZero(); d_mat_b.setZero();
    for (size_t d = 0; d < vd_mat.size(); ++d)
    {
        d_mat += symblocks[d] * vd_mat[d] * symblocks[d].transpose();
        d_mat_b += symblocks[d] * vd_mat_b[d] * symblocks[d].transpose();
    }

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
    
    for (size_t m = 0; m < irreps.size(); ++m)
    {
        vf_mat[m].noalias() = symblocks[m].transpose() * f_mat * symblocks[m];
        vf_mat_b[m].noalias() = symblocks[m].transpose() * f_mat_b * symblocks[m];
        vfp_mat[m].noalias() = vs_mat_sqrt[m].transpose() * vf_mat[m] * vs_mat_sqrt[m];
        vfp_mat_b[m].noalias() = vs_mat_sqrt[m].transpose() * vf_mat_b[m] * vs_mat_sqrt[m];
    }
    
    if (iteration > 1 && hf_settings::get_soscf()) so_scf_symmetry();

    calc_density_matrix_with_symmetry();
}
