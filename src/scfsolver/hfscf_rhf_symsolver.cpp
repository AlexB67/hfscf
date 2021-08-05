#include "../settings/hfscf_settings.hpp"
#include "hfscf_hf_print.hpp"
#include "hfscf_hf_solver.hpp"
#include <Eigen/Eigenvalues>
#include <algorithm>

using HF_SETTINGS::hf_settings;
using Eigen::Index;
using hfscfmath::index_ij;
using hfscfmath::index_ijkl;


double Mol::hf_solver::calc_initial_fock_matrix_with_symmetry()
{
    const std::vector<int> irreps = molecule->get_irreps();
    EigenVector<double> min_eval_irrep = EigenVector<double>(irreps.size());

    size_t k = 0;
    for (const auto& m : vs_mat)
    {
        const Eigen::Ref<const EigenMatrix<double>>&  ovlap = m;
        Eigen::SelfAdjointEigenSolver<EigenMatrix<double> > solver(ovlap);
        const EigenMatrix<double>& s_evecs = solver.eigenvectors();
        const EigenVector<double>& s_evals = solver.eigenvalues();
        min_eval_irrep[k] = s_evals.minCoeff();
        const Index dim = irreps[k];

        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (Index i = 0; i < dim; ++i)
            for (Index j = 0; j < dim; ++j)
            {
                vs_mat_sqrt[k](i, j) = 0;
                for (Index l = 0; l < dim; ++l)
                    vs_mat_sqrt[k](i, j) += (1.0 / sqrt(s_evals(l))) * s_evecs(i, l) * s_evecs(j, l);
            }
        ++k;
    }

    const auto& symblocks = molecule->get_sym_blocks();
    s_mat_sqrt.setZero();

    for (size_t d = 0; d < vd_mat.size(); ++d)
    {
        s_mat_sqrt += symblocks[d] * vs_mat_sqrt[d] * symblocks[d].transpose();
        vfp_mat[d] = vs_mat_sqrt[d].transpose() * vhcore_mat[d] * vs_mat_sqrt[d];
    }

    return min_eval_irrep.minCoeff();
}

void Mol::hf_solver::calc_density_matrix_with_symmetry()
{
    std::vector<std::pair<double, double>> eps
    = std::vector<std::pair<double, double>>(norbitals);

    size_t k = 0;
    Index offset = 0;
    for (const auto& m : vfp_mat)
    {
        if (iteration > 0) vd_mat_previous[k] = vd_mat[k];

        Eigen::SelfAdjointEigenSolver<EigenMatrix<double> > solver(m);
        
        if (!soscf)
        {
            const EigenMatrix<double>& f_mat_evecs = solver.eigenvectors();
            vmo_coff[k].noalias() = vs_mat_sqrt[k] * f_mat_evecs;
        }
        
        const EigenVector<double>& f_mat_evals = solver.eigenvalues();

        for (Index l = 0; l < f_mat_evals.size(); ++l)
            eps[l + offset] = std::make_pair(f_mat_evals(l), k);

        offset += f_mat_evals.size();
        ++k;
    }

    std::sort(eps.begin(), eps.end());
    eps.resize(nelectrons / 2);

    if (!pop_index.size()) pop_index = std::vector<int>(molecule->get_irreps().size());

    for (size_t occ = 0; occ < molecule->get_irreps().size(); ++occ) 
    {
        int tot = 0;
        for (const auto& s : eps)
            if ((size_t)s.second == occ) ++tot;

        pop_index[occ] = tot;
    }

    k = 0;
    for (const auto& m : vd_mat)
    {
        const Index dim = m.outerSize();

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

        // 1) We'll guard againt damping smaller than 0.2 or we'll probably never converge in a 100 cycles
        // 2) Lets not waste time recalculating density if damping is close to one because that's effectively undamped;

        double scf_damp = std::fabs(hf_settings::get_scf_damping_factor());
        
        if (iteration > 0 && std::fabs(scf_damp) > 0.2 && std::fabs(scf_damp) < 0.9)
            vd_mat[k] = (1.0 - scf_damp) * vd_mat_previous[k] + scf_damp * vd_mat[k].eval();
        
        ++k;
    }
    
    calc_scf_energy_with_symmetry();
}

void Mol::hf_solver::calc_scf_energy_with_symmetry()
{
    scf_energy_previous = scf_energy_current;
    scf_energy_current = 0;
    // f_mat = hcore_mat at iteration 0
    if (iteration > 0)
    { 
        for (size_t i = 0; i < molecule->get_irreps().size(); ++i)
            scf_energy_current += vd_mat[i].cwiseProduct(vhcore_mat[i] + vf_mat[i]).sum();
    }
    else
    {
        for (size_t i = 0; i < molecule->get_irreps().size(); ++i)
            scf_energy_current += vd_mat[i].cwiseProduct(vhcore_mat[i] + vhcore_mat[i]).sum();
    }

    scf_energies.emplace_back(scf_energy_current);
}

void Mol::hf_solver::update_fock_matrix_with_symmetry()
{
    ++iteration;
    const std::vector<int>& irreps = molecule->get_irreps();
    const auto& symblocks = molecule->get_sym_blocks();
    
    d_mat.setZero();
    for (size_t d = 0; d < vd_mat.size(); ++d)
        d_mat += symblocks[d] * vd_mat[d] * symblocks[d].transpose();
    
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

    for (size_t m = 0; m < irreps.size(); ++m)
    {
        vf_mat[m].noalias() = symblocks[m].transpose() * f_mat * symblocks[m];
        vfp_mat[m].noalias() = vs_mat_sqrt[m].transpose() * vf_mat[m] * vs_mat_sqrt[m];
    }

    if (iteration > 1 && hf_settings::get_soscf()) so_scf_symmetry();

    calc_density_matrix_with_symmetry();
}
