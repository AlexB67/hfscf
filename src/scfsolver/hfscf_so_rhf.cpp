#include "hfscf_hf_solver.hpp"
#include "../settings/hfscf_settings.hpp"
#include <iomanip>

using Eigen::Index;
using hfscfmath::index_ij;
using hfscfmath::index_ijkl;
using HF_SETTINGS::hf_settings;

// RHF
void Mol::hf_solver::so_scf()
{
    EigenMatrix<double> Fmo = mo_coff.transpose() * f_mat * mo_coff;

    const Index occ = nelectrons / 2;
    const Index virt = norbitals - occ;

    EigenMatrix<double> grad = -4.0 * Fmo.block(0, occ, occ, virt);

    double gradmax = (grad.maxCoeff() > fabs(grad.minCoeff())) ? grad.maxCoeff() : fabs(grad.minCoeff());

    if (gradmax > 0.3) return; // Do not start SO before this cutoff is reached.
   
    if (!soscf)
    {
        so_iterstart = iteration; // The SCF iteration were SO begins
        diis_ptr->set_diis_range(0); // tell diis to stop extrapolating
        soscf = true; // From here we are in SO mode and flag it to SCF to act accordingly
    }

    double graddot = (grad.cwiseProduct(grad)).sum();

    EigenMatrix<double> Jprecond = EigenMatrix<double>(occ, virt); // Jacobian preconditioner
    EigenMatrix<double> rot = EigenMatrix<double>(occ, virt); // Rotation matrix

    for (Index i = 0; i < occ; ++i)
        for (Index a = 0; a < virt; ++a)
        {
            Jprecond(i, a) = - 4.0 * (Fmo(i, i) - Fmo(occ + a, occ + a));
            rot(i, a) = grad(i, a) / Jprecond(i, a);
        }
    
    EigenMatrix<double> Ax = solve_H_rot(rot, Fmo);
    EigenMatrix<double> r = grad - Ax;
    EigenMatrix<double> z = r.cwiseProduct(Jprecond.cwiseInverse());
    EigenMatrix<double> p = z;

    double rms = sqrt((r.cwiseProduct(r)).sum() / graddot);

    if (hf_settings::get_verbosity() > 3)
    {
        std::cout << "\n\n  SOSCF iterations begin.\n";
        std::cout << "  ***********************\n";
        std::cout << "\n  Guess micro iter: " << 0 << ", RMS: " 
                  << std::fixed << std::setprecision(6) << rms << "\n"; 
    }

    int maxiter = hf_settings::get_max_soscf_iterations();
    const double rms_tol = hf_settings::get_soscf_rms_tol(); 

    for (int iter = 0; iter < maxiter; ++iter)
    {
        double  rz_old = (r.cwiseProduct(z)).sum();
        EigenMatrix<double> Ap = solve_H_rot(p, Fmo);
        double alpha = rz_old / (Ap.cwiseProduct(p)).sum();
        rot += alpha * p;
        r -= alpha * Ap;
        z = r.cwiseProduct(Jprecond.cwiseInverse());
        rms = sqrt((r.cwiseProduct(r)).sum() / graddot);

        if (hf_settings::get_verbosity() > 3)
            std::cout << "  SOSCF micro iter: " << iter + 1 << ", RMS: " 
                      << std::fixed << std::setprecision(6) << rms << "\n";

        if (rms < rms_tol) break; // abort if rms is small enough
        
        double beta = (r.cwiseProduct(z)).sum() / rz_old;
        p = z + beta * p;
    }

    if (hf_settings::get_verbosity() > 3)
    {
        std::cout << "\n  SOSCF iterations end.\n";
        std::cout << "  *********************\n";
    }

    // Generate the Rotation Matrix U exp(-rot) as a Taylor expansion
    EigenMatrix<double> U = EigenMatrix<double>::Zero(norbitals, norbitals);
    for (Index i = 0; i < occ; ++i)
        for (Index a = 0; a < virt; ++a)
        {
            U(i, a + occ) = rot(i, a);
            U(occ + a, i) = -rot(i, a);
        }
    
    U += 0.5 * U * U;
    for (Index i = 0; i < norbitals; ++i)
        U(i, i) += 1.0;
    // Like Gram Schmidt, but using householder
    EigenMatrix<double> Uorth = U.transpose().householderQr().householderQ();

    mo_coff = mo_coff * Uorth;
}


EigenMatrix<double> Mol::hf_solver::solve_H_rot(const Eigen::Ref<const EigenMatrix<double>>& rot, 
                                                const Eigen::Ref<const EigenMatrix<double>>& Fmo) const
{
    const Index occ = nelectrons / 2;
    const Index virt = norbitals - occ;

    EigenMatrix<double> F = Fmo.block(0, 0, occ, occ) * rot
                          - rot * Fmo.block(occ, occ, virt, virt);

    const EigenMatrix<double>& Cocc = mo_coff.block(0, 0, norbitals, occ);
    const EigenMatrix<double>& Cvirt = mo_coff.block(0, occ, norbitals, virt);
    EigenMatrix<double> Cright = EigenMatrix<double>(norbitals, occ);

    for (Index s = 0; s < norbitals; ++s)
        for (Index i = 0; i < occ; ++i) 
        {
            Cright(s, i) = 0;
            for (Index a = 0; a < virt; ++a) 
                Cright(s, i) -= rot(i, a) * Cvirt(s, a);
        }

    EigenMatrix<double> D =  EigenMatrix<double>(norbitals, norbitals);

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2)
    #endif
    for (Index i = 0; i < norbitals; ++i)
    {
        for (Index j = 0; j < norbitals; ++j)
        {
            D(i, j) = 0.0;
            for (Index k = 0; k < occ; ++k)
                D(i, j) += Cocc(i, k) * Cright(j, k);
        }
    }

    EigenMatrix<double> J =  EigenMatrix<double>(norbitals, norbitals);
    EigenMatrix<double> K =  EigenMatrix<double>(norbitals, norbitals);
    
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index i = 0; i < norbitals; ++i)
    {
        for (Index j = 0; j < norbitals; ++j)
        {
            J(i, j) = 0;
            K(i, j) = 0;
            for (Index k = 0; k < norbitals; ++k)
            {
                for (Index l = 0; l < norbitals; ++l)
                {
                    Index ijkl = index_ijkl(i, j, k, l);
                    Index ikjl = index_ijkl(i, k, j, l);
                    J(i, j) += D(k, l) * e_rep_mat(ijkl);
                    K(i, j) += D(k, l) * e_rep_mat(ikjl);
                }
            }
        }
    }

    F += Cocc.transpose() * (4.0 * J - K.transpose() - K) * Cvirt;
    return -4.0 * F;
}

// My own variant with symmetry support. Dunno if it's the standard way, but it works
void Mol::hf_solver::so_scf_symmetry()
{
    const std::vector<int>& irreps = molecule->get_irreps();
    std::vector<EigenMatrix<double>> Fmo = std::vector<EigenMatrix<double>>(irreps.size());
    std::vector<EigenMatrix<double>> grad = std::vector<EigenMatrix<double>>(irreps.size());
    std::vector<bool> occ_virt =  std::vector<bool>(irreps.size());
    std::vector<Index> occ =  std::vector<Index>(irreps.size());
    std::vector<Index> virt =  std::vector<Index>(irreps.size());

    for (size_t m = 0; m < irreps.size(); ++m)
    {
        Index _occ = pop_index[m]; Index _virt = irreps[m] - pop_index[m];
        (!_occ || !_virt) ? occ_virt[m] = false : occ_virt[m] = true;
        occ[m] = _occ; virt[m] = _virt;
    }

    for (size_t m = 0; m < irreps.size(); ++m)
    {
        if (!occ_virt[m]) continue;

        Fmo[m].noalias() = vmo_coff[m].transpose() * vf_mat[m] * vmo_coff[m];

        grad[m] = -4.0 * Fmo[m].block(0, occ[m], occ[m], virt[m]);

        double gradmax = (grad[m].maxCoeff() > fabs(grad[m].minCoeff())) 
                       ? grad[m].maxCoeff() : fabs(grad[m].minCoeff());
    
        if (gradmax > 0.3) return; // Do not start SO before this cutoff raising this may cause trouble
    }

    if (!soscf)
    {
        so_iterstart = iteration; // The SCF iteration were SO begins
        diis_ptr->set_diis_range(0); // tell diis to stop extrapolating
        soscf = true; // From here we are in SO mode and flag it to SCF to act accordingly
    }

    double graddot = 0;
    for (size_t m = 0; m < irreps.size(); ++m)
    {
        if (!occ_virt[m]) continue;
        graddot += (grad[m].cwiseProduct(grad[m])).sum();
    }

    std::vector<EigenMatrix<double>> Jprecond = std::vector<EigenMatrix<double>>(irreps.size()); // Jacobian preconditioner
    std::vector<EigenMatrix<double>> rot = std::vector<EigenMatrix<double>>(irreps.size()); // Rotation matrix

    for (size_t m = 0; m < irreps.size(); ++m)
    {
        if (!occ_virt[m]) continue;

        Jprecond[m] = EigenMatrix<double>(occ[m], virt[m]);
        rot[m] = EigenMatrix<double>(occ[m], virt[m]);

        for (Index i = 0; i < occ[m]; ++i)
            for (Index a = 0; a < virt[m]; ++a)
            {
                Jprecond[m](i, a) = - 4.0 * (Fmo[m](i, i) - Fmo[m](occ[m] + a, occ[m] + a));
                rot[m](i, a) = grad[m](i, a) / Jprecond[m](i, a);
            }
    }
    
    const std::vector<EigenMatrix<double>> Ax = solve_H_rot_sym(rot, Fmo);
    std::vector<EigenMatrix<double>> r = std::vector<EigenMatrix<double>>(irreps.size());
    std::vector<EigenMatrix<double>> z = std::vector<EigenMatrix<double>>(irreps.size());
    std::vector<EigenMatrix<double>> p = std::vector<EigenMatrix<double>>(irreps.size());

    double rms = 0;
    for (size_t m = 0; m < irreps.size(); ++m)
    {
        if (!occ_virt[m]) continue;
        r[m] = grad[m] - Ax[m];
        z[m] = r[m].cwiseProduct(Jprecond[m].cwiseInverse());
        p[m] = z[m];
        rms += sqrt((r[m].cwiseProduct(r[m])).sum() / graddot);
    }

    if (hf_settings::get_verbosity() > 3)
    {
        std::cout << "\n\n  SOSCF iterations begin.\n";
        std::cout << "  ***********************\n";
        std::cout << "\n  Guess micro iter: " << 0 << ", RMS: " 
                  << std::fixed << std::setprecision(6) << rms << "\n"; 
    }

    const int maxiter = hf_settings::get_max_soscf_iterations();
    const double rms_tol = hf_settings::get_soscf_rms_tol(); 

    for (int iter = 0; iter < maxiter; ++iter)
    {   
        double rz_old = 0;
        for (size_t m = 0; m < irreps.size(); ++m)
        {
            if (!occ_virt[m]) continue;
            rz_old += (r[m].cwiseProduct(z[m])).sum();
        }

        const std::vector<EigenMatrix<double>> Ap = solve_H_rot_sym(p, Fmo);

        rms = 0; double alpha = rz_old; double denom = 0;
        for (size_t m = 0; m < irreps.size(); ++m)
        {
            if (!occ_virt[m]) continue;
            denom += (Ap[m].cwiseProduct(p[m])).sum();
        }

        alpha /= denom;
        for (size_t m = 0; m < irreps.size(); ++m)
        {
            if (!occ_virt[m]) continue;
            rot[m] += alpha * p[m];
            r[m] -= alpha * Ap[m];
            z[m] = r[m].cwiseProduct(Jprecond[m].cwiseInverse());
            rms += sqrt((r[m].cwiseProduct(r[m])).sum() / graddot);
        }

        if (hf_settings::get_verbosity() > 3)
            std::cout << "  SOSCF micro iter: " << iter + 1 << ", RMS: " 
                      << std::fixed << std::setprecision(6) << rms << "\n";

        if (rms < rms_tol) break; // abort if rms is small enough

        double beta = 0;
        for (size_t m = 0; m < irreps.size(); ++m)
        {
            if (!occ_virt[m]) continue;
            beta += (r[m].cwiseProduct(z[m])).sum() / rz_old;
        }

        for (size_t m = 0; m < irreps.size(); ++m)
        {
            if (!occ_virt[m]) continue;
            p[m] = z[m] + beta * p[m];
        }
    }

    if (hf_settings::get_verbosity() > 3)
    {
        std::cout << "\n  SOSCF iterations end.\n";
        std::cout << "  *********************\n";
    }

    // Generate the Rotation Matrix U exp(-rot) as a Taylor expansion

    for (size_t m = 0; m < irreps.size(); ++m)
    {
        EigenMatrix<double> U = EigenMatrix<double>::Zero(irreps[m], irreps[m]);

        for (Index i = 0; i < occ[m]; ++i)
            for (Index a = 0; a < virt[m]; ++a)
            {
                U(i, a + occ[m]) = rot[m](i, a);
                U(occ[m] + a, i) = -rot[m](i, a);
            }
            
        U += 0.5 * U * U; // + (1.0/6.0) * U * U * U + (1.0 / 24.0) * U * U * U * U;
        // seems the first order term does the job well enough
        for (Index i = 0; i < irreps[m]; ++i)
            U(i, i) += 1.0;
        // Like Gram Schmidt, but using householder
        EigenMatrix<double> Uorth = U.transpose().householderQr().householderQ();
        vmo_coff[m] = vmo_coff[m] * Uorth;
    }
}

std::vector<EigenMatrix<double>> Mol::hf_solver::solve_H_rot_sym(const std::vector<EigenMatrix<double>>& rot, 
                                                                 const std::vector<EigenMatrix<double>>& Fmo) const
{

    const std::vector<int>& irreps = molecule->get_irreps();
    const auto& symblocks = molecule->get_sym_blocks();

    std::vector<EigenMatrix<double>> F = std::vector<EigenMatrix<double>>(irreps.size());
    std::vector<EigenMatrix<double>> dens = std::vector<EigenMatrix<double>>(irreps.size());
    std::vector<EigenMatrix<double>> Cocc = std::vector<EigenMatrix<double>>(irreps.size());
    std::vector<EigenMatrix<double>> Cvirt = std::vector<EigenMatrix<double>>(irreps.size());
    
    for (size_t m = 0; m < irreps.size(); ++m)
    {
        Index occ_ = pop_index[m]; Index virt_ = irreps[m] - pop_index[m];
        if (!occ_ || !virt_) continue;

        dens[m] = EigenMatrix<double>::Zero(irreps[m], irreps[m]);

        F[m] = Fmo[m].block(0, 0, occ_, occ_) * rot[m]
             - rot[m] * Fmo[m].block(occ_, occ_, virt_, virt_);

        Cocc[m] = vmo_coff[m].block(0, 0, irreps[m], occ_);
        Cvirt[m] = vmo_coff[m].block(0, occ_, irreps[m], virt_);
        EigenMatrix<double> Cright = EigenMatrix<double>::Zero(irreps[m], occ_);

        for (Index s = 0; s < irreps[m]; ++s)
            for (Index i = 0; i < occ_; ++i)
            {
                Cright(s, i) = 0;
                for (Index a = 0; a < virt_; ++a) 
                    Cright(s, i) -= rot[m](i, a) * Cvirt[m](s, a);
            }

        #ifdef _OPENMP
        #pragma omp parallel for collapse(2)
        #endif
        for (Index i = 0; i < irreps[m]; ++i)
        {
            for (Index j = 0; j < irreps[m]; ++j)
            {
                dens[m](i, j) = 0.0;
                for (Index k = 0; k < occ_; ++k)
                    dens[m](i, j) += Cocc[m](i, k) * Cright(j, k);
            }
        }
    }

    EigenMatrix<double> D = EigenMatrix<double>::Zero(norbitals, norbitals);
    EigenMatrix<double> J = EigenMatrix<double>(norbitals, norbitals);
    EigenMatrix<double> K = EigenMatrix<double>(norbitals, norbitals);

    for (size_t d = 0; d < irreps.size(); ++d)
    {
        Index occ_ = pop_index[d];
        if (!occ_) continue; // virt not relevant
        D += symblocks[d] * dens[d] * symblocks[d].transpose();
    }

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index i = 0; i < norbitals; ++i)
    {
        for (Index j = 0; j < norbitals; ++j)
        {
            J(i, j) = 0;
            K(i, j) = 0;
            for (Index k = 0; k < norbitals; ++k)
            {
                for (Index l = 0; l < norbitals; ++l)
                {
                    Index ijkl = index_ijkl(i, j, k, l);
                    Index ikjl = index_ijkl(i, k, j, l);
                    J(i, j) += D(k, l) * e_rep_mat(ijkl);
                    K(i, j) += D(k, l) * e_rep_mat(ikjl);
                }
            }
        }
    }

    EigenMatrix<double> G = (4.0 * J - K.transpose() - K);

    for (size_t d = 0; d < irreps.size(); ++d)
    {
        Index occ_ = pop_index[d]; Index virt_ = irreps[d] - pop_index[d];
        if (!occ_ || !virt_) continue;

        const EigenMatrix<double> Gi = symblocks[d].transpose() * G * symblocks[d];

        F[d] += Cocc[d].transpose() * (Gi) * Cvirt[d];
        F[d] *= -4.0;
    }

    return F;
}
