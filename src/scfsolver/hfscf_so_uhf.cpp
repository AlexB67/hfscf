#include "hfscf_uhf_solver.hpp"
#include "../settings/hfscf_settings.hpp"
#include <iomanip>

using Eigen::Index;
using hfscfmath::index_ij;
using hfscfmath::index_ijkl;
using HF_SETTINGS::hf_settings;

// UHF
void Mol::uhf_solver::so_scf()
{
    EigenMatrix<double> Fmo_a = mo_coff.transpose() * f_mat * mo_coff;
    EigenMatrix<double> Fmo_b = mo_coff_b.transpose() * f_mat_b * mo_coff_b;

    const Index occa = molecule->get_nalpha();
    const Index virta = norbitals - occa;
    const Index occb = molecule->get_nbeta();
    const Index virtb = norbitals - occb;

    EigenMatrix<double> grad_a = -4.0 * Fmo_a.block(0, occa, occa, virta);
    EigenMatrix<double> grad_b = -4.0 * Fmo_b.block(0, occb, occb, virtb);

    const double gradmax_a = (grad_a.maxCoeff() > fabs(grad_a.minCoeff())) ? grad_a.maxCoeff() : fabs(grad_a.minCoeff());
    const double gradmax_b = (grad_b.maxCoeff() > fabs(grad_b.minCoeff())) ? grad_b.maxCoeff() : fabs(grad_b.minCoeff());

    if (gradmax_a > 0.3 || gradmax_b > 0.3) return; // Do not start SO before this cutoff is reached.
   
    if (!soscf)
    {
        so_iterstart = iteration; // The SCF iteration were SO begins
        diis_ptr->set_diis_range(0); // tell diis to stop extrapolating
        soscf = true; // From here we are in SO mode and flag it to SCF to act accordingly
    }

    double graddot = (grad_a.cwiseProduct(grad_a)).sum() + (grad_b.cwiseProduct(grad_b)).sum();

    EigenMatrix<double> Jprecond_a = EigenMatrix<double>(occa, virta); // Jacobian preconditioner
    EigenMatrix<double> rot_a = EigenMatrix<double>(occa, virta); // Rotation matrix

    for (Index i = 0; i < occa; ++i)
        for (Index a = 0; a < virta; ++a)
        {
            Jprecond_a(i, a) = - 4.0 * (Fmo_a(i, i) - Fmo_a(occa + a, occa + a));
            rot_a(i, a) = grad_a(i, a) / Jprecond_a(i, a);
        }
    
    EigenMatrix<double> Jprecond_b = EigenMatrix<double>(occb, virtb); // Jacobian preconditioner
    EigenMatrix<double> rot_b = EigenMatrix<double>(occb, virtb); // Rotation matrix

    for (Index i = 0; i < occb; ++i)
        for (Index a = 0; a < virtb; ++a)
        {
            Jprecond_b(i, a) = - 4.0 * (Fmo_b(i, i) - Fmo_b(occb + a, occb + a));
            rot_b(i, a) = grad_b(i, a) / Jprecond_b(i, a);
        }

    EigenMatrix<double> Ax_a, Ax_b; 
    solve_H_rot(rot_a, rot_b, Fmo_a, Fmo_b, Ax_a, Ax_b);

    EigenMatrix<double> r_a = grad_a - Ax_a;
    EigenMatrix<double> z_a = r_a.cwiseProduct(Jprecond_a.cwiseInverse());
    EigenMatrix<double> p_a = z_a;
    
    EigenMatrix<double> r_b = grad_b - Ax_b;
    EigenMatrix<double> z_b = r_b.cwiseProduct(Jprecond_b.cwiseInverse());
    EigenMatrix<double> p_b = z_b;

    double rms = sqrt((r_a.cwiseProduct(r_a)).sum() / graddot)
               + sqrt((r_b.cwiseProduct(r_b)).sum() / graddot);

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
        double  rz_old = (r_a.cwiseProduct(z_a)).sum() + (r_b.cwiseProduct(z_b)).sum();
        EigenMatrix<double> Ap_a, Ap_b;
        solve_H_rot(p_a, p_b, Fmo_a, Fmo_b, Ap_a, Ap_b);

        double alpha = rz_old / (Ap_a.cwiseProduct(p_a).sum() + Ap_b.cwiseProduct(p_b).sum());
        rot_a += alpha * p_a;
        r_a -= alpha * Ap_a;
        z_a = r_a.cwiseProduct(Jprecond_a.cwiseInverse());
        
        rot_b += alpha * p_b;
        r_b -= alpha * Ap_b;
        z_b = r_b.cwiseProduct(Jprecond_b.cwiseInverse());

        rms = sqrt((r_a.cwiseProduct(r_a)).sum() / graddot) +  sqrt((r_b.cwiseProduct(r_b)).sum() / graddot);

        if (hf_settings::get_verbosity() > 3)
            std::cout << "  SOSCF micro iter: " << iter + 1 << ", RMS: " 
                      << std::fixed << std::setprecision(6) << rms << "\n";

        if (rms < rms_tol) break; // abort if rms is small enough
        
        double beta = (r_a.cwiseProduct(z_a)).sum() / rz_old + (r_b.cwiseProduct(z_b)).sum() / rz_old;
        p_a = z_a + beta * p_a;
        p_b = z_b + beta * p_b;
    }

    if (hf_settings::get_verbosity() > 3)
    {
        std::cout << "\n  SOSCF iterations end.\n";
        std::cout << "  *********************\n";
    }

    // Generate the Rotation Matrix U exp(-rot) as a Taylor expansion
    EigenMatrix<double> U = EigenMatrix<double>::Zero(norbitals, norbitals);
    for (Index i = 0; i < occa; ++i)
        for (Index a = 0; a < virta; ++a)
        {
            U(i, a + occa) = rot_a(i, a);
            U(occa + a, i) = -rot_a(i, a);
        }
    
    U += 0.5 * U * U;
    for (Index i = 0; i < norbitals; ++i)
        U(i, i) += 1.0;
    // Like Gram Schmidt, but using householder
    EigenMatrix<double> Uorth = U.transpose().householderQr().householderQ();

    mo_coff = mo_coff * Uorth;

    U.setZero();
    for (Index i = 0; i < occb; ++i)
        for (Index a = 0; a < virtb; ++a)
        {
            U(i, a + occb) = rot_b(i, a);
            U(occb + a, i) = -rot_b(i, a);
        }
    
    U += 0.5 * U * U;
    for (Index i = 0; i < norbitals; ++i)
        U(i, i) += 1.0;
    // Like Gram Schmidt, but using householder
    Uorth = U.transpose().householderQr().householderQ();

    mo_coff_b = mo_coff_b * Uorth;
}


void Mol::uhf_solver::solve_H_rot(const Eigen::Ref<const EigenMatrix<double>>& rot_a, 
                                  const Eigen::Ref<const EigenMatrix<double>>& rot_b, 
                                  const Eigen::Ref<const EigenMatrix<double>>& Fmo_a,
                                  const Eigen::Ref<const EigenMatrix<double>>& Fmo_b, 
                                  EigenMatrix<double>& Fa, EigenMatrix<double>& Fb) const
{
    const Index occa = molecule->get_nalpha();
    const Index virta = norbitals - occa;
    const Index occb = molecule->get_nbeta();
    const Index virtb = norbitals - occb;
    
    Fa = Fmo_a.block(0, 0, occa, occa) * rot_a - rot_a * Fmo_a.block(occa, occa, virta, virta);
    Fb = Fmo_b.block(0, 0, occb, occb) * rot_b - rot_b * Fmo_b.block(occb, occb, virtb, virtb);

    const EigenMatrix<double>& Cocc_a = mo_coff.block(0, 0, norbitals, occa);
    const EigenMatrix<double>& Cvirt_a = mo_coff.block(0, occa, norbitals, virta);
    EigenMatrix<double> Cright_a = EigenMatrix<double>(norbitals, occa);

    for (Index s = 0; s < norbitals; ++s)
        for (Index i = 0; i < occa; ++i) 
        {
            Cright_a(s, i) = 0;
            for (Index a = 0; a < virta; ++a) 
                Cright_a(s, i) -= rot_a(i, a) * Cvirt_a(s, a);
        }
    
    const EigenMatrix<double>& Cocc_b = mo_coff_b.block(0, 0, norbitals, occb);
    const EigenMatrix<double>& Cvirt_b = mo_coff_b.block(0, occb, norbitals, virtb);
    EigenMatrix<double> Cright_b = EigenMatrix<double>(norbitals, occb);

    for (Index s = 0; s < norbitals; ++s)
        for (Index i = 0; i < occb; ++i) 
        {
            Cright_b(s, i) = 0;
            for (Index a = 0; a < virtb; ++a) 
                Cright_b(s, i) -= rot_b(i, a) * Cvirt_b(s, a);
        }

    EigenMatrix<double> D_a = EigenMatrix<double>(norbitals, norbitals);

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2)
    #endif
    for (Index i = 0; i < norbitals; ++i)
    {
        for (Index j = 0; j < norbitals; ++j)
        {
            D_a(i, j) = 0.0;
            for (Index k = 0; k < occa; ++k)
                D_a(i, j) += Cocc_a(i, k) * Cright_a(j, k);
        }
    }

    EigenMatrix<double> D_b = EigenMatrix<double>(norbitals, norbitals);

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2)
    #endif
    for (Index i = 0; i < norbitals; ++i)
    {
        for (Index j = 0; j < norbitals; ++j)
        {
            D_b(i, j) = 0.0;
            for (Index k = 0; k < occb; ++k)
                D_b(i, j) += Cocc_b(i, k) * Cright_b(j, k);
        }
    }

    EigenMatrix<double> Jab =  EigenMatrix<double>(norbitals, norbitals);
    EigenMatrix<double> Ka  =  EigenMatrix<double>(norbitals, norbitals);
    EigenMatrix<double> Kb  =  EigenMatrix<double>(norbitals, norbitals);
    
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index i = 0; i < norbitals; ++i)
    {
        for (Index j = 0; j < norbitals; ++j)
        {
            Jab(i, j) = 0; Ka(i, j) = 0; Kb(i, j) = 0;
            for (Index k = 0; k < norbitals; ++k)
            {
                for (Index l = 0; l < norbitals; ++l)
                {
                    Index ijkl = index_ijkl(i, j, k, l);
                    Index ikjl = index_ijkl(i, k, j, l);
                    Jab(i, j) += (D_a(k, l) + D_b(k, l)) * e_rep_mat(ijkl);
                    Ka(i, j) += D_a(k, l) * e_rep_mat(ikjl);
                    Kb(i, j) += D_b(k, l) * e_rep_mat(ikjl);
                }
            }
        }
    }

    Fa += Cocc_a.transpose() * (2.0 * Jab - Ka.transpose() - Ka) * Cvirt_a;
    Fa *= -4.0;
    Fb += Cocc_b.transpose() * (2.0 * Jab - Kb.transpose() - Kb) * Cvirt_b;
    Fb *= -4.0;
}

void Mol::uhf_solver::so_scf_symmetry()
{
    const std::vector<int>& irreps = molecule->get_irreps();
    // alpha symmetry blocks
    std::vector<EigenMatrix<double>> Fmo_a = std::vector<EigenMatrix<double>>(irreps.size());
    std::vector<EigenMatrix<double>> grad_a = std::vector<EigenMatrix<double>>(irreps.size());
    std::vector<bool> occ_virta =  std::vector<bool>(irreps.size());
    std::vector<Index> occa =  std::vector<Index>(irreps.size());
    std::vector<Index> virta =  std::vector<Index>(irreps.size());

    // alpha symmetry blocks
    std::vector<EigenMatrix<double>> Fmo_b = std::vector<EigenMatrix<double>>(irreps.size());
    std::vector<EigenMatrix<double>> grad_b = std::vector<EigenMatrix<double>>(irreps.size());
    std::vector<bool> occ_virtb =  std::vector<bool>(irreps.size());
    std::vector<Index> occb =  std::vector<Index>(irreps.size());
    std::vector<Index> virtb =  std::vector<Index>(irreps.size());

    for (size_t m = 0; m < irreps.size(); ++m)
    {
        Index _occa = pop_index[m]; Index _virta = irreps[m] - pop_index[m];
        (!_occa || !_virta) ? occ_virta[m] = false : occ_virta[m] = true;
        occa[m] = _occa; virta[m] = _virta;

        Index _occb = pop_index_b[m]; Index _virtb = irreps[m] - pop_index_b[m];
        (!_occb || !_virtb) ? occ_virtb[m] = false : occ_virtb[m] = true;
        occb[m] = _occb; virtb[m] = _virtb;
    }

    for (size_t m = 0; m < irreps.size(); ++m)
    {
        if (!occ_virta[m]) continue;

        Fmo_a[m].noalias() = vmo_coff[m].transpose() * vf_mat[m] * vmo_coff[m];

        grad_a[m] = -4.0 * Fmo_a[m].block(0, occa[m], occa[m], virta[m]);

        double gradmax = (grad_a[m].maxCoeff() > fabs(grad_a[m].minCoeff())) 
                       ? grad_a[m].maxCoeff() : fabs(grad_a[m].minCoeff());
    
        if (gradmax > 0.3) return; // Do not start SO before this cutoff raising this may cause trouble
    }
    
    for (size_t m = 0; m < irreps.size(); ++m)
    {
        if (!occ_virtb[m]) continue;

        Fmo_b[m].noalias() = vmo_coff_b[m].transpose() * vf_mat_b[m] * vmo_coff_b[m];

        grad_b[m] = -4.0 * Fmo_b[m].block(0, occb[m], occb[m], virtb[m]);

        double gradmax = (grad_b[m].maxCoeff() > fabs(grad_b[m].minCoeff())) 
                       ? grad_b[m].maxCoeff() : fabs(grad_b[m].minCoeff());
    
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
        if (occ_virta[m]) graddot += (grad_a[m].cwiseProduct(grad_a[m])).sum();
        if (occ_virtb[m]) graddot += (grad_b[m].cwiseProduct(grad_b[m])).sum();
    }

    std::vector<EigenMatrix<double>> Jprecond_a = std::vector<EigenMatrix<double>>(irreps.size()); // alpha Jacobian preconditioner
    std::vector<EigenMatrix<double>> rot_a = std::vector<EigenMatrix<double>>(irreps.size()); // alpha Rotation matrix
    std::vector<EigenMatrix<double>> Jprecond_b = std::vector<EigenMatrix<double>>(irreps.size()); // beta Jacobian preconditioner
    std::vector<EigenMatrix<double>> rot_b = std::vector<EigenMatrix<double>>(irreps.size()); // beta Rotation matrix

    for (size_t m = 0; m < irreps.size(); ++m)
    {
        if (occ_virta[m])
        {
            Jprecond_a[m] = EigenMatrix<double>(occa[m], virta[m]);
            rot_a[m] = EigenMatrix<double>(occa[m], virta[m]);

            for (Index i = 0; i < occa[m]; ++i)
                for (Index a = 0; a < virta[m]; ++a)
                {
                    Jprecond_a[m](i, a) = - 4.0 * (Fmo_a[m](i, i) - Fmo_a[m](occa[m] + a, occa[m] + a));
                    rot_a[m](i, a) = grad_a[m](i, a) / Jprecond_a[m](i, a);
                }
        }

        if (occ_virtb[m])
        {
            Jprecond_b[m] = EigenMatrix<double>(occb[m], virtb[m]);
            rot_b[m] = EigenMatrix<double>(occb[m], virtb[m]);

            for (Index i = 0; i < occb[m]; ++i)
                for (Index a = 0; a < virtb[m]; ++a)
                {
                    Jprecond_b[m](i, a) = - 4.0 * (Fmo_b[m](i, i) - Fmo_b[m](occb[m] + a, occb[m] + a));
                    rot_b[m](i, a) = grad_b[m](i, a) / Jprecond_b[m](i, a);
                }
        }
    }

    std::vector<EigenMatrix<double>> Ax_a, Ax_b; 
    solve_H_rot_sym(rot_a, rot_b, Fmo_a, Fmo_b, Ax_a, Ax_b);

    // alpha intermediates
    std::vector<EigenMatrix<double>> r_a = std::vector<EigenMatrix<double>>(irreps.size());
    std::vector<EigenMatrix<double>> z_a = std::vector<EigenMatrix<double>>(irreps.size());
    std::vector<EigenMatrix<double>> p_a = std::vector<EigenMatrix<double>>(irreps.size());
    // beta intermediates
    std::vector<EigenMatrix<double>> r_b = std::vector<EigenMatrix<double>>(irreps.size());
    std::vector<EigenMatrix<double>> z_b = std::vector<EigenMatrix<double>>(irreps.size());
    std::vector<EigenMatrix<double>> p_b = std::vector<EigenMatrix<double>>(irreps.size());

    double rms = 0;
    for (size_t m = 0; m < irreps.size(); ++m)
    {
        if (occ_virta[m])
        {
            r_a[m] = grad_a[m] - Ax_a[m];
            z_a[m] = r_a[m].cwiseProduct(Jprecond_a[m].cwiseInverse());
            p_a[m] = z_a[m];
            rms += sqrt((r_a[m].cwiseProduct(r_a[m])).sum() / graddot);
        }

        if (occ_virtb[m])
        {
            r_b[m] = grad_b[m] - Ax_b[m];
            z_b[m] = r_b[m].cwiseProduct(Jprecond_b[m].cwiseInverse());
            p_b[m] = z_b[m];
            rms += sqrt((r_b[m].cwiseProduct(r_b[m])).sum() / graddot);
        }
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
            if (occ_virta[m]) rz_old += (r_a[m].cwiseProduct(z_a[m])).sum();
            if (occ_virtb[m]) rz_old += (r_b[m].cwiseProduct(z_b[m])).sum();
        }

        std::vector<EigenMatrix<double>> Ap_a, Ap_b;
        solve_H_rot_sym(p_a, p_b, Fmo_a, Fmo_b, Ap_a, Ap_b);

        rms = 0; double alpha = rz_old; double denom = 0;
        for (size_t m = 0; m < irreps.size(); ++m)
        {
            if (occ_virta[m]) denom += (Ap_a[m].cwiseProduct(p_a[m])).sum();
            if (occ_virtb[m]) denom += (Ap_b[m].cwiseProduct(p_b[m])).sum();
        }

        alpha /= denom;
        for (size_t m = 0; m < irreps.size(); ++m)
        {
            if (occ_virta[m])
            {
                rot_a[m] += alpha * p_a[m];
                r_a[m] -= alpha * Ap_a[m];
                z_a[m] = r_a[m].cwiseProduct(Jprecond_a[m].cwiseInverse());
                rms += sqrt((r_a[m].cwiseProduct(r_a[m])).sum() / graddot);
            }

            if (occ_virtb[m])
            {
                rot_b[m] += alpha * p_b[m];
                r_b[m] -= alpha * Ap_b[m];
                z_b[m] = r_b[m].cwiseProduct(Jprecond_b[m].cwiseInverse());
                rms += sqrt((r_b[m].cwiseProduct(r_b[m])).sum() / graddot);
            }
        }

        if (hf_settings::get_verbosity() > 3)
            std::cout << "  SOSCF micro iter: " << iter + 1 << ", RMS: " 
                      << std::fixed << std::setprecision(6) << rms << "\n";

        if (rms < rms_tol) break; // abort if rms is small enough

        double beta = 0;
        for (size_t m = 0; m < irreps.size(); ++m)
        {
            if (occ_virta[m]) beta += (r_a[m].cwiseProduct(z_a[m])).sum() / rz_old;
            if (occ_virtb[m]) beta += (r_b[m].cwiseProduct(z_b[m])).sum() / rz_old;
        }

        for (size_t m = 0; m < irreps.size(); ++m)
        {
            if (occ_virta[m]) p_a[m] = z_a[m] + beta * p_a[m];
            if (occ_virtb[m]) p_b[m] = z_b[m] + beta * p_b[m];
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

        for (Index i = 0; i < occa[m]; ++i)
            for (Index a = 0; a < virta[m]; ++a)
            {
                U(i, a + occa[m]) = rot_a[m](i, a);
                U(occa[m] + a, i) = -rot_a[m](i, a);
            }
            
        U += 0.5 * U * U; // + (1.0/6.0) * U * U * U + (1.0 / 24.0) * U * U * U * U;
        // seems the first order term does the job well enough
        for (Index i = 0; i < irreps[m]; ++i)
            U(i, i) += 1.0;
        // Like Gram Schmidt, but using householder
        EigenMatrix<double> Uorth = U.transpose().householderQr().householderQ();
        vmo_coff[m] = vmo_coff[m] * Uorth;
    }

    for (size_t m = 0; m < irreps.size(); ++m)
    {
        EigenMatrix<double> U = EigenMatrix<double>::Zero(irreps[m], irreps[m]);

        for (Index i = 0; i < occb[m]; ++i)
            for (Index a = 0; a < virtb[m]; ++a)
            {
                U(i, a + occb[m]) = rot_b[m](i, a);
                U(occb[m] + a, i) = -rot_b[m](i, a);
            }
            
        U += 0.5 * U * U; // + (1.0/6.0) * U * U * U + (1.0 / 24.0) * U * U * U * U;
        // seems the first order term does the job well enough
        for (Index i = 0; i < irreps[m]; ++i)
            U(i, i) += 1.0;
        // Like Gram Schmidt, but using householder
        EigenMatrix<double> Uorth = U.transpose().householderQr().householderQ();
        vmo_coff_b[m] = vmo_coff_b[m] * Uorth;
    }
}

void Mol::uhf_solver::solve_H_rot_sym(const std::vector<EigenMatrix<double>>& rot_a, 
                                      const std::vector<EigenMatrix<double>>& rot_b, 
                                      const std::vector<EigenMatrix<double>>& Fmo_a,
                                      const std::vector<EigenMatrix<double>>& Fmo_b, 
                                      std::vector<EigenMatrix<double>>& Fa, 
                                      std::vector<EigenMatrix<double>>& Fb) const
{
    const std::vector<int>& irreps = molecule->get_irreps();
    const auto& symblocks = molecule->get_sym_blocks();

    Fa = std::vector<EigenMatrix<double>>(irreps.size());
    std::vector<EigenMatrix<double>> D_a = std::vector<EigenMatrix<double>>(irreps.size());
    std::vector<EigenMatrix<double>> Cocc_a = std::vector<EigenMatrix<double>>(irreps.size());
    std::vector<EigenMatrix<double>> Cvirt_a = std::vector<EigenMatrix<double>>(irreps.size());

    Fb = std::vector<EigenMatrix<double>>(irreps.size());
    std::vector<EigenMatrix<double>> D_b = std::vector<EigenMatrix<double>>(irreps.size());
    std::vector<EigenMatrix<double>> Cocc_b = std::vector<EigenMatrix<double>>(irreps.size());
    std::vector<EigenMatrix<double>> Cvirt_b = std::vector<EigenMatrix<double>>(irreps.size());

    for (size_t m = 0; m < irreps.size(); ++m)
    {
        Index occa_ = pop_index[m]; Index virta_ = irreps[m] - pop_index[m];

        if (occa_ && virta_)
        {
            D_a[m] = EigenMatrix<double>(irreps[m], irreps[m]);
            Fa[m] = Fmo_a[m].block(0, 0, occa_, occa_) * rot_a[m]
                  - rot_a[m] * Fmo_a[m].block(occa_, occa_, virta_, virta_);

            Cocc_a[m] = vmo_coff[m].block(0, 0, irreps[m], occa_);
            Cvirt_a[m] = vmo_coff[m].block(0, occa_, irreps[m], virta_);
            EigenMatrix<double> Cright_a = EigenMatrix<double>(irreps[m], occa_);

            for (Index s = 0; s < irreps[m]; ++s)
                for (Index i = 0; i < occa_; ++i)
                {
                    Cright_a(s, i) = 0;
                    for (Index a = 0; a < virta_; ++a) 
                        Cright_a(s, i) -= rot_a[m](i, a) * Cvirt_a[m](s, a);
                }

            #ifdef _OPENMP
            #pragma omp parallel for collapse(2)
            #endif
            for (Index i = 0; i < irreps[m]; ++i)
            {
                for (Index j = 0; j < irreps[m]; ++j)
                {
                    D_a[m](i, j) = 0.0;
                    for (Index k = 0; k < occa_; ++k)
                        D_a[m](i, j) += Cocc_a[m](i, k) * Cright_a(j, k);
                }
            }
        }

        Index occb_ = pop_index_b[m]; Index virtb_ = irreps[m] - pop_index_b[m];

        if (occb_ && virtb_)
        {
            D_b[m] = EigenMatrix<double>(irreps[m], irreps[m]);
            Fb[m] = Fmo_b[m].block(0, 0, occb_, occb_) * rot_b[m]
                  - rot_b[m] * Fmo_b[m].block(occb_, occb_, virtb_, virtb_);

            Cocc_b[m] = vmo_coff_b[m].block(0, 0, irreps[m], occb_);
            Cvirt_b[m] = vmo_coff_b[m].block(0, occb_, irreps[m], virtb_);
            EigenMatrix<double> Cright_b = EigenMatrix<double>(irreps[m], occb_);

            for (Index s = 0; s < irreps[m]; ++s)
                for (Index i = 0; i < occb_; ++i)
                {
                    Cright_b(s, i) = 0;
                    for (Index a = 0; a < virtb_; ++a) 
                        Cright_b(s, i) -= rot_b[m](i, a) * Cvirt_b[m](s, a);
                }

            #ifdef _OPENMP
            #pragma omp parallel for collapse(2)
            #endif
            for (Index i = 0; i < irreps[m]; ++i)
            {
                for (Index j = 0; j < irreps[m]; ++j)
                {
                    D_b[m](i, j) = 0.0;
                    for (Index k = 0; k < occb_; ++k)
                        D_b[m](i, j) += Cocc_b[m](i, k) * Cright_b(j, k);
                }
            }
        }
    }

    EigenMatrix<double> Da = EigenMatrix<double>::Zero(norbitals, norbitals);
    EigenMatrix<double> Db = EigenMatrix<double>::Zero(norbitals, norbitals);
    for (size_t d = 0; d < irreps.size(); ++d)
    {
        Index occa_ = pop_index[d];
        if (occa_) // virt not relevant
            Da += symblocks[d] * D_a[d] * symblocks[d].transpose();
        
        Index occb_ = pop_index_b[d];
        if (occb_) // virt not relevant
            Db += symblocks[d] * D_b[d] * symblocks[d].transpose();
    }

    EigenMatrix<double> Jab =  EigenMatrix<double>(norbitals, norbitals);
    EigenMatrix<double> Ka  =  EigenMatrix<double>(norbitals, norbitals);
    EigenMatrix<double> Kb  =  EigenMatrix<double>(norbitals, norbitals);
    
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index i = 0; i < norbitals; ++i)
    {
        for (Index j = 0; j < norbitals; ++j)
        {
            Jab(i, j) = 0; Ka(i, j) = 0; Kb(i, j) = 0;
            for (Index k = 0; k < norbitals; ++k)
            {
                for (Index l = 0; l < norbitals; ++l)
                {
                    Index ijkl = index_ijkl(i, j, k, l);
                    Index ikjl = index_ijkl(i, k, j, l);
                    Jab(i, j) += (Da(k, l) + Db(k, l)) * e_rep_mat(ijkl);
                    Ka(i, j) += Da(k, l) * e_rep_mat(ikjl);
                    Kb(i, j) += Db(k, l) * e_rep_mat(ikjl);
                }
            }
        }
    }

    EigenMatrix<double> Ga = (2.0 * Jab - Ka.transpose() - Ka);
    EigenMatrix<double> Gb = (2.0 * Jab - Kb.transpose() - Kb);

    for (size_t d = 0; d < irreps.size(); ++d)
    {
        Index occa_ = pop_index[d]; Index virta_ = irreps[d] - pop_index[d];
        if (occa_ && virta_)
        {
            const EigenMatrix<double> Gi = symblocks[d].transpose() * Ga * symblocks[d];
            Fa[d] += Cocc_a[d].transpose() * (Gi) * Cvirt_a[d];
            Fa[d] *= -4.0;
        }

        Index occb_ = pop_index_b[d]; Index virtb_ = irreps[d] - pop_index_b[d];
        if (occb_ && virtb_)
        {
            const EigenMatrix<double> Gi = symblocks[d].transpose() * Gb * symblocks[d];
            Fb[d] += Cocc_b[d].transpose() * (Gi) * Cvirt_b[d];
            Fb[d] *= -4.0;
        }
    }
}
