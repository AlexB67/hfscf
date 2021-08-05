#include "hfscf_diis.hpp"
#include "settings/hfscf_settings.hpp"
#include <iomanip>
#include <iostream>

using Eigen::Index;
using HF_SETTINGS::hf_settings;

void DIIS_SOLVER::diis_solver::diis_extrapolate(EigenMatrix<double>& f_mat,
                                                const Eigen::Ref<const EigenMatrix<double> >& s_mat,
                                                const Eigen::Ref<const EigenMatrix<double> >& d_mat,
                                                const Eigen::Ref<const EigenMatrix<double> >& s_sqrt,
                                                bool print)
{

    // TODO optimise so it only calculates one row of L at each iteration.

    if(0 == m_diis_range)
    {
        // When not using DIIS we still calculate the RMS the same way instead off differences in density matrices
        // method calc_rms_density is no longer needed.
        EigenMatrix<double> FDS = f_mat * d_mat * s_mat;
        EigenMatrix<double> gradient = s_sqrt.transpose() * (FDS - FDS.transpose()) * s_sqrt;
        rms_diff_a = 0.5 * std::sqrt(gradient.cwiseAbs2().mean());
        rms_values.emplace_back(rms_diff_a);
        return;
    }
    
    if(static_cast<int>(rms_diis.size()) > m_diis_range)
    {
        // pick out the largest error vector position and delete entries
        // instead of the last vector, improves convergence.
        double rms_error = 0;
        std::vector<double>::difference_type pos = 0;
         for(size_t i = 0; i < rms_diis.size(); ++i)
         {   
             if(rms_diis[i] > rms_error)
             {
                rms_error = rms_diis[i];
                pos = static_cast<std::vector<double>::difference_type>(i);
             }
         }

        e_list.erase(e_list.begin() + pos); e_list.shrink_to_fit();
        f_list_a.erase(f_list_a.begin() + pos); f_list_a.shrink_to_fit();
        rms_diis.erase(rms_diis.begin() + pos); rms_diis.shrink_to_fit();
    }

    EigenMatrix<double> FDS = f_mat * d_mat * s_mat;
    EigenMatrix<double> gradient = s_sqrt.transpose() * (FDS - FDS.transpose()) * s_sqrt;
    //e_list.emplace_back(gradient);
    // f_list_a.emplace_back(f_mat);
    f_list_a.resize(f_list_a.size() + 1);
    f_list_a[f_list_a.size() - 1] = f_mat;
    e_list.resize(e_list.size() + 1);
    e_list[e_list.size() - 1] = gradient;
    rms_diff_a = std::sqrt(gradient.cwiseAbs2().mean());
    rms_diis.emplace_back(rms_diff_a);
    rms_values.emplace_back(rms_diff_a);

    if(rms_diis.size() > 1U) // we can start DIIS as soon as we have two error matrices.
    {
        const Index norbitals = f_mat.outerSize();
        const Index size = static_cast<Index>(e_list.size());
        EigenMatrix<double> L_mat = EigenMatrix<double>(size + 1, size + 1);
        
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic) 
        #endif
        for(Index i = 0; i < size; ++i)
            for(Eigen::Index j = 0; j < size; ++j)
            {
                L_mat(i, j) = 0;
                for(Eigen::Index k = 0; k < norbitals; ++k)
                    for(Eigen::Index l = 0; l < norbitals; ++l)
                        L_mat(i, j) += e_list[i](k, l) * e_list[j](k, l);
                        // same as double loop with L_mat(i, j) = (e_mat[i] * e_mat[j]).trace();
            }

        //normalize and scale if appropriate
        if(L_mat.cwiseAbs().sum() < diis_eps)
        {
            double maxcof = L_mat.maxCoeff();
            for(int i = 0; i < size; ++i)
                for(int j = 0; j < size; ++j)
                    L_mat(i, j) /= maxcof;
        }

        for(Index i = 0; i < size + 1; ++i)
        {
            L_mat(size, i) = -1.0;
            L_mat(i, size) = -1.0;
        }

        L_mat(size, size) = 0;

        EigenVector<double> b = EigenVector<double>::Zero(size + 1);
        b(size) = -1.0;

        EigenVector<double> coffs =  L_mat.householderQr().solve(b); // fairly stable and reasonably fast

        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (Index i = 0; i < norbitals; ++i)
            for (Index j = 0; j < norbitals; ++j) 
            {
                f_mat(i, j) = 0.0;
                for (Index k = 0; k < size; ++k) f_mat(i, j) += coffs(k) * f_list_a[k](i, j);
            }
        
        if (hf_settings::get_verbosity() > 3 && print)
        {
            std::cout << "\n  *********************";
            std::cout << "\n  * DIIS coefficients *\n";
            std::cout << "  *********************\n";
            for (Index k = 0; k < size; ++k)
                std::cout << std::setw(10) << std::setprecision(6) << coffs(k);
            
            std::cout << '\n';
        }
    }
}

// UHF
void DIIS_SOLVER::diis_solver::diis_extrapolate(EigenMatrix<double>& f_mat_a,
                                                EigenMatrix<double>& f_mat_b,
                                                const Eigen::Ref<const EigenMatrix<double> >& s_mat,
                                                const Eigen::Ref<const EigenMatrix<double> >& d_mat_a,
                                                const Eigen::Ref<const EigenMatrix<double> >& d_mat_b,
                                                const Eigen::Ref<const EigenMatrix<double> >& s_sqrt,
                                                bool print)
{
    if(0 == m_diis_range)
    {
        // When not using DIIS we still calculate the RMS the same way instead of differences in density matrices
        // method calc_rms_density is no longer needed.
        EigenMatrix<double> FDSa = f_mat_a * d_mat_a * s_mat;
        EigenMatrix<double> gradient = s_sqrt.transpose() * (FDSa - FDSa.transpose()) * s_sqrt;
        rms_diff_a = 0.5 * std::sqrt(gradient.cwiseAbs2().mean());

        EigenMatrix<double> FDSb = f_mat_b * d_mat_b * s_mat;
        gradient.noalias() = s_sqrt.transpose() * (FDSb - FDSb.transpose()) * s_sqrt;
        rms_diff_b = 0.5 * std::sqrt(gradient.cwiseAbs2().mean());
        rms_values.emplace_back(rms_diff_a + rms_diff_b);
        return;
    }
    
    if(static_cast<int>(rms_diis.size()) > m_diis_range)
    {
        // pick out the largest error vector position and delete entries
        double rms_error = 0;
        std::vector<double>::difference_type pos = 0;
        for(size_t i = 0; i < rms_diis.size(); ++i)
        {   
            if(rms_diis[i] > rms_error)
            {
                rms_error = rms_diis[i];
                pos = static_cast<std::vector<double>::difference_type>(i);
            }
        }

        e_list.erase(e_list.begin() + pos);
        f_list_a.erase(f_list_a.begin() + pos);
        f_list_b.erase(f_list_b.begin() + pos);
        rms_diis.erase(rms_diis.begin() + pos);
    }

    EigenMatrix<double> FDSa = f_mat_a * d_mat_a * s_mat;
    EigenMatrix<double> gradient = s_sqrt.transpose() * (FDSa - FDSa.transpose()) * s_sqrt;

    // Flatten the gradient matrix and build the error vector

    Index norbitals = f_mat_a.outerSize();
    EigenVector<double> grad_1d = EigenVector<double>(norbitals * norbitals + norbitals * norbitals);

    Index count = 0;
    for(Index i = 0; i < norbitals; ++ i)
        for(Index j = 0; j < norbitals; ++ j)
        {
           grad_1d(count) = gradient(i, j);
           ++count;
        }

    rms_diff_a = 0.5 * std::sqrt(gradient.cwiseAbs2().mean());
    
    EigenMatrix<double> FDSb = f_mat_b * d_mat_b * s_mat;
    gradient.noalias() = s_sqrt.transpose() * (FDSb - FDSb.transpose()) * s_sqrt;

    count = 0;
    Index offset = norbitals * norbitals;
    for(Index i = 0; i < norbitals; ++ i)
        for(Index j = 0; j < norbitals; ++ j)
        {
            grad_1d(offset + count) = gradient(i, j);
            ++count;
        }

    rms_diff_b = 0.5 * std::sqrt(gradient.cwiseAbs2().mean());

    e_list.emplace_back(grad_1d);
    f_list_a.emplace_back(f_mat_a);
    f_list_b.emplace_back(f_mat_b);
    rms_diis.emplace_back(rms_diff_a + rms_diff_b);
    rms_values.emplace_back(rms_diff_a + rms_diff_b);

    if(rms_diis.size() > 1U) // we can start DIIS as soon as we have two error matrices.
    {
        Index size = static_cast<Index>(e_list.size());
        EigenMatrix<double> L_mat = EigenMatrix<double>(size + 1, size + 1);

        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for(Index i = 0; i < size; ++i)
            for(Index j = 0; j < size; ++j)
            {
                L_mat(i, j) = 0;
                for (Index k = 0; k < grad_1d.size(); ++k)
                    L_mat(i, j) += e_list[i](k) * e_list[j](k);
            }

        //normalize and scale if appropriate 
        if(L_mat.cwiseAbs().sum() < diis_eps)
        {
            double maxcof = L_mat.maxCoeff();
            for(int i = 0; i < size; ++i)
                for(int j = 0; j < size; ++j)
                    L_mat(i, j) /= maxcof;
        }
        
        for(Index i = 0; i < size + 1; ++i)
        {
            L_mat(size, i) = -1.0;
            L_mat(i, size) = -1.0;
        }

        L_mat(size, size) = 0;

        EigenVector<double> b = EigenVector<double>::Zero(size + 1);
        b(size) = -1.0;

        EigenVector<double> coffs =  L_mat.householderQr().solve(b); // fairly stable and reasonably fast

        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (Index i = 0; i < norbitals; ++i)
            for (Index j = 0; j < norbitals; ++j)
            {
                f_mat_a(i, j) = 0;
                f_mat_b(i, j) = 0;
                for (Index k = 0; k < size; ++k)
                {
                    f_mat_a(i, j) += coffs(k) * f_list_a[k](i, j);
                    f_mat_b(i, j) += coffs(k) * f_list_b[k](i, j);
                }

            }
        
        if (hf_settings::get_verbosity() > 3 && print)
        {
            std::cout << "\n  *********************";
            std::cout << "\n  * DIIS coefficients *\n";
            std::cout << "  *********************\n";
            for (Index k = 0; k < size; ++k)
                std::cout << std::setw(10) << std::setprecision(6) << coffs(k);
            
            std::cout << '\n';
        }
    }
}
