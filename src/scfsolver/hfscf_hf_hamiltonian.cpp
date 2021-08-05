#include "../integrals/hfscf_oseri.hpp"
#include "../integrals/hfscf_osnuclear.hpp"
#include "../integrals/hfscf_osoverlap.hpp"
#include "../settings/hfscf_settings.hpp"
#include "hfscf_hf_solver.hpp"
#include <Eigen/Eigenvalues>
#include <iomanip>

using HF_SETTINGS::hf_settings;
using Eigen::Index;
using hfscfmath::index_ij;
using hfscfmath::index_ijkl;

void Mol::hf_solver::create_one_electron_hamiltonian()
{
    bool is_spherical = molecule->use_pure_am();
    std::unique_ptr<OSOVERLAP::OSOverlap> os_ptr = std::make_unique<OSOVERLAP::OSOverlap>(is_spherical);

    Index nshells = molecule->get_num_shells();
    const auto& sp = molecule->get_shell_pairs();

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif 
    for (Index i = 0; i < nshells; ++i)
        for (Index j = i; j < nshells; ++j)
            os_ptr->compute_contracted_shell(s_mat, k_mat, sp[i * nshells + j]);

    std::unique_ptr<OSNUCLEAR::OSNuclear> osnuc_ptr = std::make_unique<OSNUCLEAR::OSNuclear>(is_spherical);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif 
    for (Index i = 0; i < nshells; ++i)
        for (Index j = i; j < nshells; ++j)
                osnuc_ptr->compute_contracted_shell(v_mat, sp[i * nshells + j], molecule->get_atoms());

    hcore_mat = k_mat + v_mat;

    if (!use_sym) return;
    // block diagonlize per irrep if symmetry is true

    const std::vector<EigenMatrix<double>>& sblocks = molecule->get_sym_blocks();
    const std::vector<int>& irreps = molecule->get_irreps();

    if (!vs_mat.size())
        vs_mat = vk_mat = vv_mat = vhcore_mat = std::vector<EigenMatrix<double>>(irreps.size());
 
    size_t i = 0;
    for(const auto &sm : sblocks)
    {
        if (!vs_mat.size())
        {
            vs_mat[i] = EigenMatrix<double>(irreps[i], irreps[i]);
            vk_mat[i] = EigenMatrix<double>(irreps[i], irreps[i]);
            vv_mat[i] = EigenMatrix<double>(irreps[i], irreps[i]);
        }

        vs_mat[i] = sm.transpose() * s_mat * sm;
        vk_mat[i] = sm.transpose() * k_mat * sm;
        vv_mat[i] = sm.transpose() * v_mat * sm;
        vhcore_mat[i] = vk_mat[i] + vv_mat[i];
        ++i;
    }
}

void Mol::hf_solver::create_repulsion_matrix()
{
    const Index nshells = molecule->get_num_shells();
    const auto& sp = molecule->get_shell_pairs();
    const bool do_screen = hf_settings::get_screen_eri();
    bool is_spherical = molecule->use_pure_am();
    const double cutoff = hf_settings::get_integral_tol();

    std::unique_ptr<ERIOS::Erios> os_ptr = std::make_unique<ERIOS::Erios>(is_spherical, cutoff);

    if (do_screen) 
    {   // screen ints are always cartesian, since eri ints are evaluated with cartesian gaussians
        // before conversion to pure (if needed)
        const Index ncart = molecule->get_num_cart_orbitals();
        os_ptr->init_screen(ncart);

        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (Index i = 0; i < nshells; ++i)
            for (Index j = i; j < nshells; ++j)
                os_ptr->compute_screen_matrix(sp[i * nshells + j]);
    }

    if(hf_settings::get_scf_direct() && hf_settings::get_guess_type() != "SAD") return;
    
    e_rep_mat.setZero();
    
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index i = 0; i < nshells; ++i)
        for (Index j = i; j < nshells; ++j)
        {
            Index ij = index_ij(i, j);
            for (Index k = 0; k < nshells; ++k)
                for (Index l = k; l < nshells; ++l)
                {
                    Index kl = index_ij(k, l);
                    if(ij > kl) continue;
                        
                    os_ptr->compute_contracted_shell_quartet(e_rep_mat, sp[i * nshells + j],
                                                            sp[k * nshells + l], do_screen);
                }
        }
}
