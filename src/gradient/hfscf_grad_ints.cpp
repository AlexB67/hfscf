#include "../settings/hfscf_settings.hpp"
#include "../integrals/hfscf_osoverlap.hpp"
#include "../integrals/hfscf_oskinetic.hpp"
#include "../integrals/hfscf_oseri.hpp"
#include "../integrals/hfscf_osnuclear.hpp"
#include "../basis/hfscf_trans_basis.hpp"
#include "hfscf_gradient.hpp"
// Integral matrices

using hfscfmath::index_ij;
using HF_SETTINGS::hf_settings;
using Eigen::Index;

void Mol::scf_gradient::calc_overlap_kinetic_gradient(EigenMatrix<double>& dSdX,
                                                      EigenMatrix<double>& dTdX,
                                                      const tensor3d<double>& dSdq,
                                                      const tensor3d<double>& dTdq,
                                                      const Cart coord, 
                                                      const EigenVector<bool>& mask) const
{
    const Index num_orbitals = m_mol->get_num_orbitals();

    Index cart = 0; // X
    if (coord == Cart::Y) cart = 1;
    else if (coord == Cart::Z) cart = 2;

    for (Index i = 0; i < num_orbitals; ++i)
        for (Index j = i + 1; j < num_orbitals; ++j)
        {
            if (mask[i])
            {
                dSdX(i, j)  = dSdq(cart, i, j); // A
                dTdX(i, j)  = dTdq(cart, i, j); // A
            }
            
            if (mask[j])
            {
                dSdX(i, j) -= dSdq(cart, i, j); // A = -B translational invariance
                dTdX(i, j) -= dTdq(cart, i, j);
            }

            dSdX(j, i) = dSdX(i, j);
            dTdX(j, i) = dTdX(i, j);
        }
}

void Mol::scf_gradient::calc_overlap_centers(tensor3d<double>& dSdq) const
{
    bool is_pure = m_mol->use_pure_am();
    const Index nshells = m_mol->get_num_shells();
    const auto& sp = m_mol->get_shell_pairs();

    std::unique_ptr<OSOVERLAP::OSOverlap> overlap_ptr = std::make_unique<OSOVERLAP::OSOverlap>(is_pure);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif 
    for (Index i = 0; i < nshells; ++i)
        for (Index j = i; j < nshells; ++j)
            overlap_ptr->compute_contracted_shell_deriv1(dSdq, sp[i * nshells + j]);
}

void Mol::scf_gradient::calc_kinetic_centers(tensor3d<double>& dTdq) const
{
    bool is_pure = m_mol->use_pure_am();
    const Index nshells = m_mol->get_num_shells();
    const auto& sp = m_mol->get_shell_pairs();

    std::unique_ptr<OSKINETIC::OSKinetic> k_ptr = std::make_unique<OSKINETIC::OSKinetic>(is_pure);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif 
    for (Index i = 0; i < nshells; ++i)
        for (Index j = i; j < nshells; ++j)
            k_ptr->compute_contracted_shell_deriv1(dTdq, sp[i * nshells + j]);
}

void Mol::scf_gradient::calc_coulomb_exchange_integrals(symm4dTensor<double>& dVdX,
                                                        symm4dTensor<double>& dVdXa,
                                                        symm4dTensor<double>& dVdXb,
                                                        symm4dTensor<double>& dVdXc, 
                                                        const EigenVector<bool>& mask)
{
    const Index num_orbitals = m_mol->get_num_orbitals();
    // Note a, b, c = -d 
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index i = 0; i < num_orbitals; ++i)
        for (Index j = 0; j <= i; ++j) 
        {
            Index ij = index_ij(i, j);
            for (Index k = 0; k < num_orbitals; ++k)
                for (Index l = 0; l <= k; ++l) 
                {
                    Index kl = index_ij(k, l);
                    if (ij <= kl) 
                    {
                        if (mask[i]) dVdX(i, j, k, l) += dVdXa(i, j, k, l);
                        if (mask[j]) dVdX(i, j, k, l) += dVdXb(i, j, k, l);
                        if (mask[k]) dVdX(i, j, k, l) += dVdXc(i, j, k, l);
                        if (mask[l]) dVdX(i, j, k, l) += - dVdXa(i, j, k, l)  // d = -a -b -c
                                                         - dVdXb(i, j, k, l) - dVdXc(i, j, k, l);
                    }
                }
        }
}

void Mol::scf_gradient::calc_coulomb_exchange_integral_centers(symm4dTensor<double>& dVdXa,
                                                               symm4dTensor<double>& dVdXb,
                                                               symm4dTensor<double>& dVdXc,
                                                               symm4dTensor<double>& dVdYa,
                                                               symm4dTensor<double>& dVdYb,
                                                               symm4dTensor<double>& dVdYc,
                                                               symm4dTensor<double>& dVdZa,
                                                               symm4dTensor<double>& dVdZb,
                                                               symm4dTensor<double>& dVdZc,
                                                               const std::vector<bool>& out_of_plane)
{
    //  Calculate once for all atoms, all coordinates
    bool is_pure = m_mol->use_pure_am();
    const double cutoff = hf_settings::get_integral_tol();

    std::unique_ptr<ERIOS::Erios> os_ptr = std::make_unique<ERIOS::Erios>(is_pure, cutoff);

    const Index nshells = m_mol->get_num_shells();
    const auto& sp = m_mol->get_shell_pairs();
    
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index i = 0; i < nshells; ++i)
        for (Index j = 0; j <= i; ++j)
        {
            Index ij = index_ij(i, j);
            for (Index k = 0; k < nshells; ++k)
                for (Index l = 0; l <= k; ++l)
                {
                    Index kl = index_ij(k, l);
                    if(ij > kl) continue;
                    
                    os_ptr->compute_contracted_shell_quartet_deriv1(dVdXa, dVdXb, dVdXc,
                                                                    dVdYa, dVdYb, dVdYc,
                                                                    dVdZa, dVdZb, dVdZc, 
                                                                    sp[i * nshells + j], 
                                                                    sp[k * nshells + l],
                                                                    out_of_plane);
                }
       }
}

void Mol::scf_gradient::calc_nuclear_potential_centers(tensor3d<double>& dVdqa,
                                                       tensor3d<double>& dVdqb, 
                                                       const std::vector<bool>& out_of_plane) const
{
    const Index nshells = m_mol->get_num_shells();
    const auto& sp = m_mol->get_shell_pairs();
    const auto& atoms = m_mol->get_atoms();
    bool is_pure = m_mol->use_pure_am();
    
    std::unique_ptr<OSNUCLEAR::OSNuclear> os_ptr = std::make_unique<OSNUCLEAR::OSNuclear>(is_pure);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif 
    for (Index i = 0; i < nshells; ++i)
        for (Index j = i; j < nshells; ++j)
            os_ptr->compute_contracted_shell_deriv1(dVdqa, dVdqb, sp[nshells * i + j], 
                                                    atoms, out_of_plane);
}

void Mol::scf_gradient::calc_potential_gradient(const tensor3d<double>& dVdqa,
                                                const tensor3d<double>& dVdqb,
                                                EigenMatrix<double>& dVndX, const EigenVector<bool>& mask,
                                                const int atom, const int cart_dir) const
{
    const Index num_orbitals = m_mol->get_num_orbitals();
    const int natoms = static_cast<int>(m_mol->get_atoms().size());

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index i = 0; i < num_orbitals; ++i)
        for (Index j = i; j < num_orbitals; ++j)
        {
            for (int k = 0; k < natoms; ++k)
            {
                if(mask[i]) dVndX(i, j) -= dVdqa(natoms * cart_dir + k, i, j);
                if(mask[j]) dVndX(i, j) -= dVdqb(natoms * cart_dir + k, i, j);

                if (k == atom)
                    dVndX(i, j) += dVdqa(natoms * cart_dir + atom, i, j) 
                                 + dVdqb(natoms * cart_dir + atom, i, j);
            }
            
            dVndX(j, i) = dVndX(i, j);
        }
}
