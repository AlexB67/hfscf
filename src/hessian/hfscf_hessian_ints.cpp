#include "../integrals/hfscf_osoverlap.hpp"
#include "../integrals/hfscf_oskinetic.hpp"
#include "../integrals/hfscf_osnuclear.hpp"
#include "../integrals/hfscf_oseri.hpp"
#include "../math/hfscf_math.hpp"
#include "../basis/hfscf_trans_basis.hpp"
#include "../pretty_print/hfscf_pretty_print.hpp"
#include "../settings/hfscf_settings.hpp"
#include "hfscf_hessian.hpp"

using hfscfmath::index_ij;
using hfscfmath::Cart2;
using tensor4dmath::symm4dTensor;
using tensor4dmath::tensor4d;
using tensor4dmath::index4;
using HFCOUT::pretty_print_matrix;
using HF_SETTINGS::hf_settings;
using Eigen::Index;

void Mol::scf_hessian::calc_scf_hessian_tei()
{
    const Index num_orbitals = m_mol->get_num_orbitals();
    const Index num_shells = m_mol->get_num_shells();
    const auto &sp = m_mol->get_shell_pairs();
    const auto& atoms = m_mol->get_atoms();
    const int natoms = static_cast<int>(atoms.size());
    bool is_pure = m_mol->use_pure_am();
    const double cutoff = hf_settings::get_integral_tol();

    m_hessianC = EigenMatrix<double>::Zero(3 * natoms, 3 * natoms);
    m_hessianEx = EigenMatrix<double>::Zero(3 * natoms, 3 * natoms);
    
    std::unique_ptr<ERIOS::Erios> os_ptr = std::make_unique<ERIOS::Erios>(is_pure, cutoff);

    for(int cart2 = 0; cart2 < 3; ++cart2) // XX YY ZZ
    {   // The same permutations apply as non deriv eri integrals <ijkl> <jikl> <jilk> <jilk>
        // swap bra ket to gives another 4. 8 total.
        symm4dTensor<double> AA = symm4dTensor<double>(num_orbitals);
        symm4dTensor<double> AB = symm4dTensor<double>(num_orbitals);
        symm4dTensor<double> AC = symm4dTensor<double>(num_orbitals);
        symm4dTensor<double> BD = symm4dTensor<double>(num_orbitals);
        symm4dTensor<double> CC = symm4dTensor<double>(num_orbitals);
        symm4dTensor<double> CD = symm4dTensor<double>(num_orbitals);

        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (Index i = 0; i < num_shells; ++i)
            for (Index j = 0; j <= i; ++j)
                for (Index k = 0; k < num_shells; ++k)
                    for (Index l = 0; l <= k; ++l)
                    {
                        os_ptr->compute_contracted_shell_quartet_deriv2_qq(AA, AB, AC, BD, CC, CD, 
                                                                           sp[i * num_shells + j], 
                                                                           sp[k * num_shells + l],
                                                                           cart2);
                    }

        for(int atom = 0; atom < natoms; ++atom)
        {
            const std::vector<bool> &mask1 = (is_pure) ? smask[atom] : mask[atom];
            for (int atom2 = 0; atom2 <= atom; ++atom2)
            {
                const std::vector<bool> &mask2 = (is_pure) ? smask[atom2] : mask[atom2];
                symm4dTensor<double> eri_tensor = symm4dTensor<double>(num_orbitals);

                double tmp, ad, bb, dd, bc;
                #ifdef _OPENMP
                #pragma omp parallel for private (tmp) schedule(dynamic)
                #endif
                for (Index i = 0; i < num_orbitals; ++i)
                    for (Index j = 0; j <= i; ++j) 
                    {
                        Index ij = index_ij(i, j);
                        for (Index k = 0; k < num_orbitals; ++k)
                            for (Index l = 0; l <= k; ++l) 
                            {
                                Index kl = index_ij(k, l);
                                if (ij <= kl) // Translational invariance relationships
                                {   
                                    ad = -(AA(i, j, k, l) + AB(i, j, k, l) + AC(i, j , k, l));
                                    dd = -(ad + BD(i, j, k, l) + CD(i, j, k, l));
                                    bb = AA(i, j, k, l) + 2 * AC(i, j, k, l) + 2 * ad + CC(i, j, k, l) 
                                    + 2 * CD(i, j, k, l) + dd;
                                    bc =  -(AC(i, j, k, l) + CC(i, j, k, l) + CD(i, j, k, l));
    
                                    tmp = 0;
                                    if(atom == atom2)
                                    { 
                                        if (mask1[i]) tmp += AA(i, j, k, l); // AA
                                        if (mask1[j]) tmp += bb; // BB 
                                        if (mask1[k]) tmp += CC(i, j, k, l); // CC
                                        if (mask1[l]) tmp += dd; // DD
                                        if (mask1[i] && mask1[j]) tmp += 2 * AB(i, j, k, l); // AB
                                        if (mask1[i] && mask1[k]) tmp += 2 * AC(i, j, k, l); // AC
                                        if (mask1[i] && mask1[l]) tmp += 2 * ad; // AD
                                        if (mask1[j] && mask1[k]) tmp += 2 * bc; // BC
                                        if (mask1[j] && mask1[l]) tmp += 2 * BD(i, j, k, l); // BD
                                        if (mask1[k] && mask1[l]) tmp += 2 * CD(i, j, k, l); // CD
                                    }
                                    else
                                    {
                                        if ((mask1[i] && mask2[j]) || (mask1[j] && mask2[i])) tmp += AB(i, j, k, l); // AB
                                        if ((mask1[i] && mask2[k]) || (mask1[k] && mask2[i])) tmp += AC(i, j, k, l); // AC
                                        if ((mask1[i] && mask2[l]) || (mask1[l] && mask2[i])) tmp += ad; // AD
                                        if ((mask1[j] && mask2[k]) || (mask1[k] && mask2[j])) tmp += bc; // BC
                                        if ((mask1[j] && mask2[l]) || (mask1[l] && mask2[j])) tmp += BD(i, j, k, l); // BD
                                        if ((mask1[k] && mask2[l]) || (mask1[l] && mask2[k])) tmp += CD(i, j, k, l); // CD
                                    }

                                    eri_tensor(i, j, k, l) = tmp;
                                }
                            }
                    }

                EigenMatrix<double> dG2dX2 = EigenMatrix<double>::Zero(num_orbitals, num_orbitals);
                EigenMatrix<double> dG2dX2Coulomb = EigenMatrix<double>::Zero(num_orbitals, num_orbitals);
                EigenMatrix<double> dG2dX2Exchange = EigenMatrix<double>::Zero(num_orbitals, num_orbitals);

                #ifdef _OPENMP
                #pragma omp parallel for schedule(dynamic)
                #endif
                for (Index i = 0; i < num_orbitals; ++i)
                    for (Index j = i; j < num_orbitals; ++j) 
                    {
                        for (Index k = 0; k < num_orbitals; ++k)
                            for (Index l = 0; l < num_orbitals; ++l) 
                            {
                                dG2dX2Coulomb(i, j) += d_mat(k, l) * (2.0 * eri_tensor(i, j, k, l));
                                dG2dX2Exchange(i, j) -= d_mat(k, l) * (eri_tensor(i, k, j, l));
                            }

                        dG2dX2(i, j) = dG2dX2Coulomb(i, j) + dG2dX2Exchange(i, j);
                        dG2dX2(j, i) = dG2dX2(i, j);
                        dG2dX2Coulomb(j, i) = dG2dX2Coulomb(i, j);
                        dG2dX2Exchange(j, i) = dG2dX2Exchange(i, j);
                    }

                m_hessianC(3 * atom + cart2, 3 * atom2 + cart2) = m_hessianC(3 * atom2 + cart2, 3 * atom + cart2) 
                                                = d_mat.cwiseProduct(dG2dX2Coulomb).sum();
                m_hessianEx(3 * atom + cart2, 3 * atom2 + cart2) = m_hessianEx(3 * atom2 + cart2, 3 * atom + cart2) 
                                                = d_mat.cwiseProduct(dG2dX2Exchange).sum();
            }
        }
    }

    int maxcoords = 5;
    std::vector<bool> out_of_plane(3, false);
    if(m_mol->molecule_is_linear()) maxcoords = 0; // need to put in check if it is aligned 
    else // is it planar ?                         // molecule class aligns anyway
    {
        constexpr double tol = 1.0E-10; 
        for(int cart_dir = 0; cart_dir < 3; ++cart_dir)
        {
            for(int i = 0; i < natoms; ++i)
                if(std::fabs<double>(m_mol->get_geom()(i , cart_dir)) > tol)
                    out_of_plane[cart_dir] = true;
        }
    }

    tensor4d<double> AA = tensor4d<double>(num_orbitals); // xz xy etc.
    tensor4d<double> AB = tensor4d<double>(num_orbitals);
    tensor4d<double> AC = tensor4d<double>(num_orbitals);
                                                   
    for(int cart2 = 3; cart2 <= maxcoords; ++cart2) // XY XZ YZ etc. 45 total deriv2 terms including the above
    {
        if(!out_of_plane[0] && cart2 <= 4) continue;
        else if (!out_of_plane[1] && (cart2 == 3 || cart2 == 5)) continue;
        else if (!out_of_plane[2] && (cart2 == 4 || cart2 == 5)) continue;

        AA.setZero();
        AB.setZero();
        AC.setZero();

        // since we are doing the full loop we can use the following permutations
        // AC(i, j, k, l) = eri_pq(2);
        // BD(j, i, l, k) = eri_pq(2);
        // AC1(k, l, i, j) = eri_pq(2); 1 denotes reverse deriv order
        // The same applies to AA, AB AB1 which map to CC CD CD1 respectively by permutation,
        // thus, we are only using the AA AB and AC tensors, which contain all needed permutations
        // holding the remaining terms CC CD etc.

        // Without permutation symmetry we had
        // AA(i, j, k, l) = eri_pq(0); AB(i, j, k, l) = eri_pq(1); AC(i, j, k, l) = eri_pq(2);
        // BD(i, j, k, l) = eri_pq(3); CC(i, j, k, l) = eri(4); CD(i, j, k, l) = eri_pq(5);
        // CD1(i, j, k, l) = eri_pq(7); AB1(i, j, k, l) = eri_pq(11); AC1(i, j, k, l) = eri_pq(12);
        //            }

        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (Index i = 0; i < num_shells; ++i)
            for (Index j = 0; j < num_shells; ++j)
                for (Index k = 0; k < num_shells; ++k)
                    for (Index l = 0; l < num_shells; ++l) 
                    {
                        if (l <= k)
                        {
                            os_ptr->compute_contracted_shell_quartet_deriv2_pq(
                                AA, AB, AC, sp[i * num_shells + j], sp[k * num_shells + l], cart2, ERIOS::Permute::ALL);
                        } 
                        else 
                        {
                            os_ptr->compute_contracted_shell_quartet_deriv2_pq(
                                AA, AB, AC, sp[i * num_shells + j], sp[k * num_shells + l], cart2, ERIOS::Permute::ACONLY);
                        }
                    }

        for(int atom = 0; atom < natoms; ++atom)
        {
            const std::vector<bool> &mask1 = (is_pure) ? smask[atom] : mask[atom];

            for (int atom2 = 0; atom2 < natoms; ++atom2)
            {
                 const std::vector<bool> &mask2 = (is_pure) ? smask[atom2] : mask[atom2];

                tensor4d<double> eri_tensor = tensor4d<double>(num_orbitals);
                double tmp, ad, ad1, bb, bc, bc1, bd1, dd;
                #ifdef _OPENMP
                #pragma omp parallel for private (tmp) schedule(dynamic)
                #endif
                for (int i = 0; i < num_orbitals; ++i)
                    for (int j = 0; j < num_orbitals; ++j) 
                        for (int k = 0; k < num_orbitals; ++k)
                            for (int l = 0; l < num_orbitals; ++l)
                            {   
                                // Translational invariance relationships
                                ad  = -(AA(i, j, k, l) + AB(i, j, k, l) + AC(i, j, k, l));
                                ad1 = -(AA(i, j, k, l) + AB(j, i, k, l) + AC(k, l, i, j));
                                dd  = -(ad + AC(j, i, l, k) + AB(k, l, i, j));
                                bb  =  AA(i, j, k, l) + AC(i, j, k, l) + ad + AC(k, l, i, j) +
                                       AA(k, l, i, j) + AB(k, l, i, j) + ad1 + AB(l, k, i, j) + dd;
                                bc  = -(AC(i, j, k, l) + AA(k, l, i, j) + AB(l, k, i, j));
                                bc1 = -(AC(k, l, i, j) + AA(k, l, i, j) + AB(k, l, i, j));
                                bd1 = -(ad1 + AB(l, k, i, j) + dd);
                                tmp = 0;
                                
                                if(atom == atom2)
                                {
                                    if (mask1[i]) tmp += AA(i, j, k, l); // AA
                                    if (mask1[j]) tmp += bb; // BB(i, j, k, l); // BB 
                                    if (mask1[k]) tmp += AA(k, l, i, j); // CC(i, j, k, l); // CC
                                    if (mask1[l]) tmp += dd; // DD
                                    if (mask1[i] && mask1[j]) tmp += 2 * AB(i, j, k, l); // AB
                                    if (mask1[i] && mask1[k]) tmp += 2 * AC(i, j, k, l); // AC
                                    if (mask1[i] && mask1[l]) tmp += 2 * ad; //AD(i, j, k, l); // AD
                                    if (mask1[j] && mask1[k]) tmp += 2 * bc; // 2 * BC(i, j, k, l); // BC
                                    if (mask1[j] && mask1[l]) tmp += 2 * AC(j, i, l, k); // 2 * BD(i, j, k, l); // BD
                                    if (mask1[k] && mask1[l]) tmp += 2 * AB(k, l, i, j); // 2 * CD(i, j, k, l); // CD
                                }
                                else
                                {
                                    if (mask1[i] && mask2[j]) tmp += AB(i, j, k, l); // AB
                                    if (mask1[j] && mask2[i]) tmp += AB(j, i, k, l); //AB1(i, j, k, l); // AB1
                                    if (mask1[i] && mask2[k]) tmp += AC(i, j, k, l); // AC 
                                    if (mask1[k] && mask2[i]) tmp += AC(k, l, i, j); // AC1(i, j, k, l); // AC1
                                    if (mask1[i] && mask2[l]) tmp += ad; // AD
                                    if (mask1[l] && mask2[i]) tmp += ad1; // AD1
                                    if (mask1[j] && mask2[k]) tmp += bc; // BC(i, j, k, l); // BC 
                                    if (mask1[k] && mask2[j]) tmp += bc1; // BC1(i, j, k, l); // BC1
                                    if (mask1[j] && mask2[l]) tmp += AC(j, i, l, k); // BD(i, j, k, l); // BD 
                                    if (mask1[l] && mask2[j]) tmp += bd1; // BD1(i, j, k, l); // BD1
                                    if (mask1[k] && mask2[l]) tmp += AB(k, l, i, j);  // CD 
                                    if (mask1[l] && mask2[k]) tmp += AB(l, k, i, j); // CD1(i, j, k, l); // CD1
                                }
                                
                                eri_tensor(i, j, k, l) = tmp;
                            }

                EigenMatrix<double> dG2dX1X2 = EigenMatrix<double>::Zero(num_orbitals, num_orbitals);
                EigenMatrix<double> dG2dX1X2Coulomb = EigenMatrix<double>::Zero(num_orbitals, num_orbitals);
                EigenMatrix<double> dG2dX1X2Exchange = EigenMatrix<double>::Zero(num_orbitals, num_orbitals);

                #ifdef _OPENMP
                #pragma omp parallel for schedule(dynamic)
                #endif
                for (Index i = 0; i < num_orbitals; ++i)
                    for (Index j = 0; j < num_orbitals; ++j) 
                    {
                        for (Index k = 0; k < num_orbitals; ++k)
                            for (Index l = 0; l < num_orbitals; ++l) 
                            {
                                dG2dX1X2Coulomb(i, j) += d_mat(k, l) * (2.0 * eri_tensor(i, j, k, l));
                                dG2dX1X2Exchange(i, j) -= d_mat(k, l) * (eri_tensor(i, k, j, l));
                            }

                        dG2dX1X2(i, j) = dG2dX1X2Coulomb(i, j) + dG2dX1X2Exchange(i, j);
                    }

                if (cart2 == 3) // xy yx
                {
                    m_hessianC(3 * atom, 3 * atom2 + 1) = m_hessianC(3 * atom2 + 1, 3 * atom) 
                    = d_mat.cwiseProduct(dG2dX1X2Coulomb).sum();
                    m_hessianEx(3 * atom, 3 * atom2 + 1) = m_hessianEx(3 * atom2 + 1, 3 * atom) 
                    = d_mat.cwiseProduct(dG2dX1X2Exchange).sum();
                }
                else if (cart2 == 4) // xz zx
                {
                    m_hessianC(3 * atom, 3 * atom2 + 2) = m_hessianC(3 * atom2 + 2, 3 * atom)
                    = d_mat.cwiseProduct(dG2dX1X2Coulomb).sum();
                    m_hessianEx(3 * atom, 3 * atom2 + 2) = m_hessianEx(3 * atom2 + 2, 3 * atom) 
                    = d_mat.cwiseProduct(dG2dX1X2Exchange).sum();
                }
                else // if(cart2 == 5) // yz zy
                {
                    m_hessianC(3 * atom + 1, 3 * atom2 + 2) = m_hessianC(3 * atom2 + 2, 3 * atom + 1) 
                    = d_mat.cwiseProduct(dG2dX1X2Coulomb).sum();
                    m_hessianEx(3 * atom + 1, 3 * atom2 + 2) = m_hessianEx(3 * atom2 + 2, 3 * atom + 1) 
                    = d_mat.cwiseProduct(dG2dX1X2Exchange).sum();
                }
            }  
        }
    }

    if (hf_settings::get_verbosity() < 2) return;
    std::cout << '\n';
    std::cout << "  ****************************************\n";
    std::cout << "  *      Analytic Coulomb Hessian:       *\n";
    std::cout << "  *                                      *\n";
    std::cout << "  *      d2E/dX1dX2 / (Eh a0^-2)         *\n";
    std::cout << "  ****************************************\n";
    pretty_print_matrix<double>(m_hessianC);
    
    std::cout << '\n';
    std::cout << "  ****************************************\n";
    std::cout << "  *      Analytic Exchange Hessian:      *\n";
    std::cout << "  *                                      *\n";
    std::cout << "  *      d2E/dX1dX2 / (Eh a0^-2)         *\n";
    std::cout << "  ****************************************\n";
    pretty_print_matrix<double>(m_hessianEx);
}

void Mol::scf_hessian::calc_scf_hessian_kinetic()
{
    const auto& atoms = m_mol->get_atoms();
    const int natoms = static_cast<int>(atoms.size());
    bool is_pure = m_mol->use_pure_am();
    const Index num_orbitals = m_mol->get_num_orbitals();

    // A centers only - Translational invariance a == b

    const Index nshells = m_mol->get_num_shells();
    const auto& sp = m_mol->get_shell_pairs();

    std::unique_ptr<OSKINETIC::OSKinetic> os_ptr = std::make_unique<OSKINETIC::OSKinetic>(is_pure);
    tensor3d<double> dT2dqq = tensor3d<double>(6, num_orbitals, num_orbitals);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif 
    for (Index i = 0; i < nshells; ++i)
        for (Index j = i; j < nshells; ++j)
        {
            os_ptr->compute_contracted_shell_deriv2(dT2dqq, sp[i * nshells + j]);
        }

    m_hessianKin = EigenMatrix<double>::Zero(3 * natoms, 3 * natoms);
    const std::vector<std::vector<bool>>& msk = (is_pure) ? smask : mask;

    for(int atom = 0; atom < natoms; ++atom)
    {
        for (int atom2 = 0; atom2 <= atom; ++atom2)
        {
            EigenMatrix<double> dK2dXX = EigenMatrix<double>::Zero(num_orbitals, num_orbitals);
            EigenMatrix<double> dK2dYY = EigenMatrix<double>::Zero(num_orbitals, num_orbitals);
            EigenMatrix<double> dK2dZZ = EigenMatrix<double>::Zero(num_orbitals, num_orbitals);
            EigenMatrix<double> dK2dXY = EigenMatrix<double>::Zero(num_orbitals, num_orbitals);
            EigenMatrix<double> dK2dXZ = EigenMatrix<double>::Zero(num_orbitals, num_orbitals);
            EigenMatrix<double> dK2dYZ = EigenMatrix<double>::Zero(num_orbitals, num_orbitals);

            int same_center = 1;
            if(atom != atom2) same_center = -1;

            bool a, b;
            #ifdef _OPENMP
            #pragma omp parallel for private(a, b) schedule(dynamic)
            #endif
            for (Index i = 0; i < num_orbitals; ++i)
            {
                for (Index j = i + 1; j < num_orbitals; ++j)
                {
                    
                    if(atom == atom2) { a = msk[atom][i]; b = msk[atom][j]; }
                    else { a = msk[atom][j] && msk[atom2][i]; b = msk[atom][i] && msk[atom2][j]; }

                    if(!a && !b) continue;
                    
                    if(a && b)
                    {
                        dK2dXX(i, j) = 0; dK2dYY(i, j) = 0; dK2dZZ(i, j) = 0;
                        dK2dXY(i, j) = 0; dK2dXZ(i, j) = 0; dK2dYZ(i, j) = 0;
                    }
                    else if(a || b)
                    {
                        dK2dXX(i, j) = dT2dqq(0, i, j); dK2dYY(i, j) = dT2dqq(3, i, j); dK2dZZ(i, j) = dT2dqq(5, i, j);
                        dK2dXY(i, j) = dT2dqq(1, i, j); dK2dXZ(i, j) = dT2dqq(2, i, j); dK2dYZ(i, j) = dT2dqq(4, i, j);
                    }

                    dK2dXX(i, j) *= same_center; dK2dXX(j, i) = dK2dXX(i, j);
                    dK2dYY(i, j) *= same_center; dK2dYY(j, i) = dK2dYY(i, j);
                    dK2dZZ(i, j) *= same_center; dK2dZZ(j, i) = dK2dZZ(i, j);
                    dK2dXY(i, j) *= same_center; dK2dXY(j, i) = dK2dXY(i, j);
                    dK2dXZ(i, j) *= same_center; dK2dXZ(j, i) = dK2dXZ(i, j);
                    dK2dYZ(i, j) *= same_center; dK2dYZ(j, i) = dK2dYZ(i, j);
                }
            }

            if ("SCF" == hf_settings::get_frequencies_type())
            {
                m_hessianKin(3 * atom, 3 * atom2) = 2.0 * dK2dXX.cwiseProduct(d_mat).sum(); //XX
                m_hessianKin(3 * atom + 1, 3 * atom2 + 1) = 2.0 * dK2dYY.cwiseProduct(d_mat).sum(); // YY
                m_hessianKin(3 * atom + 2, 3 * atom2 + 2) = 2.0 * dK2dZZ.cwiseProduct(d_mat).sum(); // ZZ
                m_hessianKin(3 * atom, 3 * atom2 + 1) += 2.0 * dK2dXY.cwiseProduct(d_mat).sum(); // XY
                m_hessianKin(3 * atom, 3 * atom2 + 2) += 2.0 * dK2dXZ.cwiseProduct(d_mat).sum(); // XZ
                m_hessianKin(3 * atom + 1, 3 * atom2 + 2) += 2.0 * dK2dYZ.cwiseProduct(d_mat).sum();// YZ
            }
            else if ("MP2" == hf_settings::get_frequencies_type())
            {
                m_hessianKin(3 * atom, 3 * atom2) = 
                (mo_coff.transpose() * dK2dXX * mo_coff).cwiseProduct(Ppq).sum(); //XX
                m_hessianKin(3 * atom + 1, 3 * atom2 + 1) = 
                (mo_coff.transpose() * dK2dYY * mo_coff).cwiseProduct(Ppq).sum(); // YY
                m_hessianKin(3 * atom + 2, 3 * atom2 + 2) = 
                (mo_coff.transpose() * dK2dZZ * mo_coff).cwiseProduct(Ppq).sum(); // ZZ
                m_hessianKin(3 * atom, 3 * atom2 + 1) += 
                (mo_coff.transpose() * dK2dXY * mo_coff).cwiseProduct(Ppq).sum(); // XY
                m_hessianKin(3 * atom, 3 * atom2 + 2) +=
                (mo_coff.transpose() * dK2dXZ * mo_coff).cwiseProduct(Ppq).sum(); // XZ
                m_hessianKin(3 * atom + 1, 3 * atom2 + 2) += 
                (mo_coff.transpose() * dK2dYZ * mo_coff).cwiseProduct(Ppq).sum(); // YZ
            }
            else
            {
                std::cout << "\n  Error: Invalid Hessian request for kinetic integrals.  SCF/MP2 only.\n\n";
                exit(EXIT_FAILURE);
            }


            m_hessianKin(3 * atom2, 3 * atom) = m_hessianKin(3 * atom, 3 * atom2); // XX
            m_hessianKin(3 * atom2 + 1, 3 * atom + 1) = m_hessianKin(3 * atom + 1, 3 * atom2 + 1); // YY
            m_hessianKin(3 * atom2 + 2, 3 * atom + 2) = m_hessianKin(3 * atom + 2, 3 * atom2 + 2); // ZZ
            m_hessianKin(3 * atom2 + 2, 3 * atom) = m_hessianKin(3 * atom, 3 * atom2 + 2); // XZ
            m_hessianKin(3 * atom + 2, 3 * atom2) = m_hessianKin(3 * atom, 3 * atom2 + 2); // ZX
            m_hessianKin(3 * atom2, 3 * atom + 2) = m_hessianKin(3 * atom + 2, 3 * atom2); // ZX
            m_hessianKin(3 * atom2 + 1, 3 * atom) = m_hessianKin(3 * atom, 3 * atom2 + 1);// XY
            m_hessianKin(3 * atom + 1, 3 * atom2) = m_hessianKin(3 * atom, 3 * atom2 + 1);// YX
            m_hessianKin(3 * atom2, 3 * atom + 1) = m_hessianKin(3 * atom + 1, 3 * atom2);// YX
            m_hessianKin(3 * atom2 + 2, 3 * atom + 1) = m_hessianKin(3 * atom + 1, 3 * atom2 + 2);// YZ
            m_hessianKin(3 * atom + 2, 3 * atom2 + 1) = m_hessianKin(3 * atom + 1, 3 * atom2 + 2);// ZY
            m_hessianKin(3 * atom2 + 1, 3 * atom + 2) = m_hessianKin(3 * atom + 2, 3 * atom2 + 1);// ZY
        }
    }

    if (hf_settings::get_verbosity() < 2) return;
    std::cout << '\n';
    std::cout << "  ****************************************\n";
    std::cout << "  *      Analytic Kinetic Hessian:       *\n";
    std::cout << "  *                                      *\n";
    std::cout << "  *      d2E/dX1dX2 / (Eh a0^-2)         *\n";
    std::cout << "  ****************************************\n";
    pretty_print_matrix<double>(m_hessianKin);
}


void Mol::scf_hessian::calc_scf_hessian_overlap()
{
    const auto& atoms = m_mol->get_atoms();
    const int natoms = static_cast<int>(atoms.size());
    const Index nshells = m_mol->get_num_shells();
    const auto& sp = m_mol->get_shell_pairs();
    bool is_pure = m_mol->use_pure_am();
    const Index num_orbitals = m_mol->get_num_orbitals();
    // A centers only - Translational invariance a == b

    std::unique_ptr<OSOVERLAP::OSOverlap> os_ptr = std::make_unique<OSOVERLAP::OSOverlap>(is_pure);
    tensor3d<double> dS2dqq = tensor3d<double>(6, num_orbitals, num_orbitals);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif 
    for (Index i = 0; i < nshells; ++i)
        for (Index j = i; j < nshells; ++j)
        {
            os_ptr->compute_contracted_shell_deriv2(dS2dqq, sp[i * nshells + j]);
        }

    m_hessianOvlap = EigenMatrix<double>::Zero(3 * natoms, 3 * natoms);
    const std::vector<std::vector<bool>>& msk = (is_pure) ? smask : mask;

    for(Index atom = 0; atom < natoms; ++atom)
    {
        for (Index atom2 = 0; atom2 <= atom; ++atom2)
        {

            EigenMatrix<double> dS2dXX = EigenMatrix<double>::Zero(num_orbitals, num_orbitals);
            EigenMatrix<double> dS2dYY = EigenMatrix<double>::Zero(num_orbitals, num_orbitals);
            EigenMatrix<double> dS2dZZ = EigenMatrix<double>::Zero(num_orbitals, num_orbitals);
            EigenMatrix<double> dS2dXY = EigenMatrix<double>::Zero(num_orbitals, num_orbitals);
            EigenMatrix<double> dS2dXZ = EigenMatrix<double>::Zero(num_orbitals, num_orbitals);
            EigenMatrix<double> dS2dYZ = EigenMatrix<double>::Zero(num_orbitals, num_orbitals);
            
            int same_center = 1;
            if(atom != atom2) same_center = -1;

            bool a, b;
            #ifdef _OPENMP
            #pragma omp parallel for private(a, b) schedule(dynamic)
            #endif
            for (Index i = 0; i < num_orbitals; ++i)
            {
                for (Index j = i + 1; j < num_orbitals; ++j)
                {
                    if(atom == atom2) { a = msk[atom][i]; b = msk[atom][j]; }
                    else { a = msk[atom][j] && msk[atom2][i]; b = msk[atom][i] && msk[atom2][j]; }

                    if(!a && !b) continue;
                    
                    if(a && b)
                    {
                        dS2dXX(i, j) = 0; dS2dYY(i, j) = 0; dS2dZZ(i, j) = 0;
                        dS2dXY(i, j) = 0; dS2dXZ(i, j) = 0; dS2dYZ(i, j) = 0;
                    }
                    else if(a || b)
                    {
                        dS2dXX(i, j) = dS2dqq(0, i, j); dS2dYY(i, j) = dS2dqq(3, i, j); dS2dZZ(i, j) = dS2dqq(5, i, j);
                        dS2dXY(i, j) = dS2dqq(1, i, j); dS2dXZ(i, j) = dS2dqq(2, i, j); dS2dYZ(i, j) = dS2dqq(4, i, j);
                    }

                    dS2dXX(i, j) *= same_center; dS2dXX(j, i) = dS2dXX(i, j);
                    dS2dYY(i, j) *= same_center; dS2dYY(j, i) = dS2dYY(i, j);
                    dS2dZZ(i, j) *= same_center; dS2dZZ(j, i) = dS2dZZ(i, j);
                    dS2dXY(i, j) *= same_center; dS2dXY(j, i) = dS2dXY(i, j);
                    dS2dXZ(i, j) *= same_center; dS2dXZ(j, i) = dS2dXZ(i, j);
                    dS2dYZ(i, j) *= same_center; dS2dYZ(j, i) = dS2dYZ(i, j);
                }
            }

            if ("SCF" == hf_settings::get_frequencies_type())
            {
                m_hessianOvlap(3 * atom, 3 * atom2) = -2.0 * dS2dXX.cwiseProduct(Q_mat).sum(); // XX
                m_hessianOvlap(3 * atom + 1, 3 * atom2 + 1) = -2.0 * dS2dYY.cwiseProduct(Q_mat).sum(); // YY
                m_hessianOvlap(3 * atom + 2, 3 * atom2 + 2) = -2.0 * dS2dZZ.cwiseProduct(Q_mat).sum(); // ZZ
                m_hessianOvlap(3 * atom, 3 * atom2 + 1) += -2.0 * dS2dXY.cwiseProduct(Q_mat).sum(); // XY
                m_hessianOvlap(3 * atom, 3 * atom2 + 2) += -2.0 * dS2dXZ.cwiseProduct(Q_mat).sum(); // XZ
                m_hessianOvlap(3 * atom + 1, 3 * atom2 + 2) += -2.0 * dS2dYZ.cwiseProduct(Q_mat).sum(); // YZ
            }
            else if ("MP2" == hf_settings::get_frequencies_type())
            {
                m_hessianOvlap(3 * atom, 3 * atom2) = 
                (mo_coff.transpose() * dS2dXX * mo_coff).cwiseProduct(I).sum(); // XX
                m_hessianOvlap(3 * atom + 1, 3 * atom2 + 1) = 
                (mo_coff.transpose() * dS2dYY * mo_coff).cwiseProduct(I).sum(); // YY
                m_hessianOvlap(3 * atom + 2, 3 * atom2 + 2) = 
                (mo_coff.transpose() * dS2dZZ * mo_coff).cwiseProduct(I).sum(); // ZZ
                m_hessianOvlap(3 * atom, 3 * atom2 + 1) += 
                (mo_coff.transpose() * dS2dXY * mo_coff).cwiseProduct(I).sum(); // XY
                m_hessianOvlap(3 * atom, 3 * atom2 + 2) += 
                (mo_coff.transpose() * dS2dXZ * mo_coff).cwiseProduct(I).sum(); // XZ
                m_hessianOvlap(3 * atom + 1, 3 * atom2 + 2) +=
                (mo_coff.transpose() * dS2dYZ * mo_coff).cwiseProduct(I).sum(); // YZ
            }
            else
            {
                std::cout << "\n  Error: Invalid Hessian request for overlap integrals.  SCF/MP2 only.\n\n";
                exit(EXIT_FAILURE);
            }
            
            
            m_hessianOvlap(3 * atom2, 3 * atom) = m_hessianOvlap(3 * atom, 3 * atom2); // XX
            m_hessianOvlap(3 * atom2 + 1, 3 * atom + 1) = m_hessianOvlap(3 * atom + 1, 3 * atom2 + 1); // YY
            m_hessianOvlap(3 * atom2 + 2, 3 * atom + 2) = m_hessianOvlap(3 * atom + 2, 3 * atom2 + 2); // ZZ
            m_hessianOvlap(3 * atom2 + 2, 3 * atom) = m_hessianOvlap(3 * atom, 3 * atom2 + 2); // XZ
            m_hessianOvlap(3 * atom + 2, 3 * atom2) = m_hessianOvlap(3 * atom, 3 * atom2 + 2); // ZX
            m_hessianOvlap(3 * atom2, 3 * atom + 2) = m_hessianOvlap(3 * atom + 2, 3 * atom2); // ZX
            m_hessianOvlap(3 * atom2 + 1, 3 * atom) = m_hessianOvlap(3 * atom, 3 * atom2 + 1); // XY
            m_hessianOvlap(3 * atom + 1, 3 * atom2) = m_hessianOvlap(3 * atom, 3 * atom2 + 1); // YX
            m_hessianOvlap(3 * atom2, 3 * atom + 1) = m_hessianOvlap(3 * atom + 1, 3 * atom2); // YX
            m_hessianOvlap(3 * atom2 + 2, 3 * atom + 1) = m_hessianOvlap(3 * atom + 1, 3 * atom2 + 2); // YZ
            m_hessianOvlap(3 * atom + 2, 3 * atom2 + 1) = m_hessianOvlap(3 * atom + 1, 3 * atom2 + 2); // ZY
            m_hessianOvlap(3 * atom2 + 1, 3 * atom + 2) = m_hessianOvlap(3 * atom + 2, 3 * atom2 + 1); // ZY
        }
    }

    if (hf_settings::get_verbosity() < 2) return;
    std::cout << '\n';
    std::cout << "  ****************************************\n";
    std::cout << "  *      Analytic Overlap Hessian:       *\n";
    std::cout << "  *                                      *\n";
    std::cout << "  *      d2E/dX1dX2 / (Eh a0^-2)         *\n";
    std::cout << "  ****************************************\n";
    pretty_print_matrix<double>(m_hessianOvlap);
}

void Mol::scf_hessian::calc_scf_hessian_nuclear()
{
    EigenMatrix<double> geom = m_mol->get_geom();
    const Index natoms = m_mol->get_atoms().size();
    m_hessianNuc = EigenMatrix<double>::Zero(3 * natoms, 3 * natoms);
    const std::vector<int>& charge = m_mol->get_z_values();

    for (Index i = 0; i < natoms; ++i) 
    {
        for (Index j = 0; j < i; ++j) 
        {
            double dx = geom(i, 0) - geom(j, 0);
            double dy = geom(i, 1) - geom(j, 1);
            double dz = geom(i, 2) - geom(j, 2);

            double x2 = dx * dx;
            double y2 = dy * dy;
            double z2 = dz * dz;
            double r2 = x2 + y2 + z2;
            double r5 = r2 * r2 * sqrt(r2);
            double prefactor = charge[i] * charge[j] / r5;

            m_hessianNuc(3 * i, 3 * i) += prefactor * (3 * x2 - r2);
            m_hessianNuc(3 * i + 1, 3 * i + 1) += prefactor * (3 * y2 - r2);
            m_hessianNuc(3 * i + 2, 3 * i + 2) += prefactor * (3 * z2 - r2);
            m_hessianNuc(3 * i, 3 * i + 1) += prefactor * 3 * dx * dy;
            m_hessianNuc(3 * i, 3 * i + 2) += prefactor * 3 * dx * dz;
            m_hessianNuc(3 * i + 1, 3 * i + 2) += prefactor * 3 * dy * dz;

            m_hessianNuc(3 * j, 3 * j) += prefactor * (3 * x2 - r2);
            m_hessianNuc(3 * j + 1, 3 * j + 1) += prefactor * (3 * y2 - r2);
            m_hessianNuc(3 * j + 2, 3 * j + 2) += prefactor * (3 * z2 - r2);
            m_hessianNuc(3 * j, 3 * j + 1) += prefactor * 3 * dx * dy;
            m_hessianNuc(3 * j, 3 * j + 2) += prefactor * 3 * dx * dz;
            m_hessianNuc(3 * j + 1, 3 * j + 2) += prefactor * 3 * dy * dz;

            m_hessianNuc(3 * i, 3 * j) += -prefactor * (3 * dx * dx - r2);
            m_hessianNuc(3 * i, 3 * j + 1) += -prefactor * (3 * dx * dy);
            m_hessianNuc(3 * i, 3 * j + 2) += -prefactor * (3 * dx * dz);

            m_hessianNuc(3 * i + 1, 3 * j) += -prefactor * (3 * dy * dx);
            m_hessianNuc(3 * i + 1, 3 * j + 1) += -prefactor * (3 * dy * dy - r2);
            m_hessianNuc(3 * i + 1, 3 * j + 2) += -prefactor * 3 * dy * dz;
            
            m_hessianNuc(3 * i + 2, 3 * j) += -prefactor * 3 * dz * dx;
            m_hessianNuc(3 * i + 2, 3 * j + 1) += -prefactor * 3 * dz * dy;
            m_hessianNuc(3 * i + 2, 3 * j + 2) += -prefactor * (3 * dz * dz - r2);
        }
    }

    for(Index i = 0; i < 3 * natoms; ++i)
        for(Index j = 0; j <= i - 1; ++j)
        {
            double tmp = m_hessianNuc(j , i) + m_hessianNuc(i , j);
            m_hessianNuc(j , i) = m_hessianNuc(i , j) = tmp;
        }

    if (hf_settings::get_verbosity() < 2) return;
    std::cout << '\n';
    std::cout << "  ****************************************\n";
    std::cout << "  *      Analytic Nuclear Hessian:       *\n";
    std::cout << "  *                                      *\n";
    std::cout << "  *      d2E/dX1dX2 / (Eh a0^-2)         *\n";
    std::cout << "  ****************************************\n";
    pretty_print_matrix<double>(m_hessianNuc, 0, 0);
}

void Mol::scf_hessian::calc_scf_hessian_pot()
{
    const auto& atoms = m_mol->get_atoms();
    const int natoms = static_cast<int>(atoms.size());
    const Index nshells = m_mol->get_num_shells();
    const auto& sp = m_mol->get_shell_pairs();
    bool is_pure = m_mol->use_pure_am();
    const Index num_orbitals = m_mol->get_num_orbitals();

    std::unique_ptr<OSNUCLEAR::OSNuclear> os_ptr = std::make_unique<OSNUCLEAR::OSNuclear>(is_pure);

    m_hessianVn = EigenMatrix<double>::Zero(3 * natoms, 3 * natoms);

    int max_cart = 8;
    std::vector<bool> out_of_plane(3, false);
    if(m_mol->molecule_is_linear())  max_cart = 2;
    else // is it planar ?                         // molecule class aligns anyway
    {
        constexpr double cutoff = 1.0E-10; 
        for(int cart_dir = 0; cart_dir < 3; ++cart_dir)
        {
            for(int i = 0; i < natoms; ++i)
                if(std::fabs<double>(m_mol->get_geom()(i , cart_dir)) > cutoff)
                    out_of_plane[cart_dir] = true;
        }
    }

    for(int cart2 = 0; cart2 <= max_cart; ++cart2)
    {
        if(!out_of_plane[0] && cart2 > 2 && cart2 <= 4) continue; // planar xy
        else if (!out_of_plane[1] && cart2 > 2 && (cart2 == 3 || cart2 == 5)) continue; // planar yz xy
        else if (!out_of_plane[2] && cart2 > 2 && (cart2 == 4 || cart2 == 5)) continue; // planar yz xz

        Cart2 coord; // = Cart2::XX;

        if (cart2 == 0) coord = Cart2::XX;
        else if (cart2 == 1) coord = Cart2::YY;
        else if (cart2 == 2) coord = Cart2::ZZ;
        else if (cart2 == 3) coord = Cart2::XY;
        else if (cart2 == 4) coord = Cart2::XZ;
        else if (cart2 == 5) coord = Cart2::YZ;
        else if (cart2 == 6) coord = Cart2::ZX;
        else if (cart2 == 7) coord = Cart2::YX;
        else if (cart2 == 8) coord = Cart2::ZY;

        tensor4d1234<double> dVdqq = tensor4d1234<double>(natoms, natoms, num_orbitals, num_orbitals);

        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (Index i = 0; i < nshells; ++i)
             for (Index j = i; j < nshells; ++j) // Note use cartesian mask here even if pure
                os_ptr->compute_contracted_shell_deriv2(dVdqq, sp[nshells * i + j], atoms, coord, mask);

        for(int atom = 0; atom < natoms; ++atom)
        {
            for (int atom2 = atom; atom2 < natoms; ++atom2)
            {   // Note the sum structure since we only generate the upper triangular part for basis and atoms
                EigenMatrix<double> dVndXX = EigenMatrix<double>(num_orbitals, num_orbitals);

                for (Index i = 0; i < num_orbitals; ++i)
                    for (Index j = i; j < num_orbitals; ++j)
                    {
                        dVndXX(i, j) = dVdqq(atom, atom2, i, j);
                        dVndXX(j, i) = dVndXX(i, j);
                    }
                
                double hesspq = 0;
                
                if (hf_settings::get_frequencies_type() == "SCF")
                    hesspq = 2.0 * dVndXX.cwiseProduct(d_mat).sum();
                else if (hf_settings::get_frequencies_type() == "MP2")
                    hesspq = (mo_coff.transpose() * dVndXX * mo_coff).cwiseProduct(Ppq).sum();
                else
                {
                    std::cout << "\n  Error: Invalid Hessian request for potential integrals.  SCF/MP2 only.\n\n";
                    exit(EXIT_FAILURE);
                }
                
                if (cart2 <= 2) // XX YY ZZ
                {
                    m_hessianVn(3 * atom + cart2, 3 * atom2 + cart2) = hesspq;
                    m_hessianVn(3 * atom2 + cart2, 3 * atom + cart2) = m_hessianVn(3 * atom + cart2, 3 * atom2 + cart2);
                }
                if(cart2 == 3) // XY
                {
                    m_hessianVn(3 * atom, 3 * atom2 + 1) = hesspq;
                    m_hessianVn(3 * atom2 + 1, 3 * atom) = m_hessianVn(3 * atom, 3 * atom2 + 1);
                }
                else if(cart2 == 4) // XZ
                {
                    m_hessianVn(3 * atom, 3 * atom2 + 2) = hesspq;
                    m_hessianVn(3 * atom2 + 2, 3 * atom) = m_hessianVn(3 * atom, 3 * atom2 + 2);
                }
                else if(cart2 == 5) // YZ
                {
                    m_hessianVn(3 * atom + 1, 3 * atom2 + 2) = hesspq;
                    m_hessianVn(3 * atom2 + 2, 3 * atom + 1) = m_hessianVn(3 * atom + 1, 3 * atom2 + 2);
                }
                else if(cart2 == 6 && atom != atom2) // ZX
                {
                    m_hessianVn(3 * atom + 2, 3 * atom2) = hesspq;
                    m_hessianVn(3 * atom2, 3 * atom + 2) = m_hessianVn(3 * atom + 2, 3 * atom2);
                }
                else if(cart2 == 7 && atom != atom2) // YX
                {
                    m_hessianVn(3 * atom + 1, 3 * atom2) = hesspq;
                    m_hessianVn(3 * atom2, 3 * atom + 1) = m_hessianVn(3 * atom + 1, 3 * atom2);
                }
                else if(cart2 == 8 && atom != atom2) // ZY
                {
                    m_hessianVn(3 * atom + 2, 3 * atom2 + 1) = hesspq;
                    m_hessianVn(3 * atom2 + 1, 3 * atom + 2) = m_hessianVn(3 * atom + 2, 3 * atom2 + 1);
                }
            }
        }
    }
    
    if (hf_settings::get_verbosity() < 2) return;
    std::cout << '\n';
    std::cout << "  ****************************************\n";
    std::cout << "  *      Analytic Potential Hessian:     *\n";
    std::cout << "  *                                      *\n";
    std::cout << "  *      d2E/dX1dX2 / (Eh a0^-2)         *\n";
    std::cout << "  ****************************************\n";
    pretty_print_matrix<double>(m_hessianVn);
}

void Mol::scf_hessian::calc_scf_hessian_tei_mp2()
{
    const Index num_orbitals = m_mol->get_num_orbitals();
    const Index num_shells = m_mol->get_num_shells();
    const auto& sp = m_mol->get_shell_pairs();
    const Index occ = m_mol->get_num_electrons() / 2;
    const auto& atoms = m_mol->get_atoms();
    const int natoms = static_cast<int>(atoms.size());
    bool is_pure = m_mol->use_pure_am();
    const double cutoff = hf_settings::get_integral_tol();

    
    std::unique_ptr<BTRANS::basis_transform> ao_to_mo_ptr = 
                std::make_unique<BTRANS::basis_transform>(num_orbitals);

    m_hessianC = EigenMatrix<double>::Zero(3 * natoms, 3 * natoms);
    
    std::unique_ptr<ERIOS::Erios> os_ptr = std::make_unique<ERIOS::Erios>(is_pure, cutoff);

    for(int cart2 = 0; cart2 < 3; ++cart2) // XX YY ZZ
    {   // The same permutations apply as non deriv eri integrals <ijkl> <jikl> <jilk> <jilk>
        // swap bra ket to gives another 4. 8 total.
        symm4dTensor<double> AA = symm4dTensor<double>(num_orbitals);
        symm4dTensor<double> AB = symm4dTensor<double>(num_orbitals);
        symm4dTensor<double> AC = symm4dTensor<double>(num_orbitals);
        symm4dTensor<double> BD = symm4dTensor<double>(num_orbitals);
        symm4dTensor<double> CC = symm4dTensor<double>(num_orbitals);
        symm4dTensor<double> CD = symm4dTensor<double>(num_orbitals);

        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (Index i = 0; i < num_shells; ++i)
            for (Index j = 0; j <= i; ++j)
                for (Index k = 0; k < num_shells; ++k)
                    for (Index l = 0; l <= k; ++l)
                    {
                        os_ptr->compute_contracted_shell_quartet_deriv2_qq(AA, AB, AC, BD, CC, CD, 
                                                                           sp[i * num_shells + j], 
                                                                           sp[k * num_shells + l], 
                                                                           cart2);
                    }

        for(int atom = 0; atom < natoms; ++atom)
        {
            const std::vector<bool> &mask1 = (is_pure) ? smask[atom] : mask[atom];

            for (int atom2 = 0; atom2 <= atom; ++atom2)
            {
                const std::vector<bool> &mask2 = (is_pure) ? smask[atom2] : mask[atom2];

                symm4dTensor<double> eri_tensor = symm4dTensor<double>(num_orbitals);
                double tmp, ad, bb, dd, bc;
                #ifdef _OPENMP
                #pragma omp parallel for private (tmp) schedule(dynamic)
                #endif
                for (Index i = 0; i < num_orbitals; ++i)
                    for (Index j = 0; j <= i; ++j)
                    {
                        Index ij = index_ij(i, j);
                        for (Index k = 0; k < num_orbitals; ++k)
                            for (Index l = 0; l <= k; ++l)
                            {
                               Index kl = index_ij(k, l);
                               if (ij <= kl) // Translational invariance relationships
                               {   
                                    ad = -(AA(i, j, k, l) + AB(i, j, k, l) + AC(i, j , k, l));
                                    dd = -(ad + BD(i, j, k, l) + CD(i, j, k, l));
                                    bb = AA(i, j, k, l) + 2 * AC(i, j, k, l) + 2 * ad + CC(i, j, k, l) 
                                    + 2 * CD(i, j, k, l) + dd;
                                    bc =  -(AC(i, j, k, l) + CC(i, j, k, l) + CD(i, j, k, l));
    
                                    tmp = 0;
                                    if(atom == atom2)
                                    { 
                                        if (mask1[i]) tmp += AA(i, j, k, l); // AA
                                        if (mask1[j]) tmp += bb; // BB 
                                        if (mask1[k]) tmp += CC(i, j, k, l); // CC
                                        if (mask1[l]) tmp += dd; // DD
                                        if (mask1[i] && mask1[j]) tmp += 2 * AB(i, j, k, l); // AB
                                        if (mask1[i] && mask1[k]) tmp += 2 * AC(i, j, k, l); // AC
                                        if (mask1[i] && mask1[l]) tmp += 2 * ad; // AD
                                        if (mask1[j] && mask1[k]) tmp += 2 * bc; // BC
                                        if (mask1[j] && mask1[l]) tmp += 2 * BD(i, j, k, l); // BD
                                        if (mask1[k] && mask1[l]) tmp += 2 * CD(i, j, k, l); // CD
                                    }
                                    else
                                    {
                                        if ((mask1[i] && mask2[j]) || (mask1[j] && mask2[i])) tmp += AB(i, j, k, l); // AB
                                        if ((mask1[i] && mask2[k]) || (mask1[k] && mask2[i])) tmp += AC(i, j, k, l); // AC
                                        if ((mask1[i] && mask2[l]) || (mask1[l] && mask2[i])) tmp += ad; // AD
                                        if ((mask1[j] && mask2[k]) || (mask1[k] && mask2[j])) tmp += bc; // BC
                                        if ((mask1[j] && mask2[l]) || (mask1[l] && mask2[j])) tmp += BD(i, j, k, l); // BD
                                        if ((mask1[k] && mask2[l]) || (mask1[l] && mask2[k])) tmp += CD(i, j, k, l); // CD
                                    }

                                    eri_tensor(i, j, k, l) = tmp;
                                }
                            }
                    }

                EigenVector<double> eri_tensor_mo;
                eri_tensor_mo = EigenVector<double>::Zero((num_orbitals * (num_orbitals + 1) / 2) *
                                                         ((num_orbitals * (num_orbitals + 1) / 2) + 1) / 2);
                
                ao_to_mo_ptr->ao_to_mo_transform_2e(mo_coff, eri_tensor.get_vector_form(), eri_tensor_mo);
                
                double tei1 = 0;
                #ifdef _OPENMP
                #pragma omp parallel for reduction (+:tei1) schedule(dynamic)
                #endif
                for (Index p = 0; p < num_orbitals; ++p)
                    for (Index q = 0; q < num_orbitals; ++q)
                        for (Index i = 0; i < occ; ++i)
                        {
                            Index pqii = index4(p, q, i, i);
                            Index piiq = index4(p, i, i, q);
                            tei1 += Ppq(p, q) * (2.0 * eri_tensor_mo(pqii) - eri_tensor_mo(piiq));
                        }
                
                double tei2 = 0;
                #ifdef _OPENMP
                #pragma omp parallel for reduction (+:tei2) schedule(dynamic)
                #endif
                for (Index p = 0; p < num_orbitals; ++p)
                    for (Index q = 0; q < num_orbitals; ++q)
                        for (Index r = 0; r < num_orbitals; ++r)
                            for (Index s = 0; s < num_orbitals; ++s)
                            {
                                Index prqs = index4(p, r, q, s); // note chemist notation ints
                                tei2 += eri_tensor_mo(prqs) * Ppqrs(p, q, r, s);
                            }
    
                m_hessianC(3 * atom + cart2, 3 * atom2 + cart2) = 
                m_hessianC(3 * atom2 + cart2, 3 * atom + cart2) = tei1 + tei2;
            }
        }
    }

    int maxcoords = 5;
    std::vector<bool> out_of_plane(3, false);
    if(m_mol->molecule_is_linear()) maxcoords = 0; // need to put in check if it is aligned 
    else // is it planar ?                         // molecule class aligns anyway
    {
        constexpr double tol = 1.0E-10; 
        for(int cart_dir = 0; cart_dir < 3; ++cart_dir)
        {
            for(int i = 0; i < natoms; ++i)
                if(std::fabs<double>(m_mol->get_geom()(i , cart_dir)) > tol)
                    out_of_plane[cart_dir] = true;
        }
    }

    tensor4d<double> AA = tensor4d<double>(num_orbitals); // xz xy etc.
    tensor4d<double> AB = tensor4d<double>(num_orbitals);
    tensor4d<double> AC = tensor4d<double>(num_orbitals);
                                                   
    for(int cart2 = 3; cart2 <= maxcoords; ++cart2) // XY XZ YZ etc. 45 total deriv2 terms including the above
    {
        if(!out_of_plane[0] && cart2 <= 4) continue;
        else if (!out_of_plane[1] && (cart2 == 3 || cart2 == 5)) continue;
        else if (!out_of_plane[2] && (cart2 == 4 || cart2 == 5)) continue;

        AA.setZero();
        AB.setZero();
        AC.setZero();

        // since we are doing the full loop we can use the following permutations
        // AC(i, j, k, l) = eri_pq(2);
        // BD(j, i, l, k) = eri_pq(2);
        // AC1(k, l, i, j) = eri_pq(2); 1 denotes reverse deriv order
        // The same applies to AA, AB AB1 which map to CC CD CD1 respectively by permutation,
        // thus, we are only using AA AB and AC tensors, which contain all needed permutations

        // Without permutation symmetry we had
        // AA(i, j, k, l) = eri_pq(0); AB(i, j, k, l) = eri_pq(1); AC(i, j, k, l) = eri_pq(2);
        // BD(i, j, k, l) = eri_pq(3); CC(i, j, k, l) = eri(4); CD(i, j, k, l) = eri_pq(5);
        // CD1(i, j, k, l) = eri_pq(7); AB1(i, j, k, l) = eri_pq(11); AC1(i, j, k, l) = eri_pq(12);

        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (Index i = 0; i < num_shells; ++i)
            for (Index j = 0; j < num_shells; ++j)
                for (Index k = 0; k < num_shells; ++k)
                    for (Index l = 0; l < num_shells; ++l) 
                    {
                        if (l <= k)
                        {
                            os_ptr->compute_contracted_shell_quartet_deriv2_pq(
                                AA, AB, AC, sp[i * num_shells + j], sp[k * num_shells + l], cart2, ERIOS::Permute::ALL);
                        } 
                        else 
                        {
                            os_ptr->compute_contracted_shell_quartet_deriv2_pq(
                                AA, AB, AC, sp[i * num_shells + j], sp[k * num_shells + l], cart2, ERIOS::Permute::ACONLY);
                        }
                    }

        for(int atom = 0; atom < natoms; ++atom)
        {
            const std::vector<bool> &mask1 = (is_pure) ? smask[atom] : mask[atom];

            for (int atom2 = 0; atom2 < natoms; ++atom2)
            {
                const std::vector<bool> &mask2 = (is_pure) ? smask[atom2] : mask[atom2];
                
                tensor4d<double> eri_tensor = tensor4d<double>(num_orbitals);
                double tmp, ad, ad1, bb, bc, bc1, bd1, dd;
                #ifdef _OPENMP
                #pragma omp parallel for private (tmp) schedule(dynamic)
                #endif
                for (int i = 0; i < num_orbitals; ++i)
                    for (int j = 0; j < num_orbitals; ++j) 
                        for (int k = 0; k < num_orbitals; ++k)
                            for (int l = 0; l < num_orbitals; ++l)
                            {   
                                //if (i == j && j == k && k == l) continue; // only slows down OMP
                                
                                // Translational invariance relationships
                                ad  = -(AA(i, j, k, l) + AB(i, j, k, l) + AC(i, j, k, l));
                                ad1 = -(AA(i, j, k, l) + AB(j, i, k, l) + AC(k, l, i, j));
                                dd  = -(ad + AC(j, i, l, k) + AB(k, l, i, j));
                                bb  =  AA(i, j, k, l) + AC(i, j, k, l) + ad + AC(k, l, i, j) +
                                       AA(k, l, i, j) + AB(k, l, i, j) + ad1 + AB(l, k, i, j) + dd;
                                bc  = -(AC(i, j, k, l) + AA(k, l, i, j) + AB(l, k, i, j));
                                bc1 = -(AC(k, l, i, j) + AA(k, l, i, j) + AB(k, l, i, j));
                                bd1 = -(ad1 + AB(l, k, i, j) + dd);
                                tmp = 0;
                                
                                if(atom == atom2)
                                {
                                    if (mask1[i]) tmp += AA(i, j, k, l); // AA
                                    if (mask1[j]) tmp += bb; // BB(i, j, k, l); // BB 
                                    if (mask1[k]) tmp += AA(k, l, i, j); // CC(i, j, k, l); // CC
                                    if (mask1[l]) tmp += dd; // DD
                                }
                              
                                if (mask1[i] && mask2[j]) tmp += AB(i, j, k, l); // AB
                                if (mask1[j] && mask2[i]) tmp += AB(j, i, k, l); //AB1(i, j, k, l); // AB1
                                if (mask1[i] && mask2[k]) tmp += AC(i, j, k, l); // AC 
                                if (mask1[k] && mask2[i]) tmp += AC(k, l, i, j); // AC1(i, j, k, l); // AC1
                                if (mask1[i] && mask2[l]) tmp += ad; // AD
                                if (mask1[l] && mask2[i]) tmp += ad1; // AD1
                                if (mask1[j] && mask2[k]) tmp += bc; // BC(i, j, k, l); // BC 
                                if (mask1[k] && mask2[j]) tmp += bc1; // BC1(i, j, k, l); // BC1
                                if (mask1[j] && mask2[l]) tmp += AC(j, i, l, k); // BD(i, j, k, l); // BD 
                                if (mask1[l] && mask2[j]) tmp += bd1; // BD1(i, j, k, l); // BD1
                                if (mask1[k] && mask2[l]) tmp += AB(k, l, i, j);  // CD 
                                if (mask1[l] && mask2[k]) tmp += AB(l, k, i, j); // CD1(i, j, k, l); // CD1
                                
                                eri_tensor(i, j, k, l) = tmp;
                            }

                tensor4d<double> eri_tensor_mo;
                ao_to_mo_ptr->ao_to_mo_transform_2e(mo_coff, mo_coff, eri_tensor, eri_tensor_mo);
                
                double tei1 = 0;
                #ifdef _OPENMP
                #pragma omp parallel for reduction (+:tei1) schedule(dynamic)
                #endif
                for (Index p = 0; p < num_orbitals; ++p)
                    for (Index q = 0; q < num_orbitals; ++q)
                        for (Index i = 0; i < occ; ++i)
                            tei1 += Ppq(p, q) * (2.0 * eri_tensor_mo(p, q, i, i) - eri_tensor_mo(p, i, i, q));

                double tei2 = 0;
                #ifdef _OPENMP
                #pragma omp parallel for reduction (+:tei2) schedule(dynamic)
                #endif
                for (Index p = 0; p < num_orbitals; ++p)
                    for (Index q = 0; q < num_orbitals; ++q)
                        for (Index r = 0; r < num_orbitals; ++r)
                            for (Index s = 0; s < num_orbitals; ++s)
                                tei2 += eri_tensor_mo(p, r, q, s) * Ppqrs(p, q, r, s);

                if (cart2 == 3) // xy yx
                    m_hessianC(3 * atom, 3 * atom2 + 1) = m_hessianC(3 * atom2 + 1, 3 * atom) = tei1 + tei2;
                else if (cart2 == 4) // xz zx
                    m_hessianC(3 * atom, 3 * atom2 + 2) = m_hessianC(3 * atom2 + 2, 3 * atom) = tei1 + tei2;
                else // if(cart2 == 5) // yz zy
                    m_hessianC(3 * atom + 1, 3 * atom2 + 2) = m_hessianC(3 * atom2 + 2, 3 * atom + 1) = tei1 + tei2;

            }  
        }
    }

    if (hf_settings::get_verbosity() < 2) return;
    std::cout << '\n';
    std::cout << "  ****************************************\n";
    std::cout << "  *      Analytic TEI Hessian:           *\n";
    std::cout << "  *                                      *\n";
    std::cout << "  *      d2E/dX1dX2 / (Eh a0^-2)         *\n";
    std::cout << "  ****************************************\n";
    pretty_print_matrix<double>(m_hessianC);
}