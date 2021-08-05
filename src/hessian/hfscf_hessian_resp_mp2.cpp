#include "../settings/hfscf_settings.hpp"
#include "../gradient/hfscf_gradient.hpp"
#include "../basis/hfscf_trans_basis.hpp"
#include "../math/hfscf_tensor4d.hpp"
#include "../math/hfscf_tensors.hpp"

using hfscfmath::index_ij;
using hfscfmath::index_ijkl;
using tensor4dmath::index4;
using tensormath::tensor4d1234;
using HF_SETTINGS::hf_settings;
using Eigen::Index;

void Mol::scf_gradient::hessian_resp_terms(const Eigen::Ref<const EigenMatrix<double> >& mo_coff,
                                           const Eigen::Ref<const EigenMatrix<double> >& dTdX,
                                           const Eigen::Ref<const EigenMatrix<double> >& dVndX,
                                           const Eigen::Ref<const EigenMatrix<double> >& dSdX,
                                           const Eigen::Ref<const EigenMatrix<double> >& mo_energies,
                                           const Eigen::Ref<const EigenMatrix<double> >& Ginv,
                                           const Eigen::Ref<const EigenVector<double> >& dVdXmo,
                                           int atom, int cart_dir)
{
    constexpr double eps = 1E-12;
    const auto num_orbitals = m_mol->get_num_orbitals();
    const Index virt = num_orbitals - m_mol->get_num_electrons() / 2;
    const Index occ = num_orbitals - virt;
    const Index idx = m_mol->get_atoms().size() * cart_dir + atom;
    EigenMatrix<double> F_grad = mo_coff.transpose() * (dTdX + dVndX) * mo_coff;
    EigenMatrix<double> dSdX_mo = mo_coff.transpose() * dSdX * mo_coff;

    const auto eri_mo = [this](Index p, Index q, Index r, Index s) noexcept -> double
    {
        Index prqs = index4(p, r, q, s);
        return e_rep_mo[prqs];
    };

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index p = 0; p < num_orbitals; ++p)
        for (Index q = 0; q < num_orbitals; ++q)
            for (Index l = 0; l < occ; ++l)
            {
                Index pqll = index4(p, q, l, l);
                Index pllq = index4(p, l, l, q);
                F_grad(p, q) += 2.0 * dVdXmo(pqll) - dVdXmo(pllq);
            }

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index a = 0; a < virt; ++a)
        for (Index j = 0; j < occ; ++j) 
        {
            Bpq(idx, a, j) = mo_energies(j, j) * dSdX_mo(j, a + occ) - F_grad(a + occ, j);
            for (Index k = 0; k < occ; ++k)
                for (Index l = 0; l < occ; ++l)
                {
                    Index ajkl = index4(a + occ, j, k, l);
                    Index akjl = index4(a + occ, k, j, l);
                    Bpq(idx, a, j) += (2.0 * e_rep_mo(ajkl) - e_rep_mo(akjl)) * dSdX_mo(k, l);
                }
        }

    for (Index i = 0; i < occ; ++i)
        for (Index j = 0; j < occ; ++j)
            U(idx, i, j) = -0.5 * dSdX_mo(i, j);
    
    for (Index a = 0; a < virt; ++a)
        for (Index b = 0; b < virt; ++b)
            U(idx, a + occ, b + occ) = -0.5 * dSdX_mo(a + occ, b + occ);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index i = 0; i < occ; ++i)
        for (Index a = 0; a < virt; ++a)
            for (Index j = 0; j < occ; ++j)
                for (Index b = 0; b < virt; ++b)
                    U(idx, a + occ, i) += Bpq(idx, b, j) * Ginv(i + occ * a, j + occ * b);
    
    for (Index j = 0; j < occ; ++j)
        for (Index a = 0; a < virt; ++a)
            U(idx, j, a + occ) += -U(idx, a + occ, j) - dSdX_mo(a + occ, j);
    
    tensor4d1122<double> t2_deriv1 = tensor4d1122<double>(occ, virt);
    tensor4d1122<double> t2_tilde_deriv1 = tensor4d1122<double>(occ, virt);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(auto)
    #endif
    for (Index i = 0; i < occ; ++i)
        for (Index j = 0; j < occ; ++j)
            for (Index a = 0; a < virt; ++a)
                for (Index b = 0; b < virt; ++b)
                {
                    Index ijab = index4(i, a + occ, j, b + occ);
                    t2_deriv1(i, j, a, b) += dVdXmo(ijab);

                    for (Index p = 0; p < num_orbitals; ++p)
                    {
                        t2_deriv1(i, j, a, b) +=
                        U(idx, p, i) * eri_mo(p, j, a + occ, b + occ) +
                        U(idx, p, j) * eri_mo(i, p, a + occ, b + occ) +
                        U(idx, p, a + occ) * eri_mo(i, j, p, b + occ) +
                        U(idx, p, b + occ) * eri_mo(i, j, a + occ, p);
                    }

                    for (Index c = 0; c < virt; ++c)
                    {
                        t2_deriv1(i, j, a, b) += 
                        F_grad(a + occ, c + occ) * t2(i, j, c, b) +
                        F_grad(b + occ, c + occ) * t2(i, j, a, c);
                        
                        for (Index p = 0; p < num_orbitals; ++p)
                        {
                            t2_deriv1(i, j, a, b) += 
                            t2(i, j, c, b) * (U(idx, p, a + occ) * mo_energies(p, c + occ) +
                             U(idx, p, c + occ) * mo_energies(a + occ, p)) +
                            t2(i, j, a, c) * (U(idx, p, b + occ) * mo_energies(p, c + occ) +
                            U(idx, p, c + occ) * mo_energies(p, b + occ)); 
                        }

                        if (fabs(t2(i, j, c, b)) > eps)
                            for(Index k = 0; k < occ; ++k)
                            {
                                for(Index l = 0; l < occ; ++l)
                                {
                                    t2_deriv1(i, j, a, b) += t2(i, j, c, b) * dSdX_mo(k, l) *
                                    0.5 * (- 4.0 * eri_mo(a + occ, k, c + occ, l)
                                     + eri_mo(a + occ, c + occ, k, l)
                                     + eri_mo(a + occ, c + occ, l, k));
                                }

                                for (Index d = 0; d < virt; ++d)
                                {
                                    t2_deriv1(i, j, a, b) += t2(i, j, c, b) *
                                    (+ 4.0 * U(idx, d + occ, k) * eri_mo(a + occ, d + occ, c + occ, k)
                                    - U(idx, d + occ, k) * (eri_mo(a + occ, c + occ, d + occ, k)
                                                          + eri_mo(a + occ, c + occ, k, d + occ)));
                                }
                            }
                    }

                    for(Index k = 0; k < occ; ++k)
                    {
                         t2_deriv1(i, j, a, b) += 
                        - t2(k, j, a, b) * F_grad(k, i) - t2(i, k, a, b) * F_grad(k, j);

                        for (Index q = 0; q < num_orbitals; ++q)
                            t2_deriv1(i, j, a, b) -= t2(k, j, a, b) *
                            ( U(idx, q, k) * mo_energies(q, i) 
                            + U(idx, q, i) * mo_energies(k, q)) +
                            t2(i, k, a, b) * 
                            ( U(idx, q, k) * mo_energies(q, j)
                            + U(idx, q, j) * mo_energies(k, q));

                        for (Index c = 0; c < virt; ++c)
                        {
                            if (fabs(t2(i, j, a, c)) > eps)
                            {
                                for(Index l = 0; l < occ; ++l)
                                    t2_deriv1(i, j, a, b) += t2(i, j, a, c) * dSdX_mo(k, l) *
                                    ( - 2.0 * eri_mo(b + occ, k, c + occ, l)
                                    + 0.5 * (eri_mo(b + occ, c + occ, k, l) + eri_mo(b + occ, c + occ, l, k)));

                                for (Index d = 0; d < virt; ++d)
                                    t2_deriv1(i, j, a, b) += t2(i, j, a, c) * U(idx, d + occ, k) *
                                    ( + 4.0 * eri_mo(b + occ, d + occ, c + occ, k)
                                    - eri_mo(b + occ, c + occ, d + occ, k) - eri_mo(b + occ, c + occ, k, d + occ));
                            }
                        }

                        if (fabs(t2(k, j, a, b)) > eps)
                            for(Index l = 0; l < occ; ++l)
                            {
                                for(Index m = 0; m < occ; ++m)
                                    t2_deriv1(i, j, a, b) += dSdX_mo(l, m) * t2(k, j, a, b) *
                                    (2.0 * eri_mo(k, l, i, m) - 0.5 * eri_mo(k, i, l, m) - 0.5 * eri_mo(k, i, m, l));
                                
                                for (Index d = 0; d < virt; ++d)
                                    t2_deriv1(i, j, a, b) += U(idx, d + occ, l) * t2(k, j, a, b) *
                                    (- 4.0 * eri_mo(k, d + occ, i, l) + eri_mo(k, i, d + occ, l)
                                    + eri_mo(k, i, l, d + occ));
                            }

                        if (fabs(t2(i, k, a, b)) > eps)
                            for(Index l = 0; l < occ; ++l)
                            {   
                                for(Index m = 0; m < occ; ++m)
                                    t2_deriv1(i, j, a, b) += dSdX_mo(l, m) * t2(i, k, a, b) *
                                    (2.0 * eri_mo(k, l, j, m) - 0.5 * eri_mo(k, j, l, m) - 0.5 * eri_mo(k, j, m, l));

                                
                                for (Index d = 0; d < virt; ++d)
                                    t2_deriv1(i, j, a, b) += U(idx, d + occ, l) * t2(i, k, a, b) *
                                    (- 4.0 * eri_mo(k, d + occ, j, l) + eri_mo(k, j, d + occ, l)
                                    + eri_mo(k, j, l, d + occ));
                            }
                    }

                    t2_deriv1(i, j, a, b) /= (mo_energies(i, i) + mo_energies(j, j) 
                                    -  mo_energies(a + occ, a + occ) - mo_energies(b + occ, b + occ));
                    
                }

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index i = 0; i < occ; ++i)
        for (Index j = 0; j < occ; ++j)
            for (Index a = 0; a < virt; ++a)
                for (Index b = 0; b < virt; ++b)
                    t2_tilde_deriv1(i, j, a, b) = 2.0 * t2_deriv1(i, j, a, b) - t2_deriv1(i, j, b, a);

    EigenMatrix<double> dPij = EigenMatrix<double>::Zero(occ, occ);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index i = 0; i < occ; ++i)
        for (Index j = 0; j < occ; ++j)
            for (Index k = 0; k < occ; ++k)
                for (Index a = 0; a < virt; ++a)
                    for (Index b = 0; b < virt; ++b)
                        dPij(i, j) += - t2_deriv1(i, k, a, b) * t2_tilde(j, k, a, b)
                                      - t2_tilde_deriv1(j, k, a, b) * t2(i, k, a, b)
                                      - t2_deriv1(j, k, a, b) * t2_tilde(i, k, a, b)
                                      - t2_tilde_deriv1(i, k, a, b) * t2(j, k, a, b);
    
    EigenMatrix<double> dPab = EigenMatrix<double>::Zero(virt, virt);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index a = 0; a < virt; ++a)
        for (Index b = 0; b < virt; ++b)
            for (Index i = 0; i < occ; ++i)
                for (Index j = 0; j < occ; ++j)
                    for (Index c = 0; c < virt; ++c)
                        dPab(a, b) += t2_deriv1(i, j, a, c) * t2_tilde(i, j, b, c)
                                    + t2_tilde_deriv1(i, j, b, c) * t2(i, j, a, c)
                                    + t2_deriv1(i, j, b, c) * t2_tilde(i, j, a, c)
                                    + t2_tilde_deriv1(i, j, a, c) * t2(i, j, b, c);

    for (Index i = 0; i < occ; ++i)
        for (Index j = 0; j < occ; ++j)
            dPpq(idx, i, j) = dPij(i, j);
    
    for (Index a = 0; a < virt; ++a)
        for (Index b = 0; b < virt; ++b)
            dPpq(idx, a + occ, b + occ) = dPab(a, b);

    dPpqrs_list[idx] = tensor4d<double>(num_orbitals);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index i = 0; i < occ; ++i)
        for (Index j = 0; j < occ; ++j)
            for (Index a = 0; a < virt; ++a)
                for (Index b = 0; b < virt; ++b)
                {
                    dPpqrs_list[idx](i, j, a + occ, b + occ) += t2_tilde_deriv1(i, j, a, b);
                    dPpqrs_list[idx](b + occ, a + occ, j, i) += t2_tilde_deriv1(i, j, a, b);
                }

     #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index p = 0; p < num_orbitals; ++p)
        for (Index q = 0; q < num_orbitals; ++q) 
        {
            Fpq(idx, p, q) = F_grad(p, q);
            dSmo(idx, p, q) = dSdX_mo(p, q);
        }

    dVdXmo_list[idx] = symm4dTensor<double>(num_orbitals);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index p = 0; p < num_orbitals; ++p)
        for (Index q = 0; q <= p; ++q)
        {
            Index pq = index_ij(p, q);
            for (Index r = 0; r < num_orbitals; ++r)
                for (Index s = 0; s <= r; ++s) 
                {
                    Index rs = index_ij(r, s);
                    if (pq <= rs)
                    {
                        Index pqrs = index_ij(pq, rs);
                        dVdXmo_list[idx](p, q, r, s) = dVdXmo(pqrs);
                    }
                }
        }
}

void Mol::scf_gradient::calc_scf_hessian_response_rhf_mp2(const Eigen::Ref<const EigenMatrix<double> >& mo_energies)
{
    const int natoms = static_cast<int>(m_mol->get_atoms().size());
    const auto num_orbitals = m_mol->get_num_orbitals();
    const Index virt = num_orbitals - m_mol->get_num_electrons() / 2;
    const Index occ = num_orbitals - virt;

    std::vector<bool> out_of_plane = std::vector<bool>(3, false);
    constexpr double cutoff = 1.0E-10;
    constexpr double eps = 1.0E-12;

    for (int cart_dir = 0; cart_dir < 3; ++cart_dir) 
    {
        for (int i = 0; i < natoms; ++i)
            if (std::fabs<double>(m_mol->get_geom()(i, cart_dir)) > cutoff) 
                out_of_plane[cart_dir] = true;
    }

    hessianResp = EigenMatrix<double>::Zero(3 * natoms, 3 * natoms);

    const auto eri_mo = [this](Index p, Index q, Index r, Index s) noexcept -> double
    {
        Index prqs = index4(p, r, q, s);
        return e_rep_mo[prqs];
    };

    for (int atom1 = 0; atom1 < natoms; ++atom1)
        for (int atom2 = atom1; atom2 < natoms; ++atom2)
        {
            for (int cart1 = 0; cart1 < 3; ++cart1)
                for (int cart2 = 0; cart2 < 3; ++cart2)
                {
                    if (atom1 == atom2 && cart2 < cart1) continue;
                    else if (cart1 != cart2 && (!out_of_plane[cart1] || !out_of_plane[cart2])) continue;
                    
                    const int idx1 = natoms * cart1 + atom1;
                    const int idx2 = natoms * cart2 + atom2;
                    double tmp = 0;
                    #ifdef _OPENMP
                    #pragma omp parallel for reduction(+:tmp) schedule(dynamic)
                    #endif
                    for(Index p = 0; p < num_orbitals; ++p)
                    {
                        for(Index q = 0; q < num_orbitals; ++q)
                        {
                            tmp += 0.5 * (dPpq(idx1, p, q) * (Fpq(idx2, p, q) +
                                          U(idx2, q, p) * mo_energies(q, q) +
                                          U(idx2, p, q) * mo_energies(p, p)) +
                                          
                                          dPpq(idx2, p, q) * (Fpq(idx1, p, q) +
                                          U(idx1, q, p) * mo_energies(q, q) +
                                          U(idx1, p, q) * mo_energies(p, p)));

                            for(Index r = 0; r < num_orbitals; ++r)
                            {
                                tmp += I(p, q) * 
                                    (U(idx1, p, r) * U(idx2, q, r) +
                                     U(idx1, q, r) * U(idx2, p, r) -
                                     dSmo(idx1, p, r) * dSmo(idx2, q, r) -
                                     dSmo(idx1, q, r) * dSmo(idx2, p, r))
                                    + Ppq(p, q) *
                                    (U(idx1, r, p) * Fpq(idx2, r, q) +
                                     U(idx2, r, p) * Fpq(idx1, r, q) +
                                     U(idx1, r, q) * Fpq(idx2, p, r) +
                                     U(idx2, r, q) * Fpq(idx1, p, r));
                                
                                if (fabs(Ppq(p, q)) > eps)
                                    for(Index i = 0; i < occ; ++i)
                                    {
                                        tmp += Ppq(p, q) *
                                        (U(idx1, r, i) *
                                        (4.0 * dVdXmo_list[idx2](p, q, i, r) - dVdXmo_list[idx2](p, r, i, q)
                                                                             - dVdXmo_list[idx2](p, i, r, q)) 
                                        + U(idx2, r, i) *
                                        (4.0 * dVdXmo_list[idx1](p, q, i, r) - dVdXmo_list[idx1](p, r, i, q)
                                                                             - dVdXmo_list[idx1](p, i, r, q)));
                                    }
                                               
                                if (fabs(dPpq(idx1, p, q)) > eps)
                                    for(Index i = 0; i < occ; ++i)
                                    {
                                        tmp +=
                                        dPpq(idx1, p, q) * U(idx2, r, i) *
                                        (eri_mo(p, i, q, r) - 0.5 * eri_mo(p, i, r, q) + 
                                        eri_mo(p, r, q, i) - 0.5 * eri_mo(p, r, i, q));
                                    }
                                
                                if (fabs(dPpq(idx2, p, q)) > eps)
                                    for(Index i = 0; i < occ; ++i)
                                    {
                                        tmp +=
                                        dPpq(idx2, p, q) * U(idx1, r, i) *
                                        (eri_mo(p, i, q, r) - 0.5 * eri_mo(p, i, r, q) + 
                                        eri_mo(p, r, q, i) - 0.5 * eri_mo(p, r, i, q));
                                    }

                                for (Index s = 0; s < num_orbitals; ++s)
                                {
                                    tmp += Ppq(p, q) * mo_energies(r, s) *
                                    (U(idx1, r, p) * U(idx2, s, q) +
                                     U(idx2, r, p) * U(idx1, s, q)) +
                                    
                                    0.5 * (dPpqrs_list[idx1](p, q, r, s) * dVdXmo_list[idx2](p, r, q, s) 
                                        +  dPpqrs_list[idx2](p, q, r, s) * dVdXmo_list[idx1](p, r, q, s));
                                    
                                    if (fabs(Ppq(p, q)) > eps)
                                        for (Index i = 0; i < occ; ++i)
                                            tmp += Ppq(p, q) * (
                                            (U(idx1, r, p) * U(idx2, s, i) +
                                            U(idx2, r, p) * U(idx1, s, i)) *
                                            2.0 * (2.0 * eri_mo(r, i, q, s) - eri_mo(r, i, s, q)) +
                                            //2.0 * eri_mo(r, s, q, i) - eri_mo(r, s, i, q)) +
                                            
                                            (U(idx1, r, q) * U(idx2, s, i) +
                                            U(idx2, r, q) * U(idx1, s, i)) *
                                            2.0 * (2.0 * eri_mo(p, i, r, s) - eri_mo(p, i, s, r)) +
                                            //2.0 * eri_mo(p, s, r, i) - eri_mo(p, s, i, r)) +
                                            
                                            (U(idx1, r, i) * U(idx2, s, i) +
                                            U(idx2, r, i) * U(idx1, s, i)) *
                                            2.0 * (eri_mo(p, r, q, s) - 0.5 * eri_mo(p, r, s, q)) //+
                                            //eri_mo(p, s, q, r) - 0.5 * eri_mo(p, s, r, q))
                                            );

                                    if (fabs(Ppqrs_(p, q, r, s)) > eps)
                                        for (Index t = 0; t < num_orbitals; ++t)
                                        {
                                            tmp += Ppqrs_(p, q, r, s) *
                                            (U(idx1, t, p) * dVdXmo_list[idx2](t, r, q, s) +
                                             U(idx1, t, q) * dVdXmo_list[idx2](p, r, t, s) +
                                             U(idx1, t, r) * dVdXmo_list[idx2](p, t, q, s) +
                                             U(idx1, t, s) * dVdXmo_list[idx2](p, r, q, t) +
                                             U(idx2, t, p) * dVdXmo_list[idx1](t, r, q, s) +
                                             U(idx2, t, q) * dVdXmo_list[idx1](p, r, t, s) +
                                             U(idx2, t, r) * dVdXmo_list[idx1](p, t, q, s) +
                                             U(idx2, t, s) * dVdXmo_list[idx1](p, r, q, t));

                                            for (Index v = 0; v < num_orbitals; ++v)
                                            {
                                                tmp += Ppqrs_(p, q, r, s) *
                                                ((U(idx1, t, p) * U(idx2, v, q) +
                                                U(idx2, t, p) * U(idx1, v, q)) * eri_mo(t, v, r, s) +
                                                (U(idx1, t, p) * U(idx2, v, r) +
                                                U(idx2, t, p) * U(idx1, v, r)) * eri_mo(t, q, v, s) +
                                                (U(idx1, t, p) * U(idx2, v, s) +
                                                U(idx2, t, p) * U(idx1, v, s)) * eri_mo(t, q, r, v) +
                                                (U(idx1, t, q) * U(idx2, v, r) +
                                                U(idx2, t, q) * U(idx1, v, r)) * eri_mo(p, t, v, s) +
                                                (U(idx1, t, q) * U(idx2, v, s) +
                                                U(idx2, t, q) * U(idx1, v, s)) * eri_mo(p, t, r, v) +
                                                (U(idx1, t, r) * U(idx2, v, s) +
                                                U(idx2, t, r) * U(idx1, v, s)) * eri_mo(p, q, t, v));
                                            }
                                        }
                                    
                                    if (fabs(dPpqrs_list[idx1](p, q, r, s)) > eps)
                                        for (Index t = 0; t < num_orbitals; ++t)
                                            tmp +=
                                            0.5 * dPpqrs_list[idx1](p, q, r, s) *
                                            (U(idx2, t, p) * eri_mo(t, q, r, s) +
                                            U(idx2, t, q) * eri_mo(p, t, r, s) +
                                            U(idx2, t, r) * eri_mo(p, q, t, s) +
                                            U(idx2, t, s) * eri_mo(p, q, r, t));

                                    if (fabs(dPpqrs_list[idx2](p, q, r, s)) > eps)
                                        for (Index t = 0; t < num_orbitals; ++t)
                                            tmp +=
                                            0.5 * dPpqrs_list[idx2](p, q, r, s) *
                                            (U(idx1, t, p) * eri_mo(t, q, r, s) +
                                            U(idx1, t, q) * eri_mo(p, t, r, s) +
                                            U(idx1, t, r) * eri_mo(p, q, t, s) +
                                            U(idx1, t, s) * eri_mo(p, q, r, t));
                                }
                            }
                        }
                    }

                    hessianResp(3 * atom1 + cart1, 3 * atom2 + cart2) = tmp;
                    hessianResp(3 * atom2 + cart2, 3 * atom1 + cart1) = tmp;
                }
        }
}
