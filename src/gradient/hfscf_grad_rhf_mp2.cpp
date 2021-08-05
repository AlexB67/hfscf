#include "../settings/hfscf_settings.hpp"
#include "../molecule/hfscf_elements.hpp"
#include "../basis/hfscf_trans_basis.hpp"
#include "hfscf_gradient.hpp"
#include <iomanip>

// RHF MP2 SCF gradients

using tensor4dmath::tensor4d1122;
using tensormath::tensor4d1234;
using tensor4dmath::index4;
using Eigen::Index;
using HF_SETTINGS::hf_settings;

bool Mol::scf_gradient::calc_mp2_gradient_rhf(const Eigen::Ref<const EigenMatrix<double> >& mo_energies,
                                              const Eigen::Ref<const EigenMatrix<double> >& s_mat,
                                              const Eigen::Ref<const EigenMatrix<double> >& d_mat,
                                              const Eigen::Ref<const EigenMatrix<double> >& mo_coff,
                                              const Eigen::Ref<const EigenVector<double> >& e_rep_mat,
                                              std::vector<bool>& out_of_plane, bool check_out_of_plane,
                                              bool geom_opt, bool print_gradient)
{
    const auto num_orbitals = m_mol->get_num_orbitals();
    std::unique_ptr<BTRANS::basis_transform> ao_to_mo_ptr = std::make_unique<BTRANS::basis_transform>(num_orbitals);

    // translate AOs to MOs
    e_rep_mo = EigenVector<double>::Zero((num_orbitals * (num_orbitals + 1) / 2) *
                                        ((num_orbitals * (num_orbitals + 1) / 2) + 1) / 2);
    ao_to_mo_ptr->ao_to_mo_transform_2e(mo_coff, e_rep_mat, e_rep_mo);

    const Index virt = num_orbitals - m_mol->get_num_electrons() / 2;
    const Index occ = num_orbitals - virt;
    t2 = tensor4d1122<double>(occ, virt);
    t2_tilde = tensor4d1122<double>(occ, virt);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index i = 0; i < occ; ++i)
        for (Index j = 0; j < occ; ++j)
            for (Index a = 0; a < virt; ++a)
                for (Index b = 0; b < virt; ++b)
                {
                    Index iajb = index4(i, a + occ, j, b + occ);
                    t2(i, j, a, b) = e_rep_mo(iajb)
                    / (mo_energies(i, i) + mo_energies(j, j) 
                    - mo_energies(a + occ, a + occ) - mo_energies(b + occ, b + occ)); 
                }

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index i = 0; i < occ; ++i)
        for (Index j = 0; j < occ; ++j)
            for (Index a = 0; a < virt; ++a)
                for (Index b = 0; b < virt; ++b)
                    t2_tilde(i, j, a, b) = 2.0 * t2(i, j, a, b) - t2(i, j, b, a);

    const EigenMatrix<double> ref_opdm = mo_coff.transpose() * s_mat.transpose() * 2.0 * d_mat * s_mat * mo_coff;
    // EigenMatrix<double> ref_opdm = EigenMatrix<double>::Zero(num_orbitals, num_orbitals);
    // for(Index i = 0; i < occ; ++i) ref_opdm(i, i) = 2.0;
    
    EigenMatrix<double> Pij =  EigenMatrix<double>::Zero(occ, occ);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index i = 0; i < occ; ++i)
        for (Index j = 0; j < occ; ++j)
            for (Index a = 0; a < virt; ++a)
                for (Index b = 0; b < virt; ++b)
                    for (Index k = 0; k < occ; ++k)
                        Pij(i, j) -= (t2(i, k, a, b) * t2_tilde(j, k, a, b) 
                                   +  t2(j, k, a, b) * t2_tilde(i, k, a ,b)); 

    EigenMatrix<double> Pab =  EigenMatrix<double>::Zero(virt, virt);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index a = 0; a < virt; ++a)
        for (Index b = 0; b < virt; ++b)
            for (Index i = 0; i < occ; ++i)
                for (Index j = 0; j < occ; ++j)
                    for (Index c = 0; c < virt; ++c)
                        Pab(a, b) += t2(i, j, a, c) * t2_tilde(i, j, b, c) 
                                   + t2(i, j, b, c) * t2_tilde(i, j, a, c); 


    Ppq = ref_opdm;

    for (Index i = 0; i < occ; ++i)
        for (Index j = 0; j < occ; ++j)
            Ppq(i, j) += Pij(i, j);
    
    for (Index a = 0; a < virt; ++a)
        for (Index b = 0; b < virt; ++b)
            Ppq(a + occ, b + occ) += Pab(a, b);

    tensor4d<double> Ppqrs = tensor4d<double>(num_orbitals);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index p = 0; p < num_orbitals; ++p)
        for (Index q = 0; q < num_orbitals; ++q)
            for (Index r = 0; r < num_orbitals; ++r)
                for (Index s = 0; s < num_orbitals; ++s)
                    Ppqrs(p, q, r, s) += (- 0.50 * ref_opdm(p, r) * ref_opdm(q, s) 
                                          + 0.25 * ref_opdm(p, s) * ref_opdm(q, r));
    
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index p = 0; p < occ; ++p)
        for (Index q = 0; q < occ; ++q)
            for (Index r = 0; r < virt; ++r)
                for (Index s = 0; s < virt; ++s)
                {
                    Ppqrs(p, q, r + occ, s + occ) += t2_tilde(p, q, r, s);
                    Ppqrs(s + occ, r + occ, q, p) += t2_tilde(p, q, r, s);
                }
    EigenMatrix<double> Ip = ((Ppq + Ppq.transpose()) * mo_energies).transpose();
    
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index p = 0; p < num_orbitals; ++p)
        for (Index q = 0; q < num_orbitals; ++q)
            for (Index r = 0; r < num_orbitals; ++r)
                for (Index s = 0; s < num_orbitals; ++s)
                    for (Index t = 0; t < num_orbitals; ++t)
                    {   // Note physicist indexing from chemist ints
                        Index prst = index4(p, s, r, t);
                        Index rpst = index4(r, s, p, t);
                        Index rspt = index4(r, p, s, t);
                        //Index rstp = index4(r, t, s, p); symmetric
                        Ip(p, q) += (Ppqrs(q, r, s, t) + + Ppqrs(r, s, t, q)) * e_rep_mo(prst) 
                                   + Ppqrs(r, q, s, t) * e_rep_mo(rpst)
                                   + Ppqrs(r, s, q, t) * e_rep_mo(rspt);
                    }

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index p = 0; p < num_orbitals; ++p)
        for (Index q = 0; q < occ; ++q)
            for (Index r = 0; r < num_orbitals; ++r)
                for (Index s = 0; s < num_orbitals; ++s)
                {
                    Index rpsq = index4(r, s, p, q);
                    Index rpqs = index4(r, q, p, s);
                    Index rqps = index4(r, p, q, s);
                    Ip(p, q) += Ppq(r, s) * (4.0 * e_rep_mo(rpsq) - e_rep_mo(rpqs) -  e_rep_mo(rqps));
                }

    Ip *= -0.5;

    EigenMatrix<double> Ipp = Ip;
    for(Index a = occ; a < num_orbitals; ++a)
        for(Index i = 0; i < occ; ++i)
            Ipp(a, i) = Ip(i, a);
    
    EigenVector<double> X = EigenVector<double>::Zero(virt * occ);

    for(Index a = 0; a < virt; ++a)
        for(Index i = 0; i < occ; ++i)
            X(i + occ * a) =  Ip(i, a + occ) - Ip(a + occ, i);

    EigenMatrix<double> G = EigenMatrix<double>::Zero(occ * virt, occ * virt);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index i = 0; i < occ; ++i)
        for (Index j = 0; j < occ; ++j)
            for (Index a = 0; a < virt; ++a)
                for (Index b = 0; b < virt; ++b)
                {
                    if (i + occ * a > j + occ * b) continue;

                    Index ijab = index4(i, a + occ, j, b + occ);
                    Index ijba = index4(i, b + occ, j, a + occ);
                    Index iajb = index4(i, j, a + occ, b + occ);
                    
                    G(i + occ * a, j + occ * b) += (4.0 * e_rep_mo(ijab) - e_rep_mo(ijba) - e_rep_mo(iajb));

                    if (a == b && i == j)
                        G(i + occ * a, j + occ * b) += mo_energies(a + occ, a + occ) - mo_energies(i, i);
                    
                    G(j + occ * b, i + occ * a) = G(i + occ * a, j + occ * b);
                }

    const EigenVector<double> Zvec = G.householderQr().solve(X);

    EigenMatrix<double> Z = EigenMatrix<double>::Zero(virt, occ);
    for(Index a = 0; a < virt; ++a)
        for(Index i = 0; i < occ; ++i)
            Z(a, i) = Zvec(i + occ * a);
    
    for(Index a = 0; a < virt; ++a)
        for(Index i = 0; i < occ; ++i)
        {
            Ppq(i, a + occ) = -Z(a, i);
            Ppq(a + occ, i) = -Z(a, i);
        }

    I = Ipp;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index i = 0; i < occ; ++i)
        for (Index j = 0; j < occ; ++j)
            for (Index a = occ; a < num_orbitals; ++a)
                for (Index k = 0; k < occ; ++k)
                {
                    Index aikj = index4(a, k, i, j);
                    Index aijk = index4(a, j, i, k);
                    Index ajki = index4(a, k, j, i);
                    Index ajik = index4(a, i, j, k);
                    I(i, j) += Z(a - occ, k) * (2.0 * e_rep_mo(aikj) - e_rep_mo(aijk) +
                                                2.0 * e_rep_mo(ajki) - e_rep_mo(ajik));
                }

    for (Index i = 0; i < occ; ++i)
        for (Index a = occ; a < num_orbitals; ++a)
        {
            I(i, a) += Z(a - occ, i) * mo_energies(i, i);
            I(a, i) = I(i, a);
        }
    
    if (hf_settings::get_frequencies_type() == "MP2" && !hf_settings::get_geom_opt().length()) 
        Ppqrs_ = Ppqrs; // Ppqrs in the form used by a MP2 hessian

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index p = 0; p < num_orbitals; ++p)
        for (Index r = 0; r < num_orbitals; ++r)
            for (Index s = 0; s < occ; ++s) Ppqrs(p, s, r, s) += 2.0 * Ppq(p, r);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index p = 0; p < num_orbitals; ++p)
        for (Index q = 0; q < occ; ++q)
            for (Index s = 0; s < num_orbitals; ++s) Ppqrs(p, q, q, s) -= Ppq(p, s);

    const std::vector<int>& charge = m_mol->get_z_values();
    const auto& atoms = m_mol->get_atoms();
    const int natoms = static_cast<int>(atoms.size());
    constexpr double cutoff = 1.0E-10;

    EigenMatrix<double> geom = m_mol->get_geom();    // Geometry
    gradient = EigenMatrix<double>::Zero(natoms, 3);

    EigenMatrix<double> gradOvlap = EigenMatrix<double>::Zero(natoms, 3);
    EigenMatrix<double> gradCoreHamil = EigenMatrix<double>::Zero(natoms, 3);
    EigenMatrix<double> gradNuc = EigenMatrix<double>::Zero(natoms, 3);
    EigenMatrix<double> gradTEI = EigenMatrix<double>::Zero(natoms, 3);

    if(check_out_of_plane)
    {
        out_of_plane = std::vector<bool>(3, false);
        for(int cart_dir = 0; cart_dir < 3; ++cart_dir)
        {
            out_of_plane[cart_dir] = false;
            for(int i = 0; i < natoms; ++i)
                if(std::fabs<double>(geom(i , cart_dir)) > cutoff)
                    out_of_plane[cart_dir] = true;
        }
    }

    // TODO we can possibly generate the G matrix directly without these intermediate arrays
    tensor4dmath::symm4dTensor<double> dVdXa, dVdXb, dVdXc, dVdYa, dVdYb, dVdYc, dVdZa, dVdZb, dVdZc;
    if(out_of_plane[0]) 
    {
        dVdXa = symm4dTensor<double>(num_orbitals); // Coulomb Center A
        dVdXb = symm4dTensor<double>(num_orbitals); // Coulomb Center B
        dVdXc = symm4dTensor<double>(num_orbitals); // Coulomb Center C
    }
    
    if(out_of_plane[1]) 
    {
        dVdYa = symm4dTensor<double>(num_orbitals); // Coulomb Center A
        dVdYb = symm4dTensor<double>(num_orbitals); // Coulomb Center B
        dVdYc = symm4dTensor<double>(num_orbitals); // Coulomb Center C
    }
    
    if(out_of_plane[2]) 
    {
        dVdZa = symm4dTensor<double>(num_orbitals); // Coulomb Center A
        dVdZb = symm4dTensor<double>(num_orbitals); // Coulomb Center B
        dVdZc = symm4dTensor<double>(num_orbitals); // Coulomb Center C
    }

    tensor3d<double> dSdq = tensor3d<double>(3, num_orbitals, num_orbitals);
    calc_overlap_centers(dSdq);

    tensor3d<double> dTdq = tensor3d<double>(3, num_orbitals, num_orbitals);
    calc_kinetic_centers(dTdq);

    tensor3d<double> dVdqa = tensor3d<double>(3 * natoms, num_orbitals, num_orbitals);
    tensor3d<double> dVdqb = tensor3d<double>(3 * natoms, num_orbitals, num_orbitals);
    calc_nuclear_potential_centers(dVdqa, dVdqb, out_of_plane);
    
    calc_coulomb_exchange_integral_centers(dVdXa, dVdXb, dVdXc, 
                                           dVdYa, dVdYb, dVdYc, 
                                           dVdZa, dVdZb, dVdZc, out_of_plane);
    EigenMatrix<double> Ginv;

    if (hf_settings::get_frequencies_type() == "MP2" && !hf_settings::get_geom_opt().length())
    {
        Bpq = tensor3d<double>(3 * natoms, virt, occ);
        U = tensor3d<double>(3 * natoms, num_orbitals, num_orbitals);
        dSmo = tensor3d<double>(3 * natoms, num_orbitals, num_orbitals);
        Fpq = tensor3d<double>(3 * natoms, num_orbitals, num_orbitals);
        dPpq = tensor3d<double>(3 * natoms, num_orbitals, num_orbitals);
        Ginv = G.inverse();
        G.resize(0, 0); // free some memory, don't need it hereafter
        dPpqrs_list = std::vector<tensor4d<double>>(3 * natoms);
        dVdXmo_list = std::vector<symm4dTensor<double>>(3 * natoms);
    }

    for(int cart_dir = 0; cart_dir < 3; ++cart_dir)
    {
        hfscfmath::Cart coord;
        
        if(cart_dir == 0) 
            coord = Cart::X;
        else if (cart_dir == 1) 
            coord = Cart::Y;
        else
            coord = Cart::Z;

        // Check if the molecule is planar or linear so we skip that coordinate
        // Only if check_out_of_plane is true
   
        if (!out_of_plane[cart_dir]) continue;

        for(int atom = 0; atom < natoms; ++atom)
        {
            EigenVector<bool> mask = get_atom_mask(atom);
            EigenMatrix<double> dSdX = EigenMatrix<double>::Zero(num_orbitals, num_orbitals);
            EigenMatrix<double> dTdX = EigenMatrix<double>::Zero(num_orbitals, num_orbitals);
            calc_overlap_kinetic_gradient(dSdX, dTdX, dSdq, dTdq, coord, mask);
            
            EigenMatrix<double> dVndX = EigenMatrix<double>::Zero(num_orbitals, num_orbitals);
            calc_potential_gradient(dVdqa, dVdqb, dVndX, mask, atom, cart_dir);

            // Compute A B C D contribution to electron integrals
            tensor4dmath::symm4dTensor<double> dVdX = tensor4dmath::symm4dTensor<double>(num_orbitals);
            if(coord == Cart::X) 
                calc_coulomb_exchange_integrals(dVdX, dVdXa, dVdXb, dVdXc, mask);
            else if(coord == Cart::Y) 
                calc_coulomb_exchange_integrals(dVdX, dVdYa, dVdYb, dVdYc, mask);
            else 
                calc_coulomb_exchange_integrals(dVdX, dVdZa, dVdZb, dVdZc, mask);

            double dVNNdX = 0;
            for(int c = 0; c < natoms; ++c)
            {
                const double x_ab = m_mol->get_atoms()[atom].get_r()[cart_dir] 
                                  - m_mol->get_atoms()[c].get_r()[cart_dir];

                if (std::fabs(x_ab) > cutoff)
                {
                    const double q_a = static_cast<double>(charge[atom]);
                    const double q_b = static_cast<double>(charge[c]);
                    const double r_ab = (m_mol->get_atoms()[atom].get_r() - m_mol->get_atoms()[c].get_r()).norm();
                    dVNNdX -= x_ab * q_a * q_b / (r_ab * r_ab * r_ab);
                }
            }

            gradNuc(atom, cart_dir) = dVNNdX;
            // overlap
            gradOvlap(atom, cart_dir) = (mo_coff.transpose() * dSdX * mo_coff).cwiseProduct(I).sum();
            // Core hamiltonian
            gradCoreHamil(atom, cart_dir) = (mo_coff.transpose() * (dTdX + dVndX) * mo_coff).cwiseProduct(Ppq).sum();

            EigenVector<double> dVdXmo = EigenVector<double>::Zero((num_orbitals * (num_orbitals + 1) / 2) *
                                                                  ((num_orbitals * (num_orbitals + 1) / 2) + 1) / 2);
            ao_to_mo_ptr->ao_to_mo_transform_2e(mo_coff, dVdX.get_vector_form(), dVdXmo);

            double tei_sum = 0.0;
            #ifdef _OPENMP
            #pragma omp parallel for reduction (+:tei_sum) schedule(dynamic)
            #endif
            for (Index p = 0; p < num_orbitals; ++p)
                for (Index q = 0; q < num_orbitals; ++q)
                    for (Index r = 0; r < num_orbitals; ++r)
                        for (Index s = 0; s < num_orbitals; ++s)
                        {
                            Index pqrs = index4(p, r, q, s);
                            tei_sum += dVdXmo(pqrs) * Ppqrs(p, q, r, s);
                        }
            gradTEI(atom, cart_dir) = tei_sum;
            gradient(atom, cart_dir) = gradOvlap(atom, cart_dir) + gradCoreHamil(atom, cart_dir) + dVNNdX 
                                     + gradTEI(atom, cart_dir);
            
            if (hf_settings::get_frequencies_type() == "MP2" && !hf_settings::get_geom_opt().length())
                hessian_resp_terms(mo_coff, dTdX, dVndX, dSdX, mo_energies, Ginv, dVdXmo, atom, cart_dir);
        }
    }

    if (print_gradient)
       print_gradient_info_mp2(gradNuc, gradCoreHamil, gradOvlap, gradTEI, gradient);

    if(geom_opt)
        return check_geom_opt(charge, natoms);

    return false;
}
