#include "../settings/hfscf_settings.hpp"
#include "../gradient/hfscf_gradient.hpp"
#include "../basis/hfscf_trans_basis.hpp"
#include "../mol_properties/hfscf_properties.hpp"
#include "../integrals/hfscf_dipole.hpp"
#include "../math/hfscf_tensor4d.hpp"
#include "../math/hfscf_tensors.hpp"
#include "../pretty_print/hfscf_pretty_print.hpp"

using hfscfmath::index_ij;
using hfscfmath::index_ijkl;
using tensor4dmath::index4;
using tensormath::tensor4d1234;
using HF_SETTINGS::hf_settings;
using Eigen::Index;

// Equations from Yamaguchi Schaefer et. al. A new dimension to quantum chemistry 

void Mol::scf_gradient::calc_scf_hessian_response_rhf(const Eigen::Ref<const EigenMatrix<double> >& mo_coff,
                                                      const Eigen::Ref<const EigenMatrix<double> >& mo_energies,                                             
                                                      const Eigen::Ref<const EigenVector<double> >& eri)
{

    const Index num_orbitals = m_mol->get_num_orbitals();
    const int natoms = static_cast<int>(m_mol->get_atoms().size());

    EigenMatrix<double> geom = m_mol->get_geom();    // Geometry

    std::vector<bool> out_of_plane = std::vector<bool>(3, true);

    tensor3d<double> dSdq = tensor3d<double>(3, num_orbitals, num_orbitals);
    calc_overlap_centers(dSdq);

    tensor3d<double> dTdq = tensor3d<double>(3, num_orbitals, num_orbitals);
    calc_kinetic_centers(dTdq);

    tensor3d<double> dVdqa = tensor3d<double>(3 * natoms, num_orbitals, num_orbitals);
    tensor3d<double> dVdqb = tensor3d<double>(3 * natoms, num_orbitals, num_orbitals);
    calc_nuclear_potential_centers(dVdqa, dVdqb, out_of_plane);

    // TODO we can possibly generate the G matrix directly without these intermediate arrays
    tensor4dmath::symm4dTensor<double> dVdXa, dVdXb, dVdXc, dVdYa, dVdYb, dVdYc, dVdZa, dVdZb, dVdZc;
    dVdXa = symm4dTensor<double>(num_orbitals);  // Coulomb Center A
    dVdXb = symm4dTensor<double>(num_orbitals);  // Coulomb Center B
    dVdXc = symm4dTensor<double>(num_orbitals);  // Coulomb Center C
    dVdYa = symm4dTensor<double>(num_orbitals);  // Coulomb Center A
    dVdYb = symm4dTensor<double>(num_orbitals);  // Coulomb Center B
    dVdYc = symm4dTensor<double>(num_orbitals);  // Coulomb Center C
    dVdZa = symm4dTensor<double>(num_orbitals);  // Coulomb Center A
    dVdZb = symm4dTensor<double>(num_orbitals);  // Coulomb Center B
    dVdZc = symm4dTensor<double>(num_orbitals);  // Coulomb Center C

    calc_coulomb_exchange_integral_centers(dVdXa, dVdXb, dVdXc, 
                                           dVdYa, dVdYb, dVdYc, 
                                           dVdZa, dVdZb, dVdZc, out_of_plane);

    std::unique_ptr<BTRANS::basis_transform> ao_to_mo_ptr = std::make_unique<BTRANS::basis_transform>(num_orbitals);
    EigenVector<double> eri_mo;
    // translate AOs to MOs
    eri_mo = EigenVector<double>::Zero((num_orbitals * (num_orbitals + 1) / 2) *
                                      ((num_orbitals * (num_orbitals + 1) / 2) + 1) / 2);
    ao_to_mo_ptr->ao_to_mo_transform_2e(mo_coff, eri, eri_mo);

    const Index virt = num_orbitals - m_mol->get_num_electrons() / 2;
    const Index occ = num_orbitals - virt;

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
                    
                    G(i + occ * a, j + occ * b) += (4.0 * eri_mo(ijab) - eri_mo(ijba) - eri_mo(iajb));

                    if (a == b && i == j)
                        G(i + occ * a, j + occ * b) += mo_energies(a + occ, a + occ) - mo_energies(i, i);
                    
                    G(j + occ * b, i + occ * a) = G(i + occ * a, j + occ * b);
                }
    
    EigenMatrix<double> G_inverse = G.inverse();
    G.resize(0, 0);

    hessianResp = EigenMatrix<double>::Zero(3 * natoms, 3 * natoms);
    Bpq = tensor3d<double>(3 * natoms, virt, occ);
    U = tensor3d<double>(3 * natoms, virt, occ);
    Fpq = tensor3d<double>(3 * natoms, num_orbitals, num_orbitals);
    dSmo = tensor3d<double>(3 * natoms, num_orbitals, num_orbitals);

    for(int cart_dir = 0; cart_dir < 3; ++cart_dir)
    {
        hfscfmath::Cart coord;
        
        if(cart_dir == 0) 
            coord = Cart::X;
        else if (cart_dir == 1) 
            coord = Cart::Y;
        else
            coord = Cart::Z;

        for(int atom = 0; atom < natoms; ++atom)
        {
            EigenVector<bool> mask = get_atom_mask(atom);
            EigenMatrix<double> dSdX = EigenMatrix<double>::Zero(num_orbitals, num_orbitals);
            EigenMatrix<double> dTdX = EigenMatrix<double>::Zero(num_orbitals, num_orbitals);
            calc_overlap_kinetic_gradient(dSdX, dTdX, dSdq, dTdq, coord, mask);

            EigenMatrix<double> dVndX = EigenMatrix<double>::Zero(num_orbitals, num_orbitals);
            calc_potential_gradient(dVdqa, dVdqb, dVndX, mask, atom, cart_dir);

            EigenMatrix<double> F_grad = mo_coff.transpose() * (dTdX + dVndX) * mo_coff;

            // Compute A B C D contribution to electron integrals
            tensor4dmath::symm4dTensor<double> dVdX = tensor4dmath::symm4dTensor<double>(num_orbitals);
            if(coord == Cart::X) 
                calc_coulomb_exchange_integrals(dVdX, dVdXa, dVdXb, dVdXc, mask);
            else if(coord == Cart::Y) 
                calc_coulomb_exchange_integrals(dVdX, dVdYa, dVdYb, dVdYc, mask);
            else 
                calc_coulomb_exchange_integrals(dVdX, dVdZa, dVdZb, dVdZc, mask);

            EigenVector<double> dVdX_mo = EigenVector<double>::Zero((num_orbitals * (num_orbitals + 1) / 2) *
                                                                   ((num_orbitals * (num_orbitals + 1) / 2) + 1) / 2);;
            ao_to_mo_ptr->ao_to_mo_transform_2e(mo_coff, dVdX.get_vector_form(), dVdX_mo);
            // F_gradient
            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for (Index p = 0; p < num_orbitals; ++p)
                for (Index q = 0; q < num_orbitals; ++q)
                    for (Index l = 0; l < occ; ++l) {
                        Index pqll = index4(p, q, l, l);
                        Index pllq = index4(p, l, l, q);
                        F_grad(p, q) += 2.0 * dVdX_mo(pqll) - dVdX_mo(pllq);
                    }

            const Index idx = natoms * cart_dir + atom;
            EigenMatrix<double> dSdX_mo = mo_coff.transpose() * dSdX * mo_coff;
            // F_pq
            for(Index p = 0; p < num_orbitals; ++p)
                for(Index q = 0; q < num_orbitals; ++q)
                {
                    Fpq(idx, p, q) = F_grad(p, q);
                    dSmo(idx, p, q) = dSdX_mo(p, q);
                }

            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(Index a = 0; a < virt; ++a)
                for(Index j = 0; j < occ; ++j)
                {
                    Bpq(idx, a, j) = mo_energies(j, j) * dSdX_mo(j, a + occ) - F_grad(a + occ, j);
                    for(Index k = 0; k < occ; ++k)
                        for(Index l = 0; l < occ; ++l)
                        {
                            Index ajkl = index_ijkl(a + occ, j, k, l);
                            Index akjl = index_ijkl(a + occ, k, j, l);
                            Bpq(idx, a, j) += (2.0 * eri_mo(ajkl) - eri_mo(akjl)) * dSdX_mo(k, l);
                        }
                }

            // Solve GU = B, U = Ginv B, 1st order soln CPHF
            for(Index a = 0; a < virt; ++a)
                for(Index i = 0; i < occ; ++i)
                    for(Index b = 0; b < virt; ++b)
                        for(Index j = 0; j < occ; ++j)                   
                            U(idx, a, i) += G_inverse(i + occ * a, j + occ * b) * Bpq(idx, b, j);
        }
    }

    G_inverse.resize(0, 0);
    double tmp = 0;
    double tmp2 = 0;
    #ifdef _OPENMP
    #pragma omp parallel for private(tmp, tmp2) schedule(dynamic)
    #endif
    for (int atom1 = 0; atom1 < natoms; ++atom1)
        for (int atom2 = atom1; atom2 < natoms; ++atom2)
            for (int cart1 = 0; cart1 < 3; ++cart1)
                for (int cart2 = 0; cart2 < 3; ++cart2) 
                {
                    const int idx1 = natoms * cart1 + atom1;
                    const int idx2 = natoms * cart2 + atom2;
                    tmp = 0; tmp2 = 0;
                    for (Index p = 0; p < occ; ++p)
                        for (Index q = 0; q < occ; ++q)
                        {
                            tmp  += dSmo(idx1, p, q) * Fpq(idx2, p, q)
                                  + dSmo(idx2, p, q) * Fpq(idx1, p, q);
                            tmp2 += mo_energies(p, p) * dSmo(idx2, p, q) * dSmo(idx1, p, q);

                            for (Index r = 0; r < occ; ++r)
                                for (Index s = 0; s < occ; ++s)
                                {
                                    Index pqrs = index4(p, q, r, s);
                                    Index prqs = index4(p, r, q, s);
                                    tmp2 += dSmo(idx1, p, q) * dSmo(idx2, r, s) 
                                          * (eri_mo(pqrs) - 0.5 * eri_mo(prqs));
                                }
                        }
                    
                     for (Index a = 0; a < virt; ++a)
                        for (Index i = 0; i < occ; ++i)
                            tmp2 -= U(idx2, a, i) * Bpq(idx1, a, i);

                    hessianResp(3 * atom1 + cart1, 3 * atom2 + cart2) = -2.0 * tmp + 4.0 * tmp2;
                    hessianResp(3 * atom2 + cart2, 3 * atom1 + cart1) = hessianResp(3 * atom1 + cart1, 3 * atom2 + cart2);
                }
}

// Intensities
void Mol::scf_gradient::dip_derivs(const Eigen::Ref<const EigenMatrix<double> >& mo_coff, Index U_occ_start)
{
    if (!U.getSize())
    {
        std::clog << "\n  Error: NO CPHF coefficients available. Generate the reponse Hessian first.\n\n";
        exit(EXIT_FAILURE);
    }

    const Index num_orbitals = m_mol->get_num_orbitals();
    const Index virt = num_orbitals - m_mol->get_num_electrons() / 2;
    const Index occ = num_orbitals - virt;
    const int natoms = static_cast<int>(m_mol->get_atoms().size());

    std::unique_ptr<MolProps::Molprops> molprops = std::make_unique<MolProps::Molprops>(m_mol);
    
    molprops->create_dipole_matrix(false);
    const Eigen::Ref<const EigenMatrix<double>> mux = molprops->get_dipole_x();
    const Eigen::Ref<const EigenMatrix<double>> muy = molprops->get_dipole_y();
    const Eigen::Ref<const EigenMatrix<double>> muz = molprops->get_dipole_z();

    Index nshells = m_mol->get_num_shells();
    const auto&sp = m_mol->get_shell_pairs();
    std::vector<EigenMatrix<double>> Dpa = std::vector<EigenMatrix<double>>(9);
    std::vector<EigenMatrix<double>> Dpb = std::vector<EigenMatrix<double>>(9);

    for (size_t k = 0; k < 9; ++k)
    {
        Dpa[k] = EigenMatrix<double>(num_orbitals, num_orbitals);
        Dpb[k] = EigenMatrix<double>(num_orbitals, num_orbitals);
    }
   
    std::unique_ptr<DIPOLE::Dipole> dip_ptr = std::make_unique<DIPOLE::Dipole>(m_mol->use_pure_am());
  
    for (Index i = 0; i < nshells; ++i)
        for (Index j = i; j < nshells; ++j)
                dip_ptr->compute_contracted_shell_deriv1(Dpa, Dpb, sp[i * nshells + j]);

    std::vector<EigenMatrix<double>> dip_mo = std::vector<EigenMatrix<double>>(3);
    dip_mo[0] = -mo_coff.transpose() * mux * mo_coff;
    dip_mo[1] = -mo_coff.transpose() * muy * mo_coff;
    dip_mo[2] = -mo_coff.transpose() * muz * mo_coff;

    EigenMatrix<double> dmn = EigenMatrix<double>(num_orbitals, num_orbitals);
    dipderiv = EigenMatrix<double>::Zero(3 * natoms, 3);
    dipgrad = EigenMatrix<double>::Zero(3 * natoms, 3);

    for (Index cart = 0; cart < 3; ++cart)
    {
        for (Index atom = 0; atom < natoms; ++atom) 
        {
            const auto& mask = get_atom_mask(atom);
            
            for (Index cart2 = 0; cart2 < 3; ++cart2) 
            {   
                Index idx = natoms * cart2 + atom;

                // Eqn 17.144 page 327
                // Note: h^f ij = - d^f ij f is the electric field component 
                double dmudx = 0;
                for (Index a = 0; a < virt; ++a)
                    for (Index i = 0; i < occ; ++i) dmudx -= 4.0 * U(idx, U_occ_start + a, i) * dip_mo[cart](occ + a, i);

                for (Index i = 0; i < occ; ++i)
                    for (Index j = 0; j < occ; ++j) dmudx += 2.0 * dip_mo[cart](i, j) * dSmo(idx, i, j);

                dmn.setZero();

                for (Index i = 0; i < num_orbitals; ++i)
                    for (Index j = i; j < num_orbitals; ++j)
                    {
                        if (mask[i]) dmn(i, j) += Dpa[3 * cart + cart2](i, j);
                        if (mask[j]) dmn(i, j) += Dpb[3 * cart + cart2](i, j);
                        dmn(j, i) = dmn(i, j);
                    }

                EigenMatrix<double> dij = mo_coff.block(0, 0, num_orbitals, occ).transpose() * dmn * mo_coff;
              
                for (Index i = 0; i < occ; ++i) 
                    dipderiv(3 * atom + cart2, cart) += 2.0 * dij(i, i);
                
                dipgrad(3 * atom + cart2, cart) = dipderiv(3 * atom + cart2, cart);
                dipderiv(3 * atom + cart2, cart) += dmudx;
                
                if (cart == cart2) // Only if mu_q and field parallel. 
                    dipderiv(3 * atom + cart2, cart) += m_mol->get_z_values()[atom];

            }
        }
    }
}

void Mol::scf_gradient::print_dip_derivs()
{
    std::cout << "\n  ************************************\n";
    std::cout << "  *          Dipole gradients        *\n";
    std::cout << "  *     dmu/dX / (e a0 / bohr)       *\n";
    std::cout << "  ************************************\n";

    HFCOUT::pretty_print_matrix<double>(dipgrad);

    std::cout << "\n  ************************************\n";
    std::cout << "  *         Dipole derivatives       *\n";
    std::cout << "  *     dmu/dX / (e a0 / bohr)       *\n";
    std::cout << "  ************************************\n";

     HFCOUT::pretty_print_matrix<double>(dipderiv);
}

void Mol::scf_gradient::print_cphf_coefficients(const Index dim1, const Index dim2)
{
    std::cout << "\n  ************************************\n";
    std::cout << "  *            CPHF Solver           *\n";
    std::cout << "  *     U(a, i) matrix cefficients   *\n";
    std::cout << "  ************************************\n";
    std::cout << "\n  Method: Direct inverson\n";

    const int natoms = static_cast<int>(m_mol->get_atoms().size());

    for (int atom = 0; atom < natoms; ++atom)
        for (int cart = 0; cart < 3; ++cart)
        {
            std::cout << "\n  ******************************";
            std::cout << "\n  *   atom: " << std::setw(3) << std::left << atom;
            std::cout << " direction: ";

            switch (cart)
            {
                case 0:
                    std::cout << " X  *";
                    break;
                case 1:
                    std::cout << " Y  *";
                    break;
                case 2:
                    std::cout << " Z  *"; 
                    break;
            }

            std::cout << "\n  ******************************\n\n";

            Index offset = 0;
            Index print_cols = 6;
            Index j = 0;
            Index cols = dim1;
            Index rows = dim2;
            const int idx = natoms * cart + atom;

            while (j < cols)
            {
                if (offset + print_cols > cols - 1) print_cols = cols - offset;

                for (Index col_index = offset; col_index < offset + print_cols; ++col_index)
                    std::cout << std::right << std::setw(15) << col_index + 1 << "  ";

                std::cout << '\n';

                for (Index i = 0; i < rows; ++i)
                {
                    std::cout << std::right << std::setw(3) << i + 1;

                    for (j = offset; j < offset + print_cols; ++j)
                        std::cout << std::right << std::fixed << std::setprecision(9) 
                                  << std::setw(17) << U(idx, i, j);

                    std::cout << '\n';
                }
                offset += print_cols;
                std::cout << '\n';
            }

        }
}
