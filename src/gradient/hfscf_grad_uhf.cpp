#include "hfscf_gradient.hpp"
#include "../settings/hfscf_settings.hpp"

// UHF SCF gradients
using HF_SETTINGS::hf_settings;

bool Mol::scf_gradient::calc_scf_gradient_uhf(const Eigen::Ref<const EigenMatrix<double> >& d_mat_a,
                                              const Eigen::Ref<const EigenMatrix<double> >& d_mat_b,
                                              const Eigen::Ref<const EigenMatrix<double> >& f_mat_a,
                                              const Eigen::Ref<const EigenMatrix<double> >& f_mat_b,
                                              std::vector<bool>& out_of_plane, bool check_out_of_plane,
                                              bool geom_opt, bool print_gradient)
{
    const Index num_orbitals = m_mol->get_num_orbitals();
    const std::vector<int>& charge = m_mol->get_z_values();
    const auto& atoms = m_mol->get_atoms();
    const int natoms = static_cast<int>(atoms.size());
    constexpr double cutoff = 1.0E-10;

    EigenMatrix<double> geom = m_mol->get_geom();    // Geometry
    EigenMatrix<double> Q_mat_a =  d_mat_a * f_mat_a * d_mat_a; // weighted Q matrix
    EigenMatrix<double> Q_mat_b =  d_mat_b * f_mat_b * d_mat_b; // weighted Q matrix
    gradient = EigenMatrix<double>::Zero(natoms, 3);

    EigenMatrix<double> gradOvlap = EigenMatrix<double>::Zero(natoms, 3);
    EigenMatrix<double> gradCoreHamil = EigenMatrix<double>::Zero(natoms, 3);
    EigenMatrix<double> gradNuc = EigenMatrix<double>::Zero(natoms, 3);
    EigenMatrix<double> gradCoulomb = EigenMatrix<double>::Zero(natoms, 3);
    EigenMatrix<double> gradExchange = EigenMatrix<double>::Zero(natoms, 3);

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

        if (!out_of_plane[cart_dir]) 
            continue;

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

            EigenMatrix<double> dGdXa = EigenMatrix<double>::Zero(num_orbitals, num_orbitals);
            EigenMatrix<double> dGdXb = EigenMatrix<double>::Zero(num_orbitals, num_orbitals);
            EigenMatrix<double> dGdXCoulomb = EigenMatrix<double>::Zero(num_orbitals, num_orbitals);
            EigenMatrix<double> dGdXExchange_a = EigenMatrix<double>::Zero(num_orbitals, num_orbitals);
            EigenMatrix<double> dGdXExchange_b = EigenMatrix<double>::Zero(num_orbitals, num_orbitals);
            
            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for (int i = 0; i < num_orbitals; ++i)
                for (int j = i; j < num_orbitals; ++j) 
                {
                    for (int k = 0; k < num_orbitals; ++k)
                        for (int l = 0; l < num_orbitals; ++l) 
                        {   
                            dGdXCoulomb(i, j) += (d_mat_a(k, l) + d_mat_b(k, l)) * dVdX(i, j, k, l);
                            dGdXExchange_a(i, j) += -d_mat_a(k, l) * dVdX(i, k, j, l);
                            dGdXExchange_b(i, j) += -d_mat_b(k, l) * dVdX(i, k, j, l);

                            dGdXa(i, j) += d_mat_a(k, l) * (dVdX(i, j, k, l) - dVdX(i, k, j, l)) +
                                           d_mat_b(k, l) * dVdX(i, j, k, l);
                            dGdXb(i, j) += d_mat_b(k, l) * (dVdX(i, j, k, l) - dVdX(i, k, j, l)) +
                                           d_mat_a(k, l) * dVdX(i, j, k, l);
                        }

                    dGdXCoulomb(j, i) = dGdXCoulomb(i, j);
                    dGdXExchange_a(j, i) = dGdXExchange_a(i, j);
                    dGdXExchange_b(j, i) = dGdXExchange_b(i, j); 
                    dGdXa(j, i) = dGdXa(i, j);
                    dGdXb(j, i) = dGdXb(i, j);

                }

            gradCoulomb(atom, cart_dir) = 0.5 * (d_mat_a + d_mat_b).cwiseProduct(dGdXCoulomb).sum();
            gradExchange(atom, cart_dir) = 0.5 * (d_mat_a.cwiseProduct(dGdXExchange_a) + 
                                                  d_mat_b.cwiseProduct(dGdXExchange_b)).sum();

            dGdXa += 2.0 * (dTdX + dVndX);
            dGdXb += 2.0 * (dTdX + dVndX);

            gradOvlap(atom, cart_dir) = -dSdX.cwiseProduct(Q_mat_a + Q_mat_b).sum();
            double dEdX = (0.5 * (d_mat_a.cwiseProduct(dGdXa) + d_mat_b.cwiseProduct(dGdXb)).sum()
                               + gradOvlap(atom, cart_dir));
            
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
            gradCoreHamil(atom, cart_dir) = (dTdX + dVndX).cwiseProduct(d_mat_a + d_mat_b).sum();

            dEdX += dVNNdX;
            gradient(atom, cart_dir) = dEdX;
        }
    }

    if (print_gradient)
        print_gradient_info(gradNuc, gradCoreHamil, gradOvlap, gradCoulomb, gradExchange, gradient);

    if(geom_opt)
        return check_geom_opt(charge, natoms);

    return false;
}
