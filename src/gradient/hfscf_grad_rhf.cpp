#include "../settings/hfscf_settings.hpp"
#include "../math/hfscf_tensor4d.hpp"
#include "hfscf_gradient.hpp"

using Eigen::Index;
using HF_SETTINGS::hf_settings;

bool Mol::scf_gradient::calc_scf_gradient_rhf(const Eigen::Ref<const EigenMatrix<double> >& d_mat,
                                              const Eigen::Ref<const EigenMatrix<double> >& f_mat,
                                              std::vector<bool>& out_of_plane, bool check_out_of_plane,
                                              bool geom_opt, bool print_gradient)
{
    const Index num_orbitals = m_mol->get_num_orbitals();
    const std::vector<int>& charge = m_mol->get_z_values();
    const auto& atoms = m_mol->get_atoms();
    const int natoms = static_cast<int>(atoms.size());
    constexpr double cutoff = 1.0E-10;

    EigenMatrix<double> geom = m_mol->get_geom();    // Geometry
    EigenMatrix<double> Q_mat =  d_mat * f_mat * d_mat; // weighted Q matrix same as eps * C * C.T sum over elec in Ostlund
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

            EigenMatrix<double> dGdX = EigenMatrix<double>::Zero(num_orbitals, num_orbitals);
            EigenMatrix<double> dGdXCoulomb = EigenMatrix<double>::Zero(num_orbitals, num_orbitals);
            EigenMatrix<double> dGdXExchange = EigenMatrix<double>::Zero(num_orbitals, num_orbitals);


            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(Index i = 0; i < num_orbitals; ++i)
                for(Index j = i; j < num_orbitals; ++j)
                {
                    for(Index k = 0; k < num_orbitals; ++k)
                        for(Index l = 0; l < num_orbitals; ++l)
                        {
                            dGdXCoulomb(i, j) += d_mat(k, l) * (2.0 * dVdX(i, j, k, l));
                            dGdXExchange(i, j) -= d_mat(k, l) * (dVdX(i, k, j, l));
                        }
                    
                    dGdX(i, j) = dGdXCoulomb(i, j) + dGdXExchange(i, j);
                    dGdX(j, i) = dGdX(i, j);
                    dGdXCoulomb(j, i) = dGdXCoulomb(i, j);
                    dGdXExchange(j, i) = dGdXExchange(i, j);
                }
            
            gradCoulomb(atom, cart_dir) = d_mat.cwiseProduct(dGdXCoulomb).sum();
            gradExchange(atom, cart_dir) = d_mat.cwiseProduct(dGdXExchange).sum();

            dGdX += 2.0 * (dTdX + dVndX);

            gradOvlap(atom, cart_dir) = -2.0 * dSdX.cwiseProduct(Q_mat).sum();
            double dEdX = (d_mat.cwiseProduct(dGdX)).sum() + gradOvlap(atom, cart_dir);

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
            gradCoreHamil(atom, cart_dir) = 2.0 * (dTdX + dVndX).cwiseProduct(d_mat).sum();

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
