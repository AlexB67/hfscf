#include "../settings/hfscf_settings.hpp"
#include "../basis/hfscf_trans_basis.hpp"
#include "../integrals/hfscf_dipole.hpp"
#include "../integrals/hfscf_quadrupole.hpp"
#include "../molecule/hfscf_elements.hpp"
#include "../molecule/hfscf_constants.hpp"
#include "../pretty_print/hfscf_pretty_print.hpp"
#include "hfscf_properties.hpp"
#include <iomanip>

using HF_SETTINGS::hf_settings;
using hfscfmath::Cart;
using hfscfmath::index_ijkl;
using HFCOUT::pretty_print_matrix;
using MOLEC_CONSTANTS::au_to_debeye;
using MOLEC_CONSTANTS::bohr_to_angstrom;


void MolProps::Molprops::create_dipole_matrix(bool include_nuc_contrib /* = true */)
{
    mu_x = EigenMatrix<double>::Zero(num_orbitals, num_orbitals);
    mu_y = EigenMatrix<double>::Zero(num_orbitals, num_orbitals);
    mu_z = EigenMatrix<double>::Zero(num_orbitals, num_orbitals);

    bool is_pure = m_mol->use_pure_am();
    std::unique_ptr<DIPOLE::Dipole> dipole_ptr = std::make_unique<DIPOLE::Dipole>(is_pure);

    Vec3D mu_nuc;
    (include_nuc_contrib) ? mu_nuc = m_mol->get_center_of_charge_vector()
                          : mu_nuc = Vec3D(0.0, 0.0, 0.0);

    tensor3d<double> dp = tensor3d<double>(3, num_orbitals, num_orbitals);

    const auto& sp = m_mol->get_shell_pairs();
    const Index num_shells = m_mol->get_num_shells();


    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index i = 0; i < num_shells; ++i)
        for (Index j = i; j < num_shells; ++j)
            dipole_ptr->compute_contracted_shell(dp, mu_nuc, sp[i * num_shells + j]);
    

    for (Index i = 0; i < num_orbitals; ++i) 
        for (Index j = i; j < num_orbitals; ++j) 
        {
            mu_x(i, j) = dp(0, i, j);
            mu_y(i, j) = dp(1, i, j);
            mu_z(i, j) = dp(2, i, j);
            mu_x(j, i) = mu_x(i, j);
            mu_y(j, i) = mu_y(i, j);
            mu_z(j, i) = mu_z(i, j);
        }
}

EigenVector<double> MolProps::Molprops::get_nuclear_quadrupole_contribution() const
{
    const Eigen::Ref<const Vec3D> center = m_mol->get_center_of_mass_vector();
    EigenVector<double> qpole_nuc = EigenVector<double>::Zero(6);
    
    const Eigen::Ref<const EigenMatrix<double>> geom = m_mol->get_geom();

    for (Index i = 0; i < static_cast<Index>(m_mol->get_atoms().size()); ++i) 
    {
        Vec3D dist;
        for (Index j = 0; j < 3; ++j)
            dist[j] = geom(i, j) - center[j]; // should it be COM or center of charge ? // to Check

        qpole_nuc(0) += m_mol->get_z_values()[i] * dist[0] * dist[0];  // xx
        qpole_nuc(1) += m_mol->get_z_values()[i] * dist[0] * dist[1];  // xy
        qpole_nuc(2) += m_mol->get_z_values()[i] * dist[0] * dist[2];  // xz
        qpole_nuc(3) += m_mol->get_z_values()[i] * dist[1] * dist[1];  // yy
        qpole_nuc(4) += m_mol->get_z_values()[i] * dist[1] * dist[2];  // yz
        qpole_nuc(5) += m_mol->get_z_values()[i] * dist[2] * dist[2];  // zz
    }

    return qpole_nuc;
}

void MolProps::Molprops::create_quadrupole_tensors_rhf(const Eigen::Ref<const EigenMatrix<double> >& d_mat, 
                                                       bool print /* = true */)
{
    std::vector<EigenMatrix<double>> x1x2 = std::vector<EigenMatrix<double>>(6);

    for (size_t i = 0; i < 6; ++i)
        x1x2[i] = EigenMatrix<double>(num_orbitals, num_orbitals);

    EigenVector<double> qpole_nuc = get_nuclear_quadrupole_contribution();

    bool is_pure = m_mol->use_pure_am();
    std::unique_ptr<QPOLE::Quadrupole> qpole_ptr = std::make_unique<QPOLE::Quadrupole>(is_pure);

    const auto& sp = m_mol->get_shell_pairs();
    const Index num_shells = m_mol->get_num_shells();
    Vec3D center{0, 0, 0}; // origin 0, 0, 0 
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index i = 0; i < num_shells; ++i)
        for (Index j = i; j < num_shells; ++j)
            qpole_ptr->compute_contracted_shell(x1x2, center, sp[i * num_shells + j]);
    

    EigenMatrix<double> qp_mat = EigenMatrix<double>(num_orbitals, num_orbitals);
    quadp_moments = EigenVector<double>(6);

    EigenVector<std::string> coords = EigenVector<std::string>(6);
    coords[0] = "xx";  coords[1] = "xy";  coords[2] = "xz";
    coords[3] = "yy";  coords[4] = "yz";  coords[5] = "zz";

    for (size_t k = 0; k < 6U; ++k) 
    {
        for (Index i = 0; i < num_orbitals; ++i) 
            for (Index j = i; j < num_orbitals; ++j) 
            {
                qp_mat(i, j) = x1x2[k](i, j);
                qp_mat(j, i) = qp_mat(i, j);
            }
        
        if (print && hf_settings::get_verbosity() > 2)
        {
            std::cout << "\n  *************************************\n";
            std::cout << "  *  Quadrupole matrix                *\n";
            std::cout << "  *                                   *\n";
            std::cout << "  *   "; std::cout << " Q" << coords[k] << "(u, v) / au                 *\n";
            std::cout << "  *************************************\n";
            pretty_print_matrix<double>(qp_mat);
        }
        
        quadp_moments[k] = 2.0 * d_mat.cwiseProduct(qp_mat).sum() + qpole_nuc(k);
    }
}

void MolProps::Molprops::create_quadrupole_tensors_uhf(const Eigen::Ref<const EigenMatrix<double> >& d_mat_a,
                                                       const Eigen::Ref<const EigenMatrix<double> >& d_mat_b,
                                                       bool print)
{   // like RHF apart from the density term
    std::vector<EigenMatrix<double>> x1x2 = std::vector<EigenMatrix<double>>(6);

    for (size_t i = 0; i < 6; ++i)
        x1x2[i] = EigenMatrix<double>(num_orbitals, num_orbitals);

    EigenVector<double> qpole_nuc = get_nuclear_quadrupole_contribution();

    bool is_pure = m_mol->use_pure_am();
    std::unique_ptr<QPOLE::Quadrupole> qpole_ptr = std::make_unique<QPOLE::Quadrupole>(is_pure);

    const auto& sp = m_mol->get_shell_pairs();
    const Index num_shells = m_mol->get_num_shells();
    Vec3D center{0, 0, 0}; // origin 0, 0, 0
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index i = 0; i < num_shells; ++i)
        for (Index j = i; j < num_shells; ++j)
            qpole_ptr->compute_contracted_shell(x1x2, center, sp[i * num_shells + j]);
    

    EigenMatrix<double> qp_mat = EigenMatrix<double>(num_orbitals, num_orbitals);
    quadp_moments = EigenVector<double>(6);

    EigenVector<std::string> coords = EigenVector<std::string>(6);
    coords[0] = "XX";  coords[1] = "XY";  coords[2] = "XZ";
    coords[3] = "YY";  coords[4] = "YZ";  coords[5] = "ZZ";

    for (size_t k = 0; k < 6U; ++k) 
    {
        for (Index i = 0; i < num_orbitals; ++i) 
            for (Index j = i; j < num_orbitals; ++j) 
            {
                qp_mat(i, j) = x1x2[k](i, j);
                qp_mat(j, i) = qp_mat(i, j);
            }
        
        if (print && hf_settings::get_verbosity() > 2)
        {
            std::cout << "\n  *************************************\n";
            std::cout << "  *  Quadrupole matrix                *\n";
            std::cout << "  *                                   *\n";
            std::cout << "  *   "; std::cout << " Q" << coords[k] << "(u, v) / au                 *\n";
            std::cout << "  *************************************\n";
            pretty_print_matrix<double>(qp_mat);
        }
        
        quadp_moments[k] = (d_mat_a + d_mat_b).cwiseProduct(qp_mat).sum() + qpole_nuc(k);
    }
}

void MolProps::Molprops::create_dipole_vectors_rhf(const Eigen::Ref<const EigenMatrix<double> >& d_mat)
{
    create_dipole_matrix();

    mu_cart = Vec3D(2 * d_mat.cwiseProduct(mu_x).sum(), 
                    2 * d_mat.cwiseProduct(mu_y).sum(),
                    2 * d_mat.cwiseProduct(mu_z).sum());
}

void MolProps::Molprops::create_dipole_vectors_uhf(const Eigen::Ref<const EigenMatrix<double> >& d_mat_a,
                                                   const Eigen::Ref<const EigenMatrix<double> >& d_mat_b)
{
    create_dipole_matrix();
    
    mu_cart = Vec3D((d_mat_a + d_mat_b).cwiseProduct(mu_x).sum(),
                    (d_mat_a + d_mat_b).cwiseProduct(mu_y).sum(),
                    (d_mat_a + d_mat_b).cwiseProduct(mu_z).sum());
}

void MolProps::Molprops::population_analysis_rhf(const Eigen::Ref<const EigenMatrix<double> >& s_mat,
                                                 const Eigen::Ref<const EigenMatrix<double> >& d_mat) 
{
    const auto& zval = m_mol->get_z_values();
    const size_t natoms = zval.size();
    mul_charge = EigenVector<double>::Zero(natoms);
    low_charge = EigenVector<double>::Zero(natoms);

    EigenMatrix<double> tmp_mat = 2.0 * d_mat * s_mat;

    Eigen::SelfAdjointEigenSolver<EigenMatrix<double> > eigen_of_s(s_mat);
    EigenMatrix<double> S_sqrt = eigen_of_s.operatorSqrt();
    
    const auto& mask = (m_mol->use_pure_am()) 
                     ? m_mol->get_atom_spherical_mask()
                     : m_mol->get_atom_mask();
                     
    EigenMatrix<double> tmp2_mat = 2.0 * S_sqrt * d_mat * S_sqrt;

    int offset = 0;
    for (size_t i = 0; i < natoms; ++i)
    {
        const int basis_size = mask[i].mask_end - mask[i].mask_start + 1;
        mul_charge(i) = zval[static_cast<int>(i)];
        low_charge(i) = zval[static_cast<int>(i)];

        for (int j = offset; j < basis_size + offset; ++j) 
        {
            mul_charge(static_cast<int>(i)) -= tmp_mat(j, j);
            low_charge(static_cast<int>(i)) -= tmp2_mat(j, j);
        }

        offset += basis_size;
    }
}

void MolProps::Molprops::mayer_indices_rhf(const Eigen::Ref<const EigenMatrix<double> >& s_mat,
                                           const Eigen::Ref<const EigenMatrix<double> >& d_mat)
{
    const auto& mask = (m_mol->use_pure_am()) 
                     ? m_mol->get_atom_spherical_mask()
                     : m_mol->get_atom_mask();

    const EigenMatrix<double> ds_mat = 2.0 * d_mat * s_mat;
    const size_t natoms = m_mol->get_z_values().size();
    Eigen::Index size = static_cast<Eigen::Index>(natoms);
    mayer_indices = EigenMatrix<double>::Zero(size, size);

    int offset1 = 0;
    for (size_t i = 0; i < natoms; ++i)
    {
        const int basis_size1 = mask[i].mask_end - mask[i].mask_start + 1;
        int offset2 = 0;
        for (size_t j = 0; j < i; ++j)
        {   
            const int basis_size2 = mask[j].mask_end - mask[j].mask_start + 1;

            for (int k = offset1; k < basis_size1 + offset1; ++k)
                for (int l = offset2; l < basis_size2 + offset2; ++l)
                    mayer_indices(i, j) += ds_mat(k, l) * ds_mat(l, k);
            
            mayer_indices(j, i) = mayer_indices(i, j);
            offset2 += basis_size2;
        }
        
        offset1 += basis_size1;
    }
}

void MolProps::Molprops::population_analysis_uhf(const Eigen::Ref<const EigenMatrix<double> >& s_mat,
                                                 const Eigen::Ref<const EigenMatrix<double> >& d_mat_a,
                                                 const Eigen::Ref<const EigenMatrix<double> >& d_mat_b)
{
    const auto& zval = m_mol->get_z_values();
    size_t natoms = zval.size();
    mul_charge = EigenVector<double>::Zero(natoms);
    low_charge = EigenVector<double>::Zero(natoms);

    EigenMatrix<double> tmp_mat = (d_mat_a + d_mat_b) * s_mat;

    Eigen::SelfAdjointEigenSolver<EigenMatrix<double> > eigen_of_s(s_mat);
    EigenMatrix<double> S_sqrt = eigen_of_s.operatorSqrt();

    const auto& mask = (m_mol->use_pure_am()) 
                     ? m_mol->get_atom_spherical_mask()
                     : m_mol->get_atom_mask();

    EigenMatrix<double> tmp2_mat = S_sqrt * (d_mat_a + d_mat_b) * S_sqrt;

    int offset = 0;
    for (size_t i = 0; i < natoms; ++i) 
    {
        const int basis_size = mask[i].mask_end - mask[i].mask_start + 1;
        mul_charge(i) = zval[static_cast<int>(i)];
        low_charge(i) = zval[static_cast<int>(i)];

        for (int j = offset; j < basis_size + offset; ++j) 
        {
            mul_charge(static_cast<int>(i)) -= tmp_mat(j, j);
            low_charge(static_cast<int>(i)) -= tmp2_mat(j, j);
        }

        offset += basis_size;
    }
}

void MolProps::Molprops::mayer_indices_uhf(const Eigen::Ref<const EigenMatrix<double> >& s_mat,
                                           const Eigen::Ref<const EigenMatrix<double> >& d_mat_a,
                                           const Eigen::Ref<const EigenMatrix<double> >& d_mat_b)
{
    const auto& mask = (m_mol->use_pure_am()) 
                     ? m_mol->get_atom_spherical_mask()
                     : m_mol->get_atom_mask();

    const EigenMatrix<double> ds_mat_ab_plus  = (d_mat_a + d_mat_b) * s_mat;
    const EigenMatrix<double> ds_mat_ab_minus = (d_mat_a - d_mat_b) * s_mat;
    const size_t natoms = m_mol->get_z_values().size();
    Eigen::Index size = static_cast<Eigen::Index>(natoms);
    mayer_indices = EigenMatrix<double>::Zero(size, size);

    int offset1 = 0;
    for (size_t i = 0; i < natoms; ++i)
    {
        const int basis_size1 = mask[i].mask_end - mask[i].mask_start + 1;
        int offset2 = 0;
        for (size_t j = 0; j < i; ++j)
        {   
            const int basis_size2 = mask[j].mask_end - mask[j].mask_start + 1;

            for (int k = offset1; k < basis_size1 + offset1; ++k)
                for (int l = offset2; l < basis_size2 + offset2; ++l)
                    mayer_indices(i, j) += ds_mat_ab_plus(k, l) * ds_mat_ab_plus(l, k)
                                         + ds_mat_ab_minus(k, l) * ds_mat_ab_minus(l, k);
            mayer_indices(j, i) = mayer_indices(i, j);
            offset2 += basis_size2;
        }
        
        offset1 += basis_size1;
    }
}

void MolProps::Molprops::create_hessian_matrix(const Eigen::Ref<const EigenMatrix<double> >& C,
                                               const Eigen::Ref<const EigenMatrix<double> >& eps,
                                               const Eigen::Ref<const EigenVector<double> >& eri, 
                                               EigenMatrix<double>& G)
{
    std::unique_ptr<BTRANS::basis_transform> ao_to_mo_ptr = std::make_unique<BTRANS::basis_transform>(num_orbitals);
    EigenVector<double> eri_mo = EigenVector<double>::Zero((num_orbitals * (num_orbitals + 1) / 2) *
                                                          ((num_orbitals * (num_orbitals + 1) / 2) + 1) / 2);
    // translate AOs to MOs
    ao_to_mo_ptr->ao_to_mo_transform_2e(C, eri, eri_mo);
    if (!G.size()) G = EigenMatrix<double>::Zero(occ * virt, occ * virt);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index i = 0; i < occ; ++i)
        for (Index j = 0; j < occ; ++j)
            for (Index a = 0; a < virt; ++a)
                for (Index b = 0; b < virt; ++b)
                {
                    if (i + occ * a > j + occ * b) continue;
                    
                    Index ijab = index_ijkl(i, a + occ, j, b + occ);
                    Index ijba = index_ijkl(i, b + occ, j, a + occ);
                    Index iajb = index_ijkl(i, j, a + occ, b + occ);
                    
                    G(i + occ * a, j + occ * b) += (4.0 * eri_mo(ijab) - eri_mo(ijba) - eri_mo(iajb));

                    if (a == b && i == j)
                        G(i + occ * a, j + occ * b) += eps(a + occ, a + occ) - eps(i, i);
                    
                    G(j + occ * b, i + occ * a) = G(i + occ * a, j + occ * b);
                }
}

// from Yamaguchi Schaefer et. al. A new dimension to quantum chemistry
// equation 17.54
// cphf equation 17.174
void MolProps::Molprops::calc_static_polarizabilities_iterative(const Eigen::Ref<const EigenMatrix<double> >& C,
                                                                const Eigen::Ref<const EigenMatrix<double> >& eps,
                                                                const Eigen::Ref<const EigenVector<double> >& eri)
{
    EigenMatrix<double> C_occ = C.block(0, 0, num_orbitals, occ);
    EigenMatrix<double> C_virt = C.block(0, occ, num_orbitals, virt);

    EigenMatrix<double> mo_x = -C_occ.transpose() * mu_x * C_virt;
    EigenMatrix<double> mo_y = -C_occ.transpose() * mu_y * C_virt;
    EigenMatrix<double> mo_z = -C_occ.transpose() * mu_z * C_virt;
    
    EigenMatrix<double> U_1 = EigenMatrix<double>(occ, virt);
    EigenMatrix<double> U_2 = EigenMatrix<double>(occ, virt);
    EigenMatrix<double> U_3 = EigenMatrix<double>(occ, virt);
    EigenMatrix<double> Uold_1 = EigenMatrix<double>::Zero(occ, virt);
    EigenMatrix<double> Uold_2 = EigenMatrix<double>::Zero(occ, virt);
    EigenMatrix<double> Uold_3 = EigenMatrix<double>::Zero(occ, virt);

    EigenMatrix<double> Dx = EigenMatrix<double>(num_orbitals, num_orbitals);
    EigenMatrix<double> Dy = EigenMatrix<double>(num_orbitals, num_orbitals);
    EigenMatrix<double> Dz = EigenMatrix<double>(num_orbitals, num_orbitals);
    EigenMatrix<double> Jx = EigenMatrix<double>(num_orbitals, num_orbitals);
    EigenMatrix<double> Jy = EigenMatrix<double>(num_orbitals, num_orbitals);
    EigenMatrix<double> Jz = EigenMatrix<double>(num_orbitals, num_orbitals);
    EigenMatrix<double> Kx = EigenMatrix<double>(num_orbitals, num_orbitals);
    EigenMatrix<double> Ky = EigenMatrix<double>(num_orbitals, num_orbitals);
    EigenMatrix<double> Kz = EigenMatrix<double>(num_orbitals, num_orbitals);
    EigenMatrix<double> G = EigenMatrix<double>(num_orbitals, num_orbitals);  
    EigenMatrix<double> denom = EigenMatrix<double>(occ, virt);

    for (Index i = 0; i < occ; ++i)
        for (Index a = 0; a < virt; ++a)
        {
            denom(i, a) = - eps(i, i) + eps(occ + a, occ + a);
            U_1(i, a) = mo_x(i, a) / denom(i, a);
            U_2(i, a) = mo_y(i, a) / denom(i, a);
            U_3(i, a) = mo_z(i, a) / denom(i, a);
        }
    
    double max_rms, average_rms;
    const int cphf_max_iter = 50;
    int iter;
    for (iter = 0; iter < cphf_max_iter; ++iter)
    {
        EigenMatrix<double> C_rx = C_virt * U_1.transpose();
        EigenMatrix<double> C_ry = C_virt * U_2.transpose();
        EigenMatrix<double> C_rz = C_virt * U_3.transpose();

        #ifdef _OPENMP
        #pragma omp parallel for collapse(2)
        #endif
        for (Index i = 0; i < num_orbitals; ++i)
        {
            for (Index j = 0; j < num_orbitals; ++j)
            {
                Dx(i, j) = 0.0;  Dy(i, j) = 0.0;  Dz(i, j) = 0.0; 
                for (Index k = 0; k < occ; ++k)
                {
                    Dx(i, j) += C_occ(i, k) * C_rx(j, k);
                    Dy(i, j) += C_occ(i, k) * C_ry(j, k);
                    Dz(i, j) += C_occ(i, k) * C_rz(j, k);
                }
            }
        }

        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (Index i = 0; i < num_orbitals; ++i)
        {
            for (Index j = 0; j < num_orbitals; ++j)
            {
                Jx(i, j) = 0; Jy(i, j) = 0; Jz(i, j) = 0; 
                Kx(i, j) = 0; Ky(i, j) = 0; Kz(i, j) = 0; 
                for (Index k = 0; k < num_orbitals; ++k)
                {
                    for (Index l = 0; l < num_orbitals; ++l)
                    {
                        Index ijkl = index_ijkl(i, j, k, l);
                        Index ikjl = index_ijkl(i, k, j, l);
                        Jx(i, j) += Dx(k, l) * eri(ijkl);
                        Kx(i, j) += Dx(k, l) * eri(ikjl);
                        Jy(i, j) += Dy(k, l) * eri(ijkl);
                        Ky(i, j) += Dy(k, l) * eri(ikjl);
                        Jz(i, j) += Dz(k, l) * eri(ijkl);
                        Kz(i, j) += Dz(k, l) * eri(ikjl);
                    }
                }
            }
        }

        G = 4.0 * Jx - Kx.transpose() - Kx;
        U_1 = (mo_x - C_occ.transpose() * G * C_virt).cwiseProduct(denom.cwiseInverse());
        G = 4.0 * Jy - Ky.transpose() - Ky;
        U_2 = (mo_y - C_occ.transpose() * G * C_virt).cwiseProduct(denom.cwiseInverse());
        G = 4.0 * Jz - Kz.transpose() - Kz;
        U_3 = (mo_z - C_occ.transpose() * G * C_virt).cwiseProduct(denom.cwiseInverse());

        EigenMatrix<double> del_U1 = U_1 - Uold_1;
        EigenMatrix<double> del_U2 = U_2 - Uold_2;
        EigenMatrix<double> del_U3 = U_3 - Uold_3;

        double rms_x = del_U1.cwiseAbs().maxCoeff();
        double rms_y = del_U2.cwiseAbs().maxCoeff(); 
        double rms_z = del_U3.cwiseAbs().maxCoeff();
        max_rms = std::max(std::max(rms_x, rms_y), rms_z);
        max_rms *= max_rms;

        if (max_rms < 1e-10)
        {
            average_rms = (rms_x * rms_x + rms_y * rms_y + rms_z * rms_z) / 3.0;
            break;
        }
        else if (iter == cphf_max_iter - 1)
        {
            std::clog << "\n  Error: Polarizability determinatiom: CPHF iterations failed to converge.\n";
            exit(EXIT_FAILURE);
        }

        cphf_diis(U_1, U_2, U_3, del_U1, del_U2, del_U3);

        Uold_1 = U_1;  Uold_2 = U_2;  Uold_3 = U_3;
    }

    Eigen::Matrix3d ptensor = Eigen::Matrix3d::Zero();
    ptensor(0, 0) = U_1.cwiseProduct(mo_x).sum(); ptensor(0, 0) *= 4.0;
    ptensor(0, 1) = U_1.cwiseProduct(mo_y).sum(); ptensor(0, 1) *= 4.0;
    ptensor(0, 2) = U_1.cwiseProduct(mo_z).sum(); ptensor(0, 2) *= 4.0;
    ptensor(1, 0) = U_2.cwiseProduct(mo_x).sum(); ptensor(1, 0) *= 4.0;
    ptensor(1, 1) = U_2.cwiseProduct(mo_y).sum(); ptensor(1, 1) *= 4.0;
    ptensor(1, 2) = U_2.cwiseProduct(mo_z).sum(); ptensor(1, 2) *= 4.0;
    ptensor(2, 0) = U_3.cwiseProduct(mo_x).sum(); ptensor(2, 0) *= 4.0;
    ptensor(2, 1) = U_3.cwiseProduct(mo_y).sum(); ptensor(2, 1) *= 4.0;
    ptensor(2, 2) = U_3.cwiseProduct(mo_z).sum(); ptensor(2, 2) *= 4.0;

    std::cout << "\n  Polarizabilities: Using iterative CPHF.\n";
    std::cout << "  CPHF converged in " << iter << " iterations (DIIS range: 6).\n";
    std::cout << "  Max RMS: " << std::setprecision(6) << std::scientific << sqrt(max_rms) 
    <<  ", Average RMS: " << sqrt(average_rms) << "\n";

    if (hf_settings::get_verbosity() > 4)
    {
        std::cout << "\n  ************************************\n";
        std::cout << "  *            CPHF Solver           *\n";
        std::cout << "  *   U^f(i, a) matrix coefficients  *\n";
        std::cout << "  ************************************\n";

        std::cout << "\n  component: X\n";
        HFCOUT::pretty_print_matrix<double>(U_1);
        std::cout << "\n  component: Y\n";
        HFCOUT::pretty_print_matrix<double>(U_2);
        std::cout << "\n  component: Z\n";
        HFCOUT::pretty_print_matrix<double>(U_3);
    }

    std::cout << "\n  *************************************\n";
    std::cout << "  *      Polarizability tensor        *\n";
    std::cout << "  *                                   *\n";
    std::cout << "  *         \u0251(x, y) / au              *\n";
    std::cout << "  *************************************\n";
    pretty_print_matrix<double>(ptensor);

    Eigen::SelfAdjointEigenSolver<EigenMatrix<double> > solver(ptensor, Eigen::EigenvaluesOnly);
    Eigen::Vector3d ptensdiag = solver.eigenvalues();

    std::cout << "******************************\n";
    std::cout << "  Principal components\n";
    std::cout << "  #     \u0251(q, q) / au";
    std::cout << "\n******************************\n";

    for(int i = 0; i < 3; ++i)
        std::cout << "  " << i << "  " << std::setw(14) << std::right << std::setprecision(9) << ptensdiag(i) << "\n";

    std::cout << "\n******************************\n";
    std::cout << "  Isotropic polarizability\n";
    std::cout << "        \u0251 / au\n";
    std::cout << "******************************\n";
    std::cout << "        " << std::right << std::setprecision(9) << ptensdiag.mean() << "\n";
}

// from Yamaguchi Schaefer et. al. A new dimension to quantum chemistry
// equation 17.54
// cphf equation 17.174
void MolProps::Molprops::calc_static_polarizabilities(const Eigen::Ref<const EigenMatrix<double> >& C,
                                                      const Eigen::Ref<const EigenMatrix<double> >& eps,
                                                      const Eigen::Ref<const EigenVector<double> >& eri)
{

    EigenMatrix<double> mo_x = -C.block(0, 0, num_orbitals, occ).transpose() 
                             * mu_x * C.block(0, occ, num_orbitals, virt);
    EigenMatrix<double> mo_y = -C.block(0, 0, num_orbitals, occ).transpose() 
                             * mu_y * C.block(0, occ, num_orbitals, virt);
    EigenMatrix<double> mo_z = -C.block(0, 0, num_orbitals, occ).transpose() 
                             * mu_z * C.block(0, occ, num_orbitals, virt);
    
    EigenMatrix<double> G = EigenMatrix<double>::Zero(occ * virt, occ * virt);
    create_hessian_matrix(C, eps, eri, G);

    EigenMatrix<double> Ginv = G.inverse(); G.resize(0, 0);
    EigenMatrix<double> U_x = EigenMatrix<double>::Zero(occ, virt);
    EigenMatrix<double> U_y = EigenMatrix<double>::Zero(occ, virt);
    EigenMatrix<double> U_z = EigenMatrix<double>::Zero(occ, virt);

    for(Index i = 0; i < occ; ++i)
        for(Index a = 0; a < virt; ++a)
            for(Index j = 0; j < occ; ++j)
                for(Index b = 0; b < virt; ++b)
                {                  
                    U_x(i, a) += Ginv(i + occ * a, j + occ * b) * mo_x(j, b);
                    U_y(i, a) += Ginv(i + occ * a, j + occ * b) * mo_y(j, b);
                    U_z(i, a) += Ginv(i + occ * a, j + occ * b) * mo_z(j, b);
                }

    Eigen::Matrix3d ptensor = Eigen::Matrix3d::Zero();
    ptensor(0, 0) = U_x.cwiseProduct(mo_x).sum(); ptensor(0, 0) *= 4.0;
    ptensor(0, 1) = U_x.cwiseProduct(mo_y).sum(); ptensor(0, 1) *= 4.0;
    ptensor(0, 2) = U_x.cwiseProduct(mo_z).sum(); ptensor(0, 2) *= 4.0;
    ptensor(1, 0) = U_y.cwiseProduct(mo_x).sum(); ptensor(1, 0) *= 4.0;
    ptensor(1, 1) = U_y.cwiseProduct(mo_y).sum(); ptensor(1, 1) *= 4.0;
    ptensor(1, 2) = U_y.cwiseProduct(mo_z).sum(); ptensor(1, 2) *= 4.0;
    ptensor(2, 0) = U_z.cwiseProduct(mo_x).sum(); ptensor(2, 0) *= 4.0;
    ptensor(2, 1) = U_z.cwiseProduct(mo_y).sum(); ptensor(2, 1) *= 4.0;
    ptensor(2, 2) = U_z.cwiseProduct(mo_z).sum(); ptensor(2, 2) *= 4.0;

    std::cout << "\n Polarizabilities: Using direct CPHF (via matrix inversion).";
    std::cout << "\n Electronic Hessian and inverse: Memory size " 
              << 2 * sizeof(double) * (occ * virt) * (occ * virt) / 1048576 << "MB.\n";

    if (hf_settings::get_verbosity() > 4)
    {
        std::cout << "\n  ************************************\n";
        std::cout << "  *            CPHF Solver           *\n";
        std::cout << "  *   U^f(i, a) matrix coefficients  *\n";
        std::cout << "  ************************************\n";

        std::cout << "\n  component: X\n";
        HFCOUT::pretty_print_matrix<double>(U_x);
        std::cout << "\n  component: Y\n";
        HFCOUT::pretty_print_matrix<double>(U_y);
        std::cout << "\n  component: Z\n";
        HFCOUT::pretty_print_matrix<double>(U_z);
    }

    std::cout << "\n  *************************************\n";
    std::cout << "  *      Polarizability tensor        *\n";
    std::cout << "  *                                   *\n";
    std::cout << "  *         \u0251(x, y) / au              *\n";
    std::cout << "  *************************************\n";
    pretty_print_matrix<double>(ptensor);

    Eigen::SelfAdjointEigenSolver<EigenMatrix<double> > solver(ptensor, Eigen::EigenvaluesOnly);
    Eigen::Vector3d ptensdiag = solver.eigenvalues();

    std::cout << "******************************\n";
    std::cout << "  Principal components\n";
    std::cout << "  #     \u0251(q, q) / au";
    std::cout << "\n******************************\n";

    for(int i = 0; i < 3; ++i)
        std::cout << "  " << i << "  " << std::setw(14) << std::right << std::setprecision(9) << ptensdiag(i) << "\n";

    std::cout << "\n******************************\n";
    std::cout << "  Isotropic polarizability\n";
    std::cout << "        \u0251 / au\n";
    std::cout << "******************************\n";
    std::cout << "        " << std::right << std::setprecision(9) << ptensdiag.mean() << "\n";
}

void MolProps::Molprops::print_dipoles() const
{
    std::cout << "\n  ___Properties___\n";

    if (hf_settings::get_verbosity() > 2)
    {
        std::cout << "\n  *************************************\n";
        std::cout << "  *  dipole matrix                    *\n";
        std::cout << "  *  (includes nuclear contribution)  *\n";
        std::cout << "  *                                   *\n";
        std::cout << "  *   \u03BC_x(u, v) / au              *\n";
        std::cout << "  *************************************\n";

        pretty_print_matrix<double>(mu_x);

        std::cout << "\n  *************************************\n";
        std::cout << "  *  dipole matrix                    *\n";
        std::cout << "  *  (includes nuclear contribution)  *\n";
        std::cout << "  *                                   *\n";
        std::cout << "  *   \u03BC_y(u, v) / au              *\n";
        std::cout << "  *************************************\n";

        pretty_print_matrix<double>(mu_y);

        std::cout << "\n  *************************************\n";
        std::cout << "  *  dipole matrix                    *\n";
        std::cout << "  *  (includes nuclear contribution)  *\n";
        std::cout << "  *                                   *\n";
        std::cout << "  *   \u03BC_z(u, v) / au              *\n";
        std::cout << "  *************************************\n";

        pretty_print_matrix<double>(mu_z);
    }

    std::cout << "\n  **************************\n";
    std::cout << "  *    Dipole moments      *\n"; 
    std::cout << "  **************************\n";
    std::cout << "   \u03BC_x / au = ";
    std::cout << std::right << std::fixed << std::setprecision(8) << std::setw(13) << mu_cart(0);
    std::cout << "   \u03BC_x / D = ";
    std::cout << std::right << std::fixed << std::setprecision(8) << std::setw(13) << mu_cart(0) * au_to_debeye << '\n';
    
    std::cout << "   \u03BC_y / au = ";
    std::cout << std::right << std::fixed << std::setprecision(8) << std::setw(13) << mu_cart(1);
    std::cout << "   \u03BC_y / D = ";
    std::cout << std::right << std::fixed << std::setprecision(8) << std::setw(13) << mu_cart(1) * au_to_debeye << '\n';

    std::cout << "   \u03BC_z / au = ";
    std::cout << std::right << std::fixed << std::setprecision(8) << std::setw(13) << mu_cart(2);
    std::cout << "   \u03BC_z / D = ";
    std::cout << std::right << std::fixed << std::setprecision(8) << std::setw(13) << mu_cart(2) * au_to_debeye  << '\n';
    std::cout << "     \u03BC / au = ";
    std::cout << std::right << std::fixed << std::setprecision(8) << std::setw(13) << std::sqrt(mu_cart.squaredNorm());
    std::cout << "     \u03BC / D = ";
    std::cout << std::right << std::fixed << std::setprecision(8) << std::setw(13) 
                            << std::sqrt(mu_cart.squaredNorm()) * au_to_debeye << '\n';
}

void MolProps::Molprops::print_quadrupoles() const
{
    std::cout << "\n  *************************************\n";
    std::cout << "  *          Quadrupole moments       *\n";  
    std::cout << "  *          Q(i,j) / (Debeye A)      *\n";
    std::cout << "  *************************************\n";
    std::cout << "  " << "XX " << std::right << std::setw(13) << std::setprecision(8) 
              << quadp_moments[0] * au_to_debeye * bohr_to_angstrom;
    std::cout << std::setw(5) << "YY " << std::right << std::setw(13) << std::setprecision(8) 
              << quadp_moments[3] * au_to_debeye * bohr_to_angstrom;
    std::cout << std::setw(5) << "ZZ " << std::right << std::setw(13) << std::setprecision(8) 
              << quadp_moments[5] * au_to_debeye * bohr_to_angstrom << "\n";
    // off-diagonal
    std::cout << "  " << "XY:" << std::right << std::setw(13) << std::setprecision(8) 
              << quadp_moments[1] * au_to_debeye * bohr_to_angstrom;
    std::cout << std::setw(5) << "XZ " << std::right << std::setw(13) << std::setprecision(8) 
              << quadp_moments[2] * au_to_debeye * bohr_to_angstrom;
    std::cout << std::setw(5) << "YZ " << std::right << std::setw(13) << std::setprecision(8) 
              << quadp_moments[4] * au_to_debeye * bohr_to_angstrom << "\n";
    

    const double trace = (quadp_moments[0] + quadp_moments[3] + quadp_moments[5]) / 3.0;
    std::cout << "\n  **********************************************\n";
    std::cout << "  *  Traceless Quadrupole moment / (Debeye A)  *\n";
    std::cout << "  **********************************************\n";
    std::cout << "  " <<  "XX " << std::right << std::setw(13) << std::setprecision(8) 
              << (quadp_moments[0] - trace) * au_to_debeye * bohr_to_angstrom;
    std::cout << std::setw(5) << "YY " << std::right << std::setw(13) << std::setprecision(8) 
              << (quadp_moments[3] - trace) * au_to_debeye * bohr_to_angstrom;
    std::cout << std::setw(5) <<  "ZZ " << std::right << std::setw(13) << std::setprecision(8) 
              << (quadp_moments[5] - trace) * au_to_debeye * bohr_to_angstrom << "\n";
    // off-diagonal
    std::cout << "  " << "XY " << std::right << std::setw(13) << std::setprecision(8) 
              << quadp_moments[1] * au_to_debeye * bohr_to_angstrom;
    std::cout << std::setw(5) << "XZ " << std::right << std::setw(13) << std::setprecision(8) 
              << quadp_moments[2] * au_to_debeye * bohr_to_angstrom;
    std::cout << std::setw(5) << "YZ " << std::right << std::setw(13) << std::setprecision(8) 
              << quadp_moments[4] * au_to_debeye * bohr_to_angstrom << "\n";
}

void MolProps::Molprops::print_mayer_indices() const
{
    std::cout << "\n******************************\n";
    std::cout << "  Mayer Indices (bond orders)\n";
    std::cout << "  #  B(a, b) \n"; 
    std::cout << "******************************\n";
    HFCOUT::pretty_print_matrix<double>(mayer_indices);
}

void MolProps::Molprops::print_population_analysis() const
{
    std::cout << "\n******************************\n";
    std::cout << "  Mulliken population analysis\n";
    std::cout << "  #  atom  Q(i) / au \n"; 
    std::cout << "******************************\n";

    const auto& zval = m_mol->get_z_values();
    for(Index i = 0; i < mul_charge.size(); ++i)
    {
         std::cout << std::right << std::setw(3) <<  i << "  ";
         std::cout << ELEMENTDATA::atom_names[zval[(size_t) i] - 1];
         std::cout << std::right << std::fixed << std::setprecision(8) 
                   << std::setw(14) << mul_charge(i) << '\n';
    }

    std::cout << "\n******************************\n";
    std::cout << "  Lowdin population analysis\n";
    std::cout << "  #  atom  Q(i) / au \n"; 
    std::cout << "******************************\n";

    for(Index i = 0; i < low_charge.size(); ++i)
    {
         std::cout << std::right << std::setw(3) <<  i << "  ";
         std::cout << ELEMENTDATA::atom_names[zval[(size_t) i] - 1];
         std::cout << std::right << std::fixed << std::setprecision(8) 
                   << std::setw(14) << low_charge(i) << '\n';
    }
}

void MolProps::Molprops::cphf_diis(EigenMatrix<double>& U_1, EigenMatrix<double>& U_2, EigenMatrix<double>& U_3,
                                   const Eigen::Ref<const EigenMatrix<double> >& del_U1,
                                   const Eigen::Ref<const EigenMatrix<double> >& del_U2,
                                   const Eigen::Ref<const EigenMatrix<double> >& del_U3)
{
    size_t diis_range = 6;

    if(U1_list.size() >= diis_range)
    {
        U1_list.erase(U1_list.begin()); del_U1_list.erase(del_U1_list.begin());
        U2_list.erase(U2_list.begin()); del_U2_list.erase(del_U2_list.begin());
        U3_list.erase(U3_list.begin()); del_U3_list.erase(del_U3_list.begin());
    }

    U1_list.emplace_back(U_1); del_U1_list.emplace_back(del_U1); 
    U2_list.emplace_back(U_2); del_U2_list.emplace_back(del_U2); 
    U3_list.emplace_back(U_3); del_U3_list.emplace_back(del_U3);

    if(U1_list.size() < 2U) return; // start DIIS at 2 (note we start at 0)

    const Index size = static_cast<Index>(U1_list.size());
    EigenMatrix<double> L_mat1 = EigenMatrix<double>(size + 1, size + 1);
    EigenMatrix<double> L_mat2 = EigenMatrix<double>(size + 1, size + 1);
    EigenMatrix<double> L_mat3 = EigenMatrix<double>(size + 1, size + 1);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic) 
    #endif
    for (Index i = 0; i < size; ++i)
        for (Eigen::Index j = 0; j < size; ++j) 
        {
            L_mat1(i, j) = 0; L_mat2(i, j) = 0; L_mat3(i, j) = 0; 
            for (Eigen::Index k = 0; k < occ; ++k)
                for (Eigen::Index l = 0; l < virt; ++l)
                {
                    L_mat1(i, j) += del_U1_list[i](k, l) * del_U1_list[j](k, l);
                    L_mat2(i, j) += del_U2_list[i](k, l) * del_U2_list[j](k, l);
                    L_mat3(i, j) += del_U3_list[i](k, l) * del_U3_list[j](k, l);
                }
        }

    for (Index i = 0; i < size + 1; ++i) 
    {
        L_mat1(size, i) = -1.0; L_mat2(size, i) = -1.0; L_mat3(size, i) = -1.0;
        L_mat1(i, size) = -1.0; L_mat2(i, size) = -1.0; L_mat3(i, size) = -1.0;
    }

    L_mat1(size, size) = 0; L_mat2(size, size) = 0; L_mat3(size, size) = 0; 

    EigenVector<double> b = EigenVector<double>::Zero(size + 1);
    b(size) = -1.0;

    EigenVector<double> coffs1 = L_mat1.householderQr().solve(b);  // fairly stable and reasonably fast
    EigenVector<double> coffs2 = L_mat2.householderQr().solve(b);
    EigenVector<double> coffs3 = L_mat3.householderQr().solve(b); 

    for (Index i = 0; i < occ; ++i)
        for (Index a = 0; a < virt; ++a) 
        {
            U_1(i, a) = 0.0;
            U_2(i, a) = 0.0;
            U_3(i, a) = 0.0;
            for (Index k = 0; k < size; ++k)
            {
                U_1(i, a) += coffs1(k) * U1_list[k](i, a);
                U_2(i, a) += coffs2(k) * U2_list[k](i, a);
                U_3(i, a) += coffs3(k) * U3_list[k](i, a);
            }
        }
}
