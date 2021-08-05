#include "../molecule/hfscf_molecule.hpp"
#include "../math/hfscf_math.hpp"
#include "hfscf_osnuclear.hpp"
#include "../math/gamma.hpp"
#include "hfscf_osrecur.hpp"

using hfscfmath::rab2;
using hfscfmath::gpc;
using hfscfmath::pi;
using extramath::F_nu;
using os_recursion::osrecurpot3c;


using Eigen::Index;
void OSNUCLEAR::OSNuclear::compute_contracted_shell_deriv1(tensor3d<double>& dVa, tensor3d<double>& dVb, 
                                                           const ShellPair& sp,
                                                           const std::vector<MOLEC::Atom>& atoms,
                                                           const std::vector<bool>& coords) const
{
    // Translational invariance sets up the relation <u | dVc | v> = <du | Vc | v> + <u | Vc | v>
    const auto& sh1 = sp.m_s1;
    const auto& sh2 = sp.m_s2;
    Index nmax = 3 * (sh1.L() + sh2.L());
    Index l1_ = sh1.L(); Index m1_ = sh1.L(); Index n1_ = sh1.L();
    Index l2_ = sh2.L(); Index m2_ = sh2.L(); Index n2_ = sh2.L();

    if (coords[0]) { ++nmax; ++l1_; ++l2_;}
    if (coords[1]) { ++nmax; ++m1_; ++m2_;}  
    if (coords[2]) { ++nmax; ++n1_; ++n2_;}

    Index L1 = sh1.L() + 1; Index L2 = sh2.L() + 1;
    // dimensions enough to hold max case of all dX dY dZ;
    const Index dim1 = L1 + 1; const Index dim2 = L2 + 1;
    const Index dim11122 =  dim1 * dim1 * dim1 * dim2 * dim2;
    const Index dim1112 =  dim1 * dim1 * dim1 * dim2;
    const Index dim111 =  dim1 * dim1 * dim1;
    const Index dim11 =  dim1 * dim1;

    const auto idx = [&](Index n2, Index m2, Index l2, Index n1, Index m1, Index l1) -> Index
    {
        return  dim11122 * n2 +
                dim1112 * m2 + 
                dim111 * l2 +  
                dim11 * n1 + 
                dim1 * m1 +
                l1;
    };

    EigenMatrix<double> Vs = EigenMatrix<double>(dim2 * dim2 * dim2 * dim1 * dim1 * dim1, nmax + 1);

    const Eigen::Ref<const EigenVector<double>>& alpha1 = sh1.alpha();
    const Eigen::Ref<const EigenVector<double>>& alpha2 = sh2.alpha();
    const Eigen::Ref<const EigenVector<double>>& c1 = sh1.c();
    const Eigen::Ref<const EigenVector<double>>& c2 = sh2.c();
    const auto&id1 = sh1.get_indices();
    const auto&id2 = sh2.get_indices();
    int natoms = static_cast<int>(atoms.size());

    EigenMatrix<double> x_block = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    std::vector<EigenMatrix<double>> va_block = std::vector<EigenMatrix<double>>(3 * natoms, x_block);
    std::vector<EigenMatrix<double>> vb_block = std::vector<EigenMatrix<double>>(3 * natoms, x_block);

    int atom = 0;
    for (const auto& m : atoms)
    {
        const double charge = m.get_Z();
        
        for (Index p1 = 0; p1 < c1.size(); ++p1)
        {
            for (Index p2 = 0; p2 < c2.size(); ++p2)
            {
                const Index id = p1 * c2.size() + p2;
                const Vec3D& P = sp.P(id);
                const double gamma12 = sp.gamma_ab(id);
                const Vec3D& PA = sp.PA(id);
                const Vec3D& PB = sp.PB(id);
                const double r_pc2  = rab2(P, m.get_r());
                const double T = gamma12 * r_pc2;

                osrecurpot3c<double>(PA, PB, P - m.get_r(), gamma12, r_pc2, T,  F_nu(static_cast<double>(nmax), T), 
                                     l1_, m1_, n1_, l2_, m2_, n2_, nmax, Vs, idx);
                
                const double pfac = charge * sp.pfac2(id);

                for (const auto &i : id1)
                {
                    Index l1 = i.l; Index m1 = i.m;  Index n1 = i.n;  Index ci1 = i.ci; 
                    for (const auto &j : id2)
                    {
                        Index ci2 = j.ci; Index l2 = j.l; Index m2 = j.m; Index n2 = j.n;

                        if (coords[0])
                        {
                            // Xa
                            double dVx = 2 * alpha1(p1) * Vs(idx(n2, m2, l2, n1, m1, l1 + 1), 0);
                            if (l1) dVx -= l1 * Vs(idx(n2, m2, l2, n1, m1, l1 - 1), 0);
                            //dVa(atom, sh1.get_idx() + ci1, sh2.get_idx() + ci2) += dVx * pfac;
                            va_block[atom](ci1, ci2) += dVx * pfac;

                            // Xb
                            dVx = 2 * alpha2(p2) * Vs(idx(n2, m2, l2 + 1, n1, m1, l1), 0);
                            if (l2) dVx -= l2 * Vs(idx(n2, m2, l2 - 1, n1, m1, l1), 0);
                            // dVb(atom, sh1.get_idx() + ci1, sh2.get_idx() + ci2) += dVx * pfac;
                            vb_block[atom](ci1, ci2) += dVx * pfac;
                        }

                        if (coords[1]) 
                        {
                            // Ya
                            double dVy = 2 * alpha1(p1) * Vs(idx(n2, m2, l2, n1, m1 + 1, l1), 0);
                            if (m1) dVy -= m1 * Vs(idx(n2, m2, l2, n1, m1 - 1, l1), 0);
                            //dVa(natoms + atom, sh1.get_idx() + ci1, sh2.get_idx() + ci2) += dVy * pfac;
                            va_block[natoms + atom](ci1, ci2) += dVy * pfac;

                            // Yb
                            dVy = 2 * alpha2(p2) * Vs(idx(n2, m2 + 1, l2, n1, m1, l1), 0);
                            if (m2) dVy -= m2 * Vs(idx(n2, m2 - 1, l2, n1, m1, l1), 0);
                            //dVb(natoms + atom, sh1.get_idx() + ci1, sh2.get_idx() + ci2) += dVy * pfac;
                            vb_block[natoms + atom](ci1, ci2) += dVy * pfac;
                        }

                        if (coords[2]) 
                        {
                            // Za
                            double dVz = 2 * alpha1(p1) * Vs(idx(n2, m2, l2, n1 + 1, m1, l1), 0);
                            if (n1) dVz -= n1 * Vs(idx(n2, m2, l2, n1 - 1, m1, l1), 0);
                            //dVa(natoms * 2 + atom, sh1.get_idx() + ci1, sh2.get_idx() + ci2) += dVz * pfac;
                            va_block[2 * natoms + atom](ci1, ci2) += dVz * pfac;

                            // Zb
                            dVz = 2 * alpha2(p2) * Vs(idx(n2 + 1, m2, l2, n1, m1, l1), 0);
                            if (n2) dVz -= n2 * Vs(idx(n2 - 1, m2, l2, n1, m1, l1), 0);
                            //dVb(natoms * 2 + atom, sh1.get_idx() + ci1, sh2.get_idx() + ci2) += dVz * pfac;
                            vb_block[2 * natoms + atom](ci1, ci2) += dVz * pfac;
                        }
                    }
                }
            }
        }
        ++atom;
    }

    const int n = natoms;
    const int m = 2 * natoms;

    if (pure) // Spherical basis
    {
        const EigenMatrix<double> tr_left = sh1.get_spherical_form().transpose().eval();
        const EigenMatrix<double> tr_right = sh2.get_spherical_form().eval();

        if (coords[0])
            for (int k = 0; k < natoms; ++k)
            {
                EigenMatrix<double> vxa = tr_left * va_block[k] * tr_right;
                EigenMatrix<double> vxb = tr_left * vb_block[k] * tr_right;

                for (Index i = 0; i < sh1.get_sirange(); ++i)
                    for (Index j = 0; j < sh2.get_sirange(); ++j)
                    {   // X
                        dVa(k, sh1.get_ids() + i, sh2.get_ids() + j) = vxa(i, j);
                        dVb(k, sh1.get_ids() + i, sh2.get_ids() + j) = vxb(i, j);
                        dVa(k, sh2.get_ids() + j, sh1.get_ids() + i) = 
                        dVa(k, sh1.get_ids() + i, sh2.get_ids() + j);
                        dVb(k, sh2.get_ids() + j, sh1.get_ids() + i) = 
                        dVb(k, sh1.get_ids() + i, sh2.get_ids() + j);
                    }
            }

        if (coords[1])
            for (int k = 0; k < natoms; ++k)
            {
                EigenMatrix<double> vya = (tr_left * va_block[k + n] * tr_right);
                EigenMatrix<double> vyb = (tr_left * vb_block[k + n] * tr_right);

                for (int i = 0; i < sh1.get_sirange(); ++i)
                    for (Index j = 0; j < sh2.get_sirange(); ++j)
                    {   // Y
                        dVa(n + k, sh1.get_ids() + i, sh2.get_ids() + j) = vya(i, j);
                        dVb(n + k, sh1.get_ids() + i, sh2.get_ids() + j) = vyb(i, j);
                        dVa(n + k, sh2.get_ids() + j, sh1.get_ids() + i) = 
                        dVa(n + k, sh1.get_ids() + i, sh2.get_ids() + j);
                        dVb(n + k, sh2.get_ids() + j, sh1.get_ids() + i) = 
                        dVb(n + k, sh1.get_ids() + i, sh2.get_ids() + j);
                    }
            }

        if (coords[2])
            for (int k = 0; k < natoms; ++k)
            {
                EigenMatrix<double> vza = (tr_left * va_block[k + m] * tr_right);
                EigenMatrix<double> vzb = (tr_left * vb_block[k + m] * tr_right);

                for (Index i = 0; i < sh1.get_sirange(); ++i)
                    for (Index j = 0; j < sh2.get_sirange(); ++j)
                    {   // Z
                        dVa(m + k, sh1.get_ids() + i, sh2.get_ids() + j) = vza(i, j);
                        dVb(m + k, sh1.get_ids() + i, sh2.get_ids() + j) = vzb(i, j);
                        dVa(m + k, sh2.get_ids() + j, sh1.get_ids() + i) = 
                        dVa(m + k, sh1.get_ids() + i, sh2.get_ids() + j);
                        dVb(m + k, sh2.get_ids() + j, sh1.get_ids() + i) = 
                        dVb(m + k, sh1.get_ids() + i, sh2.get_ids() + j);
                    }
        }
    }
    else // Cartesian basis
    {
        if (coords[0])
            for (int k = 0; k < natoms; ++k)
                for (int i = 0; i < sh1.get_cirange(); ++i)
                    for (Index j = 0; j < sh2.get_cirange(); ++j)
                    {   // X
                        dVa(k, sh1.get_idx() + i, sh2.get_idx() + j) = va_block[k](i, j);
                        dVb(k, sh1.get_idx() + i, sh2.get_idx() + j) = vb_block[k](i, j);
                        dVa(k, sh2.get_idx() + j, sh1.get_idx() + i) = 
                        dVa(k, sh1.get_idx() + i, sh2.get_idx() + j);
                        dVb(k, sh2.get_idx() + j, sh1.get_idx() + i) = 
                        dVb(k, sh1.get_idx() + i, sh2.get_idx() + j);
                    }
        

        if (coords[1])
            for (int k = 0; k < natoms; ++k)
                for (Index i = 0; i < sh1.get_cirange(); ++i)
                    for (Index j = 0; j < sh2.get_cirange(); ++j)
                    {   // Y
                        dVa(n + k, sh1.get_idx() + i, sh2.get_idx() + j) = va_block[n + k](i, j);
                        dVb(n + k, sh1.get_idx() + i, sh2.get_idx() + j) = vb_block[n + k](i, j);
                        dVa(n + k, sh2.get_idx() + j, sh1.get_idx() + i) = 
                        dVa(n + k, sh1.get_idx() + i, sh2.get_idx() + j);
                        dVb(n + k, sh2.get_idx() + j, sh1.get_idx() + i) = 
                        dVb(n + k, sh1.get_idx() + i, sh2.get_idx() + j);
                    }
        
        if (coords[2])
            for (int k = 0; k < natoms; ++k)
                for (Index i = 0; i < sh1.get_cirange(); ++i)
                    for (Index j = 0; j < sh2.get_cirange(); ++j)
                    {   // Z
                        dVa(m + k, sh1.get_idx() + i, sh2.get_idx() + j) = va_block[m + k](i, j);
                        dVb(m + k, sh1.get_idx() + i, sh2.get_idx() + j) = vb_block[m + k](i, j);
                        dVa(m + k, sh2.get_idx() + j, sh1.get_idx() + i) = 
                        dVa(m + k, sh1.get_idx() + i, sh2.get_idx() + j);
                        dVb(m + k, sh2.get_idx() + j, sh1.get_idx() + i) = 
                        dVb(m + k, sh1.get_idx() + i, sh2.get_idx() + j);
                    }
    }
}
