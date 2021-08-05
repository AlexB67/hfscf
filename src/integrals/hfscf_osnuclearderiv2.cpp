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

// TODO eliminate d center and simplify mask a b c relationships
// too convoluted, but it works.
void OSNUCLEAR::OSNuclear::compute_contracted_shell_deriv2(tensor4d1234<double>& dVqq,
                                                           const ShellPair& sp,
                                                           const std::vector<MOLEC::Atom>& atoms,
                                                           const Cart2 coord,
                                                           const std::vector<std::vector<bool>>& mask) const
{
    // Translational invariance sets up the relation <u | dVc | v> = <du | Vc | v> + <u | Vc | v>
    const auto& sh1 = sp.m_s1;
    const auto& sh2 = sp.m_s2;

    Index L1 = sh1.L() + 2; Index L2 = sh2.L() + 2;
    
    const Index dim1 = L1 + 1; const Index dim2 = L2 + 1;
    const Index dim11122 =  dim1 * dim1 * dim1 * dim2 * dim2;
    const Index dim1112 =  dim1 * dim1 * dim1 * dim2;
    const Index dim111 =  dim1 * dim1 * dim1;
    const Index dim11 =  dim1 * dim1;

    const auto idx = [dim1, dim11, dim111, dim1112, dim11122](Index n2, Index m2, Index l2, 
                                                              Index n1, Index m1, Index l1) -> Index
    {
        return  dim11122 * n2 +
                dim1112 * m2 + 
                dim111 * l2 +  
                dim11 * n1 + 
                dim1 * m1 +
                l1;
    };

    const Eigen::Ref<const EigenVector<double>>& alpha1 = sh1.alpha();
    const Eigen::Ref<const EigenVector<double>>& alpha2 = sh2.alpha();
    const Eigen::Ref<const EigenVector<double>>& c1 = sh1.c();
    const Eigen::Ref<const EigenVector<double>>& c2 = sh2.c();
    const auto& id1 = sh1.get_indices();
    const auto& id2 = sh2.get_indices();

    Index l_1 = sh1.L(); Index m_1 = sh1.L(); Index n_1 = sh1.L();
    Index l_2 = sh2.L(); Index m_2 = sh2.L(); Index n_2 = sh2.L();

    if (coord == Cart2::XX) { l_1 += 2; l_2 += 2; }
    else if (coord == Cart2::YY) { m_1 += 2; m_2 += 2; }
    else if (coord == Cart2::ZZ) { n_1 += 2; n_2 += 2; }
    else if (coord == Cart2::XY || coord == Cart2::YX) { ++l_1; ++m_1; ++l_2; ++m_2;}
    else if (coord == Cart2::XZ || coord == Cart2::ZX) { ++l_1; ++n_1; ++l_2; ++n_2;}
    else if (coord == Cart2::YZ || coord == Cart2::ZY) { ++m_1; ++n_1; ++m_2; ++n_2;}


    const Index nmax = l_1 + m_1 + n_1 + l_2 + m_2 + n_2;
    EigenMatrix<double> Vs = EigenMatrix<double>(dim2 * dim2 * dim2 * dim1 * dim1 * dim1, nmax + 1);
    
    const int natoms = static_cast<int>(atoms.size());
    tensor5d<double> centers = tensor5d<double>(natoms, natoms, 8, sh1.get_cirange(), sh2.get_cirange());
    centers.setZero();

    int atom = 0;
    for (const auto& m : atoms)
    {
        for (Index p1 = 0; p1 < c1.size(); ++p1)
        {
            for (Index p2 = 0; p2 < c2.size(); ++p2)
            {
                Index id = p1 * c2.size() + p2;
                const Vec3D& P = sp.P(id);
                const double gamma12 = sp.gamma_ab(id);
                const Vec3D& PA = sp.PA(id);
                const Vec3D& PB = sp.PB(id);
                const double r_pc2  = rab2(P, m.get_r());
                const double T = gamma12 * r_pc2;

                osrecurpot3c<double>(PA, PB, P - m.get_r(), gamma12, r_pc2, T,  F_nu(static_cast<double>(nmax), T), 
                                     l_1, m_1, n_1, l_2, m_2, n_2, nmax, Vs, idx);
                
                const double prefac = m.get_Z() * sp.pfac2(id);

                for (const auto &i : id1)
                {
                    Index l1 = i.l; Index m1 = i.m;  Index n1 = i.n;  Index ci1 = i.ci; 
                    for (const auto &j : id2)
                    {
                        Index ci2 = j.ci;

                        if (!pure && sh1.get_idx() + ci1 > sh2.get_idx() + ci2) continue;

                        Index l2 = j.l; Index m2 = j.m;  Index n2 = j.n;
                        const double V = prefac;

                        if (coord == Cart2::XX)
                        {
                            // AA
                            double aa_ = 4.0 * alpha1(p1) * alpha1(p1) * Vs(idx(n2, m2, l2, n1, m1, l1 + 2), 0) -
                                        2 * alpha1(p1) * (2.0 * l1 + 1.0) * Vs(idx(n2, m2, l2, n1, m1, l1), 0);
                            if (l1 > 1) aa_ += l1 * (l1 - 1.0) * Vs(idx(n2, m2, l2, n1, m1, l1 - 2), 0);
                            // BB
                            double bb_ = 4.0 * alpha2(p2) * alpha2(p2) * Vs(idx(n2, m2, l2 + 2, n1, m1, l1), 0) -
                                        2 * alpha2(p2) * (2.0 * l2 + 1.0) * Vs(idx(n2, m2, l2, n1, m1, l1), 0);
                            if (l2 > 1) bb_ += l2 * (l2 - 1.0) * Vs(idx(n2, m2, l2 - 2, n1, m1, l1), 0);
                            // AB
                            double ab_ = 4.0 * alpha1(p1) * alpha2(p2) * Vs(idx(n2, m2, l2 + 1, n1, m1, l1 + 1), 0);
                            if (l2) ab_ += -2.0 * alpha1(p1) * l2 * Vs(idx(n2, m2, l2 - 1, n1, m1, l1 + 1), 0);
                            if (l1) ab_ += -2.0 * alpha2(p2) * l1 * Vs(idx(n2, m2, l2 + 1, n1, m1, l1 - 1), 0);
                            if (l1 && l2) ab_ += l1 * l2 * Vs(idx(n2, m2, l2 - 1, n1, m1, l1 - 1), 0);
                            
                            for (int atom1 = 0; atom1 < natoms; ++atom1)
                            {
                                const bool a = mask[atom1][sh1.get_idx()];
                                for(int atom2 = atom1; atom2 < natoms; ++atom2)
                                {
                                    const bool b = mask[atom2][sh2.get_idx()];
                                    sumXX(atom1, atom2, atom, V * aa_, V * bb_, V * ab_, 
                                          centers, ci1, ci2, a, b);
                                }
                            }
                        }
                        else if (coord == Cart2::YY)
                        {
                            // AA
                            double aa_ = 4.0 * alpha1(p1) * alpha1(p1) * Vs(idx(n2, m2, l2, n1, m1 + 2, l1), 0) -
                                2 * alpha1(p1) * (2.0 * m1 + 1.0) * Vs(idx(n2, m2, l2, n1, m1, l1), 0);
                            if (m1 > 1) aa_ += m1 * (m1 - 1.0) * Vs(idx(n2, m2, l2, n1, m1 - 2, l1), 0);
                            // BB
                            double bb_ = 4.0 * alpha2(p2) * alpha2(p2) * Vs(idx(n2, m2 + 2, l2, n1, m1, l1), 0) -
                                2 * alpha2(p2) * (2.0 * m2 + 1.0) * Vs(idx(n2, m2, l2, n1, m1, l1), 0);
                            if (m2 > 1) bb_ += m2 * (m2 - 1.0) * Vs(idx(n2, m2 - 2, l2, n1, m1, l1), 0);
                            // AB
                            double ab_ = 4.0 * alpha1(p1) * alpha2(p2) * Vs(idx(n2, m2 + 1, l2, n1, m1 + 1, l1), 0);
                            if (m2) ab_ += -2.0 * alpha1(p1) * m2 * Vs(idx(n2, m2 - 1, l2, n1, m1 + 1, l1), 0);
                            if (m1) ab_ += -2.0 * alpha2(p2) * m1 * Vs(idx(n2, m2 + 1, l2, n1, m1 - 1, l1), 0);
                            if (m1 && m2) ab_ += m1 * m2 * Vs(idx(n2, m2 - 1, l2, n1, m1 - 1, l1), 0);

                            for (int atom1 = 0; atom1 < natoms; ++atom1)
                            {
                                const bool a = mask[atom1][sh1.get_idx()];
                                for(int atom2 = atom1; atom2 < natoms; ++atom2)
                                {
                                    const bool b = mask[atom2][sh2.get_idx()];
                                    sumXX(atom1, atom2, atom, V * aa_, V * bb_, V * ab_, 
                                          centers, ci1, ci2, a, b);
                                }
                            }
                        }
                        else if (coord == Cart2::ZZ)
                        {
                            // AA
                            double aa_ = 4.0 * alpha1(p1) * alpha1(p1) * Vs(idx(n2, m2, l2, n1 + 2, m1, l1), 0) -
                                2 * alpha1(p1) * (2.0 * n1 + 1.0) * Vs(idx(n2, m2, l2, n1, m1, l1), 0);
                            if (n1 > 1) aa_ += n1 * (n1 - 1.0) * Vs(idx(n2, m2, l2, n1 - 2, m1, l1), 0);
                            // BB
                            double bb_ = 4.0 * alpha2(p2) * alpha2(p2) * Vs(idx(n2 + 2, m2, l2, n1, m1, l1), 0) -
                                2 * alpha2(p2) * (2.0 * n2 + 1.0) * Vs(idx(n2, m2, l2, n1, m1, l1), 0);
                            if (n2 > 1) bb_ += n2 * (n2 - 1.0) * Vs(idx(n2 - 2, m2, l2, n1, m1, l1), 0);
                            // AB
                            double ab_ = 4.0 * alpha1(p1) * alpha2(p2) * Vs(idx(n2 + 1, m2, l2, n1 + 1, m1, l1), 0);
                            if (n2) ab_ += -2.0 * alpha1(p1) * n2 * Vs(idx(n2 - 1, m2, l2, n1 + 1, m1, l1), 0);
                            if (n1) ab_ += -2.0 * alpha2(p2) * n1 * Vs(idx(n2 + 1, m2, l2, n1 - 1, m1, l1), 0);
                            if (n1 && n2) ab_ += n1 * n2 * Vs(idx(n2 - 1, m2, l2, n1 - 1, m1, l1), 0);

                            for (int atom1 = 0; atom1 < natoms; ++atom1) // generate upper triangular block
                            {                                            // all atoms
                                const bool a = mask[atom1][sh1.get_idx()];
                                for(int atom2 = atom1; atom2 < natoms; ++atom2)
                                {
                                    const bool b = mask[atom2][sh2.get_idx()];
                                    sumXX(atom1, atom2, atom, V * aa_, V * bb_, V * ab_, 
                                          centers, ci1, ci2, a, b);
                                }
                            }
                        }
                        else if (coord == Cart2::XY || coord == Cart2::YX)
                        {
                            // AA
                            double aa_ = 4.0 * alpha1(p1) * alpha1(p1) * Vs(idx(n2, m2, l2, n1, m1 + 1, l1 + 1), 0);
                            if (m1) aa_ -=  2.0 * alpha1(p1) * m1 * Vs(idx(n2, m2, l2, n1, m1 - 1, l1 + 1), 0);
                            if (l1) aa_ -=  2.0 * alpha1(p1) * l1 * Vs(idx(n2, m2, l2, n1, m1 + 1, l1 - 1), 0);
                            if (l1 && m1) aa_ += m1 * l1 * Vs(idx(n2, m2, l2, n1, m1 - 1, l1 - 1), 0);
                            // BB
                            double bb_ = 4.0 * alpha2(p2) * alpha2(p2) * Vs(idx(n2, m2 + 1, l2 + 1, n1, m1, l1), 0);
                            if (m2) bb_ -=  2.0 * alpha2(p2) * m2 * Vs(idx(n2, m2 - 1, l2 + 1, n1, m1, l1), 0);
                            if (l2) bb_ -=  2.0 * alpha2(p2) * l2 * Vs(idx(n2, m2 + 1, l2 - 1, n1, m1, l1), 0);
                            if (l2 && m2) bb_ += m2 * l2 * Vs(idx(n2, m2 - 1, l2 - 1, n1, m1, l1), 0);
                            // AB
                            double ab_ = 4.0 * alpha1(p1) * alpha2(p2) * Vs(idx(n2, m2 + 1, l2, n1, m1, l1 + 1), 0);
                            if (m2) ab_ += -2.0 * alpha1(p1) * m2 * Vs(idx(n2, m2 - 1, l2, n1, m1, l1 + 1), 0);
                            if (l1) ab_ += -2.0 * alpha2(p2) * l1 * Vs(idx(n2, m2 + 1, l2, n1, m1, l1 - 1), 0);
                            if (l1 && m2) ab_ += l1 * m2 * Vs(idx(n2, m2 - 1, l2, n1, m1, l1 - 1), 0);
                            // BA
                            double ba_ = 4.0 * alpha1(p1) * alpha2(p2) * Vs(idx(n2, m2, l2 + 1, n1, m1 + 1, l1), 0);
                            if (l2) ba_ += -2.0 * alpha1(p1) * l2 * Vs(idx(n2, m2, l2 - 1, n1, m1 + 1, l1), 0);
                            if (m1) ba_ += -2.0 * alpha2(p2) * m1 * Vs(idx(n2, m2, l2 + 1, n1, m1 - 1, l1), 0);
                            if (l2 && m1) ba_ += l2 * m1 * Vs(idx(n2, m2, l2 - 1, n1, m1 - 1, l1), 0);

                            if (coord == Cart2::XY)
                                for (int atom1 = 0; atom1 < natoms; ++atom1)
                                {
                                    const bool a = mask[atom1][sh1.get_idx()];
                                    for(int atom2 = atom1; atom2 < natoms; ++atom2)
                                    {
                                        const bool b = mask[atom2][sh2.get_idx()];
                                        sumXY(atom1, atom2, atom, V * aa_, V * bb_, V * ab_, V * ba_, 
                                            centers, ci1, ci2, a, b);
                                    }
                                }
                            else
                                for (int atom1 = 0; atom1 < natoms; ++atom1)
                                {
                                    const bool a = mask[atom1][sh1.get_idx()];
                                    for(int atom2 = atom1; atom2 < natoms; ++atom2)
                                    {
                                        const bool b = mask[atom2][sh2.get_idx()];        
                                        
                                        sumYX(atom1, atom2, atom, V * aa_, V * bb_, V * ab_, V * ba_, 
                                              centers, ci1, ci2, a, b);
                                    }
                                }
                        }
                        else if (coord == Cart2::XZ || coord == Cart2::ZX)
                        {
                            // AA
                            double aa_ = 4.0 * alpha1(p1) * alpha1(p1) * Vs(idx(n2, m2, l2, n1 + 1, m1, l1 + 1), 0);
                            if (n1) aa_ -= 2.0 * alpha1(p1) * n1 * Vs(idx(n2, m2, l2, n1 - 1, m1, l1 + 1), 0);
                            if (l1) aa_ -= 2.0 * alpha1(p1) * l1 * Vs(idx(n2, m2, l2, n1 + 1, m1, l1 - 1), 0);
                            if (l1 && n1) aa_ += (n1 * l1) * Vs(idx(n2, m2, l2, n1 - 1, m1, l1 - 1), 0);
                            // BB
                            double bb_ = 4.0 * alpha2(p2) * alpha2(p2) * Vs(idx(n2 + 1, m2, l2 + 1, n1, m1, l1), 0);
                            if (n2) bb_ -= 2.0 * alpha2(p2) * n2 * Vs(idx(n2 - 1, m2, l2 + 1, n1, m1, l1), 0);
                            if (l2) bb_ -= 2.0 * alpha2(p2) * l2 * Vs(idx(n2 + 1, m2, l2 - 1, n1, m1, l1), 0);
                            if (l2 && n2) bb_ += (n2 * l2) * Vs(idx(n2 - 1, m2, l2 - 1, n1, m1, l1), 0);
                            // AB
                            double ab_ = 4.0 * alpha1(p1) * alpha2(p2) * Vs(idx(n2 + 1, m2, l2, n1, m1, l1 + 1), 0);
                            if (n2) ab_ += -2.0 * alpha1(p1) * n2 * Vs(idx(n2 - 1, m2, l2, n1, m1, l1 + 1), 0);
                            if (l1) ab_ += -2.0 * alpha2(p2) * l1 * Vs(idx(n2 + 1, m2, l2, n1, m1, l1 - 1), 0);
                            if (l1 && n2) ab_ += l1 * n2 * Vs(idx(n2 - 1, m2, l2, n1, m1, l1 - 1), 0);
                            // BA
                            double ba_ = 4.0 * alpha1(p1) * alpha2(p2) * Vs(idx(n2, m2, l2 + 1, n1 + 1, m1, l1), 0);
                            if (l2) ba_ += -2.0 * alpha1(p1) * l2 * Vs(idx(n2, m2, l2 - 1, n1 + 1, m1, l1), 0);
                            if (n1) ba_ += -2.0 * alpha2(p2) * n1 * Vs(idx(n2, m2, l2 + 1, n1 - 1, m1, l1), 0);
                            if (l2 && n1) ba_ += l2 * n1 * Vs(idx(n2, m2, l2 - 1, n1 - 1, m1, l1), 0);

                            if (coord == Cart2::XZ)
                                for (int atom1 = 0; atom1 < natoms; ++atom1)
                                {
                                    const bool a = mask[atom1][sh1.get_idx()];
                                    for(int atom2 = atom1; atom2 < natoms; ++atom2)
                                    {
                                        const bool b = mask[atom2][sh2.get_idx()];
                                        sumXY(atom1, atom2, atom, V * aa_, V * bb_, V * ab_, V * ba_, 
                                            centers, ci1, ci2, a, b);
                                    }
                                }
                            else
                                for (int atom1 = 0; atom1 < natoms; ++atom1)
                                {
                                    const bool a = mask[atom1][sh1.get_idx()];
                                    for(int atom2 = atom1; atom2 < natoms; ++atom2)
                                    {
                                        const bool b = mask[atom2][sh2.get_idx()];        
                                        
                                        sumYX(atom1, atom2, atom, V * aa_, V * bb_, V * ab_, V * ba_, 
                                              centers, ci1, ci2, a, b);
                                    }
                                }
                        }
                        else if (coord == Cart2::YZ || coord == Cart2::ZY)
                        {
                            // AA
                            double aa_ = 4.0 * alpha1(p1) * alpha1(p1) * Vs(idx(n2, m2, l2, n1 + 1, m1 + 1, l1), 0);
                            if (n1) aa_ -= 2.0 * alpha1(p1) * n1 * Vs(idx(n2, m2, l2, n1 - 1, m1 + 1, l1), 0);
                            if (m1) aa_ -= 2.0 * alpha1(p1) * m1 * Vs(idx(n2, m2, l2, n1 + 1, m1 - 1, l1), 0);
                            if (m1 && n1) aa_ += (n1 * m1) * Vs(idx(n2, m2, l2, n1 - 1, m1 - 1, l1), 0);

                            // BB
                            double bb_ = 4.0 * alpha2(p2) * alpha2(p2) * Vs(idx(n2 + 1, m2 + 1, l2, n1, m1, l1), 0);
                            if (n2) bb_ -= 2.0 * alpha2(p2) * n2 * Vs(idx(n2 - 1, m2 + 1, l2, n1, m1, l1), 0);
                            if (m2) bb_ -= 2.0 * alpha2(p2) * m2 * Vs(idx(n2 + 1, m2 - 1, l2, n1, m1, l1), 0);
                            if (m2 && n2) bb_ += (n2 * m2) * Vs(idx(n2 - 1, m2 - 1, l2, n1, m1, l1), 0);

                            // AB
                            double ab_ = 4.0 * alpha1(p1) * alpha2(p2) * Vs(idx(n2 + 1, m2, l2, n1, m1 + 1, l1), 0);
                            if (n2) ab_ += -2.0 * alpha1(p1) * n2 * Vs(idx(n2 - 1, m2, l2, n1, m1 + 1, l1), 0);
                            if (m1) ab_ += -2.0 * alpha2(p2) * m1 * Vs(idx(n2 + 1, m2, l2, n1, m1 - 1, l1), 0);
                            if (m1 && n2) ab_ += (m1 * n2) * Vs(idx(n2 - 1, m2, l2, n1, m1 - 1, l1), 0);

                            // BA
                            double ba_ = 4.0 * alpha1(p1) * alpha2(p2) * Vs(idx(n2, m2 + 1, l2, n1 + 1, m1, l1), 0);
                            if (m2) ba_ += -2.0 * alpha1(p1) * m2 * Vs(idx(n2, m2 - 1, l2, n1 + 1, m1, l1), 0);
                            if (n1) ba_ += -2.0 * alpha2(p2) * n1 * Vs(idx(n2, m2 + 1, l2, n1 - 1, m1, l1), 0);
                            if (m2 && n1) ba_ += (m2 * n1) * Vs(idx(n2, m2 - 1, l2, n1 - 1, m1, l1), 0);

                            if (coord == Cart2::YZ)
                                for (int atom1 = 0; atom1 < natoms; ++atom1)
                                {
                                    const bool a = mask[atom1][sh1.get_idx()];
                                    for(int atom2 = atom1; atom2 < natoms; ++atom2)
                                    {
                                        const bool b = mask[atom2][sh2.get_idx()];
                                        sumXY(atom1, atom2, atom, V * aa_, V * bb_, V * ab_, V * ba_, 
                                            centers, ci1, ci2, a, b);
                                    }
                                }
                            else
                                for (int atom1 = 0; atom1 < natoms; ++atom1)
                                {
                                    const bool a = mask[atom1][sh1.get_idx()];
                                    for(int atom2 = atom1; atom2 < natoms; ++atom2)
                                    {
                                        const bool b = mask[atom2][sh2.get_idx()];        
                                        
                                        sumYX(atom1, atom2, atom, V * aa_, V * bb_, V * ab_, V * ba_, 
                                              centers, ci1, ci2, a, b);
                                    }
                                }
                        }
                    }
                }
            }
        }
        ++atom;
    }

    // map shell to basis

    if (!pure)
    {
        for (int atom1 = 0; atom1 < natoms; ++atom1)
        {
            const bool a = mask[atom1][sh1.get_idx()];
            for(int atom2 = atom1; atom2 < natoms; ++atom2)
            {
                const bool b =  mask[atom2][sh2.get_idx()];
                const bool c = !mask[atom1][sh2.get_idx()] && !mask[atom2][sh2.get_idx()];
                const bool d = !mask[atom1][sh1.get_idx()] && !mask[atom2][sh1.get_idx()];

                for (const auto &i : id1)
                    for (const auto &j : id2)
                        dVqq(atom1, atom2, sh1.get_idx() + i.ci, sh2.get_idx() + j.ci) =
                        -add_centers(atom1, atom2, centers, i.ci, j.ci, a, b, c, d, coord);
            }
        }
    }
    else
    {
        EigenMatrix<double> tmp =  EigenMatrix<double>(sh1.get_cirange(), sh2.get_cirange());
        const EigenMatrix<double> tr_left = sh1.get_spherical_form().transpose().eval();
        const EigenMatrix<double> tr_right = sh2.get_spherical_form().eval();

        for (int atom1 = 0; atom1 < natoms; ++atom1)
        {
            const bool a = mask[atom1][sh1.get_idx()];
            for(int atom2 = atom1; atom2 < natoms; ++atom2)
            {
                const bool b =  mask[atom2][sh2.get_idx()];
                const bool c = !mask[atom1][sh2.get_idx()] && !mask[atom2][sh2.get_idx()];
                const bool d = !mask[atom1][sh1.get_idx()] && !mask[atom2][sh1.get_idx()];

                for (const auto &i : id1)
                    for (const auto &j : id2)
                        tmp(i.ci, j.ci) =
                        -add_centers(atom1, atom2, centers, i.ci, j.ci, a, b, c, d, coord);
                
                EigenMatrix<double> tmp2 = tr_left * tmp * tr_right;

                for (Index i = 0; i < sh1.get_sirange(); ++i)
                    for (Index j = 0; j < sh2.get_sirange(); ++j)
                        dVqq(atom1, atom2, sh1.get_ids() + i, sh2.get_ids() + j) = tmp2(i, j);
            }
        }
    }
}

using OSNUCLEAR::Center; // XX YY ZZ
void OSNUCLEAR::sumXX(int atom1, int atom2, int atom, double aa_, double bb_, double ab_, 
                      tensor5d<double>& ctr, Index ci1, Index ci2, bool a, bool b)
{
    if (atom1 == atom2) 
    {
        if (atom1 == atom) ctr(atom1, atom2, cc, ci1, ci2) += aa_ + 2 * ab_ + bb_;  // CC

        if (a && b) 
        {
            if (atom1 != atom)  // AA BB AB
            {
                ctr(atom1, atom2, aa, ci1, ci2) += aa_;
                ctr(atom1, atom2, ab, ci1, ci2) += ab_;
                ctr(atom1, atom2, bb, ci1, ci2) += bb_;
            }
        } 
        else 
        {
            if (atom1 != atom) 
            {
                ctr(atom1, atom2, aa, ci1, ci2) += aa_;
                ctr(atom1, atom2, bb, ci1, ci2) += bb_;
                ctr(atom1, atom2, ab, ci1, ci2) += ab_;
            } 
            else 
            {
                ctr(atom1, atom2, aa, ci1, ci2) += bb_;
                ctr(atom1, atom2, bb, ci1, ci2) += aa_;
                ctr(atom1, atom2, ab, ci1, ci2) -= aa_ + bb_;
            }
        }
    } 
    else 
    {
        if (atom1 != atom && atom != atom2)
            ctr(atom1, atom2, ab, ci1, ci2) += ab_;
        else if (atom1 == atom) 
        {
            ctr(atom1, atom2, ab, ci1, ci2) -= bb_;         // AB
            ctr(atom1, atom2, ac, ci1, ci2) -= ab_ + aa_; // AC
            ctr(atom1, atom2, cc, ci1, ci2) += aa_ + 2 * ab_ + bb_;    // CC
        } 
        else if (atom2 == atom) 
        {
            ctr(atom1, atom2, ab, ci1, ci2) -= aa_;                    // AB
            ctr(atom1, atom2, bc, ci1, ci2) -= bb_ + ab_;            // BC
            ctr(atom1, atom2, dd, ci1, ci2) += aa_ + 2 * ab_ + bb_;  // DD
        }
    }
}
// XY XZ YZ
void OSNUCLEAR::sumXY(int atom1, int atom2, int atom, double aa_, double bb_, double ab_, double ba_,
                      tensor5d<double>& ctr, Index ci1, Index ci2, bool a, bool b)
{
    if (atom1 == atom2) 
    {
        if (a && b) 
        {
            if (atom1 == atom)
                ctr(atom1, atom2, cc, ci1, ci2) += aa_ + ab_ + ba_ + bb_;
            else 
            {
                // AA BB
                ctr(atom1, atom2, aa, ci1, ci2) += aa_;
                ctr(atom1, atom2, ab, ci1, ci2) += ab_;
                ctr(atom1, atom2, ba, ci1, ci2) += ba_;
                ctr(atom1, atom2, bb, ci1, ci2) += bb_;
            }
        } 
        else 
        {
            // AA BB
            if (atom1 != atom) 
            {
                ctr(atom1, atom2, aa, ci1, ci2) += aa_;
                ctr(atom1, atom2, bb, ci1, ci2) += bb_;
                ctr(atom1, atom2, ab, ci1, ci2) += ab_;
                ctr(atom1, atom2, ba, ci1, ci2) += ba_;
            }

            if (atom1 == atom) 
            {
                ctr(atom1, atom2, cc, ci1, ci2) += bb_ + ab_ + ba_ + aa_;  // CC
                ctr(atom1, atom2, aa, ci1, ci2) += bb_;
                ctr(atom1, atom2, bb, ci1, ci2) += aa_;
                ctr(atom1, atom2, ab, ci1, ci2) -= aa_ + bb_;
                ctr(atom1, atom2, ba, ci1, ci2) -= aa_ + bb_;
            }
        }
    } 
    else 
    {
        if (atom1 != atom && atom != atom2)
            ctr(atom1, atom2, ab, ci1, ci2) += ab_;
        else if (atom1 == atom) 
        {
            ctr(atom1, atom2, ab, ci1, ci2) -= bb_;                   // AB
            ctr(atom1, atom2, ac, ci1, ci2) -= aa_ + ba_;            // AC
            ctr(atom1, atom2, cc, ci1, ci2) += bb_ + ab_ + ba_ + aa_;  // CC
        } 
        else if (atom2 == atom) 
        {
            ctr(atom1, atom2, ab, ci1, ci2) -= aa_;                   // AB
            ctr(atom1, atom2, bc, ci1, ci2) -= bb_ + ba_;            // BC
            ctr(atom1, atom2, dd, ci1, ci2) += bb_ + ab_ + ba_ + aa_;  // DD
        } 
    }
}
// YX ZX ZY
void OSNUCLEAR::sumYX(int atom1, int atom2, int atom, double aa_, double bb_, double ab_, double ba_,
                      tensor5d<double>& ctr, Index ci1, Index ci2, bool a, bool b)
{
    if (atom1 == atom2) 
    {
        if (a && b) 
        {
            if (atom1 == atom)
                ctr(atom1, atom2, cc, ci1, ci2) += aa_ + ab_ + ba_ + bb_;
            else 
            {
                // AA BB
                ctr(atom1, atom2, aa, ci1, ci2) += aa_;
                ctr(atom1, atom2, ab, ci1, ci2) += ab_;
                ctr(atom1, atom2, ba, ci1, ci2) += ba_;
                ctr(atom1, atom2, bb, ci1, ci2) += bb_;
            }
        } 
        else 
        {
            // AA BB
            if (atom1 != atom) 
            {
                ctr(atom1, atom2, aa, ci1, ci2) += aa_;
                ctr(atom1, atom2, bb, ci1, ci2) += bb_;
                ctr(atom1, atom2, ab, ci1, ci2) += ab_;
                ctr(atom1, atom2, ba, ci1, ci2) += ba_;
            }

            if (atom1 == atom) 
            {
                ctr(atom1, atom2, cc, ci1, ci2) += bb_ + ab_ + ba_ + aa_;  // CC
                ctr(atom1, atom2, aa, ci1, ci2) += bb_;
                ctr(atom1, atom2, bb, ci1, ci2) += aa_;
                ctr(atom1, atom2, ab, ci1, ci2) -= aa_ + bb_;
                ctr(atom1, atom2, ba, ci1, ci2) -= aa_ + bb_;
            }
        }
    } 
    else 
    {
        if (atom1 != atom && atom != atom2)
            ctr(atom1, atom2, ab, ci1, ci2) += ba_;
        else if (atom1 == atom) 
        {
            ctr(atom1, atom2, ab, ci1, ci2) -= bb_;                    // AB
            ctr(atom1, atom2, ac, ci1, ci2) -= aa_ + ab_;              // AC
            ctr(atom1, atom2, cc, ci1, ci2) += bb_ + ab_ + ba_ + aa_;  // CC
        } 
        else if (atom2 == atom) 
        {
            ctr(atom1, atom2, ab, ci1, ci2) -= aa_;                    // AB
            ctr(atom1, atom2, bc, ci1, ci2) -= bb_ + ab_;              // BC
            ctr(atom1, atom2, dd, ci1, ci2) += bb_ + ab_ + ba_ + aa_;  // DD
        }
    }
}

double OSNUCLEAR::add_centers(int atom1, int atom2, const tensor5d<double>& ctr, Index ci1, Index ci2,
                              bool a, bool b, bool c, bool d, Cart2 coord)
{
    double tot = 0;

    if (atom1 == atom2) 
    {
        if (!a && !b) tot += ctr(atom1, atom2, cc, ci1, ci2);
        else if (a && b) tot += 2.0 * ctr(atom1, atom2, ab, ci1, ci2);

        if (a) tot += ctr(atom1, atom2, aa, ci1, ci2);
        if (b) tot += ctr(atom1, atom2, bb, ci1, ci2);
        if (a && b && !c && !d && (coord != Cart2::XX && coord != Cart2::YY && coord != Cart2::ZZ))
            tot = ctr(atom1, atom2, aa, ci1, ci2) + ctr(atom1, atom2, ab, ci1, ci2) 
                + ctr(atom1, atom2, ba, ci1, ci2) + ctr(atom1, atom2, bb, ci1, ci2);
    } 
    else  // if(atom1 != atom2)
    {
        if (a && b /*|| (mask1[j] && mask2[i]) */) tot += ctr(atom1, atom2, ab, ci1, ci2);
        else if (!a && !b && c && d)  tot  = 0;
        else if (!a && !b && !c && d) tot  = ctr(atom1, atom2, bc, ci1, ci2);
        else if (!a && b && !c && d)  tot -= ctr(atom1, atom2, ac, ci1, ci2) + ctr(atom1, atom2, cc, ci1, ci2);
        else if (a && !b && !c && !d) tot -= ctr(atom1, atom2, dd, ci1, ci2);
        else if (!a && b && !c && !d) tot -= ctr(atom1, atom2, cc, ci1, ci2);
        else if (a && !b && c && !d)  tot -= ctr(atom1, atom2, bc, ci1, ci2) + ctr(atom1, atom2, dd, ci1, ci2);
        else if (!a && !b && c && !d) tot += ctr(atom1, atom2, ac, ci1, ci2);
    }

    return tot;
}
