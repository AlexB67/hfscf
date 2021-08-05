#include "hfscf_oseri.hpp"
#include "../math/hfscf_math.hpp"
#include "../math/gamma.hpp"
#include "hfscf_osrecur.hpp"
#include "hfscf_transform.hpp"

using hfscfmath::rab2;
using hfscfmath::pi;
using hfscfmath::gpc;
using extramath::F_nu;
using hfscfmath::PI_25;
using os_recursion::osrecur;
using os_recursion::osrecurl13;
using os_recursion::osrecurl2;
using os_recursion::osrecurl4s0;
using os_recursion::osrecurl4s1;
using os_recursion::osrecurl4s2;
using TRANSFORM::transform;

void ERIOS::Erios::compute_contracted_shell_quartet_deriv1(symm4dTensor<double>& dVdXa,
                                                           symm4dTensor<double>& dVdXb,
                                                           symm4dTensor<double>& dVdXc,
                                                           symm4dTensor<double>& dVdYa,
                                                           symm4dTensor<double>& dVdYb,
                                                           symm4dTensor<double>& dVdYc,
                                                           symm4dTensor<double>& dVdZa,
                                                           symm4dTensor<double>& dVdZb,
                                                           symm4dTensor<double>& dVdZc,
                                                           const ShellPair& s12, const ShellPair& s34,
                                                           const std::vector<bool>& coords) const
{
    const auto& sh1 = s12.m_s1;
    const auto& sh2 = s12.m_s2;
    const auto& sh3 = s34.m_s1;
    const auto& sh4 = s34.m_s2;

    Index am1 = sh1.get_cirange();
    Index am2 = sh2.get_cirange();
    Index am3 = sh3.get_cirange();
    Index am4 = sh4.get_cirange();

    tensor4d1234<double> z_block_a, z_block_b, z_block_c;
    if (coords[2])
    {
        z_block_a = tensor4d1234<double>(am1, am2, am3, am4);
        z_block_b = tensor4d1234<double>(am1, am2, am3, am4);
        z_block_c = tensor4d1234<double>(am1, am2, am3, am4);
    }

    tensor4d1234<double> y_block_a, y_block_b, y_block_c;
    if (coords[1])
    {
        y_block_a = tensor4d1234<double>(am1, am2, am3, am4);
        y_block_b = tensor4d1234<double>(am1, am2, am3, am4);
        y_block_c = tensor4d1234<double>(am1, am2, am3, am4);
    }

    tensor4d1234<double> x_block_a, x_block_b, x_block_c;
    if (coords[0])
    {
        x_block_a = tensor4d1234<double>(am1, am2, am3, am4);
        x_block_b = tensor4d1234<double>(am1, am2, am3, am4);
        x_block_c = tensor4d1234<double>(am1, am2, am3, am4);
    }

    Index nmax = 3 * (sh4.L() + sh3.L() + sh2.L() + sh1.L()) + 1;

    tensor5d<double> Jnx, Jny, Jnz;
    
    if(coords[1] || coords[2])
    {
        Index uvmax = 2 * (sh4.L() + sh3.L() + sh2.L() + sh1.L()) + 1;
        Jnx = tensor5d<double>(sh4.L() + 1, sh3.L() + 2, sh2.L() + 2, sh1.L() + 2, nmax + 1);
        Jny = tensor5d<double>(sh4.L() + 1, sh3.L() + sh4.L() + 2, sh2.L() + 2, sh1.L() + sh2.L() + 2, uvmax + 1);
        Jnz = tensor5d<double>(sh4.L() + 1, sh3.L() + sh4.L() + 2, sh2.L() + 2, sh1.L() + sh2.L() + 2, uvmax + 1);
    }

    tensor5d<double> Jnx0, Jny0, Jnz0;

    if (coords[0]) // clumsy TODO investigate overwriting A with B and C centers see comments later
    {              // we use separate arrays to avoid zeroing ... costly in loops
        Index tvmax = 2 * (sh4.L() + sh3.L() + sh2.L() + sh1.L()) + 1;
        Index tmax  = 1 * (sh4.L() + sh3.L() + sh2.L() + sh1.L()) + 1;
        Jnx0 = tensor5d<double>(sh4.L() + 1, sh3.L() + sh4.L() + 2, sh2.L() + 2, sh1.L() + sh2.L() + 2, tmax + 1);
        Jny0 = tensor5d<double>(sh4.L() + 1, sh3.L() + 1, sh2.L() + 1, sh1.L() + 1, nmax + 1);
        Jnz0 = tensor5d<double>(sh4.L() + 1, sh3.L() + 1, sh2.L() + 1, sh1.L() + 1, tvmax + 1);
    }

    const Eigen::Ref<const EigenVector<double>>& alpha1 = sh1.alpha();
    const Eigen::Ref<const EigenVector<double>>& alpha2 = sh2.alpha();
    const Eigen::Ref<const EigenVector<double>>& alpha3 = sh3.alpha();
    const Eigen::Ref<const EigenVector<double>>& c1 = sh1.c();
    const Eigen::Ref<const EigenVector<double>>& c2 = sh2.c();
    const Eigen::Ref<const EigenVector<double>>& c3 = sh3.c();
    const Eigen::Ref<const EigenVector<double>>& c4 = sh4.c();
    const std::vector<BASIS::idq>& id1 = sh1.get_indices();
    const std::vector<BASIS::idq>& id2 = sh2.get_indices(); 
    const std::vector<BASIS::idq>& id3 = sh3.get_indices(); 
    const std::vector<BASIS::idq>& id4 = sh4.get_indices();

    const double ABx = s12.AB(0);
    const double ABy = s12.AB(1);
    const double ABz = s12.AB(2);
    const double CDx = s34.AB(0);
    const double CDy = s34.AB(1);
    const double CDz = s34.AB(2);
    const double precut = cutoff / 10000.0;

    for (Index p1 = 0; p1 < c1.size(); ++p1)
        for (Index p2 = 0; p2 < c2.size(); ++p2)
        {
            const Index id12 = p1 * c2.size() + p2;
            const double gamma_ab = s12.gamma_ab(id12);
            const double gamma_ab2 = 2 * gamma_ab;
            const Vec3D& P = s12.P(id12);
            const Vec3D& PA = s12.PA(id12);
            const Vec3D& PB = s12.PB(id12);

            for (Index p3 = 0; p3 < c3.size(); ++p3)
                for (Index p4 = 0; p4 < c4.size(); ++p4)
                {
                    const Index id34 = p3 * c4.size() + p4;
                    const Vec3D& Q = s34.P(id34);
                    const Vec3D PQ = P - Q;
                    const Vec3D QC = s34.PA(id34);
                    const Vec3D QD = s34.PB(id34);
                    const double rpq_2 = rab2(P, Q);
                    const double gamma_cd = s34.gamma_ab(id34);
                    const double gamma_cd2 = 2 * gamma_cd;
                    const double eta = gamma_ab * gamma_cd / (gamma_ab + gamma_cd);
                    const double p_r = eta / gamma_ab;
                    const double q_r = eta / gamma_cd;
                    const double pfac = s12.Kab(id12) * s34.Kab(id34)
                                      * PI_25 / (gamma_ab * gamma_cd * std::sqrt(gamma_ab + gamma_cd));
                    
                    if (fabs(pfac) < precut) continue;
                    
                    const double T = eta * rpq_2;
                    const double expT = std::exp(-T);

                    if(fabs(rpq_2) > 1E-18)
                    {
                        Jnx(0, 0, 0, 0, nmax) = F_nu(static_cast<double>(nmax), T);
                        // Downward recursion for Boys stability 
                        for (Index n = nmax - 1; n >= 0; --n)
                            Jnx(0, 0, 0, 0, n) = (1.0 / (2.0 * n + 1.0)) * (2.0 * T * Jnx(0, 0, 0, 0, n + 1) + expT);
                    }
                    else
                    {
                        Jnx(0, 0, 0, 0, nmax) = 1.0;
                        for (Index n = nmax - 1; n >= 0; --n)
                            Jnx(0, 0, 0, 0, n) = 1.0 / (2.0 * n + 1.0);
                    }

                    if (coords[1] || coords[2])
                    {
                        osrecur<double>(PQ(0), PA(0), PB(0), QC(0), Q(0) - sh4.x(), gamma_ab2, gamma_cd2, 
                                    p_r, q_r, sh1.L(), sh2.L(), sh3.L(), sh4.L(), nmax, Jnx);
                    }
                    
                    if (coords[0])
                    {
                        for (Index n = 0; n <= nmax; ++n) Jny0(0, 0, 0, 0, n) = Jnx(0, 0, 0, 0, n);

                        osrecur<double>(PQ(1), PA(1), PB(1), QC(1), Q(1) - sh4.y(), gamma_ab2, gamma_cd2, 
                                    p_r, q_r, sh1.L(), sh2.L(), sh3.L(), sh4.L(), nmax, Jny0);
                    }
                    

// ********* build shell block start ********** move to func
const auto build_shell_quartet = [&]() noexcept
{
    const double a1 = 2 * alpha1(p1); 
    const double a2 = 2 * alpha2(p2);
    const double a3 = 2 * alpha3(p3);

    for (const auto& i : id1)
    {
        Index idx1 = sh1.get_idx() + i.ci;

        for (const auto &j : id2)
        {
            Index idx2 = sh2.get_idx() + j.ci;
            if(!pure && idx1 < idx2) continue;  // permutation symmetry
            bool idx12 = idx1 == idx2;

            for (const auto& k : id3)
            {
                Index idx3 = sh3.get_idx() + k.ci;
                bool idx23 = idx2 == idx3;

                for (const auto& l : id4)
                {
                    Index idx4 = sh4.get_idx() + l.ci;
                    
                    if (!pure && idx3 < idx4) continue;  // permutation symmetry
                    else if(idx12 && idx23 && idx3 == idx4) continue; // always zero from total centers

                    /******* Z ********/

                    if (coords[2]) 
                    {
                        Index n_max = i.m + j.m + k.m + l.m + i.n + j.n + k.n + l.n + 1;  // current nmax
                                                                                    // l1 l2 l3 l4  done
                        double tmp = 0;
                        for (Index n = 0; n <= n_max; ++n)
                        {
                            Jny(0, 0, 0, 0, n) = Jnx(l.l, k.l, j.l, i.l, n);
                            tmp += fabs(Jnx(l.l, k.l, j.l, i.l, n));
                        }

                        if (tmp > cutoff)
                        {
                        osrecur<double>(PQ(1), PA(1), PB(1), QC(1), QD(1), gamma_ab2, gamma_cd2, p_r, q_r, i.m,
                                        j.m, k.m, l.m, n_max, Jny);

                        n_max -= i.m + j.m + k.m + l.m;

                        for (Index n = 0; n <= n_max; ++n)  // y to z
                            Jnz(0, 0, 0, 0, n) = Jny(l.m, k.m, j.m, i.m, n);

                        osrecurl13<double>(PQ(2), QC(2), PA(2), gamma_ab2, gamma_cd2, p_r, q_r,
                                        i.n + j.n + 1, k.n + l.n + 1, n_max, Jnz);
                        osrecurl2<double>(ABz, i.n + 1, j.n, k.n, l.n, Jnz);
                        osrecurl4s1<double>(CDz, i.n + 1, j.n, k.n, l.n, Jnz);

                        // Az
                        z_block_a(i.ci, j.ci, k.ci, l.ci) += a1 * Jnz(l.n, k.n, j.n, i.n + 1, 0) * pfac;

                        if (i.n) z_block_a(i.ci, j.ci, k.ci, l.ci) -= 
                                 (double)i.n * Jnz(l.n, k.n, j.n, i.n - 1, 0) * pfac;

                        osrecurl2<double>(ABz, i.n, j.n + 1, k.n, l.n, Jnz);
                        osrecurl4s2<double>(CDz, i.n, j.n + 1, k.n, l.n, Jnz);

                        // Bz
                        z_block_b(i.ci, j.ci, k.ci, l.ci) += a2 * Jnz(l.n, k.n, j.n + 1, i.n, 0) * pfac;

                        if (j.n) z_block_b(i.ci, j.ci, k.ci, l.ci) -=  
                                 (double)j.n * Jnz(l.n, k.n, j.n - 1, i.n, 0) * pfac;

                        osrecurl2<double>(ABz, i.n, j.n, k.n + 1, l.n, Jnz);
                        osrecurl4s0<double>(CDz, i.n, j.n, k.n + 1, l.n, Jnz);

                        // Cz
                        z_block_c(i.ci, j.ci, k.ci, l.ci) += a3 * Jnz(l.n, k.n + 1, j.n, i.n, 0) * pfac;

                        if (k.n) z_block_c(i.ci, j.ci, k.ci, l.ci) -= 
                                  (double) k.n * Jnz(l.n, k.n - 1, j.n, i.n, 0) * pfac;
                        }
                    }

                    /******* Y ********/
                    if (coords[1])
                    {
                        Index n_max = i.m + j.m + k.m + l.m + i.n + j.n + k.n + l.n + 1;  // current nmax
                        
                        double tmp = 0;
                        for (Index n = 0; n <= n_max; ++n)
                        {
                            Jnz(0, 0, 0, 0, n) = Jnx(l.l, k.l, j.l, i.l, n);
                            tmp += fabs(Jnx(l.l, k.l, j.l, i.l, n));
                        }

                        if (tmp > cutoff)
                        {
                        osrecur<double>(PQ(2), PA(2), PB(2), QC(2), QD(2), gamma_ab2, gamma_cd2, p_r, q_r, i.n,
                                        j.n, k.n, l.n, n_max, Jnz);

                        n_max -= i.n + j.n + k.n + l.n;
                        for (Index n = 0; n <= n_max; ++n)  // z to y
                            Jny(0, 0, 0, 0, n) = Jnz(l.n, k.n, j.n, i.n, n);

                        osrecurl13<double>(PQ(1), QC(1), PA(1), gamma_ab2, gamma_cd2, p_r, q_r,
                                          i.m + j.m + 1, k.m + l.m + 1, n_max, Jny);
                        osrecurl2<double>(ABy, i.m + 1, j.m, k.m, l.m, Jny);
                        osrecurl4s1<double>(CDy, i.m + 1, j.m, k.m, l.m, Jny);

                        // Ay
                        y_block_a(i.ci, j.ci, k.ci, l.ci) += a1 * Jny(l.m, k.m, j.m, i.m + 1, 0) * pfac;

                        if (i.m) y_block_a(i.ci, j.ci, k.ci, l.ci) -= 
                                  (double) i.m * Jny(l.m, k.m, j.m, i.m - 1, 0) * pfac;

                        osrecurl2<double>(ABy, i.m, j.m + 1, k.m, l.m, Jny);
                        osrecurl4s2<double>(CDy, i.m, j.m + 1, k.m, l.m, Jny);

                        // By
                        y_block_b(i.ci, j.ci, k.ci, l.ci) += a2 * Jny(l.m, k.m, j.m + 1, i.m, 0) * pfac;

                        if (j.m) y_block_b(i.ci, j.ci, k.ci, l.ci) -= 
                                  (double) j.m * Jny(l.m, k.m, j.m - 1, i.m, 0) * pfac;

                        osrecurl2<double>(ABy, i.m, j.m, k.m + 1, l.m, Jny);
                        osrecurl4s0<double>(CDy, i.m, j.m, k.m + 1, l.m, Jny);

                        // Cy
                        y_block_c(i.ci, j.ci, k.ci, l.ci) += a3 * Jny(l.m, k.m + 1, j.m, i.m, 0) * pfac;

                        if (k.m) y_block_c(i.ci, j.ci, k.ci, l.ci) -=
                                  (double) k.m * Jny(l.m, k.m - 1, j.m, i.m, 0) * pfac;
                        }
                    }
                    /****** X *******/
                    if (coords[0]) 
                    {
                        Index n_max = i.n + j.n + k.n + l.n + i.l + j.l + k.l + l.l + 1;

                        double tmp = 0;
                        for (Index n = 0; n <= n_max; ++n)
                        {
                            Jnz0(0, 0, 0, 0, n) = Jny0(l.m, k.m, j.m, i.m, n);
                            tmp += fabs(Jny0(l.m, k.m, j.m, i.m, n));
                        }

                        if (tmp > cutoff)
                        {
                        osrecur<double>(PQ(2), PA(2), PB(2), QC(2), QD(2), gamma_ab2, gamma_cd2, p_r, q_r, i.n,
                                        j.n, k.n, l.n, n_max, Jnz0);

                        n_max -= i.n + j.n + k.n + l.n;

                        for (Index n = 0; n <= n_max; ++n)  // z to y
                            Jnx0(0, 0, 0, 0, n) = Jnz0(l.n, k.n, j.n, i.n, n);

                        osrecurl13<double>(PQ(0), QC(0), PA(0), gamma_ab2, gamma_cd2, p_r, q_r,
                                        i.l + j.l + 1, k.l + l.l + 1, n_max, Jnx0);
                        osrecurl2<double>(ABx, i.l + 1, j.l, k.l, l.l, Jnx0);
                        osrecurl4s1<double>(CDx, i.l + 1, j.l, k.l, l.l, Jnx0);

                        // Ax
                        x_block_a(i.ci, j.ci, k.ci, l.ci) += a1 * Jnx0(l.l, k.l, j.l, i.l + 1, 0) * pfac;

                        if (i.l) x_block_a(i.ci, j.ci, k.ci, l.ci) -= 
                                  (double) i.l * Jnx0(l.l, k.l, j.l, i.l - 1, 0) * pfac;

                        osrecurl2<double>(ABx, i.l, j.l + 1, k.l, l.l, Jnx0);
                        osrecurl4s2<double>(CDx, i.l, j.l + 1, k.l, l.l, Jnx0);

                        // Bx
                        x_block_b(i.ci, j.ci, k.ci, l.ci) += a2 * Jnx0(l.l, k.l, j.l + 1, i.l, 0) * pfac;

                        if (j.l) x_block_b(i.ci, j.ci, k.ci, l.ci) -= j.l * Jnx0(l.l, k.l, j.l - 1, i.l, 0) * pfac;

                        osrecurl2<double>(ABx, i.l, j.l, k.l + 1, l.l, Jnx0);
                        osrecurl4s0<double>(CDx, i.l, j.l, k.l + 1, l.l, Jnx0);

                        // Cx
                        x_block_c(i.ci, j.ci, k.ci, l.ci) += a3 * Jnx0(l.l, k.l + 1, j.l, i.l, 0) * pfac;

                        if (k.l) x_block_c(i.ci, j.ci, k.ci, l.ci) -= 
                                  (double) k.l * Jnx0(l.l, k.l - 1, j.l, i.l, 0) * pfac;
                        }
                    }
                }
            }
        }
    }
};
// ********* build shell block end **********        
                    build_shell_quartet();
                }
        }
        
    // map shell block to eri integrals in basis form
    if (coords[2]) // Z
        shell_to_basis_deriv1(s12, s34, z_block_a, z_block_b, z_block_c, dVdZa, dVdZb, dVdZc);
    
    if (coords[1]) // Y
        shell_to_basis_deriv1(s12, s34, y_block_a, y_block_b, y_block_c, dVdYa, dVdYb, dVdYc);
    
    if (coords[0]) // X
        shell_to_basis_deriv1(s12, s34, x_block_a, x_block_b, x_block_c, dVdXa, dVdXb, dVdXc);
}

void ERIOS::Erios::shell_to_basis_deriv1(const ShellPair& s12, const ShellPair& s34,
                                         const tensor4d1234<double> block_a, const tensor4d1234<double> block_b,
                                         const tensor4d1234<double> block_c, symm4dTensor<double>& dXa, 
                                         symm4dTensor<double>& dXb, symm4dTensor<double>& dXc) const
{
    const auto& sh1 = s12.m_s1;
    const auto& sh2 = s12.m_s2;
    const auto& sh3 = s34.m_s1;
    const auto& sh4 = s34.m_s2;
    
    if (!pure)
    {
        for(Index p = sh1.get_idx(); p < sh1.get_idx() + sh1.get_cirange(); ++p)
            for(Index q = sh2.get_idx(); q < sh2.get_idx() + sh2.get_cirange(); ++q)
                for(Index r = sh3.get_idx(); r < sh3.get_idx() + sh3.get_cirange(); ++r)
                    for(Index s = sh4.get_idx(); s < sh4.get_idx() + sh4.get_cirange(); ++s)
                    {
                        if(p >= q && r >= s)
                        {
                            if (hfscfmath::index_ij(p, q) <= hfscfmath::index_ij(r, s))
                            {
                                dXa(p, q, r, s) = block_a(p - sh1.get_idx(), q - sh2.get_idx(), 
                                                        r - sh3.get_idx(), s - sh4.get_idx());
                                dXb(p, q, r, s) = block_b(p - sh1.get_idx(), q - sh2.get_idx(), 
                                                        r - sh3.get_idx(), s - sh4.get_idx());
                                dXc(p, q, r, s) = block_c(p - sh1.get_idx(), q - sh2.get_idx(), 
                                                        r - sh3.get_idx(), s - sh4.get_idx());
                            }
                            else  // work around A B C centers overwriting each other.. calculating too many also ?
                            {     // to look at.
                                dXc(p, q, r, s) = block_a(p - sh1.get_idx(), q - sh2.get_idx(), 
                                                        r - sh3.get_idx(), s - sh4.get_idx());
                                
                                dXb(p, q, r, s) = - block_a(p - sh1.get_idx(), q - sh2.get_idx(), 
                                                            r - sh3.get_idx(), s - sh4.get_idx())
                                                - block_b(p - sh1.get_idx(), q - sh2.get_idx(), 
                                                            r - sh3.get_idx(), s - sh4.get_idx())
                                                - block_c(p - sh1.get_idx(), q - sh2.get_idx(), 
                                                            r - sh3.get_idx(), s - sh4.get_idx());

                                dXa(p, q, r, s) = block_c(p - sh1.get_idx(), q - sh2.get_idx(), 
                                                        r - sh3.get_idx(), s - sh4.get_idx());
                            }
                            
                        }
                    }
    }
    else
    {
        tensor4d1234<double> pblock_a = tensor4d1234<double>(sh1.get_sirange(), sh2.get_sirange(), 
                                                             sh3.get_sirange(), sh4.get_sirange());
        tensor4d1234<double> pblock_b = tensor4d1234<double>(sh1.get_sirange(), sh2.get_sirange(), 
                                                             sh3.get_sirange(), sh4.get_sirange());
        tensor4d1234<double> pblock_c = tensor4d1234<double>(sh1.get_sirange(), sh2.get_sirange(), 
                                                             sh3.get_sirange(), sh4.get_sirange());
            
        transform(s12, s34, block_a, block_b, block_c, pblock_a, pblock_b, pblock_c);

        for(Index p = sh1.get_ids(); p < sh1.get_ids() + sh1.get_sirange(); ++p)
            for(Index q = sh2.get_ids(); q < sh2.get_ids() + sh2.get_sirange(); ++q)
                for(Index r = sh3.get_ids(); r < sh3.get_ids() + sh3.get_sirange(); ++r)
                    for(Index s = sh4.get_ids(); s < sh4.get_ids() + sh4.get_sirange(); ++s)
                    {
                        if(p >= q && r >= s)
                        {
                            if (hfscfmath::index_ij(p, q) <= hfscfmath::index_ij(r, s))
                            {
                                dXa(p, q, r, s) = pblock_a(p - sh1.get_ids(), q - sh2.get_ids(), 
                                                           r - sh3.get_ids(), s - sh4.get_ids());
                                dXb(p, q, r, s) = pblock_b(p - sh1.get_ids(), q - sh2.get_ids(), 
                                                           r - sh3.get_ids(), s - sh4.get_ids());
                                dXc(p, q, r, s) = pblock_c(p - sh1.get_ids(), q - sh2.get_ids(), 
                                                           r - sh3.get_ids(), s - sh4.get_ids());
                            }
                            else  // work around A B C centers overwriting each other.. calculating too many also ?
                            {     // to look at.
                                dXc(p, q, r, s) = pblock_a(p - sh1.get_ids(), q - sh2.get_ids(), 
                                                           r - sh3.get_ids(), s - sh4.get_ids());
                                
                                dXb(p, q, r, s) = - pblock_a(p - sh1.get_ids(), q - sh2.get_ids(), 
                                                             r - sh3.get_ids(), s - sh4.get_ids())
                                                  - pblock_b(p - sh1.get_ids(), q - sh2.get_ids(), 
                                                             r - sh3.get_ids(), s - sh4.get_ids())
                                                  - pblock_c(p - sh1.get_ids(), q - sh2.get_ids(), 
                                                             r - sh3.get_ids(), s - sh4.get_ids());

                                dXa(p, q, r, s) = pblock_c(p - sh1.get_ids(), q - sh2.get_ids(), 
                                                           r - sh3.get_ids(), s - sh4.get_ids());
                            }
                            
                        }
                    }
    }
}
