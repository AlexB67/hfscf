#include "hfscf_oseri.hpp"
#include "../math/hfscf_math.hpp"
#include "../math/gamma.hpp"
#include "hfscf_osrecur.hpp"

using hfscfmath::rab2;
using hfscfmath::pi;
using hfscfmath::gpc;
using extramath::F_nu;
using os_recursion::osrecurl123;
using os_recursion::osrecurl13;
using os_recursion::osrecurl4n;
using os_recursion::osrecurl24;
using tensormath::tensor4d1234;
using hfscfmath::PI_25;

void ERIOS::Erios::init_screen(const Index num_cartorbs)
{
    screen = EigenMatrix<double>(num_cartorbs, num_cartorbs);
}

void ERIOS::Erios::compute_screen_matrix(const ShellPair& s12)
{
    Index m1 = s12.m_s1.get_cirange();
    Index m2 = s12.m_s2.get_cirange();

    const auto& sh1 = s12.m_s1;
    const auto& sh2 = s12.m_s2;

    tensor4d1234<double> shell_block = tensor4d1234<double>(m1, m2, m1, m2);

    Index nmax  = 3 * (sh2.L() + sh1.L() + sh2.L() + sh1.L());
    Index uvmax = 2 * (sh2.L() + sh1.L() + sh2.L() + sh1.L());
    Index vmax  = 1 * (sh2.L() + sh1.L() + sh2.L() + sh1.L());

    const Index dimn = nmax + 1; const Index dim1 = sh1.L() + 1; const Index dim2 = sh2.L() + 1;
    const Index dim34 = sh1.L() + sh2.L() + 1; const Index dim4 = sh2.L() + 1; 
    const Index dim12 = sh1.L() + sh2.L() + 1;

    tensor5d<double> Jnx = tensor5d<double>(dim4, dim34, dim2, dim1, dimn);
    tensor5d<double> Jny = tensor5d<double>(dim4, dim34, dim2, dim1, uvmax + 1);
    tensor5d<double> Jnz = tensor5d<double>(dim4, dim34, dim2, dim12, vmax + 1);

    const Eigen::Ref<const EigenVector<double>>& c1 = sh1.c();
    const Eigen::Ref<const EigenVector<double>>& c2 = sh2.c();
    const std::vector<BASIS::idq>& id1 = sh1.get_indices();
    const std::vector<BASIS::idq>& id2 = sh2.get_indices();

    const Vec3D& AB = s12.AB;
    const Vec3D& CD = AB;

    for (Index p1 = 0; p1 < c1.size(); ++p1)
        for (Index p2 = 0; p2 < c2.size(); ++p2)
        {
            const Index id12 = p1 * c2.size() + p2;
            const double gamma_ab = s12.gamma_ab(id12);
            const double gamma_ab2 = gamma_ab + gamma_ab;
            const Vec3D& P = s12.P(id12); 
            const Vec3D& PA = s12.PA(id12);
            const Vec3D& PB = s12.PB(id12);

            for (Index p3 = 0; p3 < c1.size(); ++p3)
                for (Index p4 = 0; p4 < c2.size(); ++p4)
                {
                    const Index id34 = p3 * c2.size() + p4;
                    const Vec3D& Q = s12.P(id34);
                    const Vec3D& PQ = P - Q;
                    const Vec3D& QC = s12.PA(id34);
                    const double rpq_2 = rab2(P, Q);
                    const double gamma_cd = s12.gamma_ab(id34);
                    const double gamma_cd2 = gamma_cd + gamma_cd;
                    const double eta = gamma_ab * gamma_cd / (gamma_ab + gamma_cd);
                    const double p_r = eta / gamma_ab;
                    const double q_r = eta / gamma_cd;
                    const double pfac = s12.Kab(id12) * s12.Kab(id34) 
                                      * PI_25 / (gamma_ab * gamma_cd * std::sqrt(gamma_ab + gamma_cd));
                    
                    if (fabs(pfac) < 1E-16) continue;
                    
                    const double T = eta * rpq_2;
                    const double expT = std::exp(-T);

                    if(fabs(rpq_2) > 1E-14)
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

                    osrecurl123<double>(PQ(0), QC(0), PA(0), PB(0), gamma_ab2, gamma_cd2, 
                                        p_r, q_r, sh1.L(), sh2.L(), sh1.L() + sh2.L(), nmax, Jnx);
                        
                    for (Index l = 0; l < sh2.L(); ++l)
                        for (Index k = 0; k <= sh1.L() + sh2.L(); ++k)
                            for (Index j = 0; j <= sh2.L(); ++j)
                                for (Index i = 0; i <= sh1.L(); ++i)
                                    for (Index n = 0; n <= nmax - i - j - k - l; ++n)
                                        Jnx(l + 1, k, j, i, n) =
                                            Jnx(l, k + 1, j, i, n) + CD(0) * Jnx(l, k, j, i, n);
                    

// ********* build shell block start ********** move to func
const auto build_shell_quartet = [&]()
{
    for (const auto& i : id1)
    {
        Index idx1 = sh1.get_idx() + i.ci;

        for (const auto &j : id2)
        {
            Index idx2 = sh2.get_idx() + j.ci;
            if(idx1 > idx2) continue;  // permutation symmetry

            for (const auto& k : id1)
            {
                Index idx3 = sh1.get_idx() + k.ci;

                for (const auto& l : id2)
                {
                    Index idx4 = sh2.get_idx() + l.ci;
                    
                    if (idx3 > idx4) continue; // permutation symmetry

                    Index n_max = i.m + j.m + k.m + l.m + i.n + j.n + k.n + l.n;  // current nmax
                                                                        // l1 l2 l3 l4  done
                    double tmp = 0;
                    for (Index n = 0; n <= n_max; ++n)                    // x to y
                    {
                        Jny(0, 0, 0, 0, n) = Jnx(l.l, k.l, j.l, i.l, n);
                        tmp += fabs(Jnx(l.l, k.l, j.l, i.l, n));
                    }

                    if (fabs(tmp) < 1E-12) continue;

                    osrecurl123<double>(PQ(1), QC(1), PA(1), PB(1), gamma_ab2, gamma_cd2, p_r, q_r, i.m, j.m, k.m + l.m,
                                        n_max, Jny);
                    osrecurl4n<double>(CD(1), i.m, j.m, k.m, l.m, n_max, Jny);

                    n_max -= i.m + j.m + k.m + l.m;

                    tmp = 0;
                    for (Index n = 0; n <= n_max; ++n)  // y to z
                    {
                        Jnz(0, 0, 0, 0, n) = Jny(l.m, k.m, j.m, i.m, n);
                        tmp += fabs(Jny(l.m, k.m, j.m, i.m, n));
                    }
                    
                    if (fabs(tmp) < 1E-12) continue;

                    osrecurl13<double>(PQ(2), QC(2), PA(2), gamma_ab2, gamma_cd2, p_r, q_r, i.n + j.n, k.n + l.n,
                                       n_max, Jnz);
                    
                    osrecurl24<double>(AB(2), CD(2), i.n, j.n, k.n, l.n, Jnz);

                    shell_block(i.ci, j.ci, k.ci, l.ci) += pfac * Jnz(l.n, k.n, j.n, i.n, 0);
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
    for(Index p = sh1.get_idx(); p < sh1.get_idx() + sh1.get_cirange(); ++p)
        for(Index q = sh2.get_idx(); q < sh2.get_idx() + sh2.get_cirange(); ++q)
            if (p <= q)  // permutation symmetry
            {
                screen(p, q) = shell_block(p - sh1.get_idx(), q - sh2.get_idx(), p - sh1.get_idx(), q - sh2.get_idx());
                screen(q, p) = screen(p, q);
            }
}
