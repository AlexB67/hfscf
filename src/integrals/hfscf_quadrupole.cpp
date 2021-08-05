#include "hfscf_quadrupole.hpp"
#include "hfscf_osrecur.hpp"
#include "hfscf_transform.hpp"

using os_recursion::osrecur2center;
using TRANSFORM::transform;

// Electronic quadrupole ints @origin Q_c

void QPOLE::Quadrupole::compute_contracted_shell(std::vector<EigenMatrix<double>>& Qp, const Eigen::Ref<const Vec3D>& Q_c,
                                                 const ShellPair& sp) const
{
    const auto& sh1 = sp.m_s1;
    const auto& sh2 = sp.m_s2;

    EigenMatrix<double> Sx = EigenMatrix<double>(sh2.L() + 3, sh1.L() + 3);
    EigenMatrix<double> Sy = EigenMatrix<double>(sh2.L() + 3, sh1.L() + 3);
    EigenMatrix<double> Sz = EigenMatrix<double>(sh2.L() + 3, sh1.L() + 3);

    const Eigen::Ref<const EigenVector<double>>& c1 = sh1.c();
    const Eigen::Ref<const EigenVector<double>>& c2 = sh2.c();
    const auto&id1 = sh1.get_indices();
    const auto&id2 = sh2.get_indices();

    EigenMatrix<double> sxx = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    EigenMatrix<double> sxy = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    EigenMatrix<double> sxz = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    EigenMatrix<double> syy = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    EigenMatrix<double> syz = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    EigenMatrix<double> szz = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());

    const Vec3D &AC = sh1.r() - Q_c;
    const Vec3D &BC = sh2.r() - Q_c;

    for (Index p1 = 0; p1 < c1.size(); ++p1)
    {
        for (Index p2 = 0; p2 < c2.size(); ++p2)
        {
            const Index id = p1 * c2.size() + p2;
            const Vec3D &Rpa = sp.PA(id);
            const Vec3D &Rpb = sp.PB(id);
            const double gamma12 = sp.gamma_ab(id);
            const double gamma12inv = 1.0 / (2.0 * gamma12);
            const double pfac = sp.pfac(id);

            osrecur2center<double>(Sx, Sy, Sz, sh1.L() + 2, sh2.L() + 2, Rpa, Rpb, gamma12inv);

            for (const auto &i : id1)
            {
                Index l1 = i.l; Index m1 = i.m;  Index n1 = i.n;  Index ci1 = i.ci; 
                for (const auto &j : id2)
                {
                    Index ci2 = j.ci; Index l2 = j.l; Index m2 = j.m; Index n2 = j.n;
                    // XX
                    sxx(ci1, ci2) -=
                    (Sx(l2 + 1, l1 + 1) + Sx(l2, l1 + 1) * BC[0] + Sx(l2 + 1, l1) * AC[0] +
                    Sx(l2, l1) * AC[0] * BC[0]) * Sy(m2, m1) * Sz(n2, n1) * pfac;

                    // XY
                    sxy(ci1, ci2) -= (Sx(l2 + 1, l1) + Sx(l2, l1) * BC[0]) 
                                   * (Sy(m2 + 1, m1) + Sy(m2, m1) * BC[1]) * Sz(n2, n1) * pfac;

                    // XZ
                    sxz(ci1, ci2) -= (Sx(l2 + 1, l1) + Sx(l2, l1) * BC[0]) 
                                   * (Sz(n2 + 1, n1) + Sz(n2, n1) * BC[2]) * Sy(m2, m1) * pfac;                       
                    
                    // YY
                    syy(ci1, ci2) -=
                    (Sy(m2 + 1, m1 + 1) + Sy(m2, m1 + 1) * BC[1] + Sy(m2 + 1, m1) * AC[1] +
                    Sy(m2, m1) * AC[1] * BC[1]) * Sx(l2, l1) * Sz(n2, n1) * pfac;

                    // YZ
                    syz(ci1, ci2) -= (Sy(m2 + 1, m1) + Sy(m2, m1) * BC[1]) 
                                   * (Sz(n2 + 1, n1) + Sz(n2, n1) * BC[2]) * Sx(l2, l1) * pfac;

                    // ZZ
                    szz(ci1, ci2) -=
                    (Sz(n2 + 1, n1 + 1) + Sz(n2, n1 + 1) * BC[2] + Sz(n2 + 1, n1) * AC[2] +
                    Sz(n2, n1) * AC[2] * BC[2]) * Sx(l2, l1) * Sy(m2, m1) * pfac;
                }
            }
        }
    }

    transform(sp, sxx, Qp[0], pure);
    transform(sp, sxy, Qp[1], pure);
    transform(sp, sxz, Qp[2], pure);
    transform(sp, syy, Qp[3], pure);
    transform(sp, syz, Qp[4], pure);
    transform(sp, szz, Qp[5], pure);
}
