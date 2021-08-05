#include "hfscf_osoverlap.hpp"
#include "hfscf_osrecur.hpp"
#include "hfscf_transform.hpp"

using hfscfmath::pi;
using hfscfmath::Cart;
using os_recursion::osrecur2center;
using TRANSFORM::transform;

void OSOVERLAP::OSOverlap::compute_contracted_shell(EigenMatrix<double>& S, EigenMatrix<double>& T, 
                                                    const ShellPair& sp) const
{
    const auto& sh1 = sp.m_s1;
    const auto& sh2 = sp.m_s2;

    EigenMatrix<double> Sx = EigenMatrix<double>(sh2.L() + 3, sh1.L() + 3);
    EigenMatrix<double> Sy = EigenMatrix<double>(sh2.L() + 3, sh1.L() + 3);
    EigenMatrix<double> Sz = EigenMatrix<double>(sh2.L() + 3, sh1.L() + 3);

    EigenMatrix<double> s_block = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    EigenMatrix<double> t_block = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());

    const Eigen::Ref<const EigenVector<double>>& alpha1 = sh1.alpha();
    const Eigen::Ref<const EigenVector<double>>& c1 = sh1.c();
    const Eigen::Ref<const EigenVector<double>>& c2 = sh2.c();
    const auto&id1 = sh1.get_indices();
    const auto&id2 = sh2.get_indices();

    for (Index p1 = 0; p1 < c1.size(); ++p1)
    {
        for (Index p2 = 0; p2 < c2.size(); ++p2)
        {
            const Index id = p1 * c2.size() + p2;
            const Vec3D &Rpa = sp.PA(id);
            const Vec3D &Rpb = sp.PB(id);
            const double gamma12 = sp.gamma_ab(id);
            const double gamma12inv = 1.0 / (2.0 * gamma12);
            const double prefac = sp.pfac(id);

            osrecur2center<double>(Sx, Sy, Sz, sh1.L() + 2, sh2.L() + 2, Rpa, Rpb, gamma12inv);

            for (const auto &i : id1) // ci canonical index
            {
                Index l1 = i.l; Index m1 = i.m;  Index n1 = i.n;  Index ci1 = i.ci; 
                for (const auto &j : id2)
                {
                    Index ci2 = j.ci;
                    Index l2 = j.l; Index m2 = j.m; Index n2 = j.n;

                    s_block(ci1, ci2) += prefac * Sx(l2, l1) * Sy(m2, m1) * Sz(n2, n1);

                    t_block(ci1, ci2) -= 0.5 * prefac *
                    (D2ij(Sx, alpha1(p1), l1, l2) * Sy(m2, m1) * Sz(n2, n1) +
                    D2ij(Sy, alpha1(p1), m1, m2) * Sx(l2, l1) * Sz(n2, n1) +
                    D2ij(Sz, alpha1(p1), n1, n2) * Sy(m2, m1) * Sx(l2, l1));
                }
            }
        }
    }

    transform(sp, s_block, S, pure);
    transform(sp, t_block, T, pure);
}
