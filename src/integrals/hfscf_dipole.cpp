#include "hfscf_dipole.hpp"
#include "hfscf_osrecur.hpp"
#include "hfscf_transform.hpp"

using os_recursion::osrecur2center;
using TRANSFORM::transform;

// Electronic contribution with the nuclear center of charge taken into account
// Note you can pass Q_c as a zero vector, but you will have to add the nuclear 
// contribution (i.e. the center of charge vector) afterwards for dipole calculations if needed

void DIPOLE::Dipole::compute_contracted_shell(tensor3d<double>& Dp, const Eigen::Ref<const Vec3D>& Q_c,
                                              const ShellPair& sp) const
{
    const auto& sh1 = sp.m_s1;
    const auto& sh2 = sp.m_s2;

    EigenMatrix<double> Sx = EigenMatrix<double>(sh2.L() + 2, sh1.L() + 2);
    EigenMatrix<double> Sy = EigenMatrix<double>(sh2.L() + 2, sh1.L() + 2);
    EigenMatrix<double> Sz = EigenMatrix<double>(sh2.L() + 2, sh1.L() + 2);

    const Eigen::Ref<const EigenVector<double>>& c1 = sh1.c();
    const Eigen::Ref<const EigenVector<double>>& c2 = sh2.c();
    const auto&id1 = sh1.get_indices();
    const auto&id2 = sh2.get_indices();

    EigenMatrix<double> sx = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    EigenMatrix<double> sy = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    EigenMatrix<double> sz = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());

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

            const double PQc_x = sh1.x() - Q_c(0);
            const double PQc_y = sh1.y() - Q_c(1);
            const double PQc_z = sh1.z() - Q_c(2);

            osrecur2center<double>(Sx, Sy, Sz, sh1.L() + 1, sh2.L() + 1, Rpa, Rpb, gamma12inv);

            for (const auto &i : id1)
            {
                Index l1 = i.l; Index m1 = i.m;  Index n1 = i.n;  Index ci1 = i.ci; 
                for (const auto &j : id2)
                {
                    Index ci2 = j.ci; Index l2 = j.l; Index m2 = j.m; Index n2 = j.n;

                    sx(ci1, ci2) -= pfac * (Sx(l2, l1 + 1) + Sx(l2, l1) * PQc_x) * Sy(m2, m1) * Sz(n2, n1);
                    sy(ci1, ci2) -= pfac * (Sy(m2, m1 + 1) + Sy(m2, m1) * PQc_y) * Sx(l2, l1) * Sz(n2, n1);
                    sz(ci1, ci2) -= pfac * (Sz(n2, n1 + 1) + Sz(n2, n1) * PQc_z) * Sx(l2, l1) * Sy(m2, m1);
                }
            }
        }
    }

    transform(sp, sx, sy, sz, Dp, pure);
}
