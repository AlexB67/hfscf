#include "hfscf_oskinetic.hpp"
#include "hfscf_osrecur.hpp"

using hfscfmath::pi;
using tensormath::tensor3d;
using os_recursion::osrecur2center;
using tensormath::tensor3d;

double OSKINETIC::OSKinetic::kt(double alpha1, double alpha2, 
                                int l1, int m1, int n1, int l2, int m2, int n2, 
                                const Eigen::Ref<const EigenMatrix<double>>& Sx,
                                const Eigen::Ref<const EigenMatrix<double>>& Sy,
                                const Eigen::Ref<const EigenMatrix<double>>& Sz) const
{
    double K1, K2, K3, K4;

    K1 = (l1 == 0 || l2 == 0) ? 0.0 : Sx(l2 - 1, l1 - 1) * Sy(m2, m1) * Sz(n2, n1);
    K2 = Sx(l2 + 1, l1 + 1) * Sy(m2, m1) * Sz(n2, n1);
    K3 = (l2 == 0) ? 0.0 : Sx(l2 - 1, l1 + 1) * Sy(m2, m1) * Sz(n2, n1);
    K4 = (l1 == 0) ? 0.0 : Sx(l2 + 1, l1 - 1) * Sy(m2, m1) * Sz(n2, n1);
    const double Kx = 0.5 * l1 * l2 * K1 + 2.0 * alpha1 * alpha2 * K2 - alpha1 * l2 * K3 - l1 * alpha2 * K4;

    K1 = (m1 == 0 || m2 == 0) ? 0.0 : Sx(l2, l1) * Sy(m2 - 1, m1 - 1) * Sz(n2, n1);
    K2 = Sx(l2, l1) * Sy(m2 + 1, m1 + 1) * Sz(n2, n1);
    K3 = (m2 == 0) ? 0.0 : Sx(l2, l1) * Sy(m2 - 1, m1 + 1) * Sz(n2, n1);
    K4 = (m1 == 0) ? 0.0 : Sx(l2, l1) * Sy(m2 + 1, m1 - 1) * Sz(n2, n1);
    const double Ky = 0.5 * m1 * m2 * K1 + 2.0 * alpha1 * alpha2 * K2 - alpha1 * m2 * K3 - m1 * alpha2 * K4;

    K1 = (n1 == 0 || n2 == 0) ? 0.0 : Sx(l2, l1) * Sy(m2, m1) *  Sz(n2 - 1, n1 - 1);
    K2 = Sx(l2, l1) * Sy(m2, m1) * Sz(n2 + 1, n1 + 1);
    K3 = (n2 == 0) ? 0.0 : Sx(l2, l1) * Sy(m2, m1) * Sz(n2 - 1, n1 + 1);
    K4 = (n1 == 0) ? 0.0 : Sx(l2, l1) * Sy(m2, m1) * Sz(n2 + 1, n1 - 1);
    const double Kz = 0.5 * n1 * n2 * K1 + 2.0 * alpha1 * alpha2 * K2 - alpha1 * n2 * K3 - n1 * alpha2 * K4;

    return (Kx + Ky + Kz);
}

void OSKINETIC::OSKinetic::compute_contracted_shell_deriv2(tensor3d<double>& T,
                                                           const ShellPair& sp) const
{
    const auto& sh1 = sp.m_s1;
    const auto& sh2 = sp.m_s2;

    EigenMatrix<double> Sx = EigenMatrix<double>(sh2.L() + 4, sh1.L() + 4);
    EigenMatrix<double> Sy = EigenMatrix<double>(sh2.L() + 4, sh1.L() + 4);
    EigenMatrix<double> Sz = EigenMatrix<double>(sh2.L() + 4, sh1.L() + 4);

    const Eigen::Ref<const EigenVector<double>>& alpha1 = sh1.alpha();
    const Eigen::Ref<const EigenVector<double>>& alpha2 = sh2.alpha();
    const Eigen::Ref<const EigenVector<double>>& c1 = sh1.c();
    const Eigen::Ref<const EigenVector<double>>& c2 = sh2.c();
    const auto&id1 = sh1.get_indices();
    const auto&id2 = sh2.get_indices();

    EigenMatrix<double> txx = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    EigenMatrix<double> txy = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    EigenMatrix<double> txz = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    EigenMatrix<double> tyy = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    EigenMatrix<double> tyz = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    EigenMatrix<double> tzz = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());

    for (int p1 = 0; p1 < c1.size(); ++p1)
    {
        for (int p2 = 0; p2 < c2.size(); ++p2)
        {
            const Index id = p1 * c2.size() + p2;
            const double gamma12 = sp.gamma_ab(id);
            const double gamma12inv = 1.0 / (2.0 * gamma12);
            const Vec3D& Rpa = sp.PA(id);
            const Vec3D& Rpb = sp.PB(id);
            const double pfac = sp.pfac(id);

            osrecur2center<double>(Sx, Sy, Sz, sh1.L() + 3, sh2.L() + 3, Rpa, Rpb, gamma12inv);

            const double a11 = alpha1(p1) * alpha1(p1);
            const double a1 = alpha1(p1);
            const double a2 = alpha2(p2);

            for (const auto &i : id1)
            {
                int l1 = i.l; int m1 = i.m;  int n1 = i.n;  int ci1 = i.ci; 
                for (const auto &j : id2)
                {
                    int ci2 = j.ci; int l2 = j.l; int m2 = j.m; int n2 = j.n;

                    // Translational invariance a == b centers, only calculate a

                    const double lmn = kt(a1, a2, l1, m1, n1, l2, m2, n2, Sx, Sy, Sz);

                    // XX
                    double K_xx =
                    4.0 * a11 * kt(a1, a2, l1 + 2, m1, n1, l2, m2, n2, Sx, Sy, Sz) 
                    - 2.0 * a1 * (2 * l1 + 1) * lmn;

                    if (l1 > 1) K_xx += l1 * (l1 - 1) * kt(a1, a2, l1 - 2, m1, n1, l2, m2, n2, Sx, Sy, Sz);

                    txx(ci1, ci2) += pfac * K_xx;

                    // XY
                    double K_xy = 4.0 * a11 * kt(a1, a2, l1 + 1, m1 + 1, n1, l2, m2, n2, Sx, Sy, Sz);
                    if (l1) K_xy -= 2.0 * a1 * l1 * kt(a1, a2, l1 - 1, m1 + 1, n1, l2, m2, n2, Sx, Sy, Sz);
                    if (m1) K_xy -= 2.0 * a1 * m1 * kt(a1, a2, l1 + 1, m1 - 1, n1, l2, m2, n2, Sx, Sy, Sz);
                    if (l1 && m1) K_xy += l1 * m1 * kt(a1, a2, l1 - 1, m1 - 1, n1, l2, m2, n2, Sx, Sy, Sz);

                    txy(ci1,  ci2) += pfac * K_xy;

                    // XZ
                    double K_xz = 4.0 * a11 * kt(a1, a2, l1 + 1, m1, n1 + 1, l2, m2, n2, Sx, Sy, Sz);
                    if (l1) K_xz -= 2.0 * a1 * l1 * kt(a1, a2, l1 - 1, m1, n1 + 1, l2, m2, n2, Sx, Sy, Sz);
                    if (n1) K_xz -= 2.0 * a1 * n1 * kt(a1, a2, l1 + 1, m1, n1 - 1, l2, m2, n2, Sx, Sy, Sz);
                    if (l1 && n1) K_xz += l1 * n1 * kt(a1, a2, l1 - 1, m1, n1 - 1, l2, m2, n2, Sx, Sy, Sz);

                    txz(ci1, ci2) += pfac * K_xz;

                    // YY
                    double K_yy =
                    4.0 * a11 * kt(a1, a2, l1, m1 + 2, n1, l2, m2, n2, Sx, Sy, Sz) 
                    - 2.0 * a1 * (2 * m1 + 1) * lmn;

                    if (m1 > 1) K_yy += m1 * (m1 - 1) * kt(a1, a2, l1, m1 - 2, n1, l2, m2, n2, Sx, Sy, Sz);

                    tyy(ci1, ci2) += pfac * K_yy;

                    // YZ
                    double K_yz = 4.0 * a11 * kt(a1, a2, l1, m1 + 1, n1 + 1, l2, m2, n2, Sx, Sy, Sz);
                    if (m1) K_yz -= 2.0 * a1 * m1 * kt(a1, a2, l1, m1 - 1, n1 + 1, l2, m2, n2, Sx, Sy, Sz);
                    if (n1) K_yz -= 2.0 * a1 * n1 * kt(a1, a2, l1, m1 + 1, n1 - 1, l2, m2, n2, Sx, Sy, Sz);
                    if (m1 && n1) K_yz += m1 * n1 * kt(a1, a2, l1, m1 - 1, n1 - 1, l2, m2, n2, Sx, Sy, Sz);

                    tyz(ci1, ci2) += pfac * K_yz;

                    // ZZ
                    double K_zz = 
                    4.0 * a11 * kt(a1, a2, l1, m1, n1 + 2, l2, m2, n2, Sx, Sy, Sz) 
                    - 2.0 * a1 * (2.0 * n1 + 1.0) * lmn;

                    if (n1 > 1) K_zz += n1 * (n1 - 1) * kt(a1, a2, l1, m1, n1 - 2, l2, m2, n2, Sx, Sy, Sz);

                    tzz(ci1, ci2) += pfac * K_zz;
                }
            }
        }
    }

    if (pure) // Spherical basis
    {
        const EigenMatrix<double> tr_left = sh1.get_spherical_form().transpose().eval();
        const EigenMatrix<double> tr_right = sh2.get_spherical_form().eval();

        EigenMatrix<double> txx_pure = tr_left * txx * tr_right;
        EigenMatrix<double> txy_pure = tr_left * txy * tr_right;
        EigenMatrix<double> txz_pure = tr_left * txz * tr_right;
        EigenMatrix<double> tyy_pure = tr_left * tyy * tr_right;
        EigenMatrix<double> tyz_pure = tr_left * tyz * tr_right;
        EigenMatrix<double> tzz_pure = tr_left * tzz * tr_right;

        for (int i = 0; i < sh1.get_sirange(); ++i)
            for (int j = 0; j < sh2.get_sirange(); ++j)
            {
                T(0, sh1.get_ids() + i, sh2.get_ids() + j) = txx_pure(i, j);
                T(0, sh2.get_ids() + j, sh1.get_ids() + i) = T(0, sh1.get_ids() + i, sh2.get_ids() + j);
                T(1, sh1.get_ids() + i, sh2.get_ids() + j) = txy_pure(i, j);
                T(1, sh2.get_ids() + j, sh1.get_ids() + i) = T(1, sh1.get_ids() + i, sh2.get_ids() + j);
                T(2, sh1.get_ids() + i, sh2.get_ids() + j) = txz_pure(i, j);
                T(2, sh2.get_ids() + j, sh1.get_ids() + i) = T(2, sh1.get_ids() + i, sh2.get_ids() + j);
                T(3, sh1.get_ids() + i, sh2.get_ids() + j) = tyy_pure(i, j);
                T(3, sh2.get_ids() + j, sh1.get_ids() + i) = T(3, sh1.get_ids() + i, sh2.get_ids() + j);
                T(4, sh1.get_ids() + i, sh2.get_ids() + j) = tyz_pure(i, j);
                T(4, sh2.get_ids() + j, sh1.get_ids() + i) = T(4, sh1.get_ids() + i, sh2.get_ids() + j);
                T(5, sh1.get_ids() + i, sh2.get_ids() + j) = tzz_pure(i, j);
                T(5, sh2.get_ids() + j, sh1.get_ids() + i) = T(5, sh1.get_ids() + i, sh2.get_ids() + j);
            }
    }
    else // Cartesian basis
    {
        for (int i = 0; i < sh1.get_cirange(); ++i)
            for (int j = 0; j < sh2.get_cirange(); ++j)
            {
                T(0, sh1.get_idx() + i, sh2.get_idx() + j) = txx(i, j);
                T(0, sh2.get_idx() + j, sh1.get_idx() + i) = T(0, sh1.get_idx() + i, sh2.get_idx() + j);
                T(1, sh1.get_idx() + i, sh2.get_idx() + j) = txy(i, j);
                T(1, sh2.get_idx() + j, sh1.get_idx() + i) = T(1, sh1.get_idx() + i, sh2.get_idx() + j);
                T(2, sh1.get_idx() + i, sh2.get_idx() + j) = txz(i, j);
                T(2, sh2.get_idx() + j, sh1.get_idx() + i) = T(2, sh1.get_idx() + i, sh2.get_idx() + j);
                T(3, sh1.get_idx() + i, sh2.get_idx() + j) = tyy(i, j);
                T(3, sh2.get_idx() + j, sh1.get_idx() + i) = T(3, sh1.get_idx() + i, sh2.get_idx() + j);
                T(4, sh1.get_idx() + i, sh2.get_idx() + j) = tyz(i, j);
                T(4, sh2.get_idx() + j, sh1.get_idx() + i) = T(4, sh1.get_idx() + i, sh2.get_idx() + j);
                T(5, sh1.get_idx() + i, sh2.get_idx() + j) = tzz(i, j);
                T(5, sh2.get_idx() + j, sh1.get_idx() + i) = T(5, sh1.get_idx() + i, sh2.get_idx() + j);
            }
    }
}
