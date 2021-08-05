#include "hfscf_oskinetic.hpp"
#include "hfscf_osrecur.hpp"

using hfscfmath::pi;
using os_recursion::osrecur2center;
using OSOVERLAP::dS;

void OSKINETIC::OSKinetic::compute_contracted_shell_deriv1(tensor3d<double>& T,
                                                           const ShellPair& sp) const
{
    const auto& sh1 = sp.m_s1;
    const auto& sh2 = sp.m_s2;

    EigenMatrix<double> Sx = EigenMatrix<double>(sh2.L() + 4, sh1.L() + 4);
    EigenMatrix<double> Sy = EigenMatrix<double>(sh2.L() + 4, sh1.L() + 4);
    EigenMatrix<double> Sz = EigenMatrix<double>(sh2.L() + 4, sh1.L() + 4);

    EigenMatrix<double> tx_block = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    EigenMatrix<double> ty_block = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    EigenMatrix<double> tz_block = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());

    const Eigen::Ref<const EigenVector<double>>& alpha1 = sh1.alpha();
    const Eigen::Ref<const EigenVector<double>>& alpha2 = sh2.alpha();
    const Eigen::Ref<const EigenVector<double>>& c1 = sh1.c();
    const Eigen::Ref<const EigenVector<double>>& c2 = sh2.c();
    const auto&id1 = sh1.get_indices();
    const auto&id2 = sh2.get_indices();

    for (Index p1 = 0; p1 < c1.size(); ++p1)
    {
        for (Index p2 = 0; p2 < c2.size(); ++p2)
        {
            const Index id = p1 * c2.size() + p2;
            const double gamma12 = sp.gamma_ab(id);
            const double gamma12inv = 1.0 / (2.0 * gamma12);
            const Vec3D& Rpa = sp.PA(id);
            const Vec3D& Rpb = sp.PB(id);
            const double pfac = sp.pfac(id);
            const double B = -2.0 * std::pow(alpha2(p2), 2.0);

            osrecur2center<double>(Sx, Sy, Sz, sh1.L() + 3, sh2.L() + 3, Rpa, Rpb, gamma12inv);

            for (const auto &i : id1) // ci canonical index
            {
                Index l1 = i.l; Index m1 = i.m;  Index n1 = i.n;  Index ci1 = i.ci; 
                for (const auto &j : id2)
                {
                    Index ci2 = j.ci; Index l2 = j.l; Index m2 = j.m; Index n2 = j.n;

                    const double Ax = (2 * l2 + 1) * alpha2(p2);
                    const double Ay = (2 * m2 + 1) * alpha2(p2);
                    const double Az = (2 * n2 + 1) * alpha2(p2);
                    const double Cx = -0.5 * l2 * (l2 - 1);
                    const double Cy = -0.5 * m2 * (m2 - 1);
                    const double Cz = -0.5 * n2 * (n2 - 1);

                    // dTdX
                    const double dSl = dS(Sx, alpha1(p1), l1, l2);

                    double dTx = dSl * Ax + dS(Sx, alpha1(p1), l1, l2 + 2) * B;
                    if (l2 > 1) dTx += dS(Sx, alpha1(p1), l1, l2 - 2) * Cx;
                    dTx *= Sy(m2, m1) * Sz(n2, n1);

                    double dTy = Sy(m2, m1) * Ay + Sy(m2 + 2, m1) * B;
                    if (m2 > 1) dTy += Sy(m2 - 2, m1) * Cy;
                    dTy *= dSl * Sz(n2, n1);

                    double dTz = Sz(n2, n1) * Az + Sz(n2 + 2, n1) * B;
                    if (n2 > 1) dTz += Sz(n2 - 2, n1) * Cz;
                    dTz *= dSl * Sy(m2, m1);

                    tx_block(ci1, ci2) += pfac * (dTx + dTy + dTz);

                    // dTdY
                    const double dSm = dS(Sy, alpha1(p1), m1, m2);

                    dTy = dSm * Ay + dS(Sy, alpha1(p1), m1, m2 + 2) * B;
                    if (m2 > 1) dTy += dS(Sy, alpha1(p1), m1, m2 - 2) * Cy;
                    dTy *= Sx(l2, l1) * Sz(n2, n1);

                    dTx = Sx(l2, l1) * Ax + Sx(l2 + 2, l1) * B;
                    if (l2 > 1) dTx += Sx(l2 - 2, l1) * Cx;
                    dTx *= dSm * Sz(n2, n1);

                    dTz = Sz(n2, n1) * Az + Sz(n2 + 2, n1) * B;
                    if (n2 > 1) dTz += Sz(n2 - 2, n1) * Cz;
                    dTz *= dSm * Sx(l2, l1);

                    ty_block(ci1, ci2) += pfac * (dTx + dTy + dTz);

                    // dTdz
                    const double dSn = dS(Sz, alpha1(p1), n1, n2);

                    dTz = dSn * Az + dS(Sz, alpha1(p1), n1, n2 + 2) * B;
                    if (n2 > 1) dTz += dS(Sz, alpha1(p1), n1, n2 - 2) * Cz;
                    dTz *= Sx(l2, l1) * Sy(m2, m1);

                    dTx = Sx(l2, l1) * Ax + Sx(l2 + 2, l1) * B;
                    if (l2 > 1) dTx += Sx(l2 - 2, l1) * Cx;
                    dTx *= dSn * Sy(m2, m1);

                    dTy = Sy(m2, m1) * Ay + Sy(m2 + 2, m1) * B;
                    if (m2 > 1) dTy += Sy(m2 - 2, m1) * Cy;
                    dTy *= dSn * Sx(l2, l1);

                    tz_block(ci1, ci2) += pfac * (dTx + dTy + dTz);
                }
            }
        }
    }

    if (pure) // Spherical basis
    {
        const EigenMatrix<double> tr_left = sh1.get_spherical_form().transpose().eval();
        const EigenMatrix<double> tr_right = sh2.get_spherical_form().eval();

        EigenMatrix<double> tx_pure_block = tr_left * tx_block * tr_right;
        EigenMatrix<double> ty_pure_block = tr_left * ty_block * tr_right;
        EigenMatrix<double> tz_pure_block = tr_left * tz_block * tr_right;


        for (Index i = 0; i < sh1.get_sirange(); ++i)
            for (Index j = 0; j < sh2.get_sirange(); ++j)
            {
                T(0, sh1.get_ids() + i, sh2.get_ids() + j) = tx_pure_block(i, j);
                T(0, sh2.get_ids() + j, sh1.get_ids() + i) = T(0, sh1.get_ids() + i, sh2.get_ids() + j);
                T(1, sh1.get_ids() + i, sh2.get_ids() + j) = ty_pure_block(i, j);
                T(1, sh2.get_ids() + j, sh1.get_ids() + i) = T(1, sh1.get_ids() + i, sh2.get_ids() + j);
                T(2, sh1.get_ids() + i, sh2.get_ids() + j) = tz_pure_block(i, j);
                T(2, sh2.get_ids() + j, sh1.get_ids() + i) = T(2, sh1.get_ids() + i, sh2.get_ids() + j);
            }
    }
    else // Cartesian basis
    {
        for (Index i = 0; i < sh1.get_cirange(); ++i)
            for (Index j = 0; j < sh2.get_cirange(); ++j)
            {
                T(0, sh1.get_idx() + i, sh2.get_idx() + j) = tx_block(i, j);
                T(0, sh2.get_idx() + j, sh1.get_idx() + i) = T(0, sh1.get_idx() + i, sh2.get_idx() + j);
                T(1, sh1.get_idx() + i, sh2.get_idx() + j) = ty_block(i, j);
                T(1, sh2.get_idx() + j, sh1.get_idx() + i) = T(1, sh1.get_idx() + i, sh2.get_idx() + j);
                T(2, sh1.get_idx() + i, sh2.get_idx() + j) = tz_block(i, j);
                T(2, sh2.get_idx() + j, sh1.get_idx() + i) = T(2, sh1.get_idx() + i, sh2.get_idx() + j);
            }
    }
}
