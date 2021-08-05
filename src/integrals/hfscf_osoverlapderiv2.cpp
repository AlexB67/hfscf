#include "hfscf_osoverlap.hpp"
#include "hfscf_osrecur.hpp"
using os_recursion::osrecur2center;
using hfscfmath::pi;

void OSOVERLAP::OSOverlap::compute_contracted_shell_deriv2(tensor3d<double>& S,
                                                           const ShellPair& sp) const
{
    const auto& sh1 = sp.m_s1;
    const auto& sh2 = sp.m_s2;

    EigenMatrix<double> Sx = EigenMatrix<double>(sh2.L() + 3, sh1.L() + 3);
    EigenMatrix<double> Sy = EigenMatrix<double>(sh2.L() + 3, sh1.L() + 3);
    EigenMatrix<double> Sz = EigenMatrix<double>(sh2.L() + 3, sh1.L() + 3);

    const Eigen::Ref<const EigenVector<double>>& alpha1 = sh1.alpha();
    const Eigen::Ref<const EigenVector<double>>& alpha2 = sh2.alpha();
    const Eigen::Ref<const EigenVector<double>>& c1 = sh1.c();
    const Eigen::Ref<const EigenVector<double>>& c2 = sh2.c();
    const auto& id1 = sh1.get_indices();
    const auto& id2 = sh2.get_indices();

    EigenMatrix<double> sxx = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    EigenMatrix<double> sxy = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    EigenMatrix<double> sxz = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    EigenMatrix<double> syy = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    EigenMatrix<double> syz = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    EigenMatrix<double> szz = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());

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

            for (const auto &i : id1) // ci canonical index
            {
                Index l1 = i.l; Index m1 = i.m;  Index n1 = i.n;  Index ci1 = i.ci; 
                for (const auto &j : id2)
                {
                    Index ci2 = j.ci; Index l2 = j.l; Index m2 = j.m; Index n2 = j.n;

                    sxx(ci1, ci2) += pfac * D2ij(Sx, alpha1(p1), l1, l2) * Sy(m2, m1) * Sz(n2, n1);// XX
                    sxy(ci1, ci2) -= pfac * D11ij(Sx, Sy, alpha1(p1), alpha2(p2), l1, l2, m1, m2) * Sz(n2, n1);//XY
                    sxz(ci1, ci2) -= pfac * D11ij(Sx, Sz, alpha1(p1), alpha2(p2), l1, l2, n1, n2) * Sy(m2, m1);  // XZ
                    syy(ci1, ci2) += pfac * D2ij(Sy, alpha1(p1), m1, m2) * Sx(l2, l1) * Sz(n2, n1);  // YY
                    syz(ci1, ci2) -= pfac * D11ij(Sy, Sz, alpha1(p1), alpha2(p2), m1, m2, n1, n2) * Sx(l2, l1);  // YZ
                    szz(ci1, ci2) += pfac * D2ij(Sz, alpha1(p1), n1, n2) * Sx(l2, l1) * Sy(m2, m1);  // ZZ
                }
            }
        }
    }

    if (pure) // Spherical basis
    {
        const EigenMatrix<double> tr_left = sh1.get_spherical_form().transpose().eval();
        const EigenMatrix<double> tr_right = sh2.get_spherical_form().eval();

        EigenMatrix<double> sxx_pure = tr_left * sxx * tr_right;
        EigenMatrix<double> sxy_pure = tr_left * sxy * tr_right;
        EigenMatrix<double> sxz_pure = tr_left * sxz * tr_right;
        EigenMatrix<double> syy_pure = tr_left * syy * tr_right;
        EigenMatrix<double> syz_pure = tr_left * syz * tr_right;
        EigenMatrix<double> szz_pure = tr_left * szz * tr_right;

        for (Index i = 0; i < sh1.get_sirange(); ++i)
            for (Index j = 0; j < sh2.get_sirange(); ++j)
            {
                S(0, sh1.get_ids() + i, sh2.get_ids() + j) = sxx_pure(i, j);
                S(0, sh2.get_ids() + j, sh1.get_ids() + i) = S(0, sh1.get_ids() + i, sh2.get_ids() + j);
                S(1, sh1.get_ids() + i, sh2.get_ids() + j) = sxy_pure(i, j);
                S(1, sh2.get_ids() + j, sh1.get_ids() + i) = S(1, sh1.get_ids() + i, sh2.get_ids() + j);
                S(2, sh1.get_ids() + i, sh2.get_ids() + j) = sxz_pure(i, j);
                S(2, sh2.get_ids() + j, sh1.get_ids() + i) = S(2, sh1.get_ids() + i, sh2.get_ids() + j);
                S(3, sh1.get_ids() + i, sh2.get_ids() + j) = syy_pure(i, j);
                S(3, sh2.get_ids() + j, sh1.get_ids() + i) = S(3, sh1.get_ids() + i, sh2.get_ids() + j);
                S(4, sh1.get_ids() + i, sh2.get_ids() + j) = syz_pure(i, j);
                S(4, sh2.get_ids() + j, sh1.get_ids() + i) = S(4, sh1.get_ids() + i, sh2.get_ids() + j);
                S(5, sh1.get_ids() + i, sh2.get_ids() + j) = szz_pure(i, j);
                S(5, sh2.get_ids() + j, sh1.get_ids() + i) = S(5, sh1.get_ids() + i, sh2.get_ids() + j);
            }
    }
    else // Cartesian basis
    {
        for (Index i = 0; i < sh1.get_cirange(); ++i)
            for (Index j = 0; j < sh2.get_cirange(); ++j)
            {
                S(0, sh1.get_idx() + i, sh2.get_idx() + j) = sxx(i, j);
                S(0, sh2.get_idx() + j, sh1.get_idx() + i) = S(0, sh1.get_idx() + i, sh2.get_idx() + j);
                S(1, sh1.get_idx() + i, sh2.get_idx() + j) = sxy(i, j);
                S(1, sh2.get_idx() + j, sh1.get_idx() + i) = S(1, sh1.get_idx() + i, sh2.get_idx() + j);
                S(2, sh1.get_idx() + i, sh2.get_idx() + j) = sxz(i, j);
                S(2, sh2.get_idx() + j, sh1.get_idx() + i) = S(2, sh1.get_idx() + i, sh2.get_idx() + j);
                S(3, sh1.get_idx() + i, sh2.get_idx() + j) = syy(i, j);
                S(3, sh2.get_idx() + j, sh1.get_idx() + i) = S(3, sh1.get_idx() + i, sh2.get_idx() + j);
                S(4, sh1.get_idx() + i, sh2.get_idx() + j) = syz(i, j);
                S(4, sh2.get_idx() + j, sh1.get_idx() + i) = S(4, sh1.get_idx() + i, sh2.get_idx() + j);
                S(5, sh1.get_idx() + i, sh2.get_idx() + j) = szz(i, j);
                S(5, sh2.get_idx() + j, sh1.get_idx() + i) = S(5, sh1.get_idx() + i, sh2.get_idx() + j);
            }
    }
}
