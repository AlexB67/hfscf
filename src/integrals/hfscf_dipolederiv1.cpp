#include "hfscf_dipole.hpp"
#include "hfscf_osrecur.hpp"
#include "hfscf_transform.hpp"

using os_recursion::osrecur2center;
using TRANSFORM::transform;

void DIPOLE::Dipole::compute_contracted_shell_deriv1(std::vector<EigenMatrix<double>>& Dpa, std::vector<EigenMatrix<double>>& Dpb,
                                                     const ShellPair& sp) const
{
    const auto& sh1 = sp.m_s1;
    const auto& sh2 = sp.m_s2;

    EigenMatrix<double> Sx = EigenMatrix<double>(sh2.L() + 2, sh1.L() + 2);
    EigenMatrix<double> Sy = EigenMatrix<double>(sh2.L() + 2, sh1.L() + 2);
    EigenMatrix<double> Sz = EigenMatrix<double>(sh2.L() + 2, sh1.L() + 2);

    const Eigen::Ref<const EigenVector<double>>& alpha1 = sh1.alpha();
    const Eigen::Ref<const EigenVector<double>>& alpha2 = sh2.alpha();
    const Eigen::Ref<const EigenVector<double>>& c1 = sh1.c();
    const Eigen::Ref<const EigenVector<double>>& c2 = sh2.c();
    const auto&id1 = sh1.get_indices();
    const auto&id2 = sh2.get_indices();
    const Vec3D& A = sh1.r();
    const Vec3D& B = sh2.r();
    // mux
    EigenMatrix<double> xxa = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    EigenMatrix<double> xya = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    EigenMatrix<double> xza = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    EigenMatrix<double> xxb = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    EigenMatrix<double> xyb = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    EigenMatrix<double> xzb = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    // muy
    EigenMatrix<double> yxa = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    EigenMatrix<double> yya = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    EigenMatrix<double> yza = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    EigenMatrix<double> yxb = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    EigenMatrix<double> yyb = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    EigenMatrix<double> yzb = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    // muz
    EigenMatrix<double> zxa = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    EigenMatrix<double> zya = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    EigenMatrix<double> zza = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    EigenMatrix<double> zxb = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    EigenMatrix<double> zyb = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());
    EigenMatrix<double> zzb = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());

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

            osrecur2center<double>(Sx, Sy, Sz, sh1.L() + 1, sh2.L() + 1, Rpa, Rpb, gamma12inv);

            for (const auto &i : id1)
            {
                Index l1 = i.l; Index m1 = i.m;  Index n1 = i.n;  Index ci1 = i.ci; 
                for (const auto &j : id2)
                {
                    Index ci2 = j.ci; Index l2 = j.l; Index m2 = j.m; Index n2 = j.n;
                    double d1{0}, d2{0};
// mu_x start
                    /********** XxX **********/
                    // A
                    d1 = Sy(m2, m1) * Sz(n2, n1) * (Sx(l2 + 1, l1 + 1) + Sx(l2, l1 + 1) * B[0]);
                    
                    if (l1) d2 = Sy(m2, m1) * Sz(n2, n1) * (Sx(l2 + 1, l1 - 1) + Sx(l2, l1 - 1) * B[0]);
                      
                    xxa(ci1, ci2) -= (2.0 * alpha1(p1) * d1 - l1 * d2) * pfac;
                    /********** YxX **********/
                    // A
                    d2 = 0;
                    d1 = Sz(n2, n1) * (Sx(l2 + 1, l1) * Sy(m2, m1 + 1) + Sx(l2, l1) * Sy(m2, m1 + 1) * B[0]);
                    
                    if (m1) d2 = Sz(n2, n1) * (Sx(l2 + 1, l1) * Sy(m2, m1 - 1) + Sx(l2, l1) * Sy(m2, m1 - 1) * B[0]);
                    
                    xya(ci1, ci2) -= (2.0 * alpha1(p1) * d1 - m1 * d2) * pfac;
                    /********** ZxX **********/
                    // A
                    d2 = 0;
                    d1 =  Sy(m2, m1) * (Sx(l2 + 1, l1) * Sz(n2, n1 + 1) + Sx(l2, l1) * Sz(n2, n1 + 1) * B[0]);
                    
                    if (n1) d2 = Sy(m2, m1) * (Sx(l2 + 1, l1) * Sz(n2, n1 - 1) + Sx(l2, l1) * Sz(n2, n1 - 1) * B[0]);
                    
                    xza(ci1, ci2) -= (2.0 * alpha1(p1) * d1 - n1 * d2) * pfac;
                    /********** XxX **********/
                    // B
                    d2 = 0;
                    d1 = Sy (m2, m1) * Sz (n2, n1) * (Sx(l2 + 1, l1 + 1) + Sx(l2 + 1, l1) * A[0]);

                    if (l2) d2 = Sy(m2, m1) * (Sx(l2 - 1, l1 + 1) * Sz(n2, n1) + Sx(l2 - 1, l1) * Sz(n2, n1) * A[0]);

                    xxb(ci1, ci2) -= (2.0 * alpha2(p2) * d1 - l2 * d2) * pfac;
                    /********** XxY **********/
                    // B
                    d2 = 0.0;
                    d1 =  Sz(n2, n1) *  Sy(m2 + 1, m1) * (Sx(l2, l1 + 1) + Sx(l2, l1) * A[0]);
                    
                    if (m2) d2 = Sz(n2, n1) * (Sx(l2, l1 + 1) * Sy(m2 - 1, m1) + Sx(l2, l1) * Sy(m2 - 1, m1) * A[0]);
                    
                    xyb(ci1, ci2) -= (2.0 * alpha2(p2) * d1 - m2 * d2) * pfac;
                    /********** XxZ **********/
                    // B
                    d2 = 0.0;
                    d1 = Sy(m2, m1) * (Sx(l2, l1 + 1) * Sz(n2 + 1, n1) + Sx(l2, l1) * Sz(n2 + 1, n1) * A[0]);
                    
                    if (n2) d2 = Sy(m2, m1) * (Sx(l2, l1 + 1) * Sz(n2 - 1, n1) + Sx(l2, l1) * Sz(n2 - 1, n1) * A[0]);

                    xzb(ci1, ci2) -= (2.0 * alpha2(p2) * d1 - n2 * d2) * pfac;
// mu_y start
                    /********** YyX **********/
                    // A
                    d2 = 0;
                    d1 = Sx(l2, l1 + 1) *  Sz(n2, n1) * (Sy(m2 + 1, m1) + Sy(m2, m1) * B[1]);
                    
                    if (l1) d2 = Sx(l2, l1 - 1) *  Sz(n2, n1) * (Sy(m2 + 1, m1) + Sy(m2, m1) * B[1]);
                      
                    yxa(ci1, ci2) -= (2.0 * alpha1(p1) * d1 - l1 * d2) * pfac;
                    /********** YyY **********/
                    // A
                    d2 = 0;
                    d1 = Sx(l2, l1) * Sz(n2, n1) * (Sy(m2 + 1, m1 + 1) + Sy(m2, m1 + 1) * B[1]);
                    
                    if (m1) d2 = Sx(l2, l1) * Sz(n2, n1) * (Sy(m2 + 1, m1 - 1) + Sy(m2, m1 - 1) * B[1]);
                      
                    yya(ci1, ci2) -= (2.0 * alpha1(p1) * d1 - m1 * d2) * pfac;
                    /********** YyZ **********/
                    // A
                    d2 = 0;
                    d1 = Sx(l2, l1) * Sz (n2, n1 + 1) * (Sy (m2 + 1, m1) + Sy(m2, m1) * B[1]);
                    
                    if (n1) d2 = Sx(l2, l1) *  Sz(n2, n1 - 1) * (Sy(m2 + 1, m1) + Sy(m2, m1) * B[1]);
                      
                    yza(ci1, ci2) -= (2.0 * alpha1(p1) * d1 - n1 * d2) * pfac;
                    /********** YyX **********/
                    // B
                    d2 = 0;
                    d1 = Sx(l2 + 1, l1) * Sz (n2, n1) * (Sy(m2, m1 + 1) + Sy(m2, m1) * A[1]);
                    
                    if (l2) d2 = Sx(l2 - 1, l1) * Sz(n2, n1) * (Sy(m2, m1 + 1) + Sy(m2, m1) * A[1]);
                      
                    yxb(ci1, ci2) -= (2.0 * alpha2(p2) * d1 - l2 * d2) * pfac;
                    /********** YyY **********/
                    // B
                    d2 = 0.0;
                    d1 = Sz(n2, n1) * Sx(l2, l1) * (Sy(m2 + 1, m1 + 1) + Sy(m2 + 1, m1) * A[1]);
                    
                    if (m2) d2 = Sx(l2, l1) * Sz(n2, n1) * (Sy(m2 - 1, m1 + 1) + Sy(m2 - 1, m1) * A[1]);

                    yyb(ci1, ci2) -= (2.0 * alpha2(p2) * d1 - m2 * d2) * pfac;
                    /********** YyZ **********/
                    // B
                    d2 = 0.0;
                    d1 = Sx(l2, l1) * (Sy(m2, m1 + 1) * Sz (n2 + 1, n1) + Sy(m2, m1) * Sz(n2 + 1, n1) * A[1]);
                    
                    if (n2) d2 = Sx(l2, l1) * (Sy(m2, m1 + 1) * Sz(n2 - 1, n1) + Sy(m2, m1) * Sz(n2 - 1, n1) * A[1]);

                    yzb(ci1, ci2) -= (2.0 * alpha2(p2) * d1 - n2 * d2) * pfac;
// mu_z start
                    /********** ZzX **********/
                    // A
                    d2 = 0;
                    d1 = Sx(l2, l1 + 1) * Sy(m2, m1) * (Sz(n2 + 1, n1) + Sz(n2, n1) * B[2]);
                    
                    if (l1) d2 = Sx(l2, l1 - 1) * Sy(m2, m1) * (Sz(n2 + 1, n1) + Sz(n2, n1) * B[2]);
                      
                    zxa(ci1, ci2) -= (2.0 * alpha1(p1) * d1 - l1 * d2) * pfac;
                    /********** ZzY **********/
                    // A
                    d2 = 0;
                    d1 = Sx(l2, l1) * Sy(m2, m1 + 1) * (Sz(n2 + 1, n1) + Sz(n2, n1) * B[2]);
                    
                    if (m1) d2 = Sx(l2, l1) * Sy(m2, m1 - 1) * (Sz(n2 + 1, n1) + Sz(n2, n1) * B[2]);
                      
                    zya(ci1, ci2) -= (2.0 * alpha1(p1) * d1 - m1 * d2) * pfac;
                    /********** ZzZ **********/
                    // A
                    d2 = 0;
                    d1 = Sx(l2, l1) * Sy(m2, m1) * (Sz(n2 + 1, n1 + 1) + Sz(n2, n1 + 1) * B[2]);
                    
                    if (n1) d2 = Sx(l2, l1) * Sy(m2, m1) * (Sz(n2 + 1, n1 - 1) + Sz(n2, n1 - 1) * B[2]);
                      
                    zza(ci1, ci2) -= (2.0 * alpha1(p1) * d1 - n1 * d2) * pfac;
                    /********** ZzX **********/
                    // B
                    d2 = 0.0;
                    d1 = Sx(l2 + 1, l1) * Sy(m2, m1) * (Sz (n2, n1 + 1) + Sz(n2, n1) * A[2]);
                    
                    if (l2) d2 = Sx(l2 - 1, l1) * Sy(m2, m1) * (Sz(n2, n1 + 1) + Sz(n2, n1) * A[2]);

                    zxb(ci1, ci2) -= (2.0 * alpha2(p2) * d1 - l2 * d2) * pfac;
                    /********** ZzY **********/
                    // B
                    d2 = 0.0;
                    d1 = Sx(l2, l1) * Sy(m2 + 1, m1) * (Sz (n2, n1 + 1) + Sz(n2, n1) * A[2]); 
                    
                    if (m2) d2 = Sx(l2, l1) * Sy(m2 - 1, m1) * (Sz(n2, n1 + 1) + Sz(n2, n1) * A[2]);

                    zyb(ci1, ci2) -= (2.0 * alpha2(p2) * d1 - m2 * d2) * pfac;
                    /********** ZzZ **********/
                    // B
                    d2 = 0;
                    d1 = Sx(l2, l1) * Sy(m2, m1) * (Sz(n2 + 1, n1 + 1) + Sz(n2 + 1, n1) * A[2]); 
                    
                    if (n2) d2 = Sx(l2, l1) * Sy(m2, m1) * (Sz(n2 - 1, n1 + 1) + Sz(n2 - 1, n1) * A[2]);

                    zzb(ci1, ci2) -= (2.0 * alpha2(p2) * d1 - n2 * d2) * pfac;
                }
            }
        }
    }
    
    // mu_x
    transform(sp, xxa, Dpa[0], pure);
    transform(sp, xya, Dpa[1], pure);
    transform(sp, xza, Dpa[2], pure);
    transform(sp, xxb, Dpb[0], pure);
    transform(sp, xyb, Dpb[1], pure);
    transform(sp, xzb, Dpb[2], pure);
    //mu_y
    transform(sp, yxa, Dpa[3], pure);
    transform(sp, yya, Dpa[4], pure);
    transform(sp, yza, Dpa[5], pure);
    transform(sp, yxb, Dpb[3], pure);
    transform(sp, yyb, Dpb[4], pure);
    transform(sp, yzb, Dpb[5], pure);
    //mu_z
    transform(sp, zxa, Dpa[6], pure);
    transform(sp, zya, Dpa[7], pure);
    transform(sp, zza, Dpa[8], pure);
    transform(sp, zxb, Dpb[6], pure);
    transform(sp, zyb, Dpb[7], pure);
    transform(sp, zzb, Dpb[8], pure);
}
