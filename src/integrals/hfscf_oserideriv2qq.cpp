#include "hfscf_oseri.hpp"
#include "../math/hfscf_math.hpp"
#include "../math/gamma.hpp"
#include "hfscf_osrecur.hpp"
#include "hfscf_transform.hpp"

using hfscfmath::rab2;
using hfscfmath::pi;
using hfscfmath::gpc;
using hfscfmath::PI_25;
using extramath::F_nu;
using tensormath::tensor5d;
using os_recursion::osrecur;
using os_recursion::osrecurl123;
using os_recursion::osrecurl4n;
using os_recursion::osrecurl13;
using os_recursion::osrecurl4;
using os_recursion::osrecurl4s1;
using os_recursion::osrecurl4s2;
using os_recursion::osrecurl2;
using TRANSFORM::transform;

// ADx by trans invariance -(AAx + ABx + ACx)
// DDx by trans invariance -(ADx + BDx + CDx)
// BCx by trans invariance -(ACx + CCx + CDx)
// BBx by invariance = AAx + 2 * ACx + 2 * ADx + CCx + 2 * CDx + DDx

void ERIOS::Erios::compute_contracted_shell_quartet_deriv2_qq(symm4dTensor<double>& dV2daa,
                                                              symm4dTensor<double>& dV2dab,
                                                              symm4dTensor<double>& dV2dac,
                                                              symm4dTensor<double>& dV2dbd,
                                                              symm4dTensor<double>& dV2dcc,
                                                              symm4dTensor<double>& dV2dcd,
                                                              const ShellPair& s12, const ShellPair& s34,
                                                              const int cart2) const
{
    if (cart2 == 0) 
        compute_contracted_shell_quartet_deriv2_xx(dV2daa, dV2dab, dV2dac, dV2dbd, dV2dcc, dV2dcd, s12, s34);
    else if (cart2 == 1) 
        compute_contracted_shell_quartet_deriv2_yy(dV2daa, dV2dab, dV2dac, dV2dbd, dV2dcc, dV2dcd, s12, s34);
    else if (cart2 == 2) 
        compute_contracted_shell_quartet_deriv2_zz(dV2daa, dV2dab, dV2dac, dV2dbd, dV2dcc, dV2dcd, s12, s34);
    else
    {
        std::cout << "\n\n  Error: Invalid coordinate request for deriv2 eri.\n\n";
        exit(EXIT_FAILURE);
    }

}

void ERIOS::Erios::compute_contracted_shell_quartet_deriv2_xx(symm4dTensor<double>& dV2daa,
                                                              symm4dTensor<double>& dV2dab,
                                                              symm4dTensor<double>& dV2dac,
                                                              symm4dTensor<double>& dV2dbd,
                                                              symm4dTensor<double>& dV2dcc,
                                                              symm4dTensor<double>& dV2dcd,
                                                              const ShellPair& s12, const ShellPair& s34) const
{
    const auto& sh1 = s12.m_s1;
    const auto& sh2 = s12.m_s2;
    const auto& sh3 = s34.m_s1;
    const auto& sh4 = s34.m_s2;

    Index am1 = sh1.get_cirange();
    Index am2 = sh2.get_cirange();
    Index am3 = sh3.get_cirange();
    Index am4 = sh4.get_cirange();

    tensor4d1234<double> block_aa = tensor4d1234<double>(am1, am2, am3, am4);
    tensor4d1234<double> block_ab = tensor4d1234<double>(am1, am2, am3, am4);
    tensor4d1234<double> block_ac = tensor4d1234<double>(am1, am2, am3, am4);
    tensor4d1234<double> block_bd = tensor4d1234<double>(am1, am2, am3, am4);

    Index nmax  = 3 * (sh4.L() + sh3.L() + sh2.L() + sh1.L()) + 2;
    Index tumax = 2 * (sh4.L() + sh3.L() + sh2.L() + sh1.L()) + 2;
    Index tmax  = 1 * (sh4.L() + sh3.L() + sh2.L() + sh1.L()) + 2;
    tensor5d<double> Jnx, Jny, Jnz;
    // Enough to hold AA AB AC
    Jnx = tensor5d<double>(sh4.L() + 2, sh3.L() + sh4.L() + 2, sh2.L() + 2, sh1.L() + sh2.L() + 3, tmax + 1);
    Jny = tensor5d<double>(sh4.L() + 1, sh3.L() + sh4.L() + 1, sh2.L() + 1, sh1.L() + 1, tumax + 1);
    Jnz = tensor5d<double>(sh4.L() + 1, sh3.L() + 1, sh2.L() + 1, sh1.L() + 1, nmax + 1);

    const Eigen::Ref<const EigenVector<double>>& alpha1 = sh1.alpha();
    const Eigen::Ref<const EigenVector<double>>& alpha2 = sh2.alpha();
    const Eigen::Ref<const EigenVector<double>>& alpha3 = sh3.alpha();
    const Eigen::Ref<const EigenVector<double>>& alpha4 = sh4.alpha();
    const Eigen::Ref<const EigenVector<double>>& c1 = sh1.c();
    const Eigen::Ref<const EigenVector<double>>& c2 = sh2.c();
    const Eigen::Ref<const EigenVector<double>>& c3 = sh3.c();
    const Eigen::Ref<const EigenVector<double>>& c4 = sh4.c();
    const std::vector<BASIS::idq>& id1 = sh1.get_indices();
    const std::vector<BASIS::idq>& id2 = sh2.get_indices(); 
    const std::vector<BASIS::idq>& id3 = sh3.get_indices(); 
    const std::vector<BASIS::idq>& id4 = sh4.get_indices();

    const double CDy = s34.AB(1);
    const double ABx = s12.AB(0);
    const double CDx = s34.AB(0);
    const double precut = cutoff / 10000.0;

    for (Index p1 = 0; p1 < c1.size(); ++p1)
        for (Index p2 = 0; p2 < c2.size(); ++p2)
        {
            const Index id12 = p1 * c2.size() + p2;
            const double gamma_ab = s12.gamma_ab(id12);
            const double gamma_ab2 = gamma_ab + gamma_ab;
            const Vec3D& P = s12.P(id12); 
            const Vec3D& PA = s12.PA(id12);
            const Vec3D& PB = s12.PB(id12);

            for (Index p3 = 0; p3 < c3.size(); ++p3)
                for (Index p4 = 0; p4 < c4.size(); ++p4)
                {
                    const Index id34 = p3 * c4.size() + p4;
                    const Vec3D& Q = s34.P(id34);
                    const Vec3D& PQ = P - Q;
                    const Vec3D& QC = s34.PA(id34);
                    const Vec3D& QD = s34.PB(id34);
                    const double rpq_2 = rab2(P, Q);
                    const double gamma_cd = s34.gamma_ab(id34);
                    const double gamma_cd2 = gamma_cd + gamma_cd;
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
                        Jnz(0, 0, 0, 0, nmax) = F_nu(static_cast<double>(nmax), T);
                        // Downward recursion for Boys stability 
                        for (Index n = nmax - 1; n >= 0; --n)
                            Jnz(0, 0, 0, 0, n) = (1.0 / (2.0 * n + 1.0)) * (2.0 * T * Jnz(0, 0, 0, 0, n + 1) + expT);
                    }
                    else
                    {
                        Jnz(0, 0, 0, 0, nmax) = 1.0;
                        for (Index n = nmax - 1; n >= 0; --n)
                            Jnz(0, 0, 0, 0, n) = 1.0 / (2.0 * n + 1.0);
                    }
                    
                    osrecur<double>(PQ(2), PA(2), PB(2), QC(2), QD(2), gamma_ab2, gamma_cd2, 
                                    p_r, q_r, sh1.L(), sh2.L(), sh3.L(), sh4.L(), nmax, Jnz);
                    

// ********* build shell block start ********** move to func
const auto build_shell_quartet = [&]() noexcept
{
    const double a411 = 4.0 * alpha1(p1) * alpha1(p1);
    const double a412 = 4.0 * alpha1(p1) * alpha2(p2);
    const double a413 = 4.0 * alpha1(p1) * alpha3(p3);
    const double a424 = 4.0 * alpha2(p2) * alpha4(p4);
    const double a21  = 2.0 * alpha1(p1);
    const double a22  = 2.0 * alpha2(p2);
    const double a23  = 2.0 * alpha3(p3);
    const double a24  = 2.0 * alpha4(p4);

    for (const auto& i : id1)
    {
        Index idx1 = sh1.get_idx() + i.ci;

        for (const auto &j : id2)
        {
           Index idx2 = sh2.get_idx() + j.ci;

           if (!pure && idx1 < idx2) continue;
           bool idx12 = idx1 == idx2;

            for (const auto& k : id3)
            {
                Index idx3 = sh3.get_idx() + k.ci;
                bool idx23 = idx2 == idx3;

                for (const auto& l : id4)
                {
                    Index idx4 = sh4.get_idx() + l.ci;
                    
                    if (!pure && idx3 < idx4) continue;
                    else if(idx12 && idx23 && idx3 == idx4) continue; // always zero (all centers)

                    Index n_max = i.l + j.l + k.l + l.l + i.m + j.m + k.m + l.m + 2;  // current nmax
                                                                                    // n1 n2 n3 n4 done
                    double tmp = 0;                                                                
                    for (Index n = 0; n <= n_max; ++n)  // z to y
                    {
                        Jny(0, 0, 0, 0, n) = Jnz(l.n, k.n, j.n, i.n, n);
                        tmp += fabs(Jnz(l.n, k.n, j.n, i.n, n));
                    }

                    if (tmp < cutoff) continue;
                    
                    /******* AA ********/
                    osrecurl123<double>(PQ(1), QC(1), PA(1), PB(1), gamma_ab2, gamma_cd2, p_r, q_r, i.m,
                                    j.m, k.m + l.m, n_max, Jny);
                    osrecurl4n<double>(CDy, i.m, j.m, k.m, l.m, n_max, Jny);
                    
                    n_max -= i.m + j.m + k.m + l.m;

                    for (Index n = 0; n <= n_max; ++n)
                    {
                        Jnx(0, 0, 0, 0, n) = Jny(l.m, k.m, j.m, i.m, n);
                        tmp += fabs(Jny(l.m, k.m, j.m, i.m, n));
                    }

                    if (tmp < cutoff) continue;

                    osrecurl13<double>(PQ(0), QC(0), PA(0), gamma_ab2, gamma_cd2, p_r, q_r, 
                                      i.l + j.l + 2, k.l + l.l, n_max, Jnx);
                    osrecurl2<double>(ABx, i.l + 2, j.l, k.l, l.l, Jnx);
                    osrecurl4s1<double>(CDx, i.l + 2, j.l, k.l, l.l, Jnx);

                    block_aa(i.ci, j.ci, k.ci, l.ci) += (a411 * Jnx(l.l, k.l, j.l, i.l + 2, 0)
                   - a21 * (2.0 * i.l + 1.0) * Jnx(l.l, k.l, j.l, i.l, 0)) * pfac;

                    if (i.l > 1) block_aa(i.ci, j.ci, k.ci, l.ci) += 
                        i.l * (i.l - 1) * Jnx(l.l, k.l, j.l, i.l - 2, 0) * pfac;

                    /******* AB ********/
                    osrecurl2<double>(ABx, i.l + 1, j.l + 1, k.l, l.l, Jnx);
                    osrecurl4<double>(CDx, i.l + 1, j.l + 1, k.l, l.l, Jnx);

                    block_ab(i.ci, j.ci, k.ci, l.ci) += a412 * Jnx(l.l, k.l, j.l + 1, i.l + 1, 0) * pfac;

                    if (j.l) 
                        block_ab(i.ci, j.ci, k.ci, l.ci) -= a21 * j.l * Jnx(l.l, k.l, j.l - 1, i.l + 1, 0) * pfac;

                    if (i.l)
                        block_ab(i.ci, j.ci, k.ci, l.ci) -= a22 * i.l * Jnx(l.l, k.l, j.l + 1, i.l - 1, 0) * pfac;

                    if (i.l && j.l) 
                        block_ab(i.ci, j.ci, k.ci, l.ci) += i.l * j.l * Jnx(l.l, k.l, j.l - 1, i.l - 1, 0) * pfac;
                    
                    /******* AC ********/
                    osrecurl13<double>(PQ(0), QC(0), PA(0), gamma_ab2, gamma_cd2, p_r, q_r, 
                                      i.l + j.l + 1, k.l + l.l + 1, n_max, Jnx);
                    osrecurl2<double>(ABx, i.l + 1, j.l, k.l + 1, l.l, Jnx);
                    osrecurl4s1<double>(CDx, i.l + 1, j.l, k.l + 1, l.l, Jnx);

                    block_ac(i.ci, j.ci, k.ci, l.ci) += a413 * Jnx(l.l, k.l + 1, j.l, i.l + 1, 0) * pfac;

                    if (k.l) 
                        block_ac(i.ci, j.ci, k.ci, l.ci) -= a21 * k.l * Jnx(l.l, k.l - 1, j.l, i.l + 1, 0) * pfac;

                    if (i.l)
                        block_ac(i.ci, j.ci, k.ci, l.ci) -= a23 * i.l * Jnx(l.l, k.l + 1, j.l, i.l - 1, 0) * pfac;

                    if (i.l && k.l) 
                        block_ac(i.ci, j.ci, k.ci, l.ci) += i.l * k.l * Jnx(l.l, k.l - 1, j.l, i.l - 1, 0) * pfac;
                    
                    /******* BD ********/
                    osrecurl2<double>(ABx, i.l, j.l + 1, k.l, l.l + 1, Jnx);
                    osrecurl4s2<double>(CDx, i.l, j.l + 1, k.l, l.l + 1, Jnx);

                    block_bd(i.ci, j.ci, k.ci, l.ci) += a424 * Jnx(l.l + 1, k.l, j.l + 1, i.l, 0) * pfac;

                    if (l.l) 
                        block_bd(i.ci, j.ci, k.ci, l.ci) -= a22 * l.l * Jnx(l.l - 1, k.l, j.l + 1, i.l, 0) * pfac;

                    if (j.l)
                        block_bd(i.ci, j.ci, k.ci, l.ci) -= a24 * j.l * Jnx(l.l + 1, k.l, j.l - 1, i.l, 0) * pfac;

                    if (j.l && l.l)
                        block_bd(i.ci, j.ci, k.ci, l.ci) += j.l * l.l * Jnx(l.l - 1, k.l, j.l - 1, i.l, 0) * pfac;
                }
            }
        }
    }
};
// ********* build shell block end **********        
                    build_shell_quartet();
                }
        }

    shell_to_basis_deriv2_qq(s12, s34, block_aa, block_ab, block_ac, block_bd,
                             dV2daa, dV2dab, dV2dac, dV2dbd, dV2dcc, dV2dcd);
}


void ERIOS::Erios::compute_contracted_shell_quartet_deriv2_yy(symm4dTensor<double>& dV2daa,
                                                              symm4dTensor<double>& dV2dab,
                                                              symm4dTensor<double>& dV2dac,
                                                              symm4dTensor<double>& dV2dbd,
                                                              symm4dTensor<double>& dV2dcc,
                                                              symm4dTensor<double>& dV2dcd,
                                                              const ShellPair& s12, const ShellPair& s34) const
{
    const auto& sh1 = s12.m_s1;
    const auto& sh2 = s12.m_s2;
    const auto& sh3 = s34.m_s1;
    const auto& sh4 = s34.m_s2;

    Index am1 = sh1.get_cirange();
    Index am2 = sh2.get_cirange();
    Index am3 = sh3.get_cirange();
    Index am4 = sh4.get_cirange();

    tensor4d1234<double> block_aa = tensor4d1234<double>(am1, am2, am3, am4);
    tensor4d1234<double> block_ab = tensor4d1234<double>(am1, am2, am3, am4);
    tensor4d1234<double> block_ac = tensor4d1234<double>(am1, am2, am3, am4);
    tensor4d1234<double> block_bd = tensor4d1234<double>(am1, am2, am3, am4);

    Index nmax  = 3 * (sh4.L() + sh3.L() + sh2.L() + sh1.L()) + 2;
    Index tumax = 2 * (sh4.L() + sh3.L() + sh2.L() + sh1.L()) + 2;
    Index umax  = 1 * (sh4.L() + sh3.L() + sh2.L() + sh1.L()) + 2;
    // Enough to hold AA AB AC
    tensor5d<double> Jnx = tensor5d<double>(sh4.L() + 1, sh3.L() + sh4.L() + 1, sh2.L() + 1, sh1.L() + 1, tumax + 1);
    tensor5d<double> Jny = tensor5d<double>(sh4.L() + 2, sh3.L() + sh4.L() + 2, sh2.L() + 2, 
                                            sh1.L() + sh2.L() + 3, umax + 1);
    tensor5d<double> Jnz = tensor5d<double>(sh4.L() + 1, sh3.L() + 1, sh2.L() + 1, sh1.L() + 1, nmax + 1);

    const Eigen::Ref<const EigenVector<double>>& alpha1 = sh1.alpha();
    const Eigen::Ref<const EigenVector<double>>& alpha2 = sh2.alpha();
    const Eigen::Ref<const EigenVector<double>>& alpha3 = sh3.alpha();
    const Eigen::Ref<const EigenVector<double>>& alpha4 = sh4.alpha();
    const Eigen::Ref<const EigenVector<double>>& c1 = sh1.c();
    const Eigen::Ref<const EigenVector<double>>& c2 = sh2.c();
    const Eigen::Ref<const EigenVector<double>>& c3 = sh3.c();
    const Eigen::Ref<const EigenVector<double>>& c4 = sh4.c();
    const std::vector<BASIS::idq>& id1 = sh1.get_indices();
    const std::vector<BASIS::idq>& id2 = sh2.get_indices(); 
    const std::vector<BASIS::idq>& id3 = sh3.get_indices(); 
    const std::vector<BASIS::idq>& id4 = sh4.get_indices();

    const double CDx = s34.AB(0);
    const double CDy = s34.AB(1);
    const double ABy = s12.AB(1);
    const double precut = cutoff / 10000.0;

    for (Index p1 = 0; p1 < c1.size(); ++p1)
        for (Index p2 = 0; p2 < c2.size(); ++p2)
        {
            const Index id12 = p1 * c2.size() + p2;
            const double gamma_ab = s12.gamma_ab(id12);
            const double gamma_ab2 = gamma_ab + gamma_ab;
            const Vec3D& P = s12.P(id12); 
            const Vec3D& PA = s12.PA(id12);
            const Vec3D& PB = s12.PB(id12);

            for (Index p3 = 0; p3 < c3.size(); ++p3)
                for (Index p4 = 0; p4 < c4.size(); ++p4)
                {
                    const Index id34 = p3 * c4.size() + p4;
                    const Vec3D& Q = s34.P(id34);
                    const Vec3D& PQ = P - Q;
                    const Vec3D& QC = s34.PA(id34);
                    const Vec3D& QD = s34.PB(id34);
                    const double rpq_2 = rab2(P, Q);
                    const double gamma_cd = s34.gamma_ab(id34);
                    const double gamma_cd2 = gamma_cd + gamma_cd;
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
                        Jnz(0, 0, 0, 0, nmax) = F_nu(static_cast<double>(nmax), T);
                        // Downward recursion for Boys stability 
                        for (Index n = nmax - 1; n >= 0; --n)
                            Jnz(0, 0, 0, 0, n) = (1.0 / (2.0 * n + 1.0)) * (2.0 * T * Jnz(0, 0, 0, 0, n + 1) + expT);
                    }
                    else
                    {
                        Jnz(0, 0, 0, 0, nmax) = 1.0;
                        for (Index n = nmax - 1; n >= 0; --n)
                            Jnz(0, 0, 0, 0, n) = 1.0 / (2.0 * n + 1.0);
                    }
                    
                    osrecur<double>(PQ(2), PA(2), PB(2), QC(2), QD(2), gamma_ab2, gamma_cd2, 
                                    p_r, q_r, sh1.L(), sh2.L(), sh3.L(), sh4.L(), nmax, Jnz);
                    

// ********* build shell block start ********** move to func
const auto build_shell_quartet = [&]() noexcept
{
    const double a411 = 4.0 * alpha1(p1) * alpha1(p1);
    const double a412 = 4.0 * alpha1(p1) * alpha2(p2);
    const double a413 = 4.0 * alpha1(p1) * alpha3(p3);
    const double a424 = 4.0 * alpha2(p2) * alpha4(p4);
    const double a21  = 2.0 * alpha1(p1);
    const double a22  = 2.0 * alpha2(p2);
    const double a23  = 2.0 * alpha3(p3);
    const double a24  = 2.0 * alpha4(p4);

    for (const auto& i : id1)
    {
        Index idx1 = sh1.get_idx() + i.ci;

        for (const auto &j : id2)
        {
           Index idx2 = sh2.get_idx() + j.ci;

           if (!pure && idx1 < idx2) continue;
           bool idx12 = idx1 == idx2;

            for (const auto& k : id3)
            {
                Index idx3 = sh3.get_idx() + k.ci;
                bool idx23 = idx2 == idx3;

                for (const auto& l : id4)
                {
                    Index idx4 = sh4.get_idx() + l.ci;
                    
                    if (!pure && idx3 < idx4) continue;
                    else if(idx12 && idx23 && idx3 == idx4) continue; // always zero (all centers)

                    Index n_max = i.l + j.l + k.l + l.l + i.m + j.m + k.m + l.m + 2;  // current nmax
                                                                                    // n1 n2 n3 n4 done
                    double tmp = 0;
                    for (Index n = 0; n <= n_max; ++n)  // z to x
                    {
                        Jnx(0, 0, 0, 0, n) = Jnz(l.n, k.n, j.n, i.n, n);
                        tmp += fabs(Jnz(l.n, k.n, j.n, i.n, n));
                    }

                    if (tmp < cutoff) continue;
                    
                    /******* AA ********/
                    osrecurl123<double>(PQ(0), QC(0), PA(0), PB(0), gamma_ab2, gamma_cd2, p_r, q_r, i.l,
                                        j.l, k.l + l.l, n_max, Jnx);
                    osrecurl4n<double>(CDx, i.l, j.l, k.l, l.l, n_max, Jnx);
                    
                    n_max -= i.l + j.l + k.l + l.l;

                    for (Index n = 0; n <= n_max; ++n)
                    {
                        Jny(0, 0, 0, 0, n) = Jnx(l.l, k.l, j.l, i.l, n);
                        tmp += fabs(Jnx(l.l, k.l, j.l, i.l, n));
                    }

                    if (tmp < cutoff) continue;

                    osrecurl13<double>(PQ(1), QC(1), PA(1), gamma_ab2, gamma_cd2, p_r, q_r, i.m +
                                    j.m + 2, k.m + l.m, n_max, Jny);
                    osrecurl2<double>(ABy, i.m + 2, j.m, k.m, l.m, Jny);
                    osrecurl4s1<double>(CDy, i.m + 2, j.m, k.m, l.m, Jny);

                    block_aa(i.ci, j.ci, k.ci, l.ci) += (a411 * Jny(l.m, k.m, j.m, i.m + 2, 0)
                   - a21 * (2.0 * i.m + 1.0) * Jny(l.m, k.m, j.m, i.m, 0)) * pfac;

                    if (i.m > 1) block_aa(i.ci, j.ci, k.ci, l.ci) += 
                        i.m * (i.m - 1) * Jny(l.m, k.m, j.m, i.m - 2, 0) * pfac;

                    /******* AB ********/
                    osrecurl2<double>(ABy, i.m + 1, j.m + 1, k.m, l.m, Jny);
                    osrecurl4<double>(CDy, i.m + 1, j.m + 1, k.m, l.m, Jny);

                    block_ab(i.ci, j.ci, k.ci, l.ci) += a412 * Jny(l.m, k.m, j.m + 1, i.m + 1, 0) * pfac;

                    if (j.m) 
                        block_ab(i.ci, j.ci, k.ci, l.ci) -= a21 * j.m * Jny(l.m, k.m, j.m - 1, i.m + 1, 0) * pfac;

                    if (i.m)
                        block_ab(i.ci, j.ci, k.ci, l.ci) -= a22 * i.m * Jny(l.m, k.m, j.m + 1, i.m - 1, 0) * pfac;

                    if (i.m && j.m) 
                        block_ab(i.ci, j.ci, k.ci, l.ci) += i.m * j.m * Jny(l.m, k.m, j.m - 1, i.m - 1, 0) * pfac;
                    
                    /******* AC ********/
                    osrecurl13<double>(PQ(1), QC(1), PA(1), gamma_ab2, gamma_cd2, p_r, q_r, i.m +
                                    j.m + 1, k.m + l.m + 1, n_max, Jny);
                    osrecurl2<double>(ABy, i.m + 1, j.m, k.m + 1, l.m, Jny);
                    osrecurl4s1<double>(CDy, i.m + 1, j.m, k.m + 1, l.m, Jny);

                    block_ac(i.ci, j.ci, k.ci, l.ci) += a413 * Jny(l.m, k.m + 1, j.m, i.m + 1, 0) * pfac;

                    if (k.m) 
                        block_ac(i.ci, j.ci, k.ci, l.ci) -= a21 * k.m * Jny(l.m, k.m - 1, j.m, i.m + 1, 0) * pfac;

                    if (i.m)
                        block_ac(i.ci, j.ci, k.ci, l.ci) -= a23 * i.m * Jny(l.m, k.m + 1, j.m, i.m - 1, 0) * pfac;

                    if (i.m && k.m) 
                        block_ac(i.ci, j.ci, k.ci, l.ci) += i.m * k.m * Jny(l.m, k.m - 1, j.m, i.m - 1, 0) * pfac;
                    
                    /******* BD ********/
                    osrecurl2<double>(ABy, i.m, j.m + 1, k.m, l.m + 1, Jny);
                    osrecurl4s2<double>(CDy, i.m, j.m + 1, k.m, l.m + 1, Jny);

                    block_bd(i.ci, j.ci, k.ci, l.ci) += a424 * Jny(l.m + 1, k.m, j.m + 1, i.m, 0) * pfac;

                    if (l.m) 
                        block_bd(i.ci, j.ci, k.ci, l.ci) -= a22 * l.m * Jny(l.m - 1, k.m, j.m + 1, i.m, 0) * pfac;

                    if (j.m)
                        block_bd(i.ci, j.ci, k.ci, l.ci) -= a24 * j.m * Jny(l.m + 1, k.m, j.m - 1, i.m, 0) * pfac;

                    if (j.m && l.m) 
                        block_bd(i.ci, j.ci, k.ci, l.ci) += j.m * l.m * Jny(l.m - 1, k.m, j.m - 1, i.m, 0) * pfac;
                }
            }
        }
    }
};
// ********* build shell block end **********        
                    build_shell_quartet();
                }
        }

    shell_to_basis_deriv2_qq(s12, s34, block_aa, block_ab, block_ac, block_bd,
                             dV2daa, dV2dab, dV2dac, dV2dbd, dV2dcc, dV2dcd);
}

void ERIOS::Erios::compute_contracted_shell_quartet_deriv2_zz(symm4dTensor<double>& dV2daa,
                                                              symm4dTensor<double>& dV2dab,
                                                              symm4dTensor<double>& dV2dac,
                                                              symm4dTensor<double>& dV2dbd,
                                                              symm4dTensor<double>& dV2dcc,
                                                              symm4dTensor<double>& dV2dcd,
                                                              const ShellPair& s12, const ShellPair& s34) const
{
    const auto& sh1 = s12.m_s1;
    const auto& sh2 = s12.m_s2;
    const auto& sh3 = s34.m_s1;
    const auto& sh4 = s34.m_s2;

    Index am1 = sh1.get_cirange();
    Index am2 = sh2.get_cirange();
    Index am3 = sh3.get_cirange();
    Index am4 = sh4.get_cirange();

    tensor4d1234<double> block_aa = tensor4d1234<double>(am1, am2, am3, am4);
    tensor4d1234<double> block_ab = tensor4d1234<double>(am1, am2, am3, am4);
    tensor4d1234<double> block_ac = tensor4d1234<double>(am1, am2, am3, am4);
    tensor4d1234<double> block_bd = tensor4d1234<double>(am1, am2, am3, am4);

    Index nmax  = 3 * (sh4.L() + sh3.L() + sh2.L() + sh1.L()) + 2;
    Index uvmax = 2 * (sh4.L() + sh3.L() + sh2.L() + sh1.L()) + 2;
    Index vmax  = 1 * (sh4.L() + sh3.L() + sh2.L() + sh1.L()) + 2;

    // Enough to hold AA AB AC
    tensor5d<double> Jnx = tensor5d<double>(sh4.L() + 1, sh3.L() + 1, sh2.L() + 1, sh1.L() + 1, nmax + 1);
    tensor5d<double> Jny = tensor5d<double>(sh4.L() + 1, sh3.L() + sh4.L() + 1, sh2.L() + 1, sh1.L() + 1, uvmax + 1);
    tensor5d<double> Jnz = tensor5d<double>(sh4.L() + 2, sh3.L() + sh4.L() + 2, sh2.L() + 2, 
                                            sh1.L() + sh2.L() + 3, vmax + 1);

    const Eigen::Ref<const EigenVector<double>>& alpha1 = sh1.alpha();
    const Eigen::Ref<const EigenVector<double>>& alpha2 = sh2.alpha();
    const Eigen::Ref<const EigenVector<double>>& alpha3 = sh3.alpha();
    const Eigen::Ref<const EigenVector<double>>& alpha4 = sh4.alpha();
    const Eigen::Ref<const EigenVector<double>>& c1 = sh1.c();
    const Eigen::Ref<const EigenVector<double>>& c2 = sh2.c();
    const Eigen::Ref<const EigenVector<double>>& c3 = sh3.c();
    const Eigen::Ref<const EigenVector<double>>& c4 = sh4.c();
    const std::vector<BASIS::idq>& id1 = sh1.get_indices();
    const std::vector<BASIS::idq>& id2 = sh2.get_indices(); 
    const std::vector<BASIS::idq>& id3 = sh3.get_indices(); 
    const std::vector<BASIS::idq>& id4 = sh4.get_indices();

    const double ABz = s12.AB(2);
    const double CDy = s34.AB(1);
    const double CDz = s34.AB(2);
    const double precut = cutoff / 10000.0;

    for (Index p1 = 0; p1 < c1.size(); ++p1)
        for (Index p2 = 0; p2 < c2.size(); ++p2)
        {
            const Index id12 = p1 * c2.size() + p2;
            const double gamma_ab = s12.gamma_ab(id12);
            const double gamma_ab2 = gamma_ab + gamma_ab;
            const Vec3D& P = s12.P(id12); 
            const Vec3D& PA = s12.PA(id12);
            const Vec3D& PB = s12.PB(id12);

            for (Index p3 = 0; p3 < c3.size(); ++p3)
                for (Index p4 = 0; p4 < c4.size(); ++p4)
                {
                    const Index id34 = p3 * c4.size() + p4;
                    const Vec3D& Q = s34.P(id34);
                    const Vec3D& PQ = P - Q;
                    const Vec3D& QC = s34.PA(id34);
                    const Vec3D& QD = s34.PB(id34);
                    const double rpq_2 = rab2(P, Q);
                    const double gamma_cd = s34.gamma_ab(id34);
                    const double gamma_cd2 = gamma_cd + gamma_cd;
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
                    
                    osrecur<double>(PQ(0), PA(0), PB(0), QC(0), QD(0), gamma_ab2, gamma_cd2,
                                    p_r, q_r, sh1.L(), sh2.L(), sh3.L(), sh4.L(), nmax, Jnx);
                    

// ********* build shell block start ********** move to func
const auto build_shell_quartet = [&]() noexcept
{
    const double a411 = 4.0 * alpha1(p1) * alpha1(p1);
    const double a412 = 4.0 * alpha1(p1) * alpha2(p2);
    const double a413 = 4.0 * alpha1(p1) * alpha3(p3);
    const double a424 = 4.0 * alpha2(p2) * alpha4(p4);
    const double a21  = 2.0 * alpha1(p1);
    const double a22  = 2.0 * alpha2(p2);
    const double a23  = 2.0 * alpha3(p3);
    const double a24  = 2.0 * alpha4(p4);

    for (const auto& i : id1)
    {
        Index idx1 = sh1.get_idx() + i.ci;

        for (const auto &j : id2)
        {
           Index idx2 = sh2.get_idx() + j.ci;

           if (!pure && idx1 < idx2) continue;
           bool idx12 = idx1 == idx2;

            for (const auto& k : id3)
            {
                Index idx3 = sh3.get_idx() + k.ci;
                bool idx23 = idx2 == idx3;

                for (const auto& l : id4)
                {
                    Index idx4 = sh4.get_idx() + l.ci;
                    
                    if (!pure && idx3 < idx4) continue;
                    else if(idx12 && idx23 && idx3 == idx4) continue; // always zero (all centers)

                    Index n_max = i.n + j.n + k.n + l.n + i.m + j.m + k.m + l.m + 2;  // current nmax
                                                                                    // l1 l2 l3 l4 done
                    double tmp = 0;
                    for (Index n = 0; n <= n_max; ++n)
                    {
                        Jny(0, 0, 0, 0, n) = Jnx(l.l, k.l, j.l, i.l, n);
                        tmp += fabs(Jnx(l.l, k.l, j.l, i.l, n));
                    }

                    if (tmp < cutoff) continue;
                    
                    /******* AA ********/
                    osrecurl123<double>(PQ(1), QC(1), PA(1), PB(1), gamma_ab2, gamma_cd2, p_r, q_r, i.m,
                                    j.m, k.m + l.m, n_max, Jny);
                    osrecurl4n<double>(CDy, i.m, j.m, k.m, l.m, n_max, Jny);
                    
                    n_max -= i.m + j.m + k.m + l.m;

                    for (Index n = 0; n <= n_max; ++n)
                    {
                        Jnz(0, 0, 0, 0, n) = Jny(l.m, k.m, j.m, i.m, n);
                        tmp += fabs(Jny(l.m, k.m, j.m, i.m, n));
                    }

                    if (tmp < cutoff) continue;

                    osrecurl13<double>(PQ(2), QC(2), PA(2), gamma_ab2, gamma_cd2, p_r, q_r,
                                     i.n + j.n + 2, k.n + l.n, n_max, Jnz);
                    osrecurl2<double>(ABz, i.n + 2, j.n, k.n, l.n, Jnz);
                    osrecurl4s1<double>(CDz, i.n + 2, j.n, k.n, l.n, Jnz);

                    block_aa(i.ci, j.ci, k.ci, l.ci) += (a411 * Jnz(l.n, k.n, j.n, i.n + 2, 0)
                   - a21 * (2.0 * i.n + 1.0) * Jnz(l.n, k.n, j.n, i.n, 0)) * pfac;

                    if (i.n > 1) block_aa(i.ci, j.ci, k.ci, l.ci) += 
                        i.n * (i.n - 1) * Jnz(l.n, k.n, j.n, i.n - 2, 0) * pfac;

                    /******* AB ********/
                    osrecurl2<double>(ABz, i.n + 1, j.n + 1, k.n, l.n, Jnz);
                    osrecurl4<double>(CDz, i.n + 1, j.n + 1, k.n, l.n, Jnz);

                    block_ab(i.ci, j.ci, k.ci, l.ci) += a412 * Jnz(l.n, k.n, j.n + 1, i.n + 1, 0) * pfac;

                    if (j.n) 
                        block_ab(i.ci, j.ci, k.ci, l.ci) -= a21 * j.n * Jnz(l.n, k.n, j.n - 1, i.n + 1, 0) * pfac;

                    if (i.n)
                        block_ab(i.ci, j.ci, k.ci, l.ci) -= a22 * i.n * Jnz(l.n, k.n, j.n + 1, i.n - 1, 0) * pfac;

                    if (i.n && j.n) 
                        block_ab(i.ci, j.ci, k.ci, l.ci) += i.n * j.n * Jnz(l.n, k.n, j.n - 1, i.n - 1, 0) * pfac;
                    
                    /******* AC ********/
                    osrecurl13<double>(PQ(2), QC(2), PA(2), gamma_ab2, gamma_cd2, p_r, q_r,
                                       i.n + j.n + 1, k.n + l.n + 1, n_max, Jnz);
                    osrecurl2<double>(ABz, i.n + 1, j.n, k.n + 1, l.n, Jnz);
                    osrecurl4s1<double>(CDz, i.n + 1, j.n, k.n + 1, l.n, Jnz);

                    block_ac(i.ci, j.ci, k.ci, l.ci) += a413 * Jnz(l.n, k.n + 1, j.n, i.n + 1, 0) * pfac;

                    if (k.n)
                        block_ac(i.ci, j.ci, k.ci, l.ci) -= a21 * k.n * Jnz(l.n, k.n - 1, j.n, i.n + 1, 0) * pfac;

                    if (i.n)
                        block_ac(i.ci, j.ci, k.ci, l.ci) -= a23 * i.n * Jnz(l.n, k.n + 1, j.n, i.n - 1, 0) * pfac;

                    if (i.n && k.n)
                        block_ac(i.ci, j.ci, k.ci, l.ci) += i.n * k.n * Jnz(l.n, k.n - 1, j.n, i.n - 1, 0) * pfac;
                    
                    /******* BD ********/
                    osrecurl2<double>(ABz, i.n, j.n + 1, k.n, l.n + 1, Jnz);
                    osrecurl4s2<double>(CDz, i.n, j.n + 1, k.n, l.n + 1, Jnz);

                    block_bd(i.ci, j.ci, k.ci, l.ci) += a424 * Jnz(l.n + 1, k.n, j.n + 1, i.n, 0) * pfac;

                    if (l.n)
                        block_bd(i.ci, j.ci, k.ci, l.ci) -= a22 * l.n * Jnz(l.n - 1, k.n, j.n + 1, i.n, 0) * pfac;

                    if (j.n)
                        block_bd(i.ci, j.ci, k.ci, l.ci) -= a24 * j.n * Jnz(l.n + 1, k.n, j.n - 1, i.n, 0) * pfac;

                    if (j.n && l.n) 
                        block_bd(i.ci, j.ci, k.ci, l.ci) += j.n * l.n * Jnz(l.n - 1, k.n, j.n - 1, i.n, 0) * pfac;
                }
            }
        }
    }
};
// ********* build shell block end **********        
                    build_shell_quartet();
                }
        }

    shell_to_basis_deriv2_qq(s12, s34, block_aa, block_ab, block_ac, block_bd,
                             dV2daa, dV2dab, dV2dac, dV2dbd, dV2dcc, dV2dcd);
}

void ERIOS::Erios::shell_to_basis_deriv2_qq(const ShellPair& s12, const ShellPair& s34,
                                            const tensor4d1234<double> block_aa, const tensor4d1234<double> block_ab,
                                            const tensor4d1234<double> block_ac, const tensor4d1234<double> block_bd,
                                            symm4dTensor<double>& dV2daa, symm4dTensor<double>& dV2dab,
                                            symm4dTensor<double>& dV2dac, symm4dTensor<double>& dV2dbd,
                                            symm4dTensor<double>& dV2dcc, symm4dTensor<double>& dV2dcd) const
{
    const auto& sh1 = s12.m_s1;
    const auto& sh2 = s12.m_s2;
    const auto& sh3 = s34.m_s1;
    const auto& sh4 = s34.m_s2;

    // map shell block to eri integrals in basis form
    if (!pure)
    {
        for(Index p = sh1.get_idx(); p < sh1.get_idx() + sh1.get_cirange(); ++p)
            for(Index q = sh2.get_idx(); q < sh2.get_idx() + sh2.get_cirange(); ++q)
                for(Index r = sh3.get_idx(); r < sh3.get_idx() + sh3.get_cirange(); ++r)
                    for(Index s = sh4.get_idx(); s < sh4.get_idx() + sh4.get_cirange(); ++s)
                    {
                        if(p >= q && r >= s)
                        {
                            Index pq = hfscfmath::index_ij(p, q);
                            Index rs = hfscfmath::index_ij(r, s);

                            if (pq <= rs)
                            {
                                dV2daa(p, q, r, s) = block_aa(p - sh1.get_idx(), q - sh2.get_idx(),
                                                            r - sh3.get_idx(), s - sh4.get_idx());

                                dV2dab(p, q, r, s) = block_ab(p - sh1.get_idx(), q - sh2.get_idx(),
                                                            r - sh3.get_idx(), s - sh4.get_idx());

                                dV2dac(p, q, r, s) = block_ac(p - sh1.get_idx(), q - sh2.get_idx(),
                                                            r - sh3.get_idx(), s - sh4.get_idx());
                                
                                dV2dbd(p, q, r, s) = block_bd(p - sh1.get_idx(), q - sh2.get_idx(),
                                                            r - sh3.get_idx(), s - sh4.get_idx());
                            }

                            if (rs <= pq)
                            {
                                dV2dcc(r, s, p, q) = block_aa(p - sh1.get_idx(), q - sh2.get_idx(),
                                                            r - sh3.get_idx(), s - sh4.get_idx());
                                
                                dV2dcd(p, q, r, s) = block_ab(p - sh1.get_idx(), q - sh2.get_idx(),
                                                            r - sh3.get_idx(), s - sh4.get_idx());
                            }
                        }
                    }
    }
    else
    {
        tensor4d1234<double> pblock_aa = tensor4d1234<double>(sh1.get_sirange(), sh2.get_sirange(), 
                                                              sh3.get_sirange(), sh4.get_sirange());
        tensor4d1234<double> pblock_ab = tensor4d1234<double>(sh1.get_sirange(), sh2.get_sirange(), 
                                                              sh3.get_sirange(), sh4.get_sirange());
        tensor4d1234<double> pblock_ac = tensor4d1234<double>(sh1.get_sirange(), sh2.get_sirange(), 
                                                              sh3.get_sirange(), sh4.get_sirange());
        tensor4d1234<double> pblock_bd = tensor4d1234<double>(sh1.get_sirange(), sh2.get_sirange(), 
                                                              sh3.get_sirange(), sh4.get_sirange());

        transform(s12, s34, block_aa, pblock_aa);
        transform(s12, s34, block_ab, pblock_ab);
        transform(s12, s34, block_ac, pblock_ac);
        transform(s12, s34, block_bd, pblock_bd);

        for(Index p = sh1.get_ids(); p < sh1.get_ids() + sh1.get_sirange(); ++p)
            for(Index q = sh2.get_ids(); q < sh2.get_ids() + sh2.get_sirange(); ++q)
                for(Index r = sh3.get_ids(); r < sh3.get_ids() + sh3.get_sirange(); ++r)
                    for(Index s = sh4.get_ids(); s < sh4.get_ids() + sh4.get_sirange(); ++s)
                    {
                        if(p >= q && r >= s)
                        {
                            Index pq = hfscfmath::index_ij(p, q);
                            Index rs = hfscfmath::index_ij(r, s);

                            if (pq <= rs)
                            {
                                dV2daa(p, q, r, s) = pblock_aa(p - sh1.get_ids(), q - sh2.get_ids(),
                                                            r - sh3.get_ids(), s - sh4.get_ids());

                                dV2dab(p, q, r, s) = pblock_ab(p - sh1.get_ids(), q - sh2.get_ids(),
                                                            r - sh3.get_ids(), s - sh4.get_ids());

                                dV2dac(p, q, r, s) = pblock_ac(p - sh1.get_ids(), q - sh2.get_ids(),
                                                            r - sh3.get_ids(), s - sh4.get_ids());
                                
                                dV2dbd(p, q, r, s) = pblock_bd(p - sh1.get_ids(), q - sh2.get_ids(),
                                                            r - sh3.get_ids(), s - sh4.get_ids());
                            }

                            if (rs <= pq)
                            {
                                dV2dcc(r, s, p, q) = pblock_aa(p - sh1.get_ids(), q - sh2.get_ids(),
                                                            r - sh3.get_ids(), s - sh4.get_ids());
                                
                                dV2dcd(p, q, r, s) = pblock_ab(p - sh1.get_ids(), q - sh2.get_ids(),
                                                            r - sh3.get_ids(), s - sh4.get_ids());
                            }
                        }
                    }

    }
}
