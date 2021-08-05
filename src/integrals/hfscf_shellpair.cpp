#include "hfscf_shellpair.hpp"

using hfscfmath::gpc;
using Eigen::Index;
using hfscfmath::pi;

BASIS::ShellPair::ShellPair(const Shell& s1, const Shell& s2)
: m_s1(s1), m_s2(s2)
{
    const Index dim1 = m_s1.c().size(); 
    const Index dim2 = m_s2.c().size();
    const Eigen::Ref<const EigenVector<double>> alpha1 = m_s1.alpha();
    const Eigen::Ref<const EigenVector<double>> alpha2 = m_s2.alpha();
    const Eigen::Ref<const EigenVector<double>> c1 = m_s1.c();
    const Eigen::Ref<const EigenVector<double>> c2 = m_s2.c();

    P = EigenVector<Vec3D>(dim1 * dim2); 
    PA = EigenVector<Vec3D>(dim1 * dim2); 
    PB = EigenVector<Vec3D>(dim1 * dim2);
    gamma_ab = EigenVector<double>(dim1 * dim2);
    pfac = EigenVector<double>(dim1 * dim2);
    pfac2 = EigenVector<double>(dim1 * dim2);
    Kab = EigenVector<double>(dim1 * dim2);

    // Will never change throughout
    for(Index i = 0; i < dim1; ++i)
        for(Index j = 0; j < dim2; ++j)
            gamma_ab(i * dim2 + j) = alpha1(i) + alpha2(j);

    set_params();
}

void BASIS::ShellPair::set_params()
{
    const Index dim1 = m_s1.c().size();
    const Index dim2 = m_s2.c().size();
    const Eigen::Ref<const EigenVector<double>> alpha1 = m_s1.alpha();
    const Eigen::Ref<const EigenVector<double>> alpha2 = m_s2.alpha();
    const Eigen::Ref<const EigenVector<double>> c1 = m_s1.c();
    const Eigen::Ref<const EigenVector<double>> c2 = m_s2.c();

    AB = m_s1.r() - m_s2.r();

    const double Rab2 = hfscfmath::rab2(m_s1.r(), m_s2.r());

    for(Index i = 0; i < dim1; ++i)
        for(Index j = 0; j < dim2; ++j)
        {
            const Vec3D P_ = hfscfmath::gpc(alpha1(i), alpha2(j), m_s1.r(), m_s2.r());
            double eta = alpha1(i) * alpha2(j) / gamma_ab(i * dim2 + j);
            pfac(i * dim2 + j)  = std::pow(pi / (alpha1(i) + alpha2(j)), 1.5) * std::exp(-eta * Rab2)
                                * c1(i) * c2(j);
            pfac2(i * dim2 + j) = (2 * pi / gamma_ab(i * dim2 + j)) * std::exp(-eta * Rab2)
                                * c1(i) * c2(j);
            Kab(i * dim2 + j) = std::exp(-eta * Rab2) * c1(i) * c2(j);

            P(i * dim2 + j) = P_;
            PA(i * dim2 + j) = P_ - m_s1.r();
            PB(i * dim2 + j) = P_ - m_s2.r();
        }
}