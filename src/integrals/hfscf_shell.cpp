#include "hfscf_shell.hpp"
#include "../math/hfscf_math.hpp"
#include "../math/solid_harmonics.hpp"
#include <iostream>

using hfscfmath::dfac;
using hfscfmath::pi;

BASIS::Shell::Shell(bool ispure, const Index L, const Index idx, const Index ids, const Eigen::Ref<const Vec3D>& r, 
                    const EigenVector<double> c,
                    const EigenVector<double> alpha)

:   m_ispure(ispure),
    m_L(L),
    m_idx(idx),
    m_ids(ids),
    m_r(r),
    m_c(c),
    m_alpha(alpha)
{
    cirange = (m_L + 1) * (m_L + 2) / 2;
    sirange =  2 * m_L + 1;

    indices = std::vector<idq>(cirange);
    m_c_unscaled = m_c;

    for (Index i = 0, ci = 0; i <= m_L; ++i)
    {
        Index l = m_L - i;
        for (Index j = 0; j <= i; ++j, ++ci)
        {
            Index m = i - j;
            Index n = j;
            
            indices[ci].l = l;
            indices[ci].m = m;
            indices[ci].n = n;
            indices[ci].ci = ci;
        }
    }

    if (m_ispure) cart_to_spherical = hfscfmath::Ylm_transmat(m_L);
    
    const double m = static_cast<double>(m_L) + 1.5;
    const double m2 = 0.5 * m;
    
    double sum = 0;
    for (int j = 0; j < m_c.size(); ++j)
        for (int k = 0; k < m_c.size(); ++k)
            sum += (m_c(j) * m_c(k) * pow(m_alpha(j) * m_alpha(k), m2)) / (pow(m_alpha(j) + m_alpha(k), m));

    const double norm = 1.0 / sqrt(sum * pow(pi, 1.5) * dfac(2 * m_L - 1) / pow(2, m_L));

    for (int j = 0; j < m_c.size(); ++j) m_c(j) *= norm * pow(m_alpha(j), m2);    
}
