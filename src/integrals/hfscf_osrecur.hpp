#ifndef OSRECURALG_H
#define OSRECURALG_H

#include "../math/hfscf_tensors.hpp"
#include "../math/hfscf_math.hpp"
#include <functional>

using tensormath::tensor5d;
using tensormath::tensor3d;

template <typename T>
using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

namespace os_recursion
{

template<typename D> // 2 electron repulsion recur in one dimension. Can be used for x, y, z
void constexpr
osrecur(const D PQx, const D PAx, const D PBx, const D QCx, const D QDx, const D gamma_ab2, const D gamma_cd2, 
        const D p_r,const D q_r, const Index l1, const Index l2, const Index l3, const Index l4, 
        const Index nmax, tensor5d<D>& Jxn) noexcept
{
    if(!nmax) return;

    const D gamma_abcd2 = gamma_ab2 + gamma_cd2;

    if(l1)
        for(Index n = 0; n < nmax; ++n)
            Jxn(0, 0, 0, 1, n) = PAx * Jxn(0, 0, 0, 0, n) - p_r * PQx * Jxn(0, 0, 0, 0, n + 1);
    
    for(Index i = 1; i < l1; ++i)
        for(Index n = 0; n < nmax - i; ++n)
        {
            Jxn(0, 0, 0, i + 1, n) = PAx * Jxn(0, 0, 0, i, n) - p_r * PQx * Jxn(0, 0, 0, i, n + 1)
            + ((D)i / gamma_ab2) * (Jxn(0, 0, 0, i - 1, n) - p_r * Jxn(0, 0, 0, i - 1, n + 1));
        }

    for(Index j = 0; j < l2; ++j)
        for(Index i = 0; i <= l1; ++i)
            for(Index n = 0; n < nmax - i - j; ++n)
            {
                Jxn(0, 0, j + 1, i, n) = PBx * Jxn(0, 0, j, i, n) - p_r * PQx * Jxn(0, 0, j, i, n + 1);
                
                if (i) Jxn(0, 0, j + 1, i, n) += 
                    ((D)i / gamma_ab2) * (Jxn(0, 0, j, i - 1, n) - p_r * Jxn(0, 0, j, i - 1, n + 1));
                
                if (j) Jxn(0, 0, j + 1, i, n) += 
                    ((D)j / gamma_ab2) * (Jxn(0, 0, j - 1, i, n) - p_r * Jxn(0, 0, j - 1, i, n + 1));
            }

    for(Index k = 0; k < l3; ++k)
        for(Index j = 0; j <= l2; ++j)
            for(Index i = 0; i <= l1; ++i)
                for(Index n = 0; n < nmax - i - j - k; ++n)
                {
                    Jxn(0, k + 1, j, i, n) = QCx * Jxn(0, k, j, i, n) + q_r *  PQx * Jxn(0, k, j, i, n + 1);
                    if(i) Jxn(0, k + 1, j, i, n) += ((D)i / gamma_abcd2) * Jxn(0, k, j, i - 1, n + 1);
                    if(j) Jxn(0, k + 1, j, i, n) += ((D)j / gamma_abcd2) * Jxn(0, k, j - 1, i, n + 1);
                    
                    if(k) Jxn(0, k + 1, j, i, n) += 
                        ((D)k / gamma_cd2) * (Jxn(0, k - 1, j, i, n) - q_r * Jxn(0, k - 1, j, i, n + 1));
                }
    
    for(Index l = 0; l < l4; ++l)
        for(Index k = 0; k <= l3; ++k)
            for(Index j = 0; j <= l2; ++j)
                for(Index i = 0; i <= l1; ++i)
                    for(Index n = 0; n < nmax - i - j - k - l; ++n)
                    {
                        Jxn(l + 1, k, j, i, n) = QDx * Jxn(l, k, j, i, n) + q_r *  PQx * Jxn(l, k, j, i, n + 1);
                     
                        if(i) Jxn(l + 1, k, j, i, n) += ((D)i / gamma_abcd2) * Jxn(l, k, j, i - 1, n + 1);
                        if(j) Jxn(l + 1, k, j, i, n) += ((D)j / gamma_abcd2) * Jxn(l, k, j - 1, i, n + 1);
                        
                        if(k) Jxn(l + 1, k, j, i, n) += 
                            ((D)k / gamma_cd2) * (Jxn(l, k - 1, j, i, n) - q_r * Jxn(l, k - 1, j, i, n + 1));
                        
                        if(l) Jxn(l + 1, k, j, i, n) += 
                            ((D)l / gamma_cd2) * (Jxn(l - 1, k, j, i, n) - q_r * Jxn(l - 1, k, j, i, n + 1));
                    }
}

using Eigen::Index;
template<typename D> // 1 electron potential recur 3 dimensional
void constexpr
osrecurpot3c(const Eigen::Ref<const Vec3D>& PA, const Eigen::Ref<const Vec3D>& PB, 
             const Eigen::Ref<const Vec3D>& PC, const D gamma, 
             const D r_pc2,  const D T, const D Fnu_at_nmax, const Index l1, const Index m1, 
             const Index n1, const Index l2, const Index m2, const  Index n2, const Index nmax, EigenMatrix<D>& Jn,
             const std::function<Index(Index, Index, Index, Index, Index, Index)>& idx) noexcept
{
    if (fabs(r_pc2) > 1.0E-18)
    {
        Jn(0, nmax) = Fnu_at_nmax;
        // Downward recursion for Boys stability
        for (Index n = nmax - 1; n >= 0; --n)
            Jn(0, n) = (1.0 / (2.0 * n + 1.0)) * (2.0 * T * Jn(0, n + 1) + std::exp(-T));
    } 
    else 
    {
        Jn(0, nmax) = 1.0;
        for (Index n = nmax - 1; n >= 0; --n) Jn(0, n) = 1.0 / (2.0 * n + 1.0);
    }

    if(!nmax) return;

    const D gamma2 = gamma + gamma;

    if (l1)
        for(Index n = 0; n < nmax; ++n)
            Jn(1, n) = PA(0) * Jn(0, n) - PC(0) * Jn(0, n + 1);
    
    for(Index i = 1; i < l1; ++i)
        for(Index n = 0; n < nmax - i; ++n)
        {
            Jn(i + 1, n) = PA(0) * Jn(i, n) - PC(0) * Jn(i, n + 1)
            + ((D)i / gamma2) * (Jn(i - 1, n) - Jn(i - 1, n + 1));
        }

    for(Index j = 0; j < m1; ++j)
        for(Index i = 0; i <= l1; ++i)
        {
            const Index id1 = idx(0, 0, 0, 0, j, i);
            const Index id2 = idx(0, 0, 0, 0, j - 1, i);
            const Index id3 = idx(0, 0, 0, 0, j + 1, i);

            for(Index n = 0; n < nmax - i - j; ++n)
            {
                double tmp = PA(1) * Jn(id1, n) - PC(1) * Jn(id1, n + 1);
                if (j) tmp += ((D)j / gamma2) * (Jn(id2, n) - Jn(id2, n + 1));
                Jn(id3, n) = tmp;
            }
        }
    
    for(Index k = 0; k < n1; ++k)
        for(Index j = 0; j <= m1; ++j)
            for(Index i = 0; i <= l1; ++i)
            {
                const Index id1 = idx(0, 0, 0, k, j, i);
                const Index id2 = idx(0, 0, 0, k - 1, j, i);
                const Index id3 = idx(0, 0, 0, k + 1, j, i);

                for(Index n = 0; n < nmax - i - j - k; ++n)
                {
                    double tmp = PA(2) * Jn(id1, n) - PC(2) * Jn(id1, n + 1);
                    if (k) tmp += ((D)k / gamma2) * (Jn(id2, n) - Jn(id2, n + 1));
                    Jn(id3, n) = tmp;
                }
            }

    for(Index  l = 0; l < l2; ++l)
        for(Index k = 0; k <= n1; ++k)
            for(Index j = 0; j <= m1; ++j)
                for(Index i = 0; i <= l1; ++i)
                {
                    const Index id1 = idx(0, 0, l, k, j, i);
                    const Index id2 = idx(0, 0, l - 1, k, j, i);
                    const Index id3 = idx(0, 0, l + 1, k, j, i);

                    for(Index n = 0; n < nmax - i - j - k - l; ++n)
                    {
                        double tmp = PB(0) * Jn(id1, n) - PC(0) * Jn(id1, n + 1);
                        if (i) tmp += ((D)i / gamma2) * (Jn(id1 - 1, n) - Jn(id1 - 1, n + 1));
                        if (l) tmp += ((D)l / gamma2) * (Jn(id2, n) - Jn(id2, n + 1));
                        Jn(id3, n) = tmp;
                    }
                }

    for(Index m = 0; m < m2; ++m)
        for(Index  l = 0; l <= l2; ++l)
            for(Index k = 0; k <= n1; ++k)
                for(Index j = 0; j <= m1; ++j)
                    for(Index i = 0; i <= l1; ++i)
                    {
                        const Index id1 = idx(0, m, l, k, j, i);
                        const Index id2 = idx(0, m, l, k, j - 1, i);
                        const Index id3 = idx(0, m - 1, l, k, j, i);
                        const Index id4 = idx(0, m + 1, l, k, j, i);

                        for(Index n = 0; n < nmax - i - j - k - l - m; ++n)
                        {
                            double tmp = PB(1) * Jn(id1, n) - PC(1) * Jn(id1, n + 1);
                            if (j) tmp += ((D)j / gamma2) * (Jn(id2, n) - Jn(id2, n + 1));
                            if (m) tmp += ((D)m / gamma2) * (Jn(id3, n) - Jn(id3, n + 1));
                            Jn(id4, n) = tmp;
                        }
                    }

    for(Index o = 0; o < n2; ++o)
        for(Index m = 0; m <= m2; ++m)
            for(Index  l = 0; l <= l2; ++l)
                for(Index k = 0; k <= n1; ++k)
                    for(Index j = 0; j <= m1; ++j)
                        for(Index i = 0; i <= l1; ++i)
                        {
                            const Index id1 = idx(o, m, l, k, j, i);
                            const Index id2 = idx(o, m, l, k - 1, j, i);
                            const Index id3 = idx(o - 1, m, l, k, j, i);
                            const Index id4 = idx(o + 1, m, l, k, j, i);

                            for(Index n = 0; n < nmax - i - j - k - l - m - o; ++n)
                            {
                                double tmp = PB(2) * Jn(id1, n) - PC(2) * Jn(id1, n + 1);
                                if (k) tmp += ((D)k / gamma2) * (Jn(id2, n) - Jn(id2, n + 1));
                                if (o) tmp += ((D)o / gamma2) * (Jn(id3, n) - Jn(id3, n + 1));
                                Jn(id4, n) = tmp;
                            }
                        }
}

template<typename D> // 2 electron repulsion recur in one dimension. Can be used for x, y, z
void constexpr
osrecurl123(const D PQx, const D QCx, const D PAx, const D PBx, const D gamma_ab2, const D gamma_cd2, 
            const D p_r, const D q_r, const Index l1, const Index l2, const Index l3, const Index nmax, 
            tensor5d<D>& Jxn) noexcept
{
    if(!nmax) return;

    const D gamma_abcd2 = gamma_ab2 + gamma_cd2;

    if (l1)
        for(Index n = 0; n < nmax; ++n)
            Jxn(0, 0, 0, 1, n) = PAx * Jxn(0, 0, 0, 0, n) - p_r * PQx * Jxn(0, 0, 0, 0, n + 1);


    for(Index i = 1; i < l1; ++i)
        for(Index n = 0; n < nmax - i; ++n)
        {
            Jxn(0, 0, 0, i + 1, n) = PAx * Jxn(0, 0, 0, i, n) - p_r * PQx * Jxn(0, 0, 0, i, n + 1)
            + ((D)i / gamma_ab2) * (Jxn(0, 0, 0, i - 1, n) - p_r * Jxn(0, 0, 0, i - 1, n + 1));
        }

    for(Index j = 0; j < l2; ++j)
        for(Index i = 0; i <= l1; ++i)
            for(Index n = 0; n < nmax - i - j; ++n)
            {
                double tmp = PBx * Jxn(0, 0, j, i, n) - p_r * PQx * Jxn(0, 0, j, i, n + 1);
                if (i) tmp += ((D)i / gamma_ab2) * (Jxn(0, 0, j, i - 1, n) - p_r * Jxn(0, 0, j, i - 1, n + 1));
                if (j) tmp += ((D)j / gamma_ab2) * (Jxn(0, 0, j - 1, i, n) - p_r * Jxn(0, 0, j - 1, i, n + 1));
                Jxn(0, 0, j + 1, i, n) = tmp;
            }

    for(Index k = 0; k < l3; ++k)
        for(Index j = 0; j <= l2; ++j)
            for(Index i = 0; i <= l1; ++i)
                for(Index n = 0; n < nmax - i - j - k; ++n)
                {
                    double tmp = QCx * Jxn(0, k, j, i, n) + q_r *  PQx * Jxn(0, k, j, i, n + 1);
                    if(i) tmp += ((D)i / gamma_abcd2) * Jxn(0, k, j, i - 1, n + 1);
                    if(j) tmp += ((D)j / gamma_abcd2) * Jxn(0, k, j - 1, i, n + 1);
                    if(k) tmp += ((D)k / gamma_cd2) * (Jxn(0, k - 1, j, i, n) - q_r * Jxn(0, k - 1, j, i, n + 1));
                    Jxn(0, k + 1, j, i, n) = tmp;
                }
}

template<typename D> // 2 electron repulsion recur in one dimension. Can be used for x, y, z
void constexpr
osrecurl13(const D PQx, const D QCx, const D PAx, const D gamma_ab2, const D gamma_cd2, 
            const D p_r, const D q_r, const Index l1, const Index l3, const Index nmax, tensor5d<D>& Jxn) noexcept
{
    if(!nmax) return;

    const D gamma_abcd2 = gamma_ab2 + gamma_cd2;

    if (l1)
        for(Index n = 0; n < nmax; ++n)
            Jxn(0, 0, 0, 1, n) = PAx * Jxn(0, 0, 0, 0, n) - p_r * PQx * Jxn(0, 0, 0, 0, n + 1);


    for(Index i = 1; i < l1; ++i)
        for(Index n = 0; n < nmax - i; ++n)
        {
            Jxn(0, 0, 0, i + 1, n) = PAx * Jxn(0, 0, 0, i, n) - p_r * PQx * Jxn(0, 0, 0, i, n + 1)
            + ((D)i / gamma_ab2) * (Jxn(0, 0, 0, i - 1, n) - p_r * Jxn(0, 0, 0, i - 1, n + 1));
        }

    for(Index k = 0; k < l3; ++k)
        for(Index i = 0; i <= l1; ++i)
            for(Index n = 0; n < nmax - i - k; ++n)
                {
                    double tmp = QCx * Jxn(0, k, 0, i, n) + q_r *  PQx * Jxn(0, k, 0, i, n + 1);
                    if(i) tmp += ((D)i / gamma_abcd2) * Jxn(0, k, 0, i - 1, n + 1);
                    if(k) tmp += ((D)k / gamma_cd2) * (Jxn(0, k - 1, 0, i, n) - q_r * Jxn(0, k - 1, 0, i, n + 1));
                    Jxn(0, k + 1, 0, i, n) = tmp;
                }
}

template<typename D> // generate l4 fromm l3 + l2 no nmax for givem l1 l2. Can only be apllied to last coordinate being evaluated.
void constexpr      // where nmax is no longer needed
osrecurl4n(const D CDx, const Index l1, const Index l2, const Index l3, const Index l4, const Index nmax, 
        tensor5d<D>& Jxn) noexcept
{
    for (Index l = 0; l < l4; ++l)
        for (Index k = 0; k <= l3 + l4; ++k)
            for (Index n = 0; n <= nmax - l1 - l2 - l - k; ++n)
                Jxn(l + 1, k, l2, l1, n) = Jxn(l, k + 1, l2, l1, n) + CDx * Jxn(l, k, l2, l1, n);
}

template<typename D> //generate all l2 l4 from l1 l3 no nmax. Can only be apllied to last coordinate being evaluated
void constexpr       // where nmax is no longer needed
osrecurl24(const D ABx, const D CDx, const Index l1, const Index l2, const Index l3, const Index l4, 
           tensor5d<D>& Jxn) noexcept
{
    for (Index m = 0; m <= l3 + l4; ++m)
        for (Index l = 0; l < l2; ++l)
            for (Index k = 0; k <= l1 + l2; ++k)
                Jxn(0, m, l + 1, k, 0) = Jxn(0, m, l, k + 1, 0) + ABx * Jxn(0, m, l, k, 0);

    for (Index l = 0; l < l4; ++l)
        for (Index k = 0; k <= l3 + l4; ++k)
            Jxn(l + 1, k, l2, l1, 0) = Jxn(l, k + 1, l2, l1, 0) + CDx * Jxn(l, k, l2, l1, 0);
}

template<typename D> //generate l2 from l1 + l2 no nmax. Can only be apllied to last coordinate being evaluated
void constexpr       // where nmax is no longer needed
osrecurl2(const D ABx, const Index l1, const Index l2, const Index l3, const Index l4, tensor5d<D>& Jxn) noexcept
{
    for (Index m = 0; m <= l3 + l4; ++m)
        for (Index l = 0; l < l2; ++l)
            for (Index k = 0; k <= l1 + l2; ++k)
                Jxn(0, m, l + 1, k, 0) = Jxn(0, m, l, k + 1, 0) + ABx * Jxn(0, m, l, k, 0);
}

template<typename D> //generate l4 from l3 + l4 for all l1 l2  no nmax. Can only be apllied to last coordinate being evaluated
void constexpr       // where nmax is no longer needed
osrecurl4(const D CDx, const Index l1, const Index l2, const Index l3, const Index l4, tensor5d<D>& Jxn) noexcept
{
    for (Index l = 0; l < l4; ++l)
        for (Index k = 0; k <= l3 + l4; ++k)
            for (Index j = 0; j <= l2; ++j)
                for (Index i = 0; i <= l1; ++i)
                    Jxn(l + 1, k, j, i, 0) = Jxn(l, k + 1, j, i, 0) + CDx * Jxn(l, k, j, i, 0);
}

template<typename D> // special cases for derivs
void constexpr       // where nmax is no longer needed
osrecurl4s1(const D CDx, const Index l1, const Index l2, const Index l3, const Index l4, tensor5d<D>& Jxn) noexcept
{   
    // sum l1 for given l2
    for (Index l = 0; l < l4; ++l)
        for (Index k = 0; k <= l3 + l4; ++k)
            for (Index i = 0; i <= l1; ++i)
                Jxn(l + 1, k, l2, i, 0) = Jxn(l, k + 1, l2, i, 0) + CDx * Jxn(l, k, l2, i, 0);
}

template<typename D> // special cases for derivs
void constexpr       // where nmax is no longer needed
osrecurl4s2(const D CDx, const Index l1, const Index l2, const Index l3, const Index l4, tensor5d<D>& Jxn) noexcept
{
    // sum l2 for given l1
    for (Index l = 0; l < l4; ++l)
        for (Index k = 0; k <= l3 + l4; ++k)
            for (Index j = 0; j <= l2; ++j)
                Jxn(l + 1, k, j, l1, 0) = Jxn(l, k + 1, j, l1, 0) + CDx * Jxn(l, k, j, l1, 0);
}

template<typename D> // special cases for derivs
void constexpr       // where nmax is no longer needed
osrecurl4s0(const D CDx, const Index l1, const Index l2, const Index l3, const Index l4, tensor5d<D>& Jxn) noexcept
{
    // sum neither l1 nor l2
    for (Index l = 0; l < l4; ++l)
        for (Index k = 0; k <= l3 + l4; ++k)
            Jxn(l + 1, k, l2, l1, 0) = Jxn(l, k + 1, l2, l1, 0) + CDx * Jxn(l, k, l2, l1, 0);
}

template<typename D> 
void constexpr osrecur2center(EigenMatrix<D>& Sx, EigenMatrix<D>& Sy, EigenMatrix<D>& Sz,
                              const Index L1, const Index L2, const Eigen::Ref<const Vec3D>& Xpa, 
                              const Eigen::Ref<const Vec3D>& Xpb, const double gamma12inv) noexcept
{
    Sx(0, 0) = Sy(0, 0) = Sz(0, 0) = 1.0;

    if (L1)
    {
        Sx(0, 1) = Xpa(0);
        Sy(0, 1) = Xpa(1);
        Sz(0, 1) = Xpa(2);
    }

    for(Index i = 1; i < L1; ++i)
    {
        Sx(0, i + 1) = Xpa(0) * Sx(0, i) + gamma12inv * (D)i * Sx(0, i - 1);
        Sy(0, i + 1) = Xpa(1) * Sy(0, i) + gamma12inv * (D)i * Sy(0, i - 1);
        Sz(0, i + 1) = Xpa(2) * Sz(0, i) + gamma12inv * (D)i * Sz(0, i - 1);
    }
    

    for(Index j = 0; j < L2; ++j)
        for(Index i = 0; i <= L1; ++i)
        {
            Sx(j + 1, i) = Xpb(0) * Sx(j, i);
            Sy(j + 1, i) = Xpb(1) * Sy(j, i);
            Sz(j + 1, i) = Xpb(2) * Sz(j, i);
           
            if (i)
            { 
                Sx(j + 1, i) += gamma12inv * (D)i * Sx(j, i - 1);
                Sy(j + 1, i) += gamma12inv * (D)i * Sy(j, i - 1);
                Sz(j + 1, i) += gamma12inv * (D)i * Sz(j, i - 1);
            }
           
            if (j)
            { 
                Sx(j + 1, i) += gamma12inv * (D)j * Sx(j - 1, i);
                Sy(j + 1, i) += gamma12inv * (D)j * Sy(j - 1, i);
                Sz(j + 1, i) += gamma12inv * (D)j * Sz(j - 1, i);
            }
        }
}

}
#endif // OSRECUR_H

