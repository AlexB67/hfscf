#ifndef HFSCF_MATH
#define HFSCF_MATH
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <numeric>

// Once upon a time this file contained a lot of stuff. Still some useful relics remain.
// Do not move these into a cpp file, this gives a decent performance increase

template <typename T>
using EigenVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

using Vec3D = Eigen::Vector3d;

namespace hfscfmath {

template <typename T>
constexpr T Pi() {
    return std::acos(-1);
}

const double pi = Pi<double>();
const double PI_25 = 2.0 * std::pow(hfscfmath::Pi<double>(), 2.5);

constexpr auto fac = [](int n) noexcept -> int 
{
    if (n <= 1) return 1;

    switch (n) {
        case 2:
            return 2;
        case 3:
            return 6;
        case 4:
            return 24;
        case 5:
            return 120;
        case 6:
            return 720;
        case 7:
            return 5040;
        case 8:
            return 40320;
        case 9:
            return 362880;
        case 10:
            return 3628800;
    };

    int k = 3628800;
    for (int j = 11; j <= n; ++j) 
    {
        k *= j;
    }

    return k;
};

constexpr auto dfac(int n) -> int
{
    if (n == 1 || n <= 0) return 1;

    return n * dfac(n - 2);
}

// Gaussion product center
constexpr auto gpc = [](double alpha1, double alpha2, const Eigen::Ref<const Vec3D>& a,
                        const Eigen::Ref<const Vec3D>& b) noexcept -> Vec3D {
    const double denom = alpha1 + alpha2;
    const Vec3D r = (alpha1 * a + alpha2 * b) / denom;
    return r;
};

// Norm
constexpr auto rab2 = [](const Eigen::Ref<const Vec3D>& a, const Eigen::Ref<const Vec3D>& b) noexcept -> double 
{
    return (a - b).squaredNorm();
};

// Indices lookup for 2 electron integrals
constexpr auto index_ij = [](const Eigen::Index i, const Eigen::Index j) noexcept -> Eigen::Index 
{
    Eigen::Index ij;
    (i > j) ? ij = (i * (i + 1) >> 1) + j : ij = (j * (j + 1) >> 1) + i;
    return ij;
};

constexpr auto index_ijkl = [](const Eigen::Index i, const Eigen::Index j, const Eigen::Index k,
                               const Eigen::Index l) noexcept -> Eigen::Index {
    Eigen::Index ij;
    Eigen::Index kl;
    Eigen::Index ijkl;

    (i > j) ? ij = (i * (i + 1) >> 1) + j : ij = (j * (j + 1) >> 1) + i;
    (k > l) ? kl = (k * (k + 1) >> 1) + l : kl = (l * (l + 1) >> 1) + k;
    (ij > kl) ? ijkl = (ij * (ij + 1) >> 1) + kl : ijkl = (kl * (kl + 1) >> 1) + ij;

    return ijkl;
};

enum class Cart { X, Y, Z };

enum class Cart2 { XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ };
}  // namespace hfscfmath

#endif
// End HFSCF_MATH
