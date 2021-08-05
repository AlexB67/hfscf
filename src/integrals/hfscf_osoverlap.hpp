#ifndef HFSCF_OSOVERLAP
#define HFSCF_OSOVERLAP

#include "hfscf_shellpair.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include "../math/hfscf_tensors.hpp"

using BASIS::ShellPair;
using hfscfmath::Cart2;
using tensormath::tensor3d;

template <typename T>
using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

namespace OSOVERLAP
{
    class OSOverlap
    {
        public:
            OSOverlap() = delete;
            explicit OSOverlap(const bool is_pure) : pure(is_pure) {}
            OSOverlap(const OSOverlap&) = delete;
            OSOverlap& operator=(const OSOverlap& other) = delete;
            OSOverlap(const OSOverlap&&) = delete;
            OSOverlap&& operator=(const OSOverlap&& other) = delete;
            ~OSOverlap(){};
            
            void compute_contracted_shell(EigenMatrix<double>& S, EigenMatrix<double>& T,
                                          const ShellPair& sp) const;
            
            void compute_contracted_shell_deriv1(tensor3d<double>& S,
                                                 const ShellPair& sp) const;
            
            void compute_contracted_shell_deriv2(tensor3d<double>& S,
                                                 const ShellPair& sp) const;
        
        private:
            bool pure;
    };

// 2nd derriv d/dxx d/dyy etc.
inline constexpr auto D2ij = [](const Eigen::Ref<const EigenMatrix<double>>& S, double alpha, int i, int j) -> double
{
    double dij = 4 * alpha * alpha * S(j, i + 2) - 2 * alpha * (2 * i + 1) * S(j, i);
    if ( i > 1) dij += i * (i - 1) * S(j, i - 2);

    return dij;
};
// 2nd cross deriv d/dxy d/dyz etc
inline constexpr auto D11ij = [](const Eigen::Ref<const EigenMatrix<double>>& Si, 
                          const Eigen::Ref<const EigenMatrix<double>>& Sj, double alpha1, double alpha2,
                          int i, int j, int k, int l) -> double
{
    double dij = 4 * alpha1 * alpha2 * Si(j, i + 1) * Sj(l + 1, k);
    if (l) dij -= 2 * alpha1 * l * Si(j, i + 1) * Sj(l - 1, k);
    if (i) dij -= 2 * alpha2 * i * Si(j, i - 1) * Sj(l + 1, k);
    if (i && l) dij += i * l * Si(j, i - 1) * Sj(l - 1, k);

    return dij;
};

constexpr auto dS = [](const Eigen::Ref<const EigenMatrix<double>>& S, double alpha1, int i, int j) -> double
{
    double dSx = 2.0 * alpha1 * S(j, i + 1);
    if (i) dSx -= i * S(j, i - 1);
    
    return dSx;
};

}

#endif
// end HFSCF_OVERLAP
