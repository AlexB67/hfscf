#ifndef HFSCF_SHELL_PAIR
#define HFSCF_SHELL_PAIR
#include "hfscf_shell.hpp"

namespace BASIS
{

struct ShellPair
{
    explicit ShellPair(const Shell& s1, const Shell& s2);
    const Shell& m_s1;
    const Shell& m_s2;
    EigenVector<Vec3D> P;
    EigenVector<Vec3D> PA;
    EigenVector<Vec3D> PB;
    Vec3D AB;
    EigenVector<double> gamma_ab;
    EigenVector<double> pfac;
    EigenVector<double> pfac2;
    EigenVector<double> Kab;

    void set_params();
};
}

#endif
// end HFSCF_SHELL_PAIR
