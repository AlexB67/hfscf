#ifndef HFSCF_DIPOLE
#define HFSCF_DIPOLE

#include "hfscf_shellpair.hpp"
#include "../math/hfscf_tensors.hpp"

// forward declare
namespace MOLEC
{
    class Atom;
}

using BASIS::ShellPair;
using MOLEC::Atom;
using tensormath::tensor3d;

namespace DIPOLE
{
    class Dipole
    {
        public:
            explicit Dipole() = delete;
            explicit Dipole(const bool is_pure) : pure(is_pure) {}
            Dipole(const Dipole&) = delete;
            Dipole& operator=(const Dipole& other) = delete;
            Dipole(const Dipole&&) = delete;
            Dipole&& operator=(const Dipole&& other) = delete;
            ~Dipole(){}

            void compute_contracted_shell(tensor3d<double>& Dp,
                                          const Eigen::Ref<const Vec3D>& Q_c,
                                          const ShellPair& sp) const;
            
            void compute_contracted_shell_deriv1(std::vector<EigenMatrix<double>>& Dpa, std::vector<EigenMatrix<double>>& Dpb,
                                                 const ShellPair& sp) const;
        
        private:
            bool pure;
    };
}

#endif
// End HFSCF_DIPOLE
