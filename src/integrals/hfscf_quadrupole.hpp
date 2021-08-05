#ifndef HFSCF_QPOLE
#define HFSCF_QPOLE

#include "hfscf_shellpair.hpp"

// forward declare
namespace MOLEC
{
    class Atom;
}

using BASIS::ShellPair;
using MOLEC::Atom;

namespace QPOLE
{
    class Quadrupole
    {
        public:
            explicit Quadrupole() = delete;
            explicit Quadrupole(const bool is_pure) : pure(is_pure) {}
            Quadrupole(const Quadrupole&) = delete;
            Quadrupole& operator=(const Quadrupole& other) = delete;
            Quadrupole(const Quadrupole&&) = delete;
            Quadrupole&& operator=(const Quadrupole&& other) = delete;
            ~Quadrupole(){}

            void compute_contracted_shell(std::vector<EigenMatrix<double>>& Qp,
                                          const Eigen::Ref<const Vec3D>& Q_c,
                                          const ShellPair& sp) const;

        
        private:
            bool pure;
    };
}

#endif
// End HFSCF_QPOLE
