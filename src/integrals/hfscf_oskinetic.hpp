#ifndef HFSCF_OSKINETIC
#define HFSCF_OSKINETIC

#include "hfscf_osoverlap.hpp"

namespace OSKINETIC
{
    class OSKinetic
    {
        public:
            explicit OSKinetic(const bool is_pure) : pure(is_pure) {}
            explicit OSKinetic() = delete;
            OSKinetic(const OSKinetic&) = delete;
            OSKinetic& operator=(const OSKinetic& other) = delete;
            OSKinetic(const OSKinetic&&) = delete;
            OSKinetic&& operator=(const OSKinetic&& other) = delete;
            ~OSKinetic(){}
            
            void compute_contracted_shell_deriv1(tensor3d<double>& dT,
                                                 const ShellPair& sp) const;
            
            void compute_contracted_shell_deriv2(tensor3d<double>& ddT,
                                                 const ShellPair& sp) const;
            
            private:
                bool pure;

                double kt(double alpha1, double alpha2, 
                          int l1, int m1, int n1, int l2, int m2, int n2, 
                          const Eigen::Ref<const EigenMatrix<double>>& Sx,
                          const Eigen::Ref<const EigenMatrix<double>>& Sy,
                          const Eigen::Ref<const EigenMatrix<double>>& Sz) const;

    };
}

#endif
// end HFSCF_OSKINETIC