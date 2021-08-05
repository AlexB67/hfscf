#ifndef HFSCF_ERIOS
#define HFSCF_ERIOS

#include "hfscf_shellpair.hpp"
#include "../math/hfscf_tensors.hpp"
#include "../math/hfscf_tensor4d.hpp"

using BASIS::ShellPair;
using hfscfmath::Cart;
using tensormath::tensor5d;
using tensormath::tensor4d1234;
using tensor4dmath::symm4dTensor;
using tensor4dmath::tensor4d;
using dvec = const Eigen::Ref<const EigenVector<double>>;

template <typename T>
using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

namespace ERIOS
{
    enum class Permute
    {
        ACONLY,
        ALL
    };

    class Erios
    {
        public:
            explicit Erios() = delete;
            explicit Erios(const bool is_pure, double cut_off = 1E-14)
                          : pure(is_pure), cutoff(cut_off) { screen_tol = 100 * cutoff;}
            Erios(const Erios&) = delete;
            Erios& operator=(const Erios& other) = delete;
            Erios(const Erios&&) = delete;
            Erios&& operator=(const Erios&& other) = delete;
            ~Erios(){};

            void compute_contracted_shell_quartet(EigenVector<double>& eri,
                                                  const ShellPair& s12, const ShellPair& s34,
                                                  bool do_screen = false) const;
            
            void init_screen(const Index num_cartorbs);
            void compute_screen_matrix(const ShellPair& s12);

            // All coordinates for all atoms.  Returns A B C centers, D = -A -B -C OS version
            void compute_contracted_shell_quartet_deriv1(symm4dTensor<double>& dVdXa,
                                                         symm4dTensor<double>& dVdXb,
                                                         symm4dTensor<double>& dVdXc,
                                                         symm4dTensor<double>& dVdYa,
                                                         symm4dTensor<double>& dVdYb,
                                                         symm4dTensor<double>& dVdYc,
                                                         symm4dTensor<double>& dVdZa,
                                                         symm4dTensor<double>& dVdZb,
                                                         symm4dTensor<double>& dVdZc,
                                                         const ShellPair& s12, const ShellPair& s34,
                                                         const std::vector<bool>& coords) const;
            
            void compute_contracted_shell_quartet_deriv2_pq(tensor4d<double>& dV2daa,
                                                            tensor4d<double>& dV2dab,
                                                            tensor4d<double>& dV2dac,
                                                            const ShellPair& s12, const ShellPair& s34,
                                                            const int cart, 
                                                            const Permute perm) const;
            
            void compute_contracted_shell_quartet_deriv2_qq(symm4dTensor<double>& dV2daa,
                                                            symm4dTensor<double>& dV2dab,
                                                            symm4dTensor<double>& dV2dac,
                                                            symm4dTensor<double>& dV2dbd,
                                                            symm4dTensor<double>& dV2dcc,
                                                            symm4dTensor<double>& dV2dcd,
                                                            const ShellPair& s12, const ShellPair& s34,
                                                            const int cart) const;
        private:
            bool pure;
            double cutoff;
            double screen_tol;
            EigenMatrix<double> screen;

            void shell_to_basis_deriv1(const ShellPair& s12, const ShellPair& s34,
                                       const tensor4d1234<double> block_a, const tensor4d1234<double> block_b,
                                       const tensor4d1234<double> block_c, symm4dTensor<double>& dXa, 
                                       symm4dTensor<double>& dXb, symm4dTensor<double>& dXc) const;
            
            void shell_to_basis_deriv2_pq(const ShellPair& s12, const ShellPair& s34,
                                          const tensor4d1234<double> block_aa, const tensor4d1234<double> block_ab,
                                          const tensor4d1234<double> block_ac, tensor4d<double>& dV2daa, 
                                          tensor4d<double>& dV2dab, tensor4d<double>& dV2dac, Permute perm) const;
            
            void shell_to_basis_deriv2_qq(const ShellPair& s12, const ShellPair& s34,
                                          const tensor4d1234<double> block_aa, const tensor4d1234<double> block_ab,
                                          const tensor4d1234<double> block_ac, const tensor4d1234<double> block_bd,
                                          symm4dTensor<double>& dV2daa, symm4dTensor<double>& dV2dab,
                                          symm4dTensor<double>& dV2dac, symm4dTensor<double>& dV2dbd,
                                          symm4dTensor<double>& dV2dcc, symm4dTensor<double>& dV2dcd) const;
            
            void compute_contracted_shell_quartet_deriv2_xx(symm4dTensor<double>& dV2daa, symm4dTensor<double>& dV2dab,
                                                            symm4dTensor<double>& dV2dac, symm4dTensor<double>& dV2dbd,
                                                            symm4dTensor<double>& dV2dcc, symm4dTensor<double>& dV2dcd,
                                                            const ShellPair& s12, const ShellPair& s34) const;
            
            void compute_contracted_shell_quartet_deriv2_yy(symm4dTensor<double>& dV2daa, symm4dTensor<double>& dV2dab,
                                                            symm4dTensor<double>& dV2dac, symm4dTensor<double>& dV2dbd,
                                                            symm4dTensor<double>& dV2dcc, symm4dTensor<double>& dV2dcd,
                                                            const ShellPair& s12, const ShellPair& s34) const;
            
            void compute_contracted_shell_quartet_deriv2_zz(symm4dTensor<double>& dV2daa, symm4dTensor<double>& dV2dab,
                                                            symm4dTensor<double>& dV2dac, symm4dTensor<double>& dV2dbd,
                                                            symm4dTensor<double>& dV2dcc, symm4dTensor<double>& dV2dcd,
                                                            const ShellPair& s12, const ShellPair& s34) const;
            
            void compute_contracted_shell_quartet_deriv2_xy(tensor4d<double>& dV2daa, tensor4d<double>& dV2dab,
                                                            tensor4d<double>& dV2dac, const ShellPair& s12, const ShellPair& s34,
                                                            const Permute perm) const;
            
            void compute_contracted_shell_quartet_deriv2_xz(tensor4d<double>& dV2daa, tensor4d<double>& dV2dab,
                                                            tensor4d<double>& dV2dac, const ShellPair& s12, const ShellPair& s34,
                                                            const Permute perm) const;
            
            void compute_contracted_shell_quartet_deriv2_yz(tensor4d<double>& dV2daa, tensor4d<double>& dV2dab,
                                                            tensor4d<double>& dV2dac, const ShellPair& s12, const ShellPair& s34,
                                                            const Permute perm) const;
    };
}

#endif
// End HFSCF_ERIOS
