#ifndef SPHERICAL_TRANSFORM_H
#define SPHERICAL_TRANSFORM_H

#include "hfscf_shellpair.hpp"
#include "../math/hfscf_tensors.hpp"

using BASIS::ShellPair;
using tensormath::tensor3d;
using tensormath::tensor4d1234;

namespace TRANSFORM
{   // transform a Cartesian shell to spherical shell block
    void transform(const ShellPair& sp,  const Eigen::Ref<EigenMatrix<double>>& block, 
                   EigenMatrix<double>& M, const bool pure);
    
    void transform(const ShellPair& sp, const Eigen::Ref<EigenMatrix<double>>& sx_block,
                   const Eigen::Ref<EigenMatrix<double>>& sy_block, 
                   const Eigen::Ref<EigenMatrix<double>>& sz_block,
                   tensor3d<double>& S, const bool pure);
    
    void transform(const ShellPair& sp12, const ShellPair& sp34, const tensor4d1234<double>& block,
                   tensor4d1234<double>& pure_block);
    
    void transform(const ShellPair& sp12, const ShellPair& sp34, const tensor4d1234<double>& block1,
                   const tensor4d1234<double>& block2, const tensor4d1234<double>& block3, 
                   tensor4d1234<double>& pure_block1, tensor4d1234<double>& pure_block2,
                   tensor4d1234<double>& pure_block3);

}

#endif