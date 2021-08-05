#include "hfscf_transform.hpp"

using BASIS::Shell;


void TRANSFORM::transform(const ShellPair& sp, const Eigen::Ref<EigenMatrix<double>>& block, 
                          EigenMatrix<double>& M, const bool pure)
{
    const Shell& sh1 = sp.m_s1;
    const Shell& sh2 = sp.m_s2;

    if (pure) // Spherical basis
    {
        const EigenMatrix<double>& tr_left = sh1.get_spherical_form().transpose().eval();
        const EigenMatrix<double>& tr_right = sh2.get_spherical_form().eval();

        EigenMatrix<double> pure_block = (tr_left * block.eval() * tr_right);

        for (Index i = 0; i < sh1.get_sirange(); ++i)
            for (Index j = 0; j < sh2.get_sirange(); ++j)
            {
                M(sh1.get_ids() + i, sh2.get_ids() + j) = pure_block(i, j);
                M(sh2.get_ids() + j, sh1.get_ids() + i) = M(sh1.get_ids() + i, sh2.get_ids() + j);
            }
    }
    else // Cartesian basis
    {
        for (Index i = 0; i < sh1.get_cirange(); ++i)
            for (Index j = 0; j < sh2.get_cirange(); ++j)
            {
                M(sh1.get_idx() + i, sh2.get_idx() + j) = block(i, j);
                M(sh2.get_idx() + j, sh1.get_idx() + i) = M(sh1.get_idx() + i, sh2.get_idx() + j);
            }
    }
}

void TRANSFORM::transform(const ShellPair& sp, const Eigen::Ref<EigenMatrix<double>>& sx_block,
                          const Eigen::Ref<EigenMatrix<double>>& sy_block, 
                          const Eigen::Ref<EigenMatrix<double>>& sz_block,
                          tensor3d<double>& S, const bool pure)
{
    const auto& sh1 = sp.m_s1;
    const auto& sh2 = sp.m_s2;
    if (pure) // Spherical basis
    {
        const EigenMatrix<double>& tr_left = sh1.get_spherical_form().transpose().eval();
        const EigenMatrix<double>& tr_right = sh2.get_spherical_form().eval();

        EigenMatrix<double> sx_pure_block = (tr_left * sx_block * tr_right);
        EigenMatrix<double> sy_pure_block = (tr_left * sy_block * tr_right);
        EigenMatrix<double> sz_pure_block = (tr_left * sz_block * tr_right);


        for (Index i = 0; i < sh1.get_sirange(); ++i)
            for (Index j = 0; j < sh2.get_sirange(); ++j)
            {
                S(0, sh1.get_ids() + i, sh2.get_ids() + j) = sx_pure_block(i, j);
                S(0, sh2.get_ids() + j, sh1.get_ids() + i) = S(0, sh1.get_ids() + i, sh2.get_ids() + j);
                S(1, sh1.get_ids() + i, sh2.get_ids() + j) = sy_pure_block(i, j);
                S(1, sh2.get_ids() + j, sh1.get_ids() + i) = S(1, sh1.get_ids() + i, sh2.get_ids() + j);
                S(2, sh1.get_ids() + i, sh2.get_ids() + j) = sz_pure_block(i, j);
                S(2, sh2.get_ids() + j, sh1.get_ids() + i) = S(2, sh1.get_ids() + i, sh2.get_ids() + j);
            }
    }
    else // Cartesian basis
    {
        for (Index i = 0; i < sh1.get_cirange(); ++i)
            for (Index j = 0; j < sh2.get_cirange(); ++j)
            {
                S(0, sh1.get_idx() + i, sh2.get_idx() + j) = sx_block(i, j);
                S(0, sh2.get_idx() + j, sh1.get_idx() + i) = S(0, sh1.get_idx() + i, sh2.get_idx() + j);
                S(1, sh1.get_idx() + i, sh2.get_idx() + j) = sy_block(i, j);
                S(1, sh2.get_idx() + j, sh1.get_idx() + i) = S(1, sh1.get_idx() + i, sh2.get_idx() + j);
                S(2, sh1.get_idx() + i, sh2.get_idx() + j) = sz_block(i, j);
                S(2, sh2.get_idx() + j, sh1.get_idx() + i) = S(2, sh1.get_idx() + i, sh2.get_idx() + j);
            }
    }
}

void TRANSFORM::transform(const ShellPair& sp12, const ShellPair& sp34, const tensor4d1234<double>& block,
                          tensor4d1234<double>& pure_block)
{
    const Shell& s1 = sp12.m_s1;
    const Shell& s2 = sp12.m_s2;
    const Shell& s3 = sp34.m_s1;
    const Shell& s4 = sp34.m_s2;

    const Index num_so1 = s1.get_sirange();
    const Index num_so2 = s2.get_sirange(); 
    const Index num_so3 = s3.get_sirange(); 
    const Index num_so4 = s4.get_sirange();

    const Index num_cart1 = s1.get_cirange();
    const Index num_cart2 = s2.get_cirange(); 
    const Index num_cart3 = s3.get_cirange(); 
    const Index num_cart4 = s4.get_cirange();

    const EigenMatrix<double>& tr1 = s1.get_spherical_form().transpose();
    const EigenMatrix<double>& tr2 = s2.get_spherical_form().transpose();
    const EigenMatrix<double>& tr3 = s3.get_spherical_form().transpose();
    const EigenMatrix<double>& tr4 = s4.get_spherical_form().transpose();

    tensor4d1234<double> tmp1 = tensor4d1234<double>(num_cart1, num_cart2, num_cart3, num_cart4);
    tensor4d1234<double> tmp2 = tensor4d1234<double>(num_cart1, num_cart2, num_cart3, num_cart4);

     for (Index i = 0; i < num_so1; ++i)
        for (Index p = 0; p < num_cart1; ++p)
        {
            if (fabs(tr1(i, p)) < 1E-16) continue;  // avoid excessive looping - matrix is sparse
                                                    // big time saver
            for (Index j = 0; j < num_cart2; ++j)
                for (Index k = 0; k < num_cart3; ++k) 
                    for (Index l = 0; l < num_cart4; ++l)
                        tmp1(i, j, k, l) += block(p, j, k, l) * tr1(i, p);
        }

     for (Index i = 0; i < num_so1; ++i) 
        for (Index j = 0; j < num_so2; ++j)
            for (Index q = 0; q < num_cart2; ++q)
            {
                if (fabs(tr2(j, q)) < 1E-16) continue;

                for (Index k = 0; k < num_cart3; ++k) 
                    for (Index l = 0; l < num_cart4; ++l)  
                        tmp2(i, j, k, l) += tmp1(i, q, k, l) * tr2(j, q);
            }
    
    tmp1.setZero();

    for (Index k = 0; k < num_so3; ++k)
        for (Index r = 0; r < num_cart3; ++r)
        {
            if (fabs(tr3(k, r)) < 1E-16) continue;

            for (Index i = 0; i < num_so1; ++i)
                for (Index j = 0; j < num_so2; ++j)
                    for (Index l = 0; l < num_cart4; ++l) 
                        tmp1(i, j, k, l) += tmp2(i, j, r, l) * tr3(k, r);
        }


    for (Index l = 0; l < num_so4; ++l)
        for (Index s = 0; s < num_cart4; ++s) 
        {
            if (fabs(tr4(l, s)) < 1E-16) continue;

            for (Index i = 0; i < num_so1; ++i)
                for (Index j = 0; j < num_so2; ++j)
                    for (Index k = 0; k < num_so3; ++k)
                    {
                        pure_block(i, j, k, l) += tmp1(i, j, k, s) * tr4(l, s);
                    }
        }
}

void TRANSFORM::transform(const ShellPair& sp12, const ShellPair& sp34, const tensor4d1234<double>& block1,
                          const tensor4d1234<double>& block2, const tensor4d1234<double>& block3, 
                          tensor4d1234<double>& pure_block1, tensor4d1234<double>& pure_block2,
                          tensor4d1234<double>& pure_block3)
{
    const Shell& s1 = sp12.m_s1;
    const Shell& s2 = sp12.m_s2;
    const Shell& s3 = sp34.m_s1;
    const Shell& s4 = sp34.m_s2;

    const Index num_so1 = s1.get_sirange();
    const Index num_so2 = s2.get_sirange(); 
    const Index num_so3 = s3.get_sirange(); 
    const Index num_so4 = s4.get_sirange();

    const Index num_cart1 = s1.get_cirange();
    const Index num_cart2 = s2.get_cirange(); 
    const Index num_cart3 = s3.get_cirange(); 
    const Index num_cart4 = s4.get_cirange();

    const EigenMatrix<double>& tr1 = s1.get_spherical_form().transpose();
    const EigenMatrix<double>& tr2 = s2.get_spherical_form().transpose();
    const EigenMatrix<double>& tr3 = s3.get_spherical_form().transpose();
    const EigenMatrix<double>& tr4 = s4.get_spherical_form().transpose();

    tensor4d1234<double> tmp1a = tensor4d1234<double>(num_cart1, num_cart2, num_cart3, num_cart4);
    tensor4d1234<double> tmp2a = tensor4d1234<double>(num_cart1, num_cart2, num_cart3, num_cart4);
    tensor4d1234<double> tmp1b = tensor4d1234<double>(num_cart1, num_cart2, num_cart3, num_cart4);
    tensor4d1234<double> tmp2b = tensor4d1234<double>(num_cart1, num_cart2, num_cart3, num_cart4);
    tensor4d1234<double> tmp1c = tensor4d1234<double>(num_cart1, num_cart2, num_cart3, num_cart4);
    tensor4d1234<double> tmp2c = tensor4d1234<double>(num_cart1, num_cart2, num_cart3, num_cart4);

     for (Index i = 0; i < num_so1; ++i)
        for (Index p = 0; p < num_cart1; ++p)
        {
            if (fabs(tr1(i, p)) < 1E-16) continue;  // avoid excessive looping - matrix is sparse
                                                    // big time saver
            for (Index j = 0; j < num_cart2; ++j)
                for (Index k = 0; k < num_cart3; ++k) 
                    for (Index l = 0; l < num_cart4; ++l)
                    {
                        tmp1a(i, j, k, l) += block1(p, j, k, l) * tr1(i, p);
                        tmp1b(i, j, k, l) += block2(p, j, k, l) * tr1(i, p);
                        tmp1c(i, j, k, l) += block3(p, j, k, l) * tr1(i, p);
                    }
        }

     for (Index i = 0; i < num_so1; ++i) 
        for (Index j = 0; j < num_so2; ++j)
            for (Index q = 0; q < num_cart2; ++q)
            {
                if (fabs(tr2(j, q)) < 1E-16) continue;

                for (Index k = 0; k < num_cart3; ++k) 
                    for (Index l = 0; l < num_cart4; ++l)
                    {
                        tmp2a(i, j, k, l) += tmp1a(i, q, k, l) * tr2(j, q);
                        tmp2b(i, j, k, l) += tmp1b(i, q, k, l) * tr2(j, q);
                        tmp2c(i, j, k, l) += tmp1c(i, q, k, l) * tr2(j, q);
                    }
            }
    
    tmp1a.setZero(); tmp1b.setZero(); tmp1c.setZero(); 

    for (Index k = 0; k < num_so3; ++k)
        for (Index r = 0; r < num_cart3; ++r)
        {
            if (fabs(tr3(k, r)) < 1E-16) continue;

            for (Index i = 0; i < num_so1; ++i)
                for (Index j = 0; j < num_so2; ++j)
                    for (Index l = 0; l < num_cart4; ++l)
                    {
                        tmp1a(i, j, k, l) += tmp2a(i, j, r, l) * tr3(k, r);
                        tmp1b(i, j, k, l) += tmp2b(i, j, r, l) * tr3(k, r);
                        tmp1c(i, j, k, l) += tmp2c(i, j, r, l) * tr3(k, r);
                    }
        }


    for (Index l = 0; l < num_so4; ++l)
        for (Index s = 0; s < num_cart4; ++s) 
        {
            if (fabs(tr4(l, s)) < 1E-16) continue;

            for (Index i = 0; i < num_so1; ++i)
                for (Index j = 0; j < num_so2; ++j)
                    for (Index k = 0; k < num_so3; ++k)
                    {
                        pure_block1(i, j, k, l) += tmp1a(i, j, k, s) * tr4(l, s);
                        pure_block2(i, j, k, l) += tmp1b(i, j, k, s) * tr4(l, s);
                        pure_block3(i, j, k, l) += tmp1c(i, j, k, s) * tr4(l, s);
                    }
        }
}
