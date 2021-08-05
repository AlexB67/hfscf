#include "solid_harmonics.hpp"
#include <boost/math/special_functions/binomial.hpp>
#include <iostream>
#include <array>

double hfscfmath::Cnr(int n, int k)
{
    return boost::math::binomial_coefficient<double>(n, k);
}

double hfscfmath::dblfac(int n)
{
    return boost::math::factorial<double>(n);
}

EigenMatrix<double> hfscfmath::Ylm_transmat(int l)
{
    EigenMatrix<double> ret = EigenMatrix<double>::Zero((l + 1) * (l + 2) / 2, 2 * l + 1);
    EigenVector<double> tmp;

    for (int m = -l; m <= l; ++m)
    {
        tmp = calcYlm_coeff(l, m);
        for (int i = 0; i < tmp.size(); i++) ret(i, l + m) = tmp[i];
    }

    return ret;
}

EigenVector<double> hfscfmath::calcYlm_coeff(int l, int mval)
{
    const auto getind = [](int m_, int n_) -> int
    {
        int ii = m_ + n_;
        int jj = n_;
        return ii * (ii + 1) / 2 + jj;
    };

    const int Ncart = (l + 1) * (l + 2) / 2;

    // Returned array
    EigenVector<double> ret = EigenVector<double>::Zero(Ncart);

    int m = std::abs(mval);

    // Compute prefactor
    //double prefac = std::sqrt((2 * l + 1) / (4.0 * M_PI)) * pow(2.0, -l);
    double prefac = std::pow(2.0, -l);
    constexpr std::array cosf = {1, 0, -1, 0};
    constexpr std::array sinf = {0, 1, 0, -1};

    if (m != 0) prefac *= std::sqrt(dblfac(l - m) * 2.0 / dblfac(l + m));

    for (int k = 0; k <= (l - m) / 2; k++) 
    {
        // Compute prefactor in front
        double ffac = std::pow(-1.0, k) * Cnr(l, k) * Cnr(2 * (l - k), l);
        if (m != 0) ffac *= dblfac(l - 2 * k) / dblfac(l - 2 * k - m);
        ffac *= prefac;

        // Distribute exponents
        for (int a = 0; a <= k; a++) 
        {
            double afac = Cnr(k, a) * ffac;

            for (int b = 0; b <= a; b++)
            {
                double fac = Cnr(a, b) * afac;

                // Current exponents
                int zexp = 2 * (b - k) + l - m;
                int yexp = 2 * (a - b);
                //int xexp = 2 * (k - a);

                // Am or Bm.
                if (mval > 0) 
                {
                    // Contribution from A_m
                    for (int p = 0; p <= m; p++) 
                    {
                        // Check if term contributes
                        int cosfac = cosf[(m - p) % 4];

                        if (cosfac != 0) 
                            ret[getind(yexp + m - p, zexp)] += cosfac * Cnr(m, p) * fac;
                    }
                } 
                else if (m == 0) 
                {
                    // No A_m or B_m term.
                    ret[getind(yexp, zexp)] += fac;
                } 
                else
                {
                    // B_m
                    for (int p = 0; p <= m; p++)
                    {
                        int sinfac = sinf[(m - p) % 4];

                        if (sinfac != 0) 
                            ret[getind(yexp + m - p, zexp)] += sinfac * Cnr(m, p) * fac;
                    }  
                }     
            }         
        }             
    }

    return ret;
}
