/*
 *
 * References:
 * [1] Numerical Recipes
 * William H. Press, Saul A. Teukolsky, William T., 
 * Vetterling and Brian P. Flannery
 * 3rd edition page 261
 * ISBN-13: 978-0521880688
 *
 * The functions below have all been obtained using ref [1]
 */

#pragma once
#include <iostream>
#include <limits>
#include <algorithm>

namespace extramath
{
    double F_nu(const double m, double x);
    double gamm_inc(const double a, const double x);
    double gammp(const double m, const double x);
    double gser(const double a, const double x);
    double gammln(const double xx);
    double gcf(const double a, const double x);
    double gammpapprox(double a, double x, int psig);
}

