/*
 * References:
 * [1] Numerical Recipes
 * William H. Press, Saul A. Teukolsky, William T.,
 * Vetterling and Brian P. Flannery
 * 3rd edition page 261
 * ISBN-13: 978-0521880688
 *
 * The functions below have all been obtained using ref [1]
 * taken from hfcxx 
 * changes: constexpr where appropriate
 * Add namespace for better encapsulation
 */

#include "gamma.hpp"
#include <cmath>

double extramath::F_nu(const double m, double x)
{
    double tiny = 0.00000001;
    x = std::max(std::abs(x), tiny);
    double val = gamm_inc(m + 0.5, x);
    return 0.5 * pow(x, -m - 0.5) * val;
}

double extramath::gamm_inc(const double a, const double x)
{
    double gammap = gammp(a, x);
    double gln = gammln(a);
    return exp(gln) * gammap;
}

double extramath::gammp(const double a, double x)
{
    static constexpr int ASWITCH = 100;

    if (x < 0.0 || a <= 0.0)
    {
        std::cout << "Bad value in gammp!";
        return 0.0;
    }

    if (x == 0.0)
    {
        return 0.0;
    }
    else if (a >= ASWITCH)
    {
        return gammpapprox(a, x, 1); /* use quadrature */
    }
    else if (x < a + 1.0)
    {
        return gser(a, x); /* use the series expansion */
    }
    else
    {
        return 1.0 - gcf(a, x); /* use the continued fraction expansion */
    }
}

double extramath::gser(const double a, const double x)
{
    constexpr double EPS = std::numeric_limits<double>::epsilon();

    double sum, del, ap;
    double gln = gammln(a);
    ap = a;
    del = sum = 1.0 / a;
    for (;;)
    {
        ++ap;
        del *= x / ap;
        sum += del;
        if (fabs(del) < fabs(sum) * EPS)
        {
            return sum * exp(-x + a * log(x) - gln);
        }
    }
}

double extramath::gammln(const double xx)
{
    int j;
    double x, tmp, y, ser;
    static const double cof[14] = {57.1562356658629235, -59.5979603554754912,
                                   14.1360979747417471, -0.491913816097620199, .339946499848118887e-4,
                                   .465236289270485756e-4, -.983744753048795646e-4, .158088703224912494e-3,
                                   -.210264441724104883e-3, .217439618115212643e-3, -.164318106536763890e-3,
                                   .844182239838527433e-4, -.261908384015814087e-4, .368991826595316234e-5};
    if (xx <= 0)
    {
        std::cout << "Bad argument in gammln";
        return 0.0;
    }

    y = x = xx;
    tmp = x + 5.24218750000000000;
    tmp = (x + 0.5) * log(tmp) - tmp;
    ser = 0.999999999999997092;
    for (j = 0; j < 14; j++)
        ser += cof[j] / ++y;
    return tmp + log(2.5066282746310005 * ser / x);
}

double extramath::gcf(const double a, const double x)
{
    constexpr double EPS = std::numeric_limits<double>::epsilon();
    constexpr double FPMIN = std::numeric_limits<double>::min() / EPS;

    int i;
    double an, b, c, d, del, h;
    double gln = gammln(a);
    b = x + 1.0 - a;
    c = 1.0 / FPMIN;
    d = 1.0 / b;
    h = d;
    for (i = 1;; i++)
    {
        an = -i * (i - a);
        b += 2.0;
        d = an * d + b;
        if (fabs(d) < FPMIN)
            d = FPMIN;
        c = b + an / c;
        if (fabs(c) < FPMIN)
            c = FPMIN;
        d = 1.0 / d;
        del = d * c;
        h *= del;
        if (fabs(del - 1.0) <= EPS)
            break;
    }

    return exp(-x + a * log(x) - gln) * h;
}

double extramath::gammpapprox(double a, double x, int psig)
{
    static const int ngau = 18;

    constexpr double y[18] = {0.0021695375159141994,
                    0.011413521097787704, 0.027972308950302116, 0.051727015600492421,
                    0.082502225484340941, 0.12007019910960293, 0.16415283300752470,
                    0.21442376986779355, 0.27051082840644336, 0.33199876341447887,
                    0.39843234186401943, 0.46931971407375483, 0.54413605556657973,
                    0.62232745288031077, 0.70331500465597174, 0.78649910768313447,
                    0.87126389619061517, 0.95698180152629142};

    constexpr double w[18] = {0.0055657196642445571,
                    0.012915947284065419, 0.020181515297735382, 0.027298621498568734,
                    0.034213810770299537, 0.040875750923643261, 0.047235083490265582,
                    0.053244713977759692, 0.058860144245324798, 0.064039797355015485,
                    0.068745323835736408, 0.072941885005653087, 0.076598410645870640,
                    0.079687828912071670, 0.082187266704339706, 0.084078218979661945,
                    0.085346685739338721, 0.085983275670394821};

    int j;
    double xu, t, sum, ans;
    double a1 = a - 1.0, lna1 = log(a1), sqrta1 = sqrt(a1);
    double gln = gammln(a);

    if (x > a1)
        xu = std::max(a1 + 11.5 * sqrta1, x + 6.0 * sqrta1);
    else
        xu = std::max(0., std::min(a1 - 7.5 * sqrta1, x - 5.0 * sqrta1));
    sum = 0;

    for (j = 0; j < ngau; j++)
    {
        t = x + (xu - x) * y[j];
        sum += w[j] * exp(-(t - a1) + a1 * (log(t) - lna1));
    }

    ans = sum * (xu - x) * exp(a1 * (lna1 - 1.) - gln);
    return (psig ? (ans > 0.0 ? 1.0 - ans : -ans) : (ans >= 0.0 ? ans : 1.0 + ans));
}
