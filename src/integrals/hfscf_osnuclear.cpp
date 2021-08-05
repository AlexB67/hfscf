#include "../molecule/hfscf_molecule.hpp"
#include "../math/hfscf_math.hpp"
#include "../math/gamma.hpp"
#include "hfscf_osrecur.hpp"
#include "hfscf_osnuclear.hpp"
#include "hfscf_transform.hpp"
#include <functional>

using hfscfmath::rab2;
using hfscfmath::gpc;
using hfscfmath::pi;
using extramath::F_nu;
using os_recursion::osrecurpot3c;
using TRANSFORM::transform;

using Eigen::Index;
void OSNUCLEAR::OSNuclear::compute_contracted_shell(EigenMatrix<double>& V, const ShellPair& sp,
                                                    const std::vector<MOLEC::Atom>& atoms) const
{
    const auto& sh1 = sp.m_s1;
    const auto& sh2 = sp.m_s2;

    Index L1 = sp.m_s1.L(); Index L2 = sp.m_s2.L();
    const Index nmax = 3 * (L1 + L2);
    
    const Index dim1 = L1 + 1; const Index dim2 = L2 + 1;
    const Index dim11122 =  dim1 * dim1 * dim1 * dim2 * dim2;
    const Index dim1112 =  dim1 * dim1 * dim1 * dim2;
    const Index dim111 =  dim1 * dim1 * dim1;
    const Index dim11 =  dim1 * dim1;

    const auto idx = [&](Index n2, Index m2, Index l2, Index n1, Index m1, Index l1) -> Index
    {
        return  dim11122 * n2 +
                dim1112 * m2 + 
                dim111 * l2 +  
                dim11 * n1 + 
                dim1 * m1 +
                l1;
    };

    EigenMatrix<double> Vs = EigenMatrix<double>(dim2 * dim2 * dim2 * dim1 * dim1 * dim1, nmax + 1);
    EigenMatrix<double> v_block = EigenMatrix<double>::Zero(sh1.get_cirange(), sh2.get_cirange());

    const Eigen::Ref<const EigenVector<double>>& c1 = sp.m_s1.c();
    const Eigen::Ref<const EigenVector<double>>& c2 = sp.m_s2.c();
    const auto&id1 = sh1.get_indices();
    const auto&id2 = sh2.get_indices();

    for (const auto& m : atoms)
    {
        const double charge = m.get_Z();
        
        for (Index p1 = 0; p1 < c1.size(); ++p1)
        {
            for (Index p2 = 0; p2 < c2.size(); ++p2)
            {
                const Index id = p1 * c2.size() + p2;
                const double gamma12 = sp.gamma_ab(id);
                const double r_pc2  = rab2(sp.P(id), m.get_r());
                const double T = gamma12 * r_pc2;
                const double prefac = charge * sp.pfac2(id);

                osrecurpot3c<double>(sp.PA(id), sp.PB(id), sp.P(id) - m.get_r(), gamma12, r_pc2,
                                    T, F_nu(static_cast<double>(nmax), T), L1, L1, L1, L2, L2, L2, nmax, Vs, idx);

                for (const auto &i : id1)
                {
                    Index l1 = i.l; Index m1 = i.m;  Index n1 = i.n;  Index ci1 = i.ci; 
                    for (const auto &j : id2)
                    {
                        Index ci2 = j.ci;
                        Index l2 = j.l; Index m2 = j.m; Index n2 = j.n;

                        v_block(ci1, ci2) -= prefac * Vs(idx(n2, m2, l2, n1, m1, l1), 0);
                    }
                }
            }
        }
    }

    transform(sp, v_block, V, pure);
}
