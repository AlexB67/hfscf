#ifndef HFSCF_OSNUCLEAR
#define HFSCF_OSNUCLEAR

#include "hfscf_shellpair.hpp"
#include "../math/hfscf_tensors.hpp"
#include <vector>

// forward declare

namespace MOLEC
{
    class Atom;
}

using tensormath::tensor3d;
using tensormath::tensor4d1234;
using tensormath::tensor5d;
using BASIS::ShellPair;
using MOLEC::Atom;
using hfscfmath::Cart;
using hfscfmath::Cart2;
using std::vector;

namespace OSNUCLEAR
{
    enum Center
    {
        aa, ab, ba, bb, cc, ac, bc, dd
    };

    class OSNuclear
    {
        public:
            explicit OSNuclear(const bool is_pure) : pure(is_pure){}
            explicit OSNuclear() = delete;
            OSNuclear(const OSNuclear&) = delete;
            OSNuclear& operator=(const OSNuclear& other) = delete;
            OSNuclear(const OSNuclear&&) = delete;
            OSNuclear&& operator=(const OSNuclear&& other) = delete;
            ~OSNuclear(){};
            
            void compute_contracted_shell(EigenMatrix<double>& V, const ShellPair& sp,
                                         const std::vector<MOLEC::Atom>& atoms) const;
            
            void compute_contracted_shell_deriv1(tensor3d<double>& dVa,  tensor3d<double>& dVb, 
                                                 const ShellPair& sp,
                                                 const std::vector<MOLEC::Atom>& atoms,
                                                 const std::vector<bool>& coords) const;
            
            void compute_contracted_shell_deriv2(tensor4d1234<double>& dVqq,
                                                 const ShellPair& sp,
                                                 const std::vector<MOLEC::Atom>& atoms,
                                                 const Cart2 coord,
                                                 const std::vector<std::vector<bool>>& mask) const;
            
            private:
                bool pure;
    };

    void sumXX(int atom1, int atom2, int atom, double aa, double bb, double ab,
               tensor5d<double>& center, Index ci1, Index ci2, bool a, bool b);
    
    void sumXY(int atom1, int atom2, int atom, double aa, double bb, double ab, double ba,
               tensor5d<double>& center, Index ci1, Index ci2, bool a, bool b);
    
    void sumYX(int atom1, int atom2, int atom, double aa, double bb, double ab, double ba,
               tensor5d<double>& center,Index ci1, Index ci2, bool a, bool b);
    
    double add_centers(int atom1, int atom2, const tensor5d<double>& ctr, Index ci1, Index ci2, 
                       bool a, bool b, bool c, bool d, Cart2 coord);
}
#endif
// End HFSCF_OSNUCLEAR
