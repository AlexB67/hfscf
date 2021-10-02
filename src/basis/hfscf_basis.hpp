#ifndef HFSCF_BASIS
#define HFSCF_BASIS
#include "../integrals/hfscf_shell.hpp"
#include <iostream>
#include <sstream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <map>
#include <cmath>

template <typename T>
using Matrix1D = std::vector<T>;

using BASIS::Shell;
using ShellVector = std::vector<Shell>;

namespace BASIS
{
struct Basisshell_data
{
    public:
        explicit Basisshell_data(const int Z, const size_t atom_num, const int L,
                                 const std::vector<double>& alpha,
                                 const std::vector<double>& c)
        : m_Z(Z),
          m_atom_num(atom_num),
          m_L(L),
          m_alpha(alpha),
          m_c(c){}

        int get_Z() const {return m_Z; }
        size_t get_atom_num() const {return m_atom_num;}
        int get_L() const {return m_L;}
        size_t get_alpha_size() const {return m_alpha.size();}
        double get_alpha(const size_t i) const {return m_alpha[i];}
        double get_c(const size_t i) const {return m_c[i];}
        
    private:
        int  m_Z;
        size_t        m_atom_num;
        int           m_L;
        std::vector<double> m_alpha;
    	std::vector<double> m_c;
};

class Basisset
{
    public:
      explicit Basisset(const std::string& basis_set_name, const Matrix1D<int>& zval);
        ~Basisset() = default;
        Basisset(const Basisset&) = delete;
        Basisset& operator=(const Basisset& other) = delete;
        Basisset(const Basisset&&) = delete;
        Basisset&& operator=(const Basisset&& other) = delete;
        
        void load_basis_set(const char *filename);
        
        void emplace_back_Shell(const int L, const int idx, 
                                const Eigen::Ref<const Vec3D>& r,
                                const Eigen::Ref<const EigenVector<double>>& c,
                                const Eigen::Ref<const EigenVector<double>>& alpha);

        // must copy since shells will be destroyed ditto for shells
        ShellVector get_shells(const size_t atom_number, const Eigen::Ref<const Vec3D>& r);
        
        int get_num_gtos() const {return num_gtos;}
        int get_num_unique_gtos() const {return num_unique_gtos;}
        int get_num_shells() const {return num_shells;}
        std::string get_basis_coord_type() const {return m_basis_coord_type;}

    private:
        int num_gtos{0};  // cartesian type;
        int num_sgtos{0}; // spherical type;
        int num_unique_gtos{0};
        int num_shells{0};
        int shell_idx{0}; // cartesian index;
        int shell_ids{0}; // spherical index;
        std::string m_basis_set_name{"STO-3G"};
        std::string m_basis_coord_type{"cartesian"}; // Default: Basis set is Cartesian or Spherical;
        Matrix1D<int> m_zval;                        
        std::vector<Basisshell_data> m_shell_basis;
};
}

#endif
// end BASIS
