#ifndef CART_INT_H
#define CART_INT_H


#include "../molecule/hfscf_molecule.hpp"
#include "libirc/atom.h"
#include "libirc/connectivity.h"
#include "libirc/constants.h"
#include "libirc/conversion.h"
#include "libirc/io.h"
#include "libirc/irc.h"
#include "libirc/linalg.h"
#include "libirc/mathtools.h"
#include "libirc/molecule.h"
#include "libirc/periodic_table.h"
#include "libirc/transformation.h"
#include "libirc/wilson.h"

using namespace irc::molecule;
using namespace irc::linalg;
using namespace irc::periodic_table;
using namespace irc::connectivity;
using namespace irc::wilson;
using namespace irc::tools;
using namespace irc::atom;
using namespace irc::io;

using vec3 = Eigen::Vector3d;
using vec = Eigen::VectorXd;
using mat = Eigen::MatrixXd;

namespace CART_INT
{
    class Cart_int
    {
        public:
            explicit Cart_int(const std::shared_ptr<MOLEC::Molecule>& mol)
            : m_mol(mol){}
            ~Cart_int() = default;
            Cart_int(const Cart_int&) = delete;
            Cart_int& operator=(const Cart_int& other) = delete;
            Cart_int(const Cart_int&&) = delete;
            Cart_int&& operator=(const Cart_int&& other) = delete;
            void do_internal_coord_analysis();
            
            EigenVector<double> get_irc_gradient(const Eigen::Ref<const EigenMatrix<double> >& gradient_cart,
                                                 bool project = true) const;
                                                 
            EigenVector<double> get_cartesian_to_irc() const;
            
            void irc_to_cartesian(const Eigen::Ref<const EigenVector<double> >& irc,
                                  const Eigen::Ref<const EigenVector<double> >& del_irc,
                                  bool do_geometry_analysis = false) const;
            
            void get_guess_hessian(EigenMatrix<double>& guess_hessian, const std::string& type) const;
            
            void rfo_step(const Eigen::Ref<const EigenMatrix<double> >& gradient_cart,
                          EigenVector<double>& irc_coords, EigenVector<double>& irc_grad,
                          EigenMatrix<double>& hessian) const;

            void print_bonding_info(bool print_projector = false, bool print_wilson = false, std::string new_title = {}) const;
        
        private:
            private:
            std::shared_ptr<MOLEC::Molecule> m_mol;
            Eigen::Index num_irc{0};
            std::vector<Bond> m_bond;
            std::vector<Angle> m_angle;
            std::vector<Dihedral> m_dihedral;
            std::vector<LinearAngle<vec3>> m_linear_angle;
            std::vector<OutOfPlaneBend> m_out_of_plane_bend;
            Molecule<vec3> molecule{};
            std::unique_ptr<irc::IRC<vec3, vec, mat> > ircs;
            EigenMatrix<double> B_mat;
            EigenMatrix<double> G_mat;

            void do_bfgs_update(EigenMatrix<double>& hessian,
                                const Eigen::Ref<const EigenVector<double> >& del_irc_grad,
                                const Eigen::Ref<const EigenVector<double> >& del_irc) const;

            // void generalised_inverse(const Eigen::Ref<const EigenMatrix<double> >& in_mat, 
            //                          EigenMatrix<double>& out_mat) const;
            bool is_planar() const;
    };
}

#endif