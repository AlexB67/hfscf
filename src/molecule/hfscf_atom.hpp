#ifndef MOLEC_ATOM_H
#define MOLEC_ATOM_H

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>


template<typename T>
using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using InertiaTensor = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;

namespace MOLEC
{
    class Atom
    {
        private:
            int m_Z;
            Vec3D m_r;
            Eigen::Index m_cart_bfstart{0};
            Eigen::Index m_cart_nbfs{0};
            Eigen::Index m_pure_bfstart{0};
            Eigen::Index m_pure_nbfs{0};
            
        public:
            explicit Atom(const int Z, const Eigen::Ref<const Vec3D>& r)
            : m_Z(Z), m_r(r){}
            int get_Z() const { return m_Z;}
            const Eigen::Ref<const Vec3D> get_r() const { return m_r;}
            void set_r(const Eigen::Ref<const Vec3D>& r) { m_r = r;}

            void set_cart_basis_params(Eigen::Index bfstart, Eigen::Index nbfs) 
            { m_cart_bfstart = bfstart; m_cart_nbfs = nbfs;}

            void set_pure_basis_params(Eigen::Index bfstart, Eigen::Index nbfs) 
            { m_pure_bfstart = bfstart; m_pure_nbfs = nbfs;}

            Eigen::Index get_cart_nbfs() const { return m_cart_nbfs;}
            Eigen::Index get_pure_nbfs() const { return m_pure_nbfs;}
            Eigen::Index get_cart_offset() const { return m_cart_bfstart;}
            Eigen::Index get_pure_offset() const { return m_pure_bfstart;}

    };
}

#endif
// End MOLEC_ATOM_H
