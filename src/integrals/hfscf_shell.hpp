#ifndef HFSCF_SHELL
#define HFSCF_SHELL
#include "../math/hfscf_math.hpp"
#include <vector>

template <typename T>
using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Eigen::Index;

namespace BASIS
{

struct idq
{
    Index l;
    Index m;
    Index n;
    Index ci;
};

class Shell
{
    private:
        bool m_ispure;
        int m_L;
        Index m_idx; // cartesian index
        Index m_ids; // spherical index
        Vec3D m_r;
        EigenVector<double> m_c;
        EigenVector<double> m_alpha;
        std::vector<BASIS::idq> indices;
        EigenVector<double> m_c_unscaled;
        EigenMatrix<double> cart_to_spherical;
        int cirange;
        int sirange;

    public:
        explicit Shell(bool ispure, const Index L, const Index idx, const Index ids, const Eigen::Ref<const Vec3D>& r, 
                       const EigenVector<double> c,
                       const EigenVector<double> alpha);
        
        Index L() const {return m_L;}
        Index get_idx() const {return m_idx;}
        Index get_ids() const {return m_ids;}
        double x() const {return m_r(0);}
        double y() const {return m_r(1);}
        double z() const {return m_r(2);}
        const Eigen::Ref<const EigenVector<double>> alpha() const {return m_alpha;}
        const Eigen::Ref<const EigenVector<double>> c() const { return m_c;}
        const Eigen::Ref<const EigenVector<double>> c_unscaled() const { return m_c_unscaled;}
        const Eigen::Ref<const EigenMatrix<double>> get_spherical_form() const { return cart_to_spherical;}
        const Eigen::Ref<const Vec3D> r() const {return m_r;}
        Index get_cirange() const {return cirange;}
        Index get_sirange() const {return sirange;}
        const std::vector<BASIS::idq>& get_indices() const { return indices;}
        void set_r(const Eigen::Ref<const Vec3D>& r){m_r = r;}
};
}

#endif
// end HFSCF_SHELL
