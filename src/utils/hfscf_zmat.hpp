#ifndef ZMAT_UTIL_H
#define ZMAT_UTIL_H
#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

template<typename T>
using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

using Vec3 = Eigen::RowVector3d;

namespace ZMAT
{
	constexpr double deg_to_rad = M_PI / 180.0;
	constexpr double rad_to_deg = 180.0 / M_PI;
	constexpr double angstrom_to_bohr = 1.889725989;

	struct zparams 
	{
		std::string atom_name;
		int r_ij_num;
		int angle_num;
		int dihedral_num;
		double r_ij;
		double angle;
		double dihedral_angle;
	};

	struct cart
	{
		Vec3 x;
		Vec3 y;
		Vec3 z;
	};

    class ZtoCart
	{
		public:
      		explicit ZtoCart(const std::string& unit_type) : m_unit_type(unit_type){};
			~ZtoCart() = default;
      		ZtoCart(const ZtoCart&) = delete;
      		ZtoCart& operator=(const ZtoCart& other) = delete;
      		ZtoCart(const ZtoCart&&) = delete;
      		ZtoCart&& operator=(const ZtoCart&& other) = delete;
      		
      		void get_cartesians_from_zarray(EigenMatrix<double>& cart) const;

			void add_row(const std::string& atom_name, const std::string& r_ij_num, const std::string& angle_num, 
                         const std::string& dihedral_num, const std::string& r_ij, const std::string& angle, 
                         const std::string& dihedral_angle);

			std::vector<zparams> get_zmatrix() const { return zarray;}
      	
      	private: 
      		int m_natoms{0};
			int m_dummies{0};
			std::string m_unit_type{"angstrom"};
			std::vector<zparams> zarray;

			double udot(const Eigen::Ref<const Vec3>& r1, const Eigen::Ref<const Vec3>& r2) const;
			Vec3 ucross(const Eigen::Ref<const Vec3>& r1, const Eigen::Ref<const Vec3>& r2) const;

			cart transform_axes(const Eigen::Ref<const Vec3>& r1, 
						  		const Eigen::Ref<const Vec3>& r2,
                          		const Eigen::Ref<const Vec3>& r3) const;
	};
}

#endif
// end ZMAT_UTIL_H

