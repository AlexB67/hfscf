
#ifndef TENSOR4D_MATH_H
#define TENSOR4D_MATH_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>

// minimalist tensor arrays for Post SCF
// optimsed for p, r, q, s loop order

template <typename T>
using EigenVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
using Eigen::Index;

namespace tensor4dmath
{
    template <typename T>
    class tensor4d
    {
        private: 
            Index size{0};
            EigenVector<T> t4;

        public:
            explicit tensor4d(Index t4_size) : size(t4_size)
            {
                t4 = EigenVector<T>::Zero(size * size * size * size);
            }
            
            explicit tensor4d(){}
            ~tensor4d() = default;

            tensor4d& operator=(const tensor4d& other)
            {
                if (&other == this)
                    return *this;
                
                this->t4 = other.t4;
                this->size = other.size;

                return *this;
            }

            tensor4d(tensor4d&& other)
            {
                this->t4 = other.t4;
                other.t4 = nullptr;
                this->size = other.size;
            }

            tensor4d(const tensor4d& other)
            {
                if (&other == this)
                    return;
                
                this->t4 = other.t4;
                this->size = other.size;
            }

            const T& operator()(const Index p, const Index q, const Index r, const Index s) const
            {
                // was Index pqrs = size * p + size * size * q + size * size * size * r + s; old
                // now Index pqrs = size * size * size * p + size * q + size * size * r + s ; 
                // new ordering gives considerable speed up due to physics indexing swap of q and r
                return this->t4[size * size * size * p + size * q + size * size * r + s];
            }

            T& operator()(const Index p, const Index q, const Index r, const Index s)
            {
                return this->t4[size * size * size * p + size * q + size * size * r + s];
            }

            const T& operator()(Index index) const
            {   
                return this->t4[index];
            }

            T& operator()(Index index)
            {
                return this->t4[index];
            }

            void setDim(Index t4size)
            {
                size = t4size;
                t4 = EigenVector<T>::Zero(size * size * size * size);
            }

            void resize(Index new_dim)
            {
                size = new_dim;
                t4.EigenVector<T>::resize(size * size * size * size);
                t4.setZero();
            }

            Index getSize()
            {
                return size * size * size * size;
            }

            void setZero()
            {
                t4.setZero();
            }
    };

    constexpr auto index2 = [](const Index i, const Index j) noexcept -> Index 
    {
        Index ij; // surprisingly bitshift gives slight speed up when used with symm4dTensor
        (i > j) ? ij = (i * (i + 1) >> 1) + j : ij = (j * (j + 1) >> 1) + i;
        return ij;
    };

    constexpr auto index4 = [](const Index i, const Index j, const Index k, const Index l) noexcept -> Index 
    {
        Eigen::Index ij;
        Eigen::Index kl;
        Eigen::Index ijkl;

        (i > j) ? ij = (i * (i + 1) >> 1) + j : ij = (j * (j + 1) >> 1) + i;
        (k > l) ? kl = (k * (k + 1) >> 1) + l : kl = (l * (l + 1) >> 1) + k;
        (ij > kl) ? ijkl = (ij * (ij + 1) >> 1) + kl : ijkl = (kl * (kl + 1) >> 1) + ij;
        
        return ijkl;
    };

    // includes skew symmetric via asymm call, not used so far
    template <typename T>
    class symm2dTensor
    {
        private: 
            Index size{0};
            EigenVector<T> t2;

        public:
            explicit symm2dTensor(Index t2_size) : size(t2_size)
            {
               t2 = EigenVector<T>::Zero(size * (size + 1) / 2);
            }
            
            explicit symm2dTensor(){}
            ~symm2dTensor() = default;

            symm2dTensor& operator=(const symm2dTensor& other)
            {
                if (&other == this)
                    return *this;
                
                this->t2 = other.t2;
                this->size = other.size;

                return *this;
            }

            symm2dTensor(symm2dTensor&& other)
            {
                this->t2 = other.t2;
                other.t2 = nullptr;
                this->size = other.size;
            }

            symm2dTensor(const symm2dTensor& other)
            {
                if (&other == this)
                    return;
                
                this->t2 = other.t2;
                this->size = other.size;
            }

            const T& operator()(const Index p, const Index q) const
            {
                const Index pq = index2(p, q);
                return this->t2[pq]; 
            }                           

            T& operator()(const Index p, const Index q)
            {
                const Index pq = index2(p, q);
                return this->t2[pq];
            }                 

            const T asymm(const Index p, const Index q) const noexcept
            {
                const Index pq = index2(p, q);

                if(p > q) return -this->t2[pq];

                return this->t2[pq]; 
            }

            void setDim(Index t2size)
            {
                t2 = EigenVector<T>::Zero(t2size * (t2size + 1 ) / 2);
                size = t2size;
            }

            void setZero()
            {
                t2.setZero();
            }

            Index getSize()
            {
                return size * (size + 1) / 2;
            }
    };

    // Uses symmetry with pqrs indices to save substantial memory.
    // Slower versus Tensor4d since indices have to be calculated.


    template <typename T>
    class symm4dTensor // includes skew symmetric via asymm call if used appropriately. See trans_basis class
    {
        private: 
            Index size{0};
            EigenVector<T> t4;

        public:
            explicit symm4dTensor(Index t4_size) : size(t4_size)
            {
                t4 = EigenVector<T>::Zero(size * (size + 1) *  (size * size + size + 2 ) / 8);
            }
            
            explicit symm4dTensor(){}
            ~symm4dTensor() = default;

            symm4dTensor& operator=(const symm4dTensor& other)
            {
                if (&other == this)
                    return *this;
                
                this->t4 = other.t4;
                this->size = other.size;

                return *this;
            }

            symm4dTensor(symm4dTensor&& other)
            {
                this->t4 = other.t4;
                other.t4 = nullptr;
                this->size = other.size;
            }

            symm4dTensor(const symm4dTensor& other)
            {
                if (&other == this)
                    return;
                
                this->t4 = other.t4;
                this->size = other.size;
            }

            const T& operator()(Index p, Index q, Index r, Index s) const
            {
                const Index pqrs = index4(p, q, r, s);
                return t4[pqrs];
            }                           

            T& operator()(Index p, Index q, Index r, Index s)
            {
                const Index pqrs = index4(p, q, r, s);
                return t4[pqrs];
            }

            const T& operator[](Index pqrs) const
            {
                return t4[pqrs];
            }                           

            T& operator[](Index pqrs)
            {
                return t4[pqrs];
            }

            const T asymm(const Index p, const Index q, const Index r, const Index s) const noexcept
            {
                int sign = 1;

                if(p < q) sign *= -1;
                if(r < s) sign *= -1;

                const Index pqrs = index4(p, q, r, s);

                return sign * t4[pqrs];
            }

            void setDim(Index t4size)
            {
                size = t4size;
                t4 = EigenVector<T>::Zero(size * (size + 1) *  (size * size + size + 2) / 8);
            }

            void setZero()
            {
                t4.setZero();
            }

            Index getSize()
            {
                return (size * (size + 1) *  (size * size + size + 2) / 8);
            }

            void resize(Index new_dim)
            {
                size = new_dim;
                t4.EigenVector<T>::resize(size * (size + 1) *  (size * size + size + 2) / 8);
                t4.setZero();
            }

            const Eigen::Ref<const EigenVector<T> > get_vector_form()
            {
                return t4;
            }
    };


    // Tensor/4D matrix of the form T^ij^kl where i,l 0 < dim1 and j,k 0 < dim2
    template <typename T>
    class tensor4d1221
    {
        private: 
            Index m_dim1{0};
            Index m_dim2{0};
            EigenVector<T> t4;

        public:
            explicit tensor4d1221(Index dim1, Index dim2) : m_dim1(dim1), m_dim2(dim2)
            {
              t4 = EigenVector<T>::Zero(m_dim1 * m_dim2 * m_dim2 * m_dim1);
            }
            
            explicit tensor4d1221(){}
            ~tensor4d1221() = default;

            tensor4d1221& operator=(const tensor4d1221& other)
            {
                if (&other == this)
                    return *this;
                
                this->t4 = other.t4;
                this->m_dim1 = other.m_dim1;
                this->m_dim2 = other.m_dim2;

                return *this;
            }

            tensor4d1221(tensor4d1221&& other)
            {
                this->t4 = other.t4;
                other.t4 = nullptr;
                this->m_dim1 = other.m_dim1;
                this->m_dim2 = other.m_dim2;
            }

            tensor4d1221(const tensor4d1221& other)
            {
                if (&other == this)
                    return;
                
                this->t4 = other.t4;
                this->m_dim1 = other.m_dim1;
                this->m_dim2 = other.m_dim2;
            }

            const T& operator()(const Index p, const Index q, const Index r, const Index s) const
            {
               return t4[m_dim1 * m_dim2 * m_dim2 * p + m_dim1 * m_dim2 * q + m_dim1 * r + s];
            }
            
            T& operator()(const Index p, const Index q, const Index r, const Index s)
            {
               return t4[m_dim1 * m_dim2 * m_dim2 * p + m_dim1 * m_dim2 * q + m_dim1 * r + s];
            }

            void setDim(const Index dim1, const Index dim2)
            {
                m_dim1 = dim1; 
                m_dim2 = dim2;

               t4 = EigenVector<T>::Zero(m_dim1 * m_dim2 * m_dim2 * m_dim1);
            }

            void setZero()
            {
                t4.setZero();
            }
    };
    // Tensor/4d matrix of the form T^ij^kl where i,j 0 < dim1 and k, l 0 < dim2
    template <typename T>
    class tensor4d1122
    {
        private: 
            Index m_dim1{0};
            Index m_dim2{0};
            EigenVector<T> t4;

        public:
            explicit tensor4d1122(Index dim1, Index dim2) : m_dim1(dim1), m_dim2(dim2)
            {
               t4 = EigenVector<T>::Zero(m_dim1 * m_dim1 * m_dim2 * m_dim2);
            }
            
            explicit tensor4d1122(){}
            ~tensor4d1122() = default;

            tensor4d1122& operator=(const tensor4d1122& other)
            {
                if (&other == this)
                    return *this;
                
                this->t4 = other.t4;
                this->m_dim1 = other.m_dim1;
                this->m_dim2 = other.m_dim2;

                return *this;
            }

            tensor4d1122(tensor4d1122&& other)
            {
                this->t4 = other.t4;
                other.t4 = nullptr;
                this->m_dim1 = other.m_dim1;
                this->m_dim2 = other.m_dim2;
            }

            tensor4d1122(const tensor4d1122& other)
            {
                if (&other == this)
                    return;
                
                this->t4 = other.t4;
                this->m_dim1 = other.m_dim1;
                this->m_dim2 = other.m_dim2;
            }

            const T& operator()(const Index p, const Index q, const Index r, const Index s) const
            {
                // note the storage order as before for increased speed in post scf
                return t4[m_dim1 * m_dim2 * m_dim2 * p + m_dim2 * m_dim1 * r + m_dim2 * q + s];
                //return t4[m_dim1 * m_dim2 * m_dim2 * p + m_dim2 * m_dim2 * q + m_dim2 * r + s]; // original order
            }
            
            T& operator()(const Index p, const Index q, const Index r, const Index s)
            {
                return t4[m_dim1 * m_dim2 * m_dim2 * p + m_dim2 * m_dim1 * r + m_dim2 * q + s];
                // return t4[m_dim1 * m_dim2 * m_dim2 * p + m_dim2 * m_dim2 * q + m_dim2 * r + s]; // original order
            }

            void setDim(const Index dim1, const Index dim2)
            {
                m_dim1 = dim1; 
                m_dim2 = dim2;

                t4 = EigenVector<T>::Zero(m_dim1 * m_dim1 * m_dim2 * m_dim2);
            }

            void setZero()
            {
                t4.setZero();
            }

            const Eigen::Ref<const EigenVector<T> > get_vector_form() const
            {
                return t4;
            }
    };
}

#endif
// TENSOR_MATH_H
