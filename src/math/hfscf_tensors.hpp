
#ifndef TENSOR_MATH_H
#define TENSOR_MATH_H

// used by integration routines
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>

template <typename T>
using EigenVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
using Eigen::Index;

namespace tensormath
{
    template <typename T>
    class tensor3d
    {
        private: 
            Index m_dim1{0};
            Index m_dim2{0};
            Index m_dim3{0};
            EigenVector<T> t3;

        public:
            explicit tensor3d(Index dim1, Index dim2, Index dim3) : m_dim1(dim1), m_dim2(dim2), m_dim3(dim3)
            {
                t3 = EigenVector<T>::Zero(m_dim1 * m_dim2 * m_dim3);
            }
            
            explicit tensor3d(){}
            ~tensor3d(){};

            tensor3d& operator=(const tensor3d& other)
            {
                if (&other == this)
                    return *this;
                
                this->t3 = other.t3;
                this->m_dim1 = other.m_dim1;
                this->m_dim2 = other.m_dim2;
                this->m_dim3 = other.m_dim3;

                return *this;
            }

            tensor3d(tensor3d&& other)
            {
                this->t3 = other.t3;
                other.t3 = nullptr;
                this->m_dim1 = other.m_dim1;
                this->m_dim2 = other.m_dim2;
                this->m_dim3 = other.m_dim3;
            }

            tensor3d(const tensor3d& other)
            {
                if (&other == this)
                    return;
                
                this->t3 = other.t3;
                this->m_dim1 = other.m_dim1;
                this->m_dim2 = other.m_dim2;
                this->m_dim3 = other.m_dim3;

            }

            const T& operator()(const Index p, const Index q, const Index r) const
            {
                return this->t3[p * m_dim3 * m_dim2 + q * m_dim3 + r];
            }

            T& operator()(const Index p, const Index q, const Index r)
            {
                return this->t3[p * m_dim3 * m_dim2 + q * m_dim3 + r];
            }

            void setDim(Index dim1, Index dim2, Index dim3)
            {
                m_dim1 = dim1;
                m_dim2 = dim2;
                m_dim3 = dim3;
                t3 = EigenVector<T>::Zero(m_dim1 * m_dim2 * m_dim3);
            }

            Index getSize()
            {
                return m_dim1 * m_dim2 * m_dim3;
            }

            void setZero()
            {
                t3.setZero();
            }
    };

    template <typename T>
    class tensor4d1234
    {
        private: 
            Index m_dim1{0};
            Index m_dim2{0};
            Index m_dim3{0};
            Index m_dim4{0};
            EigenVector<T> t4;

        public:
            explicit tensor4d1234(Index dim1, Index dim2, Index dim3, Index dim4) 
            : m_dim1(dim1), m_dim2(dim2), m_dim3(dim3), m_dim4(dim4)
            {
                t4 = EigenVector<T>::Zero(m_dim1 * m_dim2 * m_dim3 * m_dim4);
            }
            
            explicit tensor4d1234(){}
            ~tensor4d1234() {};


            tensor4d1234& operator=(const tensor4d1234& other)
            {
                if (&other == this)
                    return *this;
                
                this->t4 = other.t4;
                this->m_dim1 = other.m_dim1;
                this->m_dim2 = other.m_dim2;
                this->m_dim3 = other.m_dim3;
                this->m_dim4 = other.m_dim4;

                return *this;
            }

            tensor4d1234(tensor4d1234&& other)
            {
                this->t4 = other.t4;
                other.t4 = nullptr;
                this->m_dim1 = other.m_dim1;
                this->m_dim2 = other.m_dim2;
                this->m_dim3 = other.m_dim3;
                this->m_dim4 = other.m_dim4;
            }

            tensor4d1234(const tensor4d1234& other)
            {
                if (&other == this)
                    return;
                
                this->t4 = other.t4;
                this->m_dim1 = other.m_dim1;
                this->m_dim2 = other.m_dim2;
                this->m_dim3 = other.m_dim3;
                this->m_dim4 = other.m_dim4;

            }

            tensor4d1234(const tensor4d1234&& other)
            {
                if (&other == this)
                    return;
                
                this->t4 = other.t4;
                this->m_dim1 = other.m_dim1;
                this->m_dim2 = other.m_dim2;
                this->m_dim3 = other.m_dim3;
                this->m_dim4 = other.m_dim4;

            }

            const T& operator()(const Index p, const Index q, const Index r, const Index s) const
            {
                return this->t4[p * m_dim4 * m_dim3 * m_dim2 + q * m_dim4 * m_dim3 + r * m_dim4 + s];
            }

            T& operator()(const Index p, const Index q, const Index r, const Index s)
            {
                return this->t4[p * m_dim4 * m_dim3 * m_dim2 + q * m_dim4 * m_dim3 + r * m_dim4 + s];
            }

            void setDim(Index dim1, Index dim2, Index dim3, Index dim4)
            {
                m_dim1 = dim1;
                m_dim2 = dim2;
                m_dim3 = dim3;
                m_dim4 = dim4;
                t4 = EigenVector<T>::Zero(m_dim1 * m_dim2 * m_dim3 * m_dim4);
            }

            Index getSize()
            {
                return m_dim1 * m_dim2 * m_dim3 * m_dim4;
            }

            void setZero()
            {
                t4.setZero();
            }
    };

    template <typename T>
    class tensor5d
    {
        private: 
            Index m_dim1{0};
            Index m_dim2{0};
            Index m_dim3{0};
            Index m_dim4{0};
            Index m_dim5{0};
            Index m_dim5432{0};
            EigenVector<T> t5;

        public:
            explicit tensor5d(Index dim1, Index dim2, Index dim3, Index dim4, Index dim5) 
            : m_dim1(dim1), m_dim2(dim2), m_dim3(dim3), m_dim4(dim4), m_dim5(dim5)
            {
                m_dim5432 = m_dim2 * m_dim3 * m_dim4 * m_dim5;
                t5 = EigenVector<T>(m_dim1 * m_dim5432);
            }
            
            explicit tensor5d(){}
            ~tensor5d(){};

            tensor5d& operator=(const tensor5d& other)
            {
                if (&other == this)
                    return *this;
                
                this->t5 = other.t5;
                this->m_dim1 = other.m_dim1;
                this->m_dim2 = other.m_dim2;
                this->m_dim3 = other.m_dim3;
                this->m_dim4 = other.m_dim4;
                this->m_dim5 = other.m_dim5;
                this->m_dim5432 = other.m_dim5432;

                return *this;
            }

            tensor5d(tensor5d&& other)
            {
                this->t5 = other.t5;
                other.t5 = nullptr;
                this->m_dim1 = other.m_dim1;
                this->m_dim2 = other.m_dim2;
                this->m_dim3 = other.m_dim3;
                this->m_dim4 = other.m_dim4;
                this->m_dim5 = other.m_dim5;
                this->m_dim5432 = other.m_dim5432;
            }

            tensor5d(const tensor5d& other)
            {
                if (&other == this)
                    return;
                
                this->t5 = other.t5;
                this->m_dim1 = other.m_dim1;
                this->m_dim2 = other.m_dim2;
                this->m_dim3 = other.m_dim3;
                this->m_dim4 = other.m_dim4;
                this->m_dim5 = other.m_dim5;
                this->m_dim5432 = other.m_dim5432;
            }

            tensor5d(const tensor5d&& other)
            {
                if (&other == this)
                    return;
                
                this->t5 = other.t5;
                this->m_dim1 = other.m_dim1;
                this->m_dim2 = other.m_dim2;
                this->m_dim3 = other.m_dim3;
                this->m_dim4 = other.m_dim4;
                this->m_dim5 = other.m_dim5;
                this->m_dim5432 = other.m_dim5432;
            }

            T& operator()(const Index p, const Index q, const Index r, const Index s, const Index t)
            {
                return this->t5[p * m_dim5432 + q * m_dim5 * m_dim4 * m_dim3 + 
                                r * m_dim5 * m_dim4 + m_dim5 * s + t];
            }

            const T& operator()(const Index p, const Index q, const Index r, const Index s, const Index t) const
            {
                return this->t5[p * m_dim5432 + q * m_dim5 * m_dim4 * m_dim3 + 
                                r * m_dim5 * m_dim4 + m_dim5 * s + t];
            }

            void setDim(Index dim1, Index dim2, Index dim3, Index dim4, Index dim5)
            {
                m_dim1 = dim1;
                m_dim2 = dim2;
                m_dim3 = dim3;
                m_dim4 = dim4;
                m_dim5 = dim5;
                m_dim5432 = m_dim2 * m_dim3 * m_dim4 * m_dim5;
                t5 = EigenVector<T>(m_dim1 * m_dim2 * m_dim3 *  m_dim4 * m_dim5);
            }

            Index getSize()
            {
                return m_dim1 * m_dim5432;
            }

            void setZero()
            {
                t5.setZero();
            }
    };
}

#endif
// TENSOR_MATH_H
