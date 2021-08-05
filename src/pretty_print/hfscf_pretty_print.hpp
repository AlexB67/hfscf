#ifndef PRETTY_PRINT_H
#define PRETTY_PRINT_H

#include <iomanip>
#include <iostream>
#include <Eigen/Core>
#include <vector>

namespace HFCOUT
{

template <typename T>
using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

using Eigen::Index;

template<typename T>
void pretty_print_matrix(const Eigen::Ref<const EigenMatrix<T> >& mat, Index rows = 0,  Index cols = 0)
{
    Index offset = 0;
    Index print_cols = 6;
    Index j = 0;
    if(!cols) cols = mat.innerSize();
    if(!rows) rows = mat.outerSize();

    while (j < cols) 
    {
        if (offset + print_cols > cols - 1) print_cols = cols - offset;

        for (Index col_index = offset; col_index < offset + print_cols; ++col_index)
            std::cout << std::right << std::setw(15) << col_index + 1 << "  ";

        std::cout << '\n';

        for (Index i = 0; i < rows; ++i) 
        {
            std::cout << std::right << std::setw(3) << i + 1;

            for (j = offset; j < offset + print_cols; ++j)
                std::cout << std::right << std::fixed << std::setprecision(9) << std::setw(17) << mat(i, j);

            std::cout << '\n';
        }
        offset += print_cols;
        std::cout << '\n';
    }
}

template<typename T>
void pretty_print_matrix(const std::vector<EigenMatrix<T>>& vec_mat, Index rows = 0,  Index cols = 0)
{
    for (size_t i = 0; i < vec_mat.size(); ++i)
    {
        const Eigen::Ref<const EigenMatrix<T> >& mat = vec_mat[i];
        std::cout << "\n  Irrep: " << i + 1 << " Dimensions: " << mat.outerSize() 
                  << "x" << mat.innerSize() << "\n\n";
        
        pretty_print_matrix<T>(mat, rows, cols);
    }
}

}
#endif