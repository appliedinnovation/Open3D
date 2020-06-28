// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#pragma once

#include <memory>
#include <vector>

#include "open3d/geometry/Image.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/utility/Eigen.h"

/// @cond
namespace Eigen {

typedef Eigen::Matrix<double, 14, 14> Matrix14d;
typedef Eigen::Matrix<double, 14, 1> Vector14d;
typedef Eigen::Matrix<int, 14, 1> Vector14i;

}  // namespace Eigen
/// @endcond

namespace open3d {
namespace pipelines {
namespace color_map {

class ImageWarpingField;

/// Function to compute i-th row of J and r
/// the vector form of J_r is basically 6x1 matrix, but it can be
/// easily extendable to 6xn matrix.
/// See RGBDOdometryJacobianFromHybridTerm for this case.
void ComputeJacobianAndResidualRigid(
        int row,
        Eigen::Vector6d& J_r,
        double& r,
        const geometry::TriangleMesh& mesh,
        const std::vector<double>& proxy_intensity,
        const std::shared_ptr<geometry::Image>& images_gray,
        const std::shared_ptr<geometry::Image>& images_dx,
        const std::shared_ptr<geometry::Image>& images_dy,
        const Eigen::Matrix4d& intrinsic,
        const Eigen::Matrix4d& extrinsic,
        const std::vector<int>& visibility_image_to_vertex,
        const int image_boundary_margin);

/// Function to compute i-th row of J and r
/// The vector form of J_r is basically 14x1 matrix.
/// This function can take additional matrix multiplication pattern
/// to avoid full matrix multiplication
void ComputeJacobianAndResidualNonRigid(
        int row,
        Eigen::Vector14d& J_r,
        double& r,
        Eigen::Vector14i& pattern,
        const geometry::TriangleMesh& mesh,
        const std::vector<double>& proxy_intensity,
        const std::shared_ptr<geometry::Image>& images_gray,
        const std::shared_ptr<geometry::Image>& images_dx,
        const std::shared_ptr<geometry::Image>& images_dy,
        const ImageWarpingField& warping_fields,
        const ImageWarpingField& warping_fields_init,
        const Eigen::Matrix4d& intrinsic,
        const Eigen::Matrix4d& extrinsic,
        const std::vector<int>& visibility_image_to_vertex,
        const int image_boundary_margin);

/// Function to compute JTJ and Jtr
/// Input: function pointer f and total number of rows of Jacobian matrix
/// Output: JTJ, JTr, sum of r^2
/// Note: this function is almost identical to the functions in
/// Utility/Eigen.h/cpp, but this function takes additional multiplication
/// pattern that can produce JTJ having hundreds of rows and columns.
template <typename VecInTypeDouble,
          typename VecInTypeInt,
          typename MatOutType,
          typename VecOutType>
std::tuple<MatOutType, VecOutType, double> ComputeJTJandJTrNonRigid(
        std::function<void(int, VecInTypeDouble&, double&, VecInTypeInt&)> f,
        int iteration_num,
        int nonrigidval,
        bool verbose /*=true*/) {
    MatOutType JTJ(6 + nonrigidval, 6 + nonrigidval);
    VecOutType JTr(6 + nonrigidval);
    double r2_sum = 0.0;
    JTJ.setZero();
    JTr.setZero();
#ifdef _OPENMP
#pragma omp parallel
    {
#endif
        MatOutType JTJ_private(6 + nonrigidval, 6 + nonrigidval);
        VecOutType JTr_private(6 + nonrigidval);
        double r2_sum_private = 0.0;
        JTJ_private.setZero();
        JTr_private.setZero();
        VecInTypeDouble J_r;
        VecInTypeInt pattern;
        double r;
#ifdef _OPENMP
#pragma omp for nowait
#endif
        for (int i = 0; i < iteration_num; i++) {
            f(i, J_r, r, pattern);
            for (auto x = 0; x < J_r.size(); x++) {
                for (auto y = 0; y < J_r.size(); y++) {
                    JTJ_private(pattern(x), pattern(y)) += J_r(x) * J_r(y);
                }
            }
            for (auto x = 0; x < J_r.size(); x++) {
                JTr_private(pattern(x)) += r * J_r(x);
            }
            r2_sum_private += r * r;
        }
#ifdef _OPENMP
#pragma omp critical
        {
#endif
            JTJ += JTJ_private;
            JTr += JTr_private;
            r2_sum += r2_sum_private;
#ifdef _OPENMP
        }
    }
#endif
    if (verbose) {
        utility::LogDebug("Residual : {:.2e} (# of elements : {:d})",
                          r2_sum / (double)iteration_num, iteration_num);
    }
    return std::make_tuple(std::move(JTJ), std::move(JTr), r2_sum);
}

}  // namespace color_map
}  // namespace pipelines
}  // namespace open3d
