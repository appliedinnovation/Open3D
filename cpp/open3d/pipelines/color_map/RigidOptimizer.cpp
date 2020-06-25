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

#include <memory>
#include <vector>

#include "open3d/pipelines/color_map/JacobianHelper.h"
#include "open3d/pipelines/color_map/RigidOptimizer.h"
#include "open3d/pipelines/color_map/TriangleMeshAndImageUtilities.h"

namespace open3d {
namespace pipelines {
namespace color_map {

void RigidOptimizer::Run(const RigidOptimizerOption& option) {
    utility::LogDebug("[ColorMapOptimization] :: MakingMasks");
    std::vector<std::shared_ptr<geometry::Image>> images_mask =
            CreateDepthBoundaryMasks(
                    images_depth_,
                    option.depth_threshold_for_discontinuity_check_,
                    option.half_dilation_kernel_size_for_discontinuity_map_);

    utility::LogDebug("[ColorMapOptimization] :: VisibilityCheck");
    std::vector<std::vector<int>> visibility_vertex_to_image;
    std::vector<std::vector<int>> visibility_image_to_vertex;
    std::tie(visibility_vertex_to_image, visibility_image_to_vertex) =
            CreateVertexAndImageVisibility(
                    *mesh_, images_depth_, images_mask, *camera_trajectory_,
                    option.maximum_allowable_depth_,
                    option.depth_threshold_for_visibility_check_);

    utility::LogDebug("[ColorMapOptimization] :: Run Rigid Optimization");
    std::vector<double> proxy_intensity;
    int total_num_ = 0;
    int n_camera = int(camera_trajectory_->parameters_.size());
    SetProxyIntensityForVertex(*mesh_, images_gray_, *camera_trajectory_,
                               visibility_vertex_to_image, proxy_intensity,
                               option.image_boundary_margin_);
    for (int itr = 0; itr < option.maximum_iteration_; itr++) {
        utility::LogDebug("[Iteration {:04d}] ", itr + 1);
        double residual = 0.0;
        total_num_ = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int c = 0; c < n_camera; c++) {
            Eigen::Matrix4d pose;
            pose = camera_trajectory_->parameters_[c].extrinsic_;

            auto intrinsic = camera_trajectory_->parameters_[c]
                                     .intrinsic_.intrinsic_matrix_;
            auto extrinsic = camera_trajectory_->parameters_[c].extrinsic_;
            Eigen::Matrix4d intr = Eigen::Matrix4d::Zero();
            intr.block<3, 3>(0, 0) = intrinsic;
            intr(3, 3) = 1.0;

            auto f_lambda = [&](int i, Eigen::Vector6d& J_r, double& r) {
                ComputeJacobianAndResidualRigid(
                        i, J_r, r, *mesh_, proxy_intensity, images_gray_[c],
                        images_dx_[c], images_dy_[c], intr, extrinsic,
                        visibility_image_to_vertex[c],
                        option.image_boundary_margin_);
            };
            Eigen::Matrix6d JTJ;
            Eigen::Vector6d JTr;
            double r2;
            std::tie(JTJ, JTr, r2) =
                    utility::ComputeJTJandJTr<Eigen::Matrix6d, Eigen::Vector6d>(
                            f_lambda, int(visibility_image_to_vertex[c].size()),
                            false);

            bool is_success;
            Eigen::Matrix4d delta;
            std::tie(is_success, delta) =
                    utility::SolveJacobianSystemAndObtainExtrinsicMatrix(JTJ,
                                                                         JTr);
            pose = delta * pose;
            camera_trajectory_->parameters_[c].extrinsic_ = pose;
#ifdef _OPENMP
#pragma omp critical
#endif
            {
                residual += r2;
                total_num_ += int(visibility_image_to_vertex[c].size());
            }
        }
        utility::LogDebug("Residual error : {:.6f} (avg : {:.6f})", residual,
                          residual / total_num_);
        SetProxyIntensityForVertex(*mesh_, images_gray_, *camera_trajectory_,
                                   visibility_vertex_to_image, proxy_intensity,
                                   option.image_boundary_margin_);
    }

    utility::LogDebug("[ColorMapOptimization] :: Set Mesh Color");
    SetGeometryColorAverage(*mesh_, images_color_, *camera_trajectory_,
                            visibility_vertex_to_image,
                            option.image_boundary_margin_,
                            option.invisible_vertex_color_knn_);
}

}  // namespace color_map
}  // namespace pipelines
}  // namespace open3d
