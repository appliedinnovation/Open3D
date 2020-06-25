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

#include "open3d/pipelines/color_map/ImageWarpingField.h"
#include "open3d/pipelines/color_map/JacobianHelper.h"
#include "open3d/pipelines/color_map/NonRigidOptimizer.h"
#include "open3d/pipelines/color_map/TriangleMeshAndImageUtilities.h"

namespace open3d {
namespace pipelines {
namespace color_map {

static std::vector<ImageWarpingField> CreateWarpingFields(
        const std::vector<std::shared_ptr<geometry::Image>>& images,
        int number_of_vertical_anchors) {
    std::vector<ImageWarpingField> fields;
    for (size_t i = 0; i < images.size(); i++) {
        int width = images[i]->width_;
        int height = images[i]->height_;
        fields.push_back(
                ImageWarpingField(width, height, number_of_vertical_anchors));
    }
    return fields;
}

void NonRigidOptimizer::Run(const NonRigidOptimizerOption& option) {
    utility::LogDebug("[ColorMapOptimization] :: MakingMasks");
    auto images_mask = CreateDepthBoundaryMasks(
            images_depth_, option.depth_threshold_for_discontinuity_check_,
            option.half_dilation_kernel_size_for_discontinuity_map_);

    utility::LogDebug("[ColorMapOptimization] :: VisibilityCheck");
    std::vector<std::vector<int>> visibility_vertex_to_image;
    std::vector<std::vector<int>> visibility_image_to_vertex;
    std::tie(visibility_vertex_to_image, visibility_image_to_vertex) =
            CreateVertexAndImageVisibility(
                    *mesh_, images_depth_, images_mask, *camera_trajectory_,
                    option.maximum_allowable_depth_,
                    option.depth_threshold_for_visibility_check_);

    utility::LogDebug("[ColorMapOptimization] :: Run Non-Rigid Optimization");
    auto warping_fields = CreateWarpingFields(
            images_gray_, option.number_of_vertical_anchors_);
    auto warping_fields_init = CreateWarpingFields(
            images_gray_, option.number_of_vertical_anchors_);
    std::vector<double> proxy_intensity;
    auto n_vertex = mesh_->vertices_.size();
    int n_camera = int(camera_trajectory_->parameters_.size());
    SetProxyIntensityForVertex(*mesh_, images_gray_, warping_fields,
                               *camera_trajectory_, visibility_vertex_to_image,
                               proxy_intensity, option.image_boundary_margin_);
    for (int itr = 0; itr < option.maximum_iteration_; itr++) {
        utility::LogDebug("[Iteration {:04d}] ", itr + 1);
        double residual = 0.0;
        double residual_reg = 0.0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int c = 0; c < n_camera; c++) {
            int nonrigidval = warping_fields[c].anchor_w_ *
                              warping_fields[c].anchor_h_ * 2;
            double rr_reg = 0.0;

            Eigen::Matrix4d pose;
            pose = camera_trajectory_->parameters_[c].extrinsic_;

            auto intrinsic = camera_trajectory_->parameters_[c]
                                     .intrinsic_.intrinsic_matrix_;
            auto extrinsic = camera_trajectory_->parameters_[c].extrinsic_;
            Eigen::Matrix4d intr = Eigen::Matrix4d::Zero();
            intr.block<3, 3>(0, 0) = intrinsic;
            intr(3, 3) = 1.0;

            auto f_lambda = [&](int i, Eigen::Vector14d& J_r, double& r,
                                Eigen::Vector14i& pattern) {
                ComputeJacobianAndResidualNonRigid(
                        i, J_r, r, pattern, *mesh_, proxy_intensity,
                        images_gray_[c], images_dx_[c], images_dy_[c],
                        warping_fields[c], warping_fields_init[c], intr,
                        extrinsic, visibility_image_to_vertex[c],
                        option.image_boundary_margin_);
            };
            Eigen::MatrixXd JTJ;
            Eigen::VectorXd JTr;
            double r2;
            std::tie(JTJ, JTr, r2) =
                    ComputeJTJandJTrNonRigid<Eigen::Vector14d, Eigen::Vector14i,
                                             Eigen::MatrixXd, Eigen::VectorXd>(
                            f_lambda, int(visibility_image_to_vertex[c].size()),
                            nonrigidval, false);

            double weight = option.non_rigid_anchor_point_weight_ *
                            visibility_image_to_vertex[c].size() / n_vertex;
            for (int j = 0; j < nonrigidval; j++) {
                double r = weight * (warping_fields[c].flow_(j) -
                                     warping_fields_init[c].flow_(j));
                JTJ(6 + j, 6 + j) += weight * weight;
                JTr(6 + j) += weight * r;
                rr_reg += r * r;
            }

            bool success;
            Eigen::VectorXd result;
            std::tie(success, result) = utility::SolveLinearSystemPSD(
                    JTJ, -JTr, /*prefer_sparse=*/false,
                    /*check_symmetric=*/false,
                    /*check_det=*/false, /*check_psd=*/false);
            Eigen::Vector6d result_pose;
            result_pose << result.block(0, 0, 6, 1);
            auto delta = utility::TransformVector6dToMatrix4d(result_pose);
            pose = delta * pose;

            for (int j = 0; j < nonrigidval; j++) {
                warping_fields[c].flow_(j) += result(6 + j);
            }
            camera_trajectory_->parameters_[c].extrinsic_ = pose;

#ifdef _OPENMP
#pragma omp critical
#endif
            {
                residual += r2;
                residual_reg += rr_reg;
            }
        }
        utility::LogDebug("Residual error : {:.6f}, reg : {:.6f}", residual,
                          residual_reg);
        SetProxyIntensityForVertex(*mesh_, images_gray_, warping_fields,
                                   *camera_trajectory_,
                                   visibility_vertex_to_image, proxy_intensity,
                                   option.image_boundary_margin_);
    }

    utility::LogDebug("[ColorMapOptimization] :: Set Mesh Color");
    SetGeometryColorAverage(*mesh_, images_color_, warping_fields,
                            *camera_trajectory_, visibility_vertex_to_image,
                            option.image_boundary_margin_,
                            option.invisible_vertex_color_knn_);
}

}  // namespace color_map
}  // namespace pipelines
}  // namespace open3d
