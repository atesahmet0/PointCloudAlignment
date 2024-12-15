import open3d as o3d
import numpy as np
import time


def preprocess_point_cloud(pcd, voxel_size):
    print(f"\nPreprocessing point cloud with voxel size = {voxel_size}")
    print(f"Original point cloud has {len(pcd.points)} points.")

    # Downsample the point cloud
    pcd_down = pcd.voxel_down_sample(voxel_size)
    print(f"Downsampled point cloud to {len(pcd_down.points)} points.")

    # Estimate normals
    print("Estimating normals...")
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2, max_nn=30
        )
    )
    print(f"Normals estimated. Point cloud now has {len(pcd_down.normals)} normals.")

    # Aggressive outlier removal
    print("Removing outliers...")
    # Statistical outlier removal
    pcd_down, ind_stat = pcd_down.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=1.0  # Aggressive threshold
    )
    print(f"Statistical outlier removal: Kept {len(pcd_down.points)} points.")

    # Radius outlier removal
    pcd_down, ind_radius = pcd_down.remove_radius_outlier(
        nb_points=16, radius=voxel_size * 2  # Aggressive threshold
    )
    print(f"Radius outlier removal: Kept {len(pcd_down.points)} points.")

    return pcd_down


def refine_registration(source, target, initial_transformation, voxel_size):
    print("\nRefining registration using ICP...")

    # Ensure normals are computed for both source and target
    print("Ensuring normals are computed for both source and target point clouds...")
    target.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2, max_nn=30
        )
    )
    source.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2, max_nn=30
        )
    )

    # Set the ICP distance threshold
    distance_threshold = voxel_size * 3
    print(f"ICP distance threshold: {distance_threshold}")

    # Perform ICP refinement with higher accuracy
    try:
        start_time = time.time()
        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            distance_threshold,
            initial_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=2000,  # Higher iterations for finer alignment
                relative_fitness=1e-6,
                relative_rmse=1e-6
            )
        )
        print(f"ICP refinement completed in {time.time() - start_time:.2f} seconds.")
        print("Refined Transformation Matrix:")
        print(result.transformation)
        print(f"ICP Fitness: {result.fitness:.4f}, Inlier RMSE: {result.inlier_rmse:.4f}")
        return result
    except Exception as e:
        print(f"Error during ICP refinement: {e}")
        raise


def compute_fpfh(pcd_down, voxel_size):
    print(f"\nComputing FPFH features for the point cloud (voxel size = {voxel_size})")
    
    # Estimate normals (if not already computed)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal, max_nn=30
        )
    )

    # Compute FPFH features
    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature, max_nn=100
        )
    )
    print(f"FPFH features computed. Feature vector size: {np.asarray(fpfh.data).shape}")
    return fpfh


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    print("\nExecuting global registration using RANSAC...")
    distance_threshold = voxel_size * 1.5
    print(f"Distance threshold for RANSAC: {distance_threshold}")

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    print(f"Global registration completed.")
    print("RANSAC Transformation Matrix:")
    print(result.transformation)
    return result


if __name__ == "__main__":
    # Load source and target point clouds
    print("Loading point clouds...")
    source_file = "source.ply"
    target_file = "target.ply"

    source = o3d.io.read_point_cloud(source_file)
    target = o3d.io.read_point_cloud(target_file)

    print(f"Loaded source point cloud: {len(source.points)} points.")
    print(f"Loaded target point cloud: {len(target.points)} points.")

    # Preprocess point clouds
    voxel_size = 10  # Smaller voxel size for more detail
    source_down = preprocess_point_cloud(source, voxel_size)
    target_down = preprocess_point_cloud(target, voxel_size)

    # Compute features
    source_fpfh = compute_fpfh(source_down, voxel_size)
    target_fpfh = compute_fpfh(target_down, voxel_size)

    # Perform global registration
    global_result = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    print("\nInitial alignment complete.")
    print("Transformation Matrix from Global Registration:")
    print(global_result.transformation)

    # Refine the registration using ICP
    refined_result = refine_registration(source, target, global_result.transformation, voxel_size)
    print("\nFinal alignment complete.")
    print("Final Transformation Matrix:")
    print(refined_result.transformation)

    # Save the aligned source point cloud
    print("\nSaving aligned point clouds...")
    aligned_source = source.transform(refined_result.transformation)
    o3d.io.write_point_cloud("aligned_source.ply", aligned_source)
    print("Aligned source point cloud saved as 'aligned_source.ply'.")

    # Visualize the final alignment
    print("\nVisualizing the aligned point clouds...")
    o3d.visualization.draw_geometries([aligned_source, target])
