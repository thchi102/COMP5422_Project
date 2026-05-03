import os
import cv2
import numpy as np
import mmengine
from mmengine.utils.dl_utils import TimeCounter

from vlm_grounder.utils import (
    calculate_aabb,
    remove_statistical_outliers,
    remove_truncated_outliers,
)

class BBoxGenerator:
    def __init__(
        self,
        scene_infos,
        post_process_erosion=False,
        post_process_dilation=False,
        kernel_size=3,
        post_process_component=False,
        post_process_component_num=1,
        point_filter_nb=20,
        point_filter_std=1.0,
        point_filter_type="statistical",
        point_filter_tx=0.05,
        point_filter_ty=0.05,
        point_filter_tz=0.05,
        project_color_image=False,
    ):
        """
        Initialize the BBoxGenerator.
        
        Args:
            scene_infos: SceneInfoHandler instance to handle 3D projections.
            post_process_*: Optional morphological operations if your masks need cleanup.
            point_filter_*: Configurations for 3D point cloud outlier removal.
        """
        self.scene_infos = scene_infos
        
        self.post_process_erosion = post_process_erosion
        self.post_process_dilation = post_process_dilation
        self.kernel_size = kernel_size
        self.post_process_component = post_process_component
        self.post_process_component_num = post_process_component_num
        
        if (
            self.post_process_erosion
            or self.post_process_dilation
            or self.post_process_component
        ):
            self.post_process = True
        else:
            self.post_process = False

        self.point_filter_nb = point_filter_nb
        self.point_filter_std = point_filter_std
        self.point_filter_type = point_filter_type
        self.point_filter_tx = point_filter_tx
        self.point_filter_ty = point_filter_ty
        self.point_filter_tz = point_filter_tz
        self.project_color_image = project_color_image

    def post_process_mask(self, mask):
        """
        Process a binary mask to smooth and optionally remove small components.

        Args:
            mask (np.array): A 2D boolean numpy array.

        Returns:
            np.array: The processed mask as a boolean numpy array.
        """
        # Convert boolean mask to uint8
        img = np.uint8(mask) * 255

        # Define the kernel for morphological operations
        kernel = np.ones((self.kernel_size * 2 + 1, self.kernel_size * 2 + 1), np.uint8)

        # Apply morphological erosion if requested
        if self.post_process_erosion:
            img = cv2.erode(img, kernel, iterations=1)

        # Apply morphological dilation if requested
        if self.post_process_dilation:
            img = cv2.dilate(img, kernel, iterations=1)

        # Find all connected components
        num_labels, labels_im = cv2.connectedComponents(
            img
        )  # label 0 is background, so start from 1
        if self.post_process_component and num_labels > 1:
            # Calculate the area of each component and sort them, keeping the largest k
            component_areas = [
                (label, np.sum(labels_im == label)) for label in range(1, num_labels)
            ]
            component_areas.sort(key=lambda x: x[1], reverse=True)
            largest_components = [
                x[0] for x in component_areas[: self.post_process_component_num]
            ]
            img = np.isin(labels_im, largest_components).astype(np.uint8)

        # Return the processed image as a boolean mask
        return img.astype(bool)

    def ensemble_pred_points(
        self,
        scene_id,
        image_masks,
    ):
        """
        Project the provided masks from different images into 3D and ensemble them.

        Args:
            scene_id (str): The scene ID.
            image_masks (dict): A dictionary mapping image_id (str or int) to its corresponding 
                                segmentation mask (2D boolean numpy array).
                                
        Returns:
            np.ndarray: The ensembled 3D points.
        """
        ensemble_points = []
        
        for image_id, mask in image_masks.items():
            # Optional post-processing if masks are noisy
            if self.post_process:
                mask = self.post_process_mask(mask)
                
            # Project the 2D mask pixels to 3D points using camera intrinsics/extrinsics
            current_aligned_points_3d = self.scene_infos.project_image_to_3d_with_mask(
                scene_id=scene_id,
                image_id=image_id,
                mask=mask,
                with_color=self.project_color_image,
            )

            if current_aligned_points_3d is not None and len(current_aligned_points_3d) > 0:
                ensemble_points.append(current_aligned_points_3d)

        if not ensemble_points:
            return np.array([])
            
        aligned_points_3d = np.concatenate(ensemble_points, axis=0)
        return aligned_points_3d

    def generate_3d_bbox(
        self,
        scene_id,
        image_masks,
        intermediate_output_dir=None,
        query="",
    ):
        """
        Generate a 3D bounding box directly from pre-segmented outputs.

        Args:
            scene_id (str): The scene ID.
            image_masks (dict): A dictionary mapping image_id (str or int) to its corresponding 
                                segmentation mask (2D boolean numpy array).
            intermediate_output_dir (str, optional): Directory to save intermediate point clouds.
            query (str, optional): The query string for logging.

        Returns:
            np.ndarray: The predicted 3D bounding box (AABB), or None if invalid.
        """
        # 1. Project all masks to 3D and combine them
        with TimeCounter(tag="EnsemblePredPoints", log_interval=60):
            aligned_points_3d = self.ensemble_pred_points(
                scene_id=scene_id,
                image_masks=image_masks,
            )

        if aligned_points_3d.shape[0] == 0:
            print(f"[Box projection] No points projected for {scene_id}: {query}.")
            return None

        # 2. Filter out noisy points
        with TimeCounter(tag="RemoveOutliers", log_interval=60):
            if self.point_filter_type == "statistical":
                aligned_points_3d_filtered = remove_statistical_outliers(
                    aligned_points_3d,
                    nb_neighbors=self.point_filter_nb,
                    std_ratio=self.point_filter_std,
                )
            elif self.point_filter_type == "truncated":
                aligned_points_3d_filtered = remove_truncated_outliers(
                    aligned_points_3d,
                    tx=self.point_filter_tx,
                    ty=self.point_filter_ty,
                    tz=self.point_filter_tz,
                )
            elif self.point_filter_type == "none":
                aligned_points_3d_filtered = aligned_points_3d
            else:
                raise NotImplementedError(
                    f"Point filter type {self.point_filter_type} is not implemented."
                )

        # 3. (Optional) Save intermediate point clouds
        if intermediate_output_dir is not None:
            aligned_points_3d_output_dir = os.path.join(
                intermediate_output_dir, "projected_points"
            )
            mmengine.mkdir_or_exist(aligned_points_3d_output_dir)
            np.save(
                f"{aligned_points_3d_output_dir}/ensemble_points.npy",
                aligned_points_3d,
            )
            np.save(
                f"{aligned_points_3d_output_dir}/ensemble_points_filtered.npy",
                aligned_points_3d_filtered,
            )

        # 4. Validate points
        if (
            aligned_points_3d_filtered.shape[0] == 0
            or np.isnan(aligned_points_3d).any()
            or np.isnan(aligned_points_3d_filtered).any()
        ):
            print(
                f"[Box projection] Filtered points are empty or have NaN for {scene_id}: {query}."
            )
            return None

        # 5. Calculate Axis-Aligned Bounding Box (AABB)
        pred_bbox = calculate_aabb(aligned_points_3d_filtered)

        return pred_bbox
