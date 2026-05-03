from vlm_grounder.utils import SceneInfoHandler
from vlm_grounder.grounder.visual_grounder_new import BBoxGenerator

# 1. Initialize the handlers (paths depend on your dataset setup)
scene_infos = SceneInfoHandler("data/scannet/scannet_instance_data/scenes_train_val_info_w_images.pkl")

# 2. Instantiate the BBoxGenerator
bbox_generator = BBoxGenerator(
    scene_infos=scene_infos,
    point_filter_type="statistical"
)

# 3. Generate the 3D BBox from your segmented output (image_masks)
# image_masks is a dictionary mapping image_id to its corresponding segmentation mask (2D boolean numpy array)
image_masks = {
    "150": my_2d_segmentation_mask,
    "140": my_2d_segmentation_mask_140,
    "145": my_2d_segmentation_mask_145,
    "155": my_2d_segmentation_mask_155,
    "160": my_2d_segmentation_mask_160,
}

pred_3d_bbox = bbox_generator.generate_3d_bbox(
    scene_id="scene0000_00",
    image_masks=image_masks,
    intermediate_output_dir="./temp_outputs",
    query="a chair near the table"
)
print("Generated 3D BBox:", pred_3d_bbox)