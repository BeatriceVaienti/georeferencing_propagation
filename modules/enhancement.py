# strategy 2 + 1 we actually compute the full bbox and align them with a transformation, get the superpoint results, and then only match them locally
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.strtree import STRtree
from collections import Counter
from scipy.spatial import distance
from modules.homologous_points_detection import run_superglue_matching, matches_to_dataframe, compute_matches
from sklearn.cluster import DBSCAN
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from skimage.transform import ProjectiveTransform, warp
from modules.MapDataset import MapDataset, MapInfo
from shapely.geometry import Polygon, MultiPoint
from shapely.ops import unary_union
from modules.georeferencing_propagation import meters_to_tensors
import logging
import copy
colors = {
    'pink_dark' : '#f75785',
    'pink_light' : '#f8b0be',
    'aqua_dark' : '#009da5',
    'aqua_light' : '#3cc5be',
    'orange_dark' : '#ffa631',
    'orange_light' : '#ffd766',
    'red_dark' : '#e84743',
    'red_light' : '#ed8e83',
    'blue_dark': '#489fee',
    'blue_light': '#8fcfff',
    'dark_grey': '#413d3a',
    'light_grey': '#cac7c7',
}

def create_patches_from_bbox(bbox, patch_size=400, overlap=100):
    """
    Creates patches of a given size with specified overlap within a bounding box.
    """
    min_x, min_y, max_x, max_y = bbox
    step = patch_size - overlap
    patches = []
    x = min_x
    while x < max_x:
        y = min_y
        while y < max_y:
            patch_max_x = min(x + patch_size, max_x)
            patch_max_y = min(y + patch_size, max_y)
            patches.append((x, y, patch_max_x, patch_max_y))
            y += step
        x += step
    return patches

def extract_keypoints_and_descriptors(map_dataset):
    """
    Extracts keypoints, descriptors, and scores from a given MapDataset object.
    """
    return (map_dataset.superpoint_results.keypoints[0],  # Assuming keypoints are stored in a list
            map_dataset.superpoint_results.descriptors[0],
            map_dataset.superpoint_results.scores[0])

def match_keypoints_with_superglue(map1, map2, min_score=0.1):
    """
    Matches keypoints using SuperGlue between two MapDataset objects.
    """
    matches, kp1, kp2 = run_superglue_matching(map1, map2, min_score=min_score)
    return matches, kp1, kp2

def filter_matches_with_delaunay(matches_df, similarity_threshold=0.5, plot=False):
    """
    Filters matches using Delaunay triangulation based on similarity of triangles.
    """
    kp1_coords = matches_df[['kp1_x', 'kp1_y']].values
    kp2_coords = matches_df[['kp2_x', 'kp2_y']].values

    if len(kp1_coords) < 3:
        return matches_df

    tri_kp1 = Delaunay(kp1_coords)
    points_to_remove = set()
    for simplex in tri_kp1.simplices:
        sides_kp1 = [np.linalg.norm(kp1_coords[simplex[i]] - kp1_coords[simplex[j]]) for i in range(3) for j in range(i + 1, 3)]
        sides_kp2 = [np.linalg.norm(kp2_coords[simplex[i]] - kp2_coords[simplex[j]]) for i in range(3) for j in range(i + 1, 3)]
        normalized_sides_kp1 = [s / np.mean(sides_kp1) for s in sides_kp1]
        normalized_sides_kp2 = [s / np.mean(sides_kp2) for s in sides_kp2]
        side_ratios = [s2 / s1 if s1 > 0 else 0 for s1, s2 in zip(normalized_sides_kp1, normalized_sides_kp2)]
        similarity_deviation = max(side_ratios) - min(side_ratios)
        if similarity_deviation > similarity_threshold:
            max_ratio_index = np.argmax(side_ratios)
            min_ratio_index = np.argmin(side_ratios)
            problematic_point = simplex[max(max_ratio_index, min_ratio_index)]
            points_to_remove.add(problematic_point)

    filtered_matches = matches_df.drop(list(points_to_remove), errors="ignore").reset_index(drop=True)
    return filtered_matches

def process_patches_for_matches(target_map, base_map, patches):
    """
    Processes patches for matching using SuperPoint and SuperGlue, then filters matches using Delaunay triangulation.
    Creates temporary MapDataset objects from the original images instead of tensors.
    """
    all_matches = []
    for patch in patches:
        x_min, y_min, x_max, y_max = patch

        # Extract patches from the original images, not tensors
        target_patch = target_map.extract_patch_from_image(x_min, y_min, x_max, y_max)
        base_patch = base_map.extract_patch_from_image(x_min, y_min, x_max, y_max)

        # Extract keypoints, descriptors, and scores from patches
        kp1, desc1, scores1 = extract_keypoints_and_descriptors(target_patch)
        kp2, desc2, scores2 = extract_keypoints_and_descriptors(base_patch)

        # Match keypoints using SuperGlue
        matches, _, _ = match_keypoints_with_superglue(target_patch, base_patch)
        print(f"Matches found: {len(matches)}")
        # Convert matches to a DataFrame for further processing
        matches_df = matches_to_dataframe(matches, kp1, kp2)

        # Filter matches using Delaunay triangulation
        filtered_matches = filter_matches_with_delaunay(matches_df)
        all_matches.append(filtered_matches)

    enhanced_matches = pd.concat(all_matches, ignore_index=True)
    return enhanced_matches

def create_bbox_with_offset(filtered_points_image, offset=100, plot=False, image=None, mask=None, curr_map=None):
    """
    Creates a bounding box around filtered points with an optional offset in the image space.
    Ensures the bounding box does not cross the mask boundary using binary mask operations.
    Optionally plots the result.

    Parameters:
    - filtered_points_image: Numpy array of points (x, y) in image coordinates.
    - mask: Binary mask where the valid area is non-zero (optional).
    - offset: Offset to expand the bounding box (in pixels).
    - plot: Boolean flag to enable plotting of the image, bbox, and mask.
    - image: The original image to plot, if available (optional).

    Returns:
    - bbox_image_space: Tuple (min_x, min_y, max_x, max_y) defining the bounding box in image space.
    """
    # Calculate initial bounding box with offset
    min_x, min_y = np.min(filtered_points_image, axis=0)
    max_x, max_y = np.max(filtered_points_image, axis=0)
    min_x = max(0, int(min_x - offset))
    min_y = max(0, int(min_y - offset))
    max_x = int(max_x + offset)
    max_y = int(max_y + offset)

    # Optional plotting
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        print(curr_map.map_info)
        if curr_map.map_info.image_path is not None:
            image = cv2.imread(curr_map.map_info.image_path, cv2.IMREAD_GRAYSCALE)
            ax.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
        elif curr_map.map_info.mask_path is not None:
            mask = cv2.imread(curr_map.map_info.mask_path, cv2.IMREAD_GRAYSCALE)
            ax.imshow(mask, cmap='gray', alpha=0.5)  # Display mask if image is not available
        else:
            ax.set_title("No image or mask to display")

        # Plot bounding box
        rect = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                             linewidth=2, edgecolor='r', facecolor='none', label='Bounding Box')
        ax.add_patch(rect)
        
        # Plot filtered points
        ax.scatter(filtered_points_image[:, 0], filtered_points_image[:, 1], c=colors['orange_light'], s=5, label='Filtered Points')

        # Overlay mask if available
        if mask is not None:
            ax.imshow(mask, cmap='gray', alpha=0.3)
            ax.set_title("Image with Bounding Box and Mask")
        else:
            ax.set_title("Image with Bounding Box")

        ax.set_xlim([0, mask.shape[1] if mask is not None else image.shape[1]])
        ax.set_ylim([mask.shape[0] if mask is not None else image.shape[0], 0])  # Invert y-axis for image display
        ax.legend()
        plt.show()

    return (min_x, min_y, max_x, max_y)

def get_bbox(map, points_tensor_space, offset_tensor_space, plot=False):
    """
    Creates a bounding box around the filtered points in the image space.
    """
    # Convert the points to image space
    tensor_to_image_transform = map.tensor_to_image_transform
    points_image_space = apply_transform(points_tensor_space.reshape(1, -1, 2), tensor_to_image_transform).reshape(-1, 2)
    #mask = cv2.imread(map.map_info.mask_path, cv2.IMREAD_GRAYSCALE) if map.map_info.mask_path else None

    scale_x = np.linalg.norm(tensor_to_image_transform[:, 0])  # Scale factor for x-direction
    scale_y = np.linalg.norm(tensor_to_image_transform[:, 1])  # Scale factor for y-direction
    average_scale = (scale_x + scale_y) / 2.0  # Average scale factor

    # Convert the offset to image space
    offset_image_space = offset_tensor_space * average_scale

    # Create the bounding box
    bbox_image_space = create_bbox_with_offset(points_image_space, offset=offset_image_space, plot=plot, image=map.image, mask=map.mask, curr_map=map)
    return bbox_image_space

def create_patches_from_bbox(bbox, patch_length, overlap, tensor_to_image_transform, plot=False, image=None):
    """
    Creates patches within a given bounding box, ensuring that patches are of the specified size.
    The overlap between patches is adjusted as needed to cover the bounding box completely.
    
    Parameters:
    - bbox: Tuple (min_x, min_y, max_x, max_y) defining the bounding box in image space.
    - patch_length: Length of each patch (in meters).
    - overlap: Desired overlap between patches (in meters). Actual overlap may vary.
    - tensor_to_image_transform: 2x3 transformation matrix from tensor to image space.
    - plot: Boolean flag to enable plotting.
    - image: The original image to plot, if available (optional).
    
    Returns:
    - patches: List of tuples defining the patches (min_x, min_y, max_x, max_y) in image space.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from math import ceil

    min_x, min_y, max_x, max_y = bbox

    # Convert patch length from meters to image space
    scale_x = np.linalg.norm(tensor_to_image_transform[:, 0])  # Scale factor for x-direction
    scale_y = np.linalg.norm(tensor_to_image_transform[:, 1])  # Scale factor for y-direction
    average_scale = (scale_x + scale_y) / 2.0  # Average scale factor
    patch_length_image_space = patch_length * average_scale

    # Compute the width and height of the bbox in image space
    width = max_x - min_x
    height = max_y - min_y

    # Compute the number of patches needed in x and y directions
    num_patches_x = int(np.ceil(width / patch_length_image_space))
    num_patches_y = int(np.ceil(height / patch_length_image_space))

    # Adjusted overlap to make patches fit exactly within the bbox
    if num_patches_x > 1:
        total_coverage_x = num_patches_x * patch_length_image_space
        overlap_x = (total_coverage_x - width) / (num_patches_x - 1)
    else:
        overlap_x = 0

    if num_patches_y > 1:
        total_coverage_y = num_patches_y * patch_length_image_space
        overlap_y = (total_coverage_y - height) / (num_patches_y - 1)
    else:
        overlap_y = 0

    patches = []
    for i in range(num_patches_x):
        x = min_x + i * (patch_length_image_space - overlap_x)
        for j in range(num_patches_y):
            y = min_y + j * (patch_length_image_space - overlap_y)
            patch_min_x = x
            patch_min_y = y
            patch_max_x = x + patch_length_image_space
            patch_max_y = y + patch_length_image_space

            # Ensure patches are within the bbox
            if patch_max_x > max_x:
                patch_min_x = max_x - patch_length_image_space
                patch_max_x = max_x
            if patch_max_y > max_y:
                patch_min_y = max_y - patch_length_image_space
                patch_max_y = max_y

            # Correct any potential negative values if bbox is smaller than patch size
            patch_min_x = max(patch_min_x, min_x)
            patch_min_y = max(patch_min_y, min_y)
            patch_max_x = min(patch_max_x, max_x)
            patch_max_y = min(patch_max_y, max_y)

            patches.append((int(patch_min_x), int(patch_min_y), int(patch_max_x), int(patch_max_y)))

    # Optional plotting
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        if image is not None:
            ax.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
        ax.set_title("Patches within Bounding Box")
        for patch in patches:
            px_min, py_min, px_max, py_max = patch
            rect = plt.Rectangle((px_min, py_min), px_max - px_min, py_max - py_min,
                                 linewidth=1, edgecolor=colors['blue_dark'], facecolor='none', linestyle='--', label='Patch')
            ax.add_patch(rect)

        # Draw the bounding box
        rect_bbox = plt.Rectangle((min_x, min_y), width, height,
                                  linewidth=2, edgecolor=colors['red_dark'], facecolor='none', label='Bounding Box')
        ax.add_patch(rect_bbox)

        # Ensure no duplicate legends
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        ax.set_xlim([0, image.shape[1] if image is not None else max_x])
        ax.set_ylim([image.shape[0] if image is not None else max_y, 0])  # Invert y-axis for image display
        plt.show()

    return patches

def create_patches_from_bbox_given_overlap(bbox, patch_length, overlap, tensor_to_image_transform, plot=False, image=None):
    """
    Creates patches within a given bounding box, with a specified patch length and overlap.
    The dimensions are transformed to the image space for visualization.

    Parameters:
    - bbox: Tuple (min_x, min_y, max_x, max_y) defining the bounding box in image space.
    - patch_length: Length of each patch (in meters).
    - overlap: Overlap between patches (in meters).
    - tensor_to_image_transform: 2x3 transformation matrix from tensor to image space.
    - plot: Boolean flag to enable plotting.
    - image: The original image to plot, if available (optional).

    Returns:
    - patches: List of tuples defining the patches (min_x, min_y, max_x, max_y) in image space.
    """

    min_x, min_y, max_x, max_y = bbox

    # Convert patch length and overlap from meters to tensor space
    scale_x = np.linalg.norm(tensor_to_image_transform[:, 0])  # Scale factor for x-direction
    scale_y = np.linalg.norm(tensor_to_image_transform[:, 1])  # Scale factor for y-direction
    average_scale = (scale_x + scale_y) / 2.0  # Average scale factor
    patch_length_image_space = patch_length * average_scale
    overlap_image_space = overlap * average_scale

    patches = []
    x = min_x
    while x < max_x:
        y = min_y
        while y < max_y:
            patch_max_x = min(x + patch_length_image_space, max_x)
            patch_max_y = min(y + patch_length_image_space, max_y)
            patches.append((int(x), int(y), int(patch_max_x), int(patch_max_y)))
            y += patch_length_image_space - overlap_image_space
        x += patch_length_image_space - overlap_image_space

    # Optional plotting
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        if image is not None:
            ax.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
        ax.set_title("Patches within Bounding Box")
        for patch in patches:
            px_min, py_min, px_max, py_max = patch
            rect = plt.Rectangle((px_min, py_min), px_max - px_min, py_max - py_min,
                                 linewidth=1, edgecolor='b', facecolor='none', linestyle='--', label='Patch')
            ax.add_patch(rect)

        # Ensure no duplicate legends
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        ax.set_xlim([0, image.shape[1] if image is not None else max_x])
        ax.set_ylim([image.shape[0] if image is not None else max_y, 0])  # Invert y-axis for image display
        plt.show()

    return patches

#####

def enhance_matches_with_patches(
    initial_matches_df, match_result, target_map, base_dataset,
    patch_length=400, overlap=100, offset=100, plot=True
):
    
    """
    Enhances matches using a patch-based approach with Delaunay filtering.
    Defines patches in the target image, finds corresponding locations in the base image
    using thin plate spline (TPS) transformation, and creates warped patches for visual comparison.
    """

    # Retrieve the base map corresponding to the match
    base_map = next(
        (map_obj for map_obj in base_dataset if map_obj.map_info.folder == match_result.base_folder),
        None
    )
    if base_map is None:
        print("Base map not found.")
        return match_result

    # Read masks if available
    if target_map.map_info.mask_path is not None:
        target_map.mask = cv2.imread(target_map.map_info.mask_path, cv2.IMREAD_GRAYSCALE) if target_map.map_info.mask_path else None
    if base_map.map_info.mask_path is not None:
        base_map.mask = cv2.imread(base_map.map_info.mask_path, cv2.IMREAD_GRAYSCALE) if base_map.map_info.mask_path else None

        ##
    
    filtered_points_target = initial_matches_df[['kp1_x', 'kp1_y']].values
    filtered_points_base = initial_matches_df[['kp2_x', 'kp2_y']].values

    # Transform keypoints to image space
    filtered_points_target_image_space = apply_transform(
        filtered_points_target.reshape(1, -1, 2), target_map.tensor_to_image_transform
    ).reshape(-1, 2)
    filtered_points_base_image_space = apply_transform(
        filtered_points_base.reshape(1, -1, 2), base_map.tensor_to_image_transform
    ).reshape(-1, 2)

    if len(filtered_points_target) < 3 or len(filtered_points_base) < 3:
        logging.error("Not enough points for thin plate spline transformation.")
        return None

    # Create the TPS interpolator from target to base coordinates
    tps_transform = RBFInterpolator(
        filtered_points_target_image_space, filtered_points_base_image_space,
        kernel='thin_plate_spline', smoothing=1e-5
    )

    # Create the bounding box and patches for the target image
    offset_tensor_target, _ = meters_to_tensors(match_result, base_dataset, offset)
    patch_length_tensor, _ = meters_to_tensors(match_result, base_dataset, patch_length)
    overlap_tensor, _ = meters_to_tensors(match_result, base_dataset, overlap)
    bbox_target_image_space = get_bbox(target_map, filtered_points_target, offset_tensor_target, plot=plot)
   
    patches_target = create_patches_from_bbox(
        bbox_target_image_space, patch_length_tensor, overlap_tensor,
        target_map.tensor_to_image_transform, plot=plot, image=target_map.image
    )

    # Compute the inverse of tensor_to_image_transform for the target map
    target_map_image_to_tensor_transform = target_map.image_to_tensor_transform
    base_map_image_to_tensor_transform =  base_map.image_to_tensor_transform

    # Initialize a list to collect matches from all patches
    all_overall_matches_df = []

    for patch_idx, (min_x, min_y, max_x, max_y) in enumerate(patches_target):
        #print(f"Processing patch {patch_idx + 1}/{len(patches_target)}")

        # Define the corners of the patch in the target image space
        patch_corners_target_image = np.array([
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y]
        ], dtype='float32')

        # Collect points within the patch
        points_in_patch_target = np.array([
            point for point in filtered_points_target_image_space
            if min_x <= point[0] <= max_x and min_y <= point[1] <= max_y
        ])
        indices_in_patch = [
            i for i, point in enumerate(filtered_points_target_image_space)
            if min_x <= point[0] <= max_x and min_y <= point[1] <= max_y
        ]
        points_in_patch_base = filtered_points_base_image_space[indices_in_patch]

        if len(points_in_patch_target) < 3 or len(points_in_patch_base) < 3:
            print(f"Insufficient points in patch {patch_idx + 1} for transformation.")
            print("check whether to keep or not this functionality")
            continue
        # Transform the corners to the base image using the TPS transformation
        patch_corners_base_image = tps_transform(patch_corners_target_image)
        # Compute the transformation from the full target map to the temp target map
        # Since we are cropping, the transformation is a translation
        transformation_target_to_temp = np.array([
            [1, 0, -min_x],
            [0, 1, -min_y],
            [0, 0, 1]
        ], dtype=np.float32)

        # For the base map, we have a combination of the TPS transformation and cropping
        # We need to compute the overall transformation from the full base map to the temp base map
        # First, compute the projective transformation from the base image to the temp base patch
        width = int(max_x - min_x)
        height = int(max_y - min_y)
        dst_corners = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ], dtype='float32')

        # Include additional points for a better estimation
        src_points = np.concatenate([patch_corners_base_image, points_in_patch_base], axis=0)
        dst_points = np.concatenate([dst_corners, points_in_patch_target - np.array([min_x, min_y])], axis=0)

        # Estimate the projective transformation
        transform = ProjectiveTransform()
        transform.estimate(src_points, dst_points)

        # Compute the transformation matrix from the full base map to the temp base map
        transformation_base_to_temp = transform.params

        # Extract the patches from the images
        target_patch = target_map.image[min_y:max_y, min_x:max_x]
        warped_patch_base = warp(
            base_map.image, transform.inverse, output_shape=(height, width)
        )
        base_patch = (warped_patch_base * 255).astype(base_map.image.dtype)

        # Extract masks for the patches
        target_patch_mask = target_map.mask[min_y:max_y, min_x:max_x] if target_map.mask is not None else None 
        if target_patch_mask is None:
            print('No mask available for the target map')
        if base_map.mask is not None:
            warped_mask_base = warp(
                base_map.mask, transform.inverse, output_shape=(height, width), preserve_range=True
            )
            base_patch_mask = warped_mask_base.astype(base_map.mask.dtype)
        else:
            base_patch_mask = None
            print("No mask available for the base map.")


        target_info = MapInfo(
            folder=target_map.map_info.folder,
            author=target_map.map_info.author,
            year=target_map.map_info.year,
            image_path=None,
            mask_path=None,)
        
        temp_target_map = MapDataset(
            map_info=target_info,
            gcps=target_map.gcps,
            image=target_patch,
            mask=target_patch_mask,
            north_rotation_angle = None
        )

        base_info = MapInfo(
            folder=base_map.map_info.folder,
            author=base_map.map_info.author,
            year=base_map.map_info.year,
            image_path=None,
            mask_path=None,)
        
        temp_base_map = MapDataset(
            map_info=base_info,
            gcps=base_map.gcps,
            image=base_patch,
            mask=base_patch_mask,
            north_rotation_angle = None
        )
        #######
        ### here we want to create a subgrid with partially overlapping patches that will work as a sliding window where to compare the superpoint results
        temp_target_map.run_superpoint_pipeline()
        temp_base_map.run_superpoint_pipeline()
        # Get the tensors of the temporary maps
        target_tensor = temp_target_map.tensor
        base_tensor = temp_base_map.tensor

        # Shape of the tensors
        target_height, target_width = target_tensor.shape[-2], target_tensor.shape[-1]
        base_height, base_width = base_tensor.shape[-2], base_tensor.shape[-1]

        start_x_position_target, start_y_position_target  = 0, 0
        start_x_position_base, start_y_position_base = int(start_x_position_target * base_width / target_width), int(start_y_position_target * base_height / target_height)

        # Initialize lists to collect matches and bounds
        matches_dfs = []
        sliding_windows_target = []  # List of bounds for each sub-patch in target
        sliding_windows_base = []    # List of bounds for each sub-patch in base

        # Loop over sub-patches
        #for idx_x, start_x_t in enumerate(start_x_positions_target):
        end_x_t = start_x_position_target + target_width
        end_x_t = min(end_x_t, target_width)
        start_x_b = start_x_position_base
        end_x_b = start_x_b + base_width
        end_x_b = min(end_x_b, base_width)
        #for idx_y, start_y_t in enumerate(start_y_positions_target):
        end_y_t = start_y_position_target + target_height
        end_y_t = min(end_y_t, target_height)
        start_y_b = start_y_position_base
        end_y_b = start_y_b + base_height
        end_y_b = min(end_y_b, base_height)

        # Define bounds for the sub-patch
        bounds_target = (start_x_position_target, start_y_position_target, end_x_t, end_y_t)
        bounds_base = (start_x_b, start_y_b, end_x_b, end_y_b)

        # Store bounds
        sliding_windows_target.append(bounds_target)
        sliding_windows_base.append(bounds_base)

        # Run local SuperGlue matching using the temp maps
        matches, kp1, kp2 = run_local_superglue_matching(
            temp_target_map, temp_base_map, bounds_target, bounds_base, min_score=0.1
        )

        # Convert matches to DataFrame
        matches_df = matches_to_dataframe(matches, kp1, kp2)

        # After processing all sub-patches in this patch
        if matches_df is not None:

            kp1_temp = matches_df[['kp1_x', 'kp1_y']].values.astype(np.float32)
            kp2_temp = matches_df[['kp2_x', 'kp2_y']].values.astype(np.float32)

            temp_base_tensor_to_image_transform = temp_base_map.tensor_to_image_transform
            temp_target_tensor_to_image_transform = temp_target_map.tensor_to_image_transform
            kp1_temp_image = apply_transform(     # Transform keypoints to image space
                kp1_temp.reshape(-1, 1, 2), temp_target_tensor_to_image_transform
            ).reshape(-1, 2)
            kp2_temp_image = apply_transform(
                kp2_temp.reshape(-1, 1, 2), temp_base_tensor_to_image_transform
            ).reshape(-1, 2)

            # Transform keypoints back to full map image space
            kp1_full_image = apply_transform(
                kp1_temp_image.reshape(-1, 1, 2), np.linalg.inv(transformation_target_to_temp)
            ).reshape(-1, 2)
            kp2_full_image = apply_transform(
                kp2_temp_image.reshape(-1, 1, 2), np.linalg.inv(transformation_base_to_temp)
            ).reshape(-1, 2)

            # Transform from image space to tensor space
            kp1_full_tensor = apply_transform(
                kp1_full_image.reshape(-1, 1, 2), target_map_image_to_tensor_transform
            ).reshape(-1, 2)
            kp2_full_tensor = apply_transform(
                kp2_full_image.reshape(-1, 1, 2), base_map_image_to_tensor_transform
            ).reshape(-1, 2)

            # Confirm lengths match
            assert len(kp1_full_tensor) == len(matches_df), "Lengths do not match after transformation."

            # Update the dataframe with the transformed keypoints
            matches_df['kp1_x'] = kp1_full_tensor[:, 0]
            matches_df['kp1_y'] = kp1_full_tensor[:, 1]
            matches_df['kp2_x'] = kp2_full_tensor[:, 0]
            matches_df['kp2_y'] = kp2_full_tensor[:, 1]

            # Remove or set indices to NaNs since they don't correspond to original SuperPoint indices
            if 'kp1_idx' in matches_df.columns:
                matches_df['kp1_idx'] = np.nan
            if 'kp2_idx' in matches_df.columns:
                matches_df['kp2_idx'] = np.nan

            # Append to the list collecting all matches
            all_overall_matches_df.append(matches_df)

            if plot:
                # Plot the matches collected for this patch between the two patches
                plot_matches_on_patch(
                    temp_target_map, temp_base_map,
                    matches_df
                )
        else:
            print(f"No matches found in patch {patch_idx + 1}.")

    # After processing all patches
    if all_overall_matches_df:
        # Concatenate all matches from all patches
        enhanced_matches_df = pd.concat(all_overall_matches_df, ignore_index=True)

        # **Add the initial matches from delaunay_filtered_matches**
        initial_matches_df = initial_matches_df.copy()

        # Ensure that 'kp1_idx' and 'kp2_idx' columns are present and set to NaN
        initial_matches_df['kp1_idx'] = np.nan
        initial_matches_df['kp2_idx'] = np.nan

        # **Concatenate the initial matches with the enhanced matches**
        final_overall_matches_df = pd.concat([initial_matches_df, enhanced_matches_df], ignore_index=True)

        # **Final Check: Remove Duplicate Keypoints in Target Image**
        # Define threshold distance (in pixels) for considering keypoints as duplicates
        threshold_distance = 1  # Adjust this value as needed

        # Extract keypoints and match scores
        kp1_coords = final_overall_matches_df[['kp1_x', 'kp1_y']].values
        match_scores = final_overall_matches_df['match_score'].values
        #print(f"Initial matches: {len(initial_matches_df)}, Enhanced matches: {len(enhanced_matches_df)}")
        # Perform clustering on kp1_coords
        dbscan = DBSCAN(eps=threshold_distance, min_samples=1)
        clusters = dbscan.fit_predict(kp1_coords)

        # For each cluster, keep the match with the highest match_score
        indices_to_keep = []
        for cluster_id in set(clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            cluster_scores = match_scores[cluster_indices]
            best_index_in_cluster = cluster_indices[np.argmax(cluster_scores)]
            indices_to_keep.append(best_index_in_cluster)
        # Create a new DataFrame with only the selected matches
        final_matches_df = final_overall_matches_df.iloc[indices_to_keep].reset_index(drop=True)

        if plot:
            # Plot the matches on the full maps
            plot_matches_on_full_maps(
                target_map, base_map,
                final_matches_df
            )
        return final_matches_df
    else:
        # If no enhanced matches were found, use the initial matches
        print("No matches found in patches, using initial matches only.")
        initial_matches_df = match_result.delaunay_filtered_matches.copy()
        initial_matches_df['kp1_idx'] = np.nan
        initial_matches_df['kp2_idx'] = np.nan

        if plot:
            # Plot the matches on the full maps
            plot_matches_on_full_maps(
                target_map, base_map,
                initial_matches_df
            )
        return initial_matches_df

    

def apply_transform(points, transform_matrix):
    if transform_matrix.shape == (2, 3):
        # Use cv2.transform for affine transformations
        transformed_points = cv2.transform(points.reshape(-1, 1, 2), transform_matrix).reshape(-1, 2)
    elif transform_matrix.shape == (3, 3):
        # Use cv2.perspectiveTransform for homographies
        transformed_points = cv2.perspectiveTransform(points.reshape(-1, 1, 2), transform_matrix).reshape(-1, 2)
    else:
        raise ValueError(f"Unsupported transform matrix shape: {transform_matrix.shape}")
    return transformed_points

# import superglue
import torch
from SuperGluePretrainedNetwork.models.superglue import SuperGlue
# Initialize and load the SuperGlue model with configuration

superglue_config = {
    'weights': 'outdoor',  # Load the outdoor pretrained weights
    'sinkhorn_iterations': 100,
    'match_threshold': 0.2
}

superglue = SuperGlue(superglue_config)
superglue.load_state_dict(torch.load('./SuperGluePretrainedNetwork/models/weights/superglue_outdoor.pth'))
superglue.eval()  # Set to evaluation mode


def get_subgrid_size_and_stride(tensor_width, tensor_height, grid_subdivisions, overlap_fraction):
    subgrid_size_x = int(tensor_width / grid_subdivisions)
    subgrid_size_y = int(tensor_height / grid_subdivisions)
    overlap_pixels_x = int(subgrid_size_x * overlap_fraction)
    overlap_pixels_y = int(subgrid_size_y * overlap_fraction)
    stride_x = subgrid_size_x - overlap_pixels_x
    stride_y = subgrid_size_y - overlap_pixels_y
    return subgrid_size_x, subgrid_size_y, stride_x, stride_y

def generate_sliding_windows(start, end, window_size, stride):
    positions = []
    pos = start
    while pos + window_size <= end:
        positions.append(pos)
        pos += stride
    if pos < end:
        positions.append(end - window_size)
    return positions
def run_local_superglue_matching(map1, map2, bounds_target, bounds_base, min_score=0.1):
    """
    Uses precomputed keypoints and descriptors from two MapDataset objects
    to run SuperGlue inference and find matches within specified bounds.

    Parameters:
    - map1, map2: Two MapDataset objects to compare.
    - bounds_target, bounds_base: Bounds for filtering keypoints in the tensor space
    - min_score: Minimum score for filtering keypoints.

    Returns:
    - matches: The matching results from SuperGlue.
    - kp1, kp2: Keypoints from map1 and map2.
    """
    with torch.no_grad():
        image0_tensor = map1.tensor
        image1_tensor = map2.tensor

        # Use full SuperPoint results
        map1_superpoint_results = map1.superpoint_results
        map2_superpoint_results = map2.superpoint_results

        # Extract keypoints, descriptors, and scores
        kp1 = map1_superpoint_results.keypoints[0]
        kp2 = map2_superpoint_results.keypoints[0]
        desc1 = map1_superpoint_results.descriptors[0]
        desc2 = map2_superpoint_results.descriptors[0]
        scores1 = map1_superpoint_results.scores[0]
        scores2 = map2_superpoint_results.scores[0]

        # Convert bounds to tensors for consistent comparison
        min_x_t, min_y_t, max_x_t, max_y_t = bounds_target
        min_x_b, min_y_b, max_x_b, max_y_b = bounds_base

        # Filter keypoints that are within bounds
        in_bounds_kp1 = (kp1[:, 0] >= min_x_t) & (kp1[:, 0] <= max_x_t) & \
                        (kp1[:, 1] >= min_y_t) & (kp1[:, 1] <= max_y_t)
        in_bounds_kp2 = (kp2[:, 0] >= min_x_b) & (kp2[:, 0] <= max_x_b) & \
                        (kp2[:, 1] >= min_y_b) & (kp2[:, 1] <= max_y_b)

        # Filter keypoints, descriptors, and scores based on in-bounds condition
        kp1 = kp1[in_bounds_kp1]
        kp2 = kp2[in_bounds_kp2]
        desc1 = desc1[in_bounds_kp1]
        desc2 = desc2[in_bounds_kp2]
        scores1 = scores1[in_bounds_kp1]
        scores2 = scores2[in_bounds_kp2]

        # Further filter by scores
        valid1 = scores1 > min_score
        valid2 = scores2 > min_score
        kp1 = kp1[valid1]
        kp2 = kp2[valid2]
        desc1 = desc1[valid1]
        desc2 = desc2[valid2]
        scores1 = scores1[valid1]
        scores2 = scores2[valid2]

        # Check if there are enough keypoints after filtering
        if kp1.shape[0] == 0 or kp2.shape[0] == 0:
            print("No valid keypoints after filtering.")
            # Create empty matches dictionary with expected keys
            matches = {
                'matches0': [torch.empty(0, dtype=torch.int)],
                'matching_scores0': [torch.empty(0, dtype=torch.float)]
            }
            return matches, [kp1], [kp2]
        
        # Prepare data for SuperGlue
        sg_data = {
            'keypoints0': kp1.unsqueeze(0),
            'keypoints1': kp2.unsqueeze(0),
            'descriptors0': desc1.unsqueeze(0).transpose(1, 2),
            'descriptors1': desc2.unsqueeze(0).transpose(1, 2),
            'scores0': scores1.unsqueeze(0),
            'scores1': scores2.unsqueeze(0),
            'image0': image0_tensor,
            'image1': image1_tensor
        }

        # Run SuperGlue
        matches = superglue(sg_data)


    return matches, sg_data['keypoints0'], sg_data['keypoints1']

def plot_matches_on_patch(target_map, base_map, matches_df):
    """
    Plots all matches collected for the entire patch, visualizing match scores.

    Parameters:
    - target_map, base_map: MapDataset objects for target and base patches (temp maps).
    - matches_df: DataFrame containing all matches for the patch.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2

    # Combine images side by side
    image1 = target_map.image
    image2 = base_map.image
    combined_image = np.hstack((image1, image2))

    # Get the width of the target image for adjusting base keypoints
    target_image_width = image1.shape[1]

    # Check if matches_df is empty
    if matches_df.empty:
        print("No matches to plot for this patch.")
        # Optionally, display the combined image without matches
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
        ax.set_title("No Matches Found in This Patch")
        ax.axis('off')
        plt.tight_layout()
        plt.show()
        return

    # Extract keypoints and match scores from the dataframe
    kp1 = matches_df[['kp1_x', 'kp1_y']].values  # Keypoints in tensor space (target)

    kp2 = matches_df[['kp2_x', 'kp2_y']].values  # Keypoints in tensor space (base)

    match_scores = matches_df['match_score'].values  # Match scores

    # Ensure that the arrays have valid data (no NaNs)
    valid_indices = ~np.isnan(kp1).any(axis=1) & ~np.isnan(kp2).any(axis=1) & ~np.isnan(match_scores)
    kp1 = kp1[valid_indices]
    kp2 = kp2[valid_indices]
    match_scores = match_scores[valid_indices]

    # Check if there are any valid matches after filtering
    if kp1.size == 0 or kp2.size == 0 or match_scores.size == 0:
        print("No valid matches to plot after filtering.")
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
        ax.set_title("No Valid Matches Found in This Patch")
        ax.axis('off')
        plt.tight_layout()
        plt.show()
        return

    # Normalize match scores for visualization
    score_range = match_scores.max() - match_scores.min()
    if score_range != 0:
        normalized_scores = (match_scores - match_scores.min()) / score_range
    else:
        normalized_scores = np.ones_like(match_scores)

    # Retrieve the tensor-to-image transformations
    target_tensor_to_image_transform = target_map.tensor_to_image_transform
    base_tensor_to_image_transform = base_map.tensor_to_image_transform

    # Transform keypoints into image space
    kp1_image_space = apply_transform(kp1, target_tensor_to_image_transform)
    
    kp2_image_space = apply_transform(kp2, base_tensor_to_image_transform)

    # Adjust base keypoints x-coordinate for side-by-side combined image
    kp2_image_space[:, 0] += target_image_width

    # Ensure all arrays are of the same length
    min_length = min(len(kp1_image_space), len(kp2_image_space), len(normalized_scores))
    if min_length == 0:
        print("No matches to plot after processing.")
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
        ax.set_title("No Matches Found in This Patch")
        ax.axis('off')
        plt.tight_layout()
        plt.show()
        return

    # Truncate arrays to the minimum length
    kp1_image_space = kp1_image_space[:min_length]
    kp2_image_space = kp2_image_space[:min_length]
    normalized_scores = normalized_scores[:min_length]

    # Plotting
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
    ax.set_title("Matches on Patch")
    ax.axis('off')

    # Plot keypoints and matches with varying alpha based on match score
    for target_point, base_point, score in zip(kp1_image_space, kp2_image_space, normalized_scores):

        ax.scatter(*target_point, c=colors['aqua_dark'], s=3, alpha=score)
        ax.scatter(*base_point, c=colors['pink_dark'], s=3, alpha=score)
        ax.plot(
            [target_point[0], base_point[0]],
            [target_point[1], base_point[1]],
            'w-', linewidth=0.5, alpha=score
        )

    plt.tight_layout()
    plt.show()


def plot_matches_on_subpatch(target_map, base_map, matches_df, bounds_target, bounds_base):
    """
    Plots matches for a selected sub-patch, in the context of the full images,
    visualizing match scores.
    """
    # Combine images side by side
    image1 = target_map.image
    image2 = base_map.image
    combined_image = np.hstack((image1, image2))

    # Get the width of the target image for adjusting base keypoints
    target_image_width = image1.shape[1]

    # Retrieve the tensor-to-image transformations
    target_tensor_to_image_transform = target_map.tensor_to_image_transform
    base_tensor_to_image_transform = base_map.tensor_to_image_transform

    # Plotting
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
    ax.set_title("Matches on Selected Sub-Patch")
    ax.axis('off')

    # Draw rectangle for the selected sub-patch on the target image
    start_x_t, start_y_t, end_x_t, end_y_t = bounds_target
    top_left_t = np.array([[start_x_t, start_y_t]], dtype=np.float32)
    bottom_right_t = np.array([[end_x_t, end_y_t]], dtype=np.float32)
    top_left_img_t = apply_transform(top_left_t.reshape(-1, 1, 2), target_tensor_to_image_transform).reshape(-1, 2)[0]
    bottom_right_img_t = apply_transform(bottom_right_t.reshape(-1, 1, 2), target_tensor_to_image_transform).reshape(-1, 2)[0]
    rect_t = plt.Rectangle(top_left_img_t, bottom_right_img_t[0] - top_left_img_t[0], bottom_right_img_t[1] - top_left_img_t[1],
                           edgecolor=colors['orange_light'], facecolor='none', linewidth=2)
    ax.add_patch(rect_t)

    # Draw rectangle for the selected sub-patch on the base image
    start_x_b, start_y_b, end_x_b, end_y_b = bounds_base
    top_left_b = np.array([[start_x_b, start_y_b]], dtype=np.float32)
    bottom_right_b = np.array([[end_x_b, end_y_b]], dtype=np.float32)
    top_left_img_b = apply_transform(top_left_b.reshape(-1, 1, 2), base_tensor_to_image_transform).reshape(-1, 2)[0]
    bottom_right_img_b = apply_transform(bottom_right_b.reshape(-1, 1, 2), base_tensor_to_image_transform).reshape(-1, 2)[0]
    # Adjust x-coordinate for combined image
    top_left_img_b[0] += target_image_width
    bottom_right_img_b[0] += target_image_width
    rect_b = plt.Rectangle(top_left_img_b, bottom_right_img_b[0] - top_left_img_b[0], bottom_right_img_b[1] - top_left_img_b[1],
                           edgecolor=colors['orange_light'], facecolor='none', linewidth=2)
    ax.add_patch(rect_b)

    # Check if there are matches to plot
    if not matches_df.empty:
        # Extract keypoints and match scores from the dataframe
        kp1 = matches_df[['kp1_x', 'kp1_y']].values  # Keypoints in tensor space (target)
        kp2 = matches_df[['kp2_x', 'kp2_y']].values  # Keypoints in tensor space (base)
        match_scores = matches_df['match_score'].values  # Match scores

        # Filter out invalid match scores
        valid_scores_mask = ~np.isnan(match_scores) & ~np.isinf(match_scores)
        if not np.any(valid_scores_mask):
            print("All match scores are invalid (nan or inf).")
            return

        # Apply the mask to match_scores and keypoints
        match_scores = match_scores[valid_scores_mask]
        kp1 = kp1[valid_scores_mask]
        kp2 = kp2[valid_scores_mask]

        # Normalize match scores for visualization
        match_scores_min = np.nanmin(match_scores)
        match_scores_max = np.nanmax(match_scores)
        score_range = match_scores_max - match_scores_min

        if np.isnan(score_range) or score_range == 0:
            normalized_scores = np.ones_like(match_scores)
        else:
            normalized_scores = (match_scores - match_scores_min) / score_range

        # Transform keypoints into image space
        kp1_image_space = apply_transform(
            kp1.reshape(-1, 1, 2).astype(np.float32), target_tensor_to_image_transform
        ).reshape(-1, 2)
        kp2_image_space = apply_transform(
            kp2.reshape(-1, 1, 2).astype(np.float32), base_tensor_to_image_transform
        ).reshape(-1, 2)

        # Adjust base keypoints x-coordinate for side-by-side combined image
        kp2_image_space[:, 0] += target_image_width

        # Plot keypoints and matches with varying alpha based on match score
        for idx in range(len(kp1_image_space)):
            target_point = kp1_image_space[idx]
            base_point = kp2_image_space[idx]
            score = normalized_scores[idx]

            # Ensure alpha is within [0, 1]
            if np.isnan(score) or np.isinf(score):
                score = 1.0
            else:
                score = np.clip(score, 0.0, 1.0)
            ax.scatter(*target_point, c=colors['aqua_dark'], s=3, alpha=score)
            ax.scatter(*base_point, c=colors['pink_dark'], s=3, alpha=score)
            ax.plot(
                [target_point[0], base_point[0]],
                [target_point[1], base_point[1]],
                'w-', linewidth=0.5, alpha=score
            )
    else:
        print("No matches to plot for the selected sub-patch.")

    plt.tight_layout()
    plt.show()

def plot_matches_on_full_maps(target_map: MapDataset, base_map: MapDataset, matches_df: pd.DataFrame):
    """
    Plots the matches across the full target and base maps by resizing images to the same height.

    Parameters:
    - target_map: The MapDataset object for the target map.
    - base_map: The MapDataset object for the base map.
    - matches_df: DataFrame containing matches with keypoints in tensor coordinates.
    """
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np

    # Load images
    image1 = target_map.image
    image2 = base_map.image

    if image1 is None or image2 is None:
        raise ValueError("Target or base map image is not loaded.")

    #eventually used to downscale the saved image, can be used to go back to the original shapes
    # Get original heights and widths
    h1, w1 = image1.shape[:2] 
    h2, w2 = image2.shape[:2]

    # Resize images to the same height while maintaining aspect ratio
    desired_height = max(h1, h2)
    scaling_factor1 = desired_height / h1
    scaling_factor2 = desired_height / h2

    new_w1 = int(w1 * scaling_factor1)
    new_w2 = int(w2 * scaling_factor2)

    image1_resized = cv2.resize(image1, (new_w1, desired_height), interpolation=cv2.INTER_AREA)
    image2_resized = cv2.resize(image2, (new_w2, desired_height), interpolation=cv2.INTER_AREA)

    # Combine images side by side
    combined_image = np.hstack((image1_resized, image2_resized))

    # Adjust base map keypoints' x-coordinates for side-by-side layout
    width_image1 = image1_resized.shape[1]

    # Transform keypoints to original image space
    if matches_df.empty:
        print("No matches to plot on the full maps.")
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
        ax.set_title("No Matches Found")
        ax.axis('off')
        plt.tight_layout()
        plt.show()
        return

    # Extract keypoints and match scores
    kp1_tensor = matches_df[['kp1_x', 'kp1_y']].values.astype(np.float32)
    kp2_tensor = matches_df[['kp2_x', 'kp2_y']].values.astype(np.float32)
    match_scores = matches_df['match_score'].values

    kp1_image_space = apply_transform(kp1_tensor, target_map.tensor_to_image_transform)
    kp2_image_space = apply_transform(kp2_tensor, base_map.tensor_to_image_transform)

    # Apply resizing scaling factors
    kp1_image_space *= scaling_factor1
    kp2_image_space *= scaling_factor2
    kp2_image_space[:, 0] += width_image1  # Adjust x-coordinate for combined layout

    # Normalize match scores for alpha transparency
    score_range = np.nanmax(match_scores) - np.nanmin(match_scores)
    normalized_scores = (
        (match_scores - np.nanmin(match_scores)) / score_range if score_range > 0 else np.ones_like(match_scores)
    )

    # Plot combined image and matches
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
    ax.set_title("Matches on Full Maps")
    ax.axis('off')

    for idx in range(len(kp1_image_space)):
        target_point = kp1_image_space[idx]
        base_point = kp2_image_space[idx]
        score = normalized_scores[idx]

        ax.scatter(*target_point, c=colors['aqua_dark'], s=3, alpha=score)
        ax.scatter(*base_point, c=colors['pink_dark'], s=3, alpha=score)
        ax.plot(
            [target_point[0], base_point[0]],
            [target_point[1], base_point[1]],
            'w-', linewidth=0.5, alpha=score
        )

    plt.tight_layout()
    plt.show()

# read the evaluation dataset
#import pickle
#evaluation_dataset = pickle.load(open('./output/evaluation_dataset.pkl', 'rb'))

#eval_map = evaluation_dataset[2]
# as an eval map take the folder 1860_pierotti
#eval_map = [map_obj for map_obj in evaluation_dataset if map_obj.folder == '1860_pierotti'][0]
#match_result = eval_map.best_matches_result[0]
#initial_matches_df = match_result.delaunay_filtered_matches
#match_result.enhanced_matches = enhance_matches_with_patches(initial_matches_df, match_result, eval_map, base_dataset, patch_length=1600, overlap=500, grid_subdivisions=1, plot=True)



