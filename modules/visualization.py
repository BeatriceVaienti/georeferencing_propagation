import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from shapely.geometry import Polygon as ShapelyPolygon
from matplotlib.patches import Polygon
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
def plot_filtered_match(match_result, target_map, base_dataset):
    """
    Plots the matches between an evaluation map and its best match, showing inliers and outliers.

    Parameters:
    - match_result: A MatchingResult object containing the filtered match result for a single evaluation map.
    - evaluation_dataset: List of MapDataset objects for evaluation.
    - base_dataset: List of MapDataset objects for base maps.
    """
    # Retrieve eval_map and base_map using their folder names
    eval_map_folder = match_result.evaluation_folder
    base_map_folder = match_result.base_folder
    base_map = next((map_obj for map_obj in base_dataset if map_obj.map_info.folder == base_map_folder), None)
    
    if target_map is None or base_map is None:
        print(f"Could not find maps for {eval_map_folder} or {base_map_folder}.")
        return

    # Load images
    image1 = cv2.imread(target_map.map_info.image_path)
    image2 = cv2.imread(base_map.map_info.image_path)

    if image1 is None or image2 is None:
        print(f"Failed to load images for {target_map.map_info.folder} and {base_map.map_info.folder}")
        return


    # Convert images to RGB
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    # Extract keypoint coordinates (already in tensor space)
    kp1_coords = match_result.superglue_matches[['kp1_x', 'kp1_y']].values
    kp2_coords = match_result.superglue_matches[['kp2_x', 'kp2_y']].values
    inlier_indices = match_result.ransac_filtered_matches.index.values  # Inlier indices

    # Transform the keypoints to the original image space using the transformation matrix
    transformation_matrix_target = target_map.tensor_to_image_transform
    transformation_matrix_base = base_map.tensor_to_image_transform
    kp1_coords = cv2.transform(kp1_coords.reshape(-1, 1, 2), transformation_matrix_target).reshape(-1, 2)
    kp2_coords = cv2.transform(kp2_coords.reshape(-1, 1, 2), transformation_matrix_base).reshape(-1, 2)

    # Combine resized images side by side
    h1, w1, _ = image1_rgb.shape
    h2, w2, _ = image2_rgb.shape
    H = max(h1, h2)
    W = w1 + w2
    combined_image = np.zeros((H, W, 3), dtype=np.uint8)
    combined_image[:h1, :w1, :] = image1_rgb
    combined_image[:h2, w1:w1 + w2, :] = image2_rgb

    plt.figure(figsize=(15, 10))
    plt.imshow(combined_image)

    # Shift kp2 x-coordinates by w1 to align with combined image
    kp2_coords_shifted = kp2_coords.copy()
    kp2_coords_shifted[:, 0] += w1

    # Plot inliers in green
    for idx in inlier_indices:
        x1, y1 = kp1_coords[idx]
        x2, y2 = kp2_coords_shifted[idx]
        plt.plot([x1, x2], [y1, y2], color='green', linewidth=0.5)
        plt.scatter([x1, x2], [y1, y2], color='green', s=10)

    # Plot outliers in red
    outlier_indices = np.setdiff1d(match_result.superglue_matches.index.values, inlier_indices)
    for idx in outlier_indices:
        x1, y1 = kp1_coords[idx]
        x2, y2 = kp2_coords_shifted[idx]
        plt.plot([x1, x2], [y1, y2], color='red', linewidth=0.5)
        plt.scatter([x1, x2], [y1, y2], color='red', s=10)

    plt.axis('off')
    plt.title(f"Matches between {target_map.map_info.folder} and {base_map.map_info.folder}\nInliers (Green), Outliers (Red)")
    plt.show()

def plot_delaunay_matches(match_result, match_result_1, match_result_2, target_map, base_dataset):
    """
    Plots the Delaunay triangulation of matched keypoints between two images.

    The function displays two rows of plots:
    - match 1
    - match 2
    """
    # Retrieve eval_map and base_map using their folder names
    target_map_folder = match_result.evaluation_folder
    base_map_folder = match_result.base_folder
    base_map = next((map_obj for map_obj in base_dataset if map_obj.map_info.folder == base_map_folder), None)

    if target_map is None or base_map is None:
        print(f"Could not find maps for {target_map_folder} or {base_map_folder}.")
        return

    # Load images
    image1 = cv2.imread(target_map.map_info.image_path)
    image2 = cv2.imread(base_map.map_info.image_path)
    if image1 is None or image2 is None:
        print(f"Failed to load images for {target_map.map_info.folder} and {base_map.map_info.folder}")
        return

    # Convert images to RGB
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    # Darken the images
    dark_factor = 0.65  # Adjust this factor to control darkness (0.0 to 1.0)
    image1_rgb = (image1_rgb * dark_factor).astype(np.uint8)
    image2_rgb = (image2_rgb * dark_factor).astype(np.uint8)

    # Extract keypoint coordinates
    kp1_coords_1 = match_result_1[['kp1_x', 'kp1_y']].values.astype(np.float32)
    kp2_coords_1 = match_result_1[['kp2_x', 'kp2_y']].values.astype(np.float32)
    kp1_coords_2 = match_result_2[['kp1_x', 'kp1_y']].values.astype(np.float32)
    kp2_coords_2 = match_result_2[['kp2_x', 'kp2_y']].values.astype(np.float32)

    # Transform keypoints
    transformation_matrix_target = target_map.tensor_to_image_transform[:2, :]
    transformation_matrix_base = base_map.tensor_to_image_transform[:2, :]

    kp1_coords_1 = cv2.transform(kp1_coords_1.reshape(-1, 1, 2), transformation_matrix_target).reshape(-1, 2)
    kp2_coords_1 = cv2.transform(kp2_coords_1.reshape(-1, 1, 2), transformation_matrix_base).reshape(-1, 2)
    kp1_coords_2 = cv2.transform(kp1_coords_2.reshape(-1, 1, 2), transformation_matrix_target).reshape(-1, 2)
    kp2_coords_2 = cv2.transform(kp2_coords_2.reshape(-1, 1, 2), transformation_matrix_base).reshape(-1, 2)

    # Delaunay triangulation requires at least 3 points
    if len(kp1_coords_1) < 3 or len(kp1_coords_2) < 3:
        print("Not enough points for Delaunay triangulation.")
        return

    # Prepare plot
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # Colors
    color1= colors['aqua_light']
    color2 = colors['pink_dark']

    def plot_delaunay_with_overlaps(ax, img, keypoints, triangles, title):
        
        # make the image darker
        ax.imshow(img)

        # Create polygons from triangles
        polygons = [ShapelyPolygon(keypoints[simplex]) for simplex in triangles.simplices]

        # Identify overlapping triangles
        overlaps = []
        for i in range(len(polygons)):
            for j in range(i + 1, len(polygons)):
                if polygons[i].intersects(polygons[j]) and polygons[i].intersection(polygons[j]).area > 0:
                    overlaps.extend([i, j])
        overlaps = list(set(overlaps))

        # Plot triangles, highlighting overlaps
        for i, simplex in enumerate(triangles.simplices):
            triangle_points = keypoints[simplex]
            if i in overlaps:
                poly = Polygon(triangle_points, closed=True, facecolor=color2, alpha=0.5)
                ax.add_patch(poly)
            ax.plot(triangle_points[:, 0], triangle_points[:, 1], c=color1, linewidth=1.5)

        ax.set_title(title, fontsize=14)
        ax.axis('off')
        return overlaps

    # Row 1: Match 1
    tri_all = Delaunay(kp1_coords_1)
    axes[0, 0].scatter(kp1_coords_1[:, 0], kp1_coords_1[:, 1], c='blue', s=5)
    plot_delaunay_with_overlaps(axes[0, 0], image1_rgb, kp1_coords_1, tri_all, "Delaunay on All Matches (Image 1)")

    axes[0, 1].scatter(kp2_coords_1[:, 0], kp2_coords_1[:, 1], c='green', s=5)
    overlaps_1 = plot_delaunay_with_overlaps(
        axes[0, 1], image2_rgb, kp2_coords_1, tri_all, "Delaunay on All Matches (Image 2)"
    )
    print(f"Number of overlapping triangles in Match 1: {len(overlaps_1)}")

    # Row 2: Match 2
    tri_2 = Delaunay(kp1_coords_2)
    axes[1, 0].scatter(kp1_coords_2[:, 0], kp1_coords_2[:, 1], c='blue', s=5)
    plot_delaunay_with_overlaps(axes[1, 0], image1_rgb, kp1_coords_2, tri_2, "Delaunay on Inliers (Image 1)")

    axes[1, 1].scatter(kp2_coords_2[:, 0], kp2_coords_2[:, 1], c='green', s=5)
    overlaps_2 = plot_delaunay_with_overlaps(
        axes[1, 1], image2_rgb, kp2_coords_2, tri_2, "Delaunay on Inliers (Image 2)"
    )
    print(f"Number of overlapping triangles in Match 2: {len(overlaps_2)}")

    plt.tight_layout()
    plt.show()
