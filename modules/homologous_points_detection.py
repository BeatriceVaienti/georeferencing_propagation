from SuperGluePretrainedNetwork.models.superglue import SuperGlue
import torch
import numpy as np
from typing import List
from tqdm import tqdm
from modules.MapDataset import MapDataset
from modules.MatchingResult import MatchingResult
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.strtree import STRtree
from collections import Counter
import copy
import gc  # Garbage collector for memory management
import cv2
import matplotlib.pyplot as plt
from skimage.measure import ransac
from skimage.transform import AffineTransform
from scipy.spatial import Delaunay
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.strtree import STRtree
from collections import Counter
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.strtree import STRtree
from collections import Counter

from collections import defaultdict


from collections import Counter
import numpy as np


superglue_config = {
    'weights': 'outdoor',  # Load the outdoor pretrained weights
    'sinkhorn_iterations': 20,
    'match_threshold': 0.2
}

# Initialize and load the SuperGlue model with configuration
superglue = SuperGlue(superglue_config)
superglue.load_state_dict(torch.load('./SuperGluePretrainedNetwork/models/weights/superglue_outdoor.pth'))
superglue.eval()  # Set to evaluation mode


def run_superglue_matching(map1, map2, min_score=0.1):
    """
    Uses precomputed keypoints and descriptors from two MapDataset objects
    to run SuperGlue inference and find matches.
    
    Parameters:
    - map1, map2: Two MapDataset objects to compare.

    Returns:
    - matches: The matching results from SuperGlue.
    - kp1, kp2: Keypoints from map1 and map2.
    """
    with torch.no_grad():
        # Ensure the tensors have the correct dimensions
        image0_tensor = map1.tensor
        image1_tensor = map2.tensor

        # Convert keypoints, descriptors, and scores to tensors if they are not already
        map1_superpoint_results = map1.superpoint_results
        map2_superpoint_results = map2.superpoint_results
        kp1 = map1_superpoint_results.keypoints[0] if isinstance(map1_superpoint_results.keypoints[0], torch.Tensor) else torch.from_numpy(np.array(map1_superpoint_results.keypoints[0]))
        kp2 = map2_superpoint_results.keypoints[0] if isinstance(map2_superpoint_results.keypoints[0], torch.Tensor) else torch.from_numpy(np.array(map2_superpoint_results.keypoints[0]))
        desc1 = map1_superpoint_results.descriptors[0] if isinstance(map1_superpoint_results.descriptors[0], torch.Tensor) else torch.from_numpy(np.array(map1_superpoint_results.descriptors[0]))
        desc2 = map2_superpoint_results.descriptors[0] if isinstance(map2_superpoint_results.descriptors[0], torch.Tensor) else torch.from_numpy(np.array(map2_superpoint_results.descriptors[0]))
        scores1 = map1_superpoint_results.scores[0] if isinstance(map1_superpoint_results.scores[0], torch.Tensor) else torch.from_numpy(np.array(map1_superpoint_results.scores[0]))
        scores2 = map2_superpoint_results.scores[0] if isinstance(map2_superpoint_results.scores[0], torch.Tensor) else torch.from_numpy(np.array(map2_superpoint_results.scores[0]))

        # Filter out low-score keypoints
        valid1 = scores1 > min_score
        valid2 = scores2 > min_score
        kp1 = kp1[valid1]
        kp2 = kp2[valid2]
        desc1 = desc1[valid1]
        desc2 = desc2[valid2]
        scores1 = scores1[valid1]
        scores2 = scores2[valid2]

        # Prepare data for SuperGlue
        sg_data = {
            'keypoints0': kp1.unsqueeze(0),   # [1, num_keypoints, 2]
            'keypoints1': kp2.unsqueeze(0),   # [1, num_keypoints, 2]
            # Transpose descriptors to [1, 256, num_keypoints]
            'descriptors0': desc1.unsqueeze(0).transpose(1, 2),  # [1, 256, num_keypoints]
            'descriptors1': desc2.unsqueeze(0).transpose(1, 2),  # [1, 256, num_keypoints]
            'scores0': scores1.unsqueeze(0),  # [1, num_keypoints]
            'scores1': scores2.unsqueeze(0),  # [1, num_keypoints]
            'image0': image0_tensor,  # [1, 1, H, W]
            'image1': image1_tensor   # [1, 1, H, W]
        }
        # Run SuperGlue
        matches = superglue(sg_data)
        
    return matches, sg_data['keypoints0'], sg_data['keypoints1']


def find_single_best_match(target_map: MapDataset, base_set: List[MapDataset], min_score=0.1):
    """
    Finds the best matching map in the base set for a single map in the target set
    using precomputed SuperPoint data and SuperGlue.

    Parameters:
    - target_map: The MapDataset object for evaluation.
    - base_set: List of MapDataset objects for the base dataset.

    Returns:
    - None (the result is stored inside the MapDataset object)
    """
    _, match_results = compute_stats_and_matches(target_map, base_set, min_score=min_score)
    best_match_result = max(match_results, key=lambda x: x['valid_matches'])
    return best_match_result

def find_best_matches(target_set: List[MapDataset], base_set: List[MapDataset], threshold_number_matches: int = 100, number_best_results = 3, evaluation = True, min_score=0.1):
    """
    Finds the best matching maps or single map in the base set for each map in the target set
    which could either be an evaluation set, or the one for inference.

    if the function is set to "evaluation" mode it will collect multiple matches:
    - best number_best_results (3) matches
    - matches with more than threshold_number_matches (100) matches 
    (with no overlap, i.e. if the same map is in both categories, it will be counted only once)
    using precomputed SuperPoint data and SuperGlue.

    Parameters:
    - evaluation_set: List of MapDataset objects for evaluation.
    - base_set: List of MapDataset objects for the base dataset.
    - threshold_number_matches: Minimum number of matches to consider for additional matches.

    Returns:
    - None (the results are stored inside the MapDataset objects)
    """
    for target_map in tqdm(target_set, desc="Evaluating maps"):
        matches_stats_list, match_results = compute_stats_and_matches(target_map, base_set, min_score=min_score) # we collect the stats on the number of zero-shot matches that we obtain between each pair of maps and the point pairs with scores
        target_map.initial_matches_stats = pd.DataFrame(matches_stats_list) # we store the stats in the target map object
        store_match_results(target_map, match_results, threshold_number_matches, number_best_results, evaluation) # we store the match results in the target map object


def store_match_results(target_map: MapDataset, match_results: List[dict], threshold_number_matches: int, number_best_results: int, evaluation: bool):
    """
    Stores the match results in the target map object.

    Parameters:
    - target_map: The evaluation MapDataset object.
    - match_results: List of dictionaries with match results.
    - threshold_number_matches: Minimum number of matches to consider for additional matches (used in evaluation mode).
    - number_best_results: Number of best matches to store (used in evaluation mode).
    - evaluation: Boolean flag to indicate if the evaluation mode is active.
    """
    # Sort match_results by valid_matches in descending order
    match_results_sorted = sorted(match_results, key=lambda x: x['valid_matches'], reverse=True)

    if evaluation:
        match_results_selected = match_results_sorted[:number_best_results]
        # add more in case after the number best_results we have more than threshold_number_matches
        for match_data in match_results_sorted[number_best_results:]:
            if match_data['valid_matches'] >= threshold_number_matches:
                match_results_selected.append(match_data)
    else:
        # only keep the best match
        match_results_selected = match_results_sorted[:1]

    # for each match_results_selected we create a MatchingResult object and we collect it as target_map.best_matches_result (list of MatchingResult objects, one for each base map, it will only have one element if evaluation is False)
    matches_results = []
    for match_data in match_results_selected:
        match_results = create_matching_result(target_map, match_data)
        matches_results.append(match_results)
    
    target_map.best_matches_result = matches_results


def create_matching_result(target_map: MapDataset, match_data: dict) -> MatchingResult:
    """
    Creates a MatchingResult object from match data.

    Parameters:
    - eval_map: The evaluation MapDataset object.
    - match_data: Dictionary containing match data.

    Returns:
    - MatchingResult object.
    """
    return MatchingResult(
        evaluation_folder=target_map.map_info.folder,
        base_folder=match_data['base_map'].map_info.folder,
        superglue_matches=match_data['superglue_matches_df'],
        delaunay_filtered_matches=None,  
        enhanced_matches=None,
        enhanced_delaunay_filtered_matches=None,
        gcp_propagated=None
    )
from copy import deepcopy

def rotate_map_dataset(map_dataset: MapDataset, num_rotations: int) -> MapDataset:
    """
    Rotates the map dataset clockwise by 90 degrees num_rotations times.
    Recomputes the tensor and SuperPoint features after rotation.

    Parameters:
    - map_dataset: MapDataset to rotate
    - num_rotations: Number of times to rotate by 90 degrees (clockwise).

    Returns:
    - A new MapDataset that is the rotated version of the original.
    """

    num_rotations = num_rotations % 4
    if num_rotations == 0:
        map_dataset.number_of_rotations = 0
        return map_dataset

    rotated_map = deepcopy(map_dataset)
    rotated_map.image = np.ascontiguousarray(np.rot90(map_dataset.image, k=-num_rotations)) if map_dataset.image is not None else None
    rotated_map.mask = np.ascontiguousarray(np.rot90(map_dataset.mask, k=-num_rotations)) if map_dataset.mask is not None else None

    rotated_map.generate_tensor()
    rotated_map.run_superpoint_inference()
    rotated_map.remove_points_near_mask()
    rotated_map.number_of_rotations = num_rotations
    rotated_map.rotated_versions = {}
    
    return rotated_map


def compute_stats_and_matches(target_map: MapDataset, base_set: List[MapDataset], min_score=0.1):
    """
    Processes an evaluation map by matching it against all base maps, considering rotations.
    Stores rotated versions only if they are the best matching maps.

    Parameters:
    - target_map: The target MapDataset object.
    - base_set: List of base MapDataset objects.

    Returns:
    - matches_stats_list: List of dictionaries with match statistics.
    - match_results: List of dictionaries with match results.
    """
    matches_stats_list = []
    match_results = []

    for base_map in tqdm(base_set, desc="Comparing with base maps", leave=False):

        # Perform SuperGlue matching
        valid_matches, superglue_matches_df = compute_matches(target_map, base_map, min_score=min_score)

        # Update the stats list with the best result
        matches_stats_list.append({
            'folder': base_map.map_info.folder,
            'number_of_superglue_matches': valid_matches
        })

        # Collect data for match results
        match_results.append({
            'base_map': base_map,  # Store the rotated map with the best match
            'valid_matches': valid_matches,
            'superglue_matches_df': superglue_matches_df
        })

        # Free up memory by deleting temporary variables
        del superglue_matches_df
        gc.collect()
    return matches_stats_list, match_results

def compute_matches(target_map: MapDataset, base_map: MapDataset, min_score=0.1):
    """
    Computes the SuperGlue matches between the target map and a base map.

    Parameters:
    - eval_map: The target MapDataset object.
    - base_map: The base MapDataset object.

    Returns:
    - valid_matches: Number of valid matches.
    - superglue_matches_df: DataFrame of matches.
    """
    # Use precomputed data to run SuperGlue matching
    matches, kp1, kp2 = run_superglue_matching(target_map, base_map, min_score=min_score)

    # Count the number of valid matches
    if 'matches0' in matches:
        valid_matches = (matches['matches0'][0] > -1).sum().item()
    else:
        valid_matches = 0

    # Convert matches to DataFrame
    superglue_matches_df = matches_to_dataframe(matches, kp1, kp2)

    return valid_matches, superglue_matches_df

def matches_to_dataframe(matches, kp1, kp2):
    """
    Converts matches and keypoints into a DataFrame.

    Parameters:
    - matches: Dictionary returned by SuperGlue matching.
    - kp1: Keypoints from the first image (target_map).
    - kp2: Keypoints from the second image (base_map).

    Returns:
    - df_matches: DataFrame with matched keypoints coordinates and scores.
    """

    # Extract data
    matches0 = matches['matches0'][0].cpu().numpy()  # Shape [N], values are indices into kp2 or -1
    matching_scores0 = matches['matching_scores0'][0].cpu().numpy()  # Shape [N], values are scores

    kp1_coords = kp1[0].cpu().numpy()  # Shape [N, 2]
    kp2_coords = kp2[0].cpu().numpy()  # Shape [M, 2]

    # Find valid matches
    valid_idx1 = np.where(matches0 > -1)[0]
    valid_idx2 = matches0[valid_idx1]

    matched_kp1 = kp1_coords[valid_idx1]
    matched_kp2 = kp2_coords[valid_idx2]
    matched_scores = matching_scores0[valid_idx1]

    # Create DataFrame
    df_matches = pd.DataFrame({
        'kp1_idx': valid_idx1,
        'kp2_idx': valid_idx2,
        'kp1_x': matched_kp1[:, 0],
        'kp1_y': matched_kp1[:, 1],
        'kp2_x': matched_kp2[:, 0],
        'kp2_y': matched_kp2[:, 1],
        'match_score': matched_scores
    })

    return df_matches



def filter_matches_with_delaunay(matches_to_filter, similarity_threshold=0.5, plot=False):
    """
    Filters matches based on Delaunay triangulation and similarity of triangles.

    Parameters:
    - matches_to_filter: DataFrame containing the match coordinates 'kp1_x', 'kp1_y', 'kp2_x', 'kp2_y'.
    - similarity_threshold: Threshold for deviation from similarity (default 0.2).
    - plot: If True, plots up to the first three triangles flagged for removal.

    Returns:
    - filtered_matches: DataFrame with filtered matches after removing outliers.
    """
    kp1_coords = matches_to_filter[['kp1_x', 'kp1_y']].values
    kp2_coords = matches_to_filter[['kp2_x', 'kp2_y']].values

    
    if len(kp1_coords) < 3:
        #print("Not enough points for Delaunay triangulation.")
        return matches_to_filter

    # Initial Delaunay triangulation on kp1
    initial_tri_kp1 = Delaunay(kp1_coords)

    points_to_remove = set()
    triangles_plotted = 0  # Counter to limit the number of triangles plotted

    for simplex in initial_tri_kp1.simplices:

        # Calculate side lengths for kp1 and kp2 triangles
        sides_kp1 = [np.linalg.norm(kp1_coords[simplex[i]] - kp1_coords[simplex[j]]) for i in range(3) for j in range(i + 1, 3)]
        sides_kp2 = [np.linalg.norm(kp2_coords[simplex[i]] - kp2_coords[simplex[j]]) for i in range(3) for j in range(i + 1, 3)]

        normalized_sides_kp1 = [s / np.mean(sides_kp1) for s in sides_kp1]
        normalized_sides_kp2 = [s / np.mean(sides_kp2) for s in sides_kp2]

        # Calculate the ratio for each corresponding side
        side_ratios = [s2 / s1 if s1 > 0 else 0 for s1, s2 in zip(normalized_sides_kp1, normalized_sides_kp2)]
        
        # Calculate d = max(ratios) - min(ratios)
        similarity_deviation = max(side_ratios) - min(side_ratios)
        
        # If deviation from similarity exceeds the threshold, mark the triangle as problematic
        if similarity_deviation > similarity_threshold:
            max_ratio_index = np.argmax(side_ratios)
            min_ratio_index = np.argmin(side_ratios)

            # Determine the vertex between the max and min ratio sides
            if (max_ratio_index, min_ratio_index) in [(0, 1), (1, 0)]:
                problematic_point = simplex[1]
            elif (max_ratio_index, min_ratio_index) in [(1, 2), (2, 1)]:
                problematic_point = simplex[2]
            else:  # (0, 2) or (2, 0)
                problematic_point = simplex[0]

            points_to_remove.add(problematic_point)

            # Plot up to the first three triangles flagged for removal
            if plot and triangles_plotted < 3:
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))

                # Labels for vertices
                labels = ['A', 'B', 'C']
                
                # Triangle from the initial triangulation on kp1
                triangle_kp1 = kp1_coords[simplex]
                axes[0].plot(triangle_kp1[:, 0], triangle_kp1[:, 1], 'o-', color='blue', label="Triangle Vertices (kp1)")
                axes[0].set_title("Triangle in Target Image (kp1)")
                
                # Annotate vertices with labels A, B, C
                for i, (x, y) in enumerate(triangle_kp1):
                    axes[0].text(x, y, labels[i], fontsize=12, ha='right', color='blue')
                
                # Annotate each side with its ratio and highlight max/min sides
                for i, side_ratio in enumerate(side_ratios):
                    pt1, pt2 = triangle_kp1[i], triangle_kp1[(i + 1) % 3]
                    mid_x, mid_y = (pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2
                    color = 'red' if i == max_ratio_index or i == min_ratio_index else 'black'
                    axes[0].plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=color, linestyle='--' if color == 'red' else '-', linewidth=1)
                    axes[0].text(mid_x, mid_y, f'{side_ratio:.2f}', color='red' if color == 'red' else 'black', fontsize=10, ha='center')

                # Mark the problematic point in kp1
                outlier_kp1 = triangle_kp1[simplex.tolist().index(problematic_point)]
                axes[0].plot(outlier_kp1[0], outlier_kp1[1], 'ro', label="Problematic Point")
                axes[0].legend()

                # Triangle from the corresponding points in kp2
                triangle_kp2 = kp2_coords[simplex]
                axes[1].plot(triangle_kp2[:, 0], triangle_kp2[:, 1], 'o-', color='green', label="Triangle Vertices (kp2)")
                axes[1].set_title("Corresponding Triangle in Base Image (kp2)")

                # Annotate vertices with labels A, B, C
                for i, (x, y) in enumerate(triangle_kp2):
                    axes[1].text(x, y, labels[i], fontsize=12, ha='right', color='green')

                # Annotate each side with its ratio and highlight max/min sides
                for i, side_ratio in enumerate(side_ratios):
                    pt1, pt2 = triangle_kp2[i], triangle_kp2[(i + 1) % 3]
                    mid_x, mid_y = (pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2
                    color = 'red' if i == max_ratio_index or i == min_ratio_index else 'black'
                    axes[1].plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=color, linestyle='--' if color == 'red' else '-', linewidth=1)
                    axes[1].text(mid_x, mid_y, f'{side_ratio:.2f}', color='red' if color == 'red' else 'black', fontsize=10, ha='center')

                plt.show()

                triangles_plotted += 1  # Increment the plot counter

    # Filter out the matches involving problematic points
    filtered_matches = matches_to_filter.drop(list(points_to_remove), errors="ignore").reset_index(drop=True)

    return filtered_matches



def find_overlapping_triangles(match_result_2, target_map, base_map):
    """
    finds the overlapping triangles in a match

    """

    # Extract keypoint coordinates (already in tensor space)
    kp1_coords = match_result_2[['kp1_x', 'kp1_y']].values.astype(np.float32)
    kp2_coords = match_result_2[['kp2_x', 'kp2_y']].values.astype(np.float32)
    transformation_matrix_target = target_map.tensor_to_image_transform[:2, :]
    transformation_matrix_base = base_map.tensor_to_image_transform[:2, :]
    kp1_coords = cv2.transform(kp1_coords.reshape(-1, 1, 2), transformation_matrix_target).reshape(-1, 2)
    kp2_coords = cv2.transform(kp2_coords.reshape(-1, 1, 2), transformation_matrix_base).reshape(-1, 2)

    tri_all = Delaunay(kp1_coords)
    # Apply same triangulation structure
    overlapping_triangles = []
    polygons = []
    for simplex in tri_all.simplices:
        triangle_points = kp2_coords[simplex]
        polygon = ShapelyPolygon(triangle_points)
        polygons.append(polygon)

    # Identify overlapping triangles
    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            if polygons[i].intersects(polygons[j]):
                intersection_area = polygons[i].intersection(polygons[j]).area
                if intersection_area > 0:
                    overlapping_triangles.extend([i, j])
    overlapping_triangles = list(set(overlapping_triangles))
    return overlapping_triangles

def find_overlapping_triangles_numpy(kp1_coords_transformed, kp2_coords_transformed, overlap_threshold=0.1):
    """
    Finds the overlapping triangles using pre-transformed keypoints.
    
    Parameters:
    - kp1_coords_transformed: NumPy array of shape (N, 2) with transformed kp1 coordinates.
    - kp2_coords_transformed: NumPy array of shape (N, 2) with transformed kp2 coordinates.
    - overlap_threshold: Minimum intersection area to consider as overlap.
    
    Returns:
    - overlapping_triangles: List of indices of overlapping triangles.
    """
    if len(kp1_coords_transformed) < 3:
        return []
    
    tri_all = Delaunay(kp1_coords_transformed)
    simplices = tri_all.simplices
    
    # Create Shapely polygons for kp2 triangles
    polygons = [ShapelyPolygon(kp2_coords_transformed[simplex]) for simplex in simplices]
    
    # Build spatial index
    spatial_index = STRtree(polygons)
    
    overlapping_triangles = set()
    for i, polygon in enumerate(polygons):
        # Query potential overlaps
        possible_overlap_indices = spatial_index.query(polygon)
        for j in possible_overlap_indices:
            if i >= j:
                continue  # Avoid duplicate checks and self-intersection
            other_polygon = polygons[j]
            if polygon.intersects(other_polygon):
                intersection_area = polygon.intersection(other_polygon).area
                if intersection_area > overlap_threshold:
                    overlapping_triangles.update([i, j])
    
    return list(overlapping_triangles)


def remove_remaining_overlaps(matches_df, target_map, base_map, overlap_threshold=0.1):
    """
    Removes remaining overlapping triangles by deleting points connected to the highest number of overlaps.
    
    Parameters:
    - matches_df: DataFrame containing match coordinates 'kp1_x', 'kp1_y', 'kp2_x', 'kp2_y'.
    - target_map: Object containing 'tensor_to_image_transform' for the target map.
    - base_map: Object containing 'tensor_to_image_transform' for the base map.
    - overlap_threshold: Minimum intersection area to consider as overlap.
    
    Returns:
    - filtered_matches_df: DataFrame with overlapping triangles removed.
    """
    filtered_matches_df = matches_df.copy().reset_index(drop=True)
    
    # Precompute transformed keypoints
    kp1_coords = filtered_matches_df[['kp1_x', 'kp1_y']].values.astype(np.float32)
    kp2_coords = filtered_matches_df[['kp2_x', 'kp2_y']].values.astype(np.float32)
    transformation_matrix_target = target_map.tensor_to_image_transform[:2, :]
    transformation_matrix_base = base_map.tensor_to_image_transform[:2, :]
    
    kp1_transformed = cv2.transform(kp1_coords.reshape(-1, 1, 2), transformation_matrix_target).reshape(-1, 2)
    kp2_transformed = cv2.transform(kp2_coords.reshape(-1, 1, 2), transformation_matrix_base).reshape(-1, 2)
    
    while len(filtered_matches_df) >= 3:
        # Find overlapping triangles using pre-transformed keypoints
        overlapping_triangles = find_overlapping_triangles_numpy(kp1_transformed, kp2_transformed, overlap_threshold)
        
        if not overlapping_triangles:
            break  # No overlaps detected
        
        # Count overlaps per point
        overlap_counts = Counter()
        tri_all = Delaunay(kp1_transformed)
        simplices = tri_all.simplices
        for simplex_idx in overlapping_triangles:
            simplex = simplices[simplex_idx]
            overlap_counts.update(simplex)
        
        if not overlap_counts:
            break
        
        # Identify the point with the highest overlap count
        most_overlapping_point = overlap_counts.most_common(1)[0][0]
        
        # Remove the corresponding match (row) from the DataFrame
        filtered_matches_df = filtered_matches_df.drop(most_overlapping_point).reset_index(drop=True)
        
        # Also remove the corresponding transformed keypoints
        kp1_transformed = np.delete(kp1_transformed, most_overlapping_point, axis=0)
        kp2_transformed = np.delete(kp2_transformed, most_overlapping_point, axis=0)
    

    return filtered_matches_df

def filter_eval_map_with_delaunay(eval_map, base_dataset, matches_to_filter = 'delaunay_filtered_matches' , similarity_threshold = 0.3, min_score_match = 0.1):
    """
    Filters matches based on Delaunay triangulation and similarity of triangles.
    we can choose to filter the matches from the initial matches, the ransac filtered matches, the enhanced matches or the enhanced ransac filtered matches.
    matches_to_filter can be 'superglue_matches', 'ransac_filtered_matches', 'enhanced_matches' or 'enhanced_ransac_filtered_matches'
    """
    
    for match_result in eval_map.best_matches_result:
        base_map = next((map_obj for map_obj in base_dataset if map_obj.map_info.folder == match_result.base_folder), None)
        
        # Extract matches from match_result.superglue_matches (DataFrame)
        if matches_to_filter == 'superglue_matches':
            matches_df = match_result.superglue_matches
        elif matches_to_filter == 'ransac_filtered_matches':
            matches_df = match_result.ransac_filtered_matches
        elif matches_to_filter == 'enhanced_matches':
            matches_df = match_result.enhanced_matches
        elif matches_to_filter == 'enhanced_ransac_filtered_matches':
            matches_df = match_result.enhanced_ransac_filtered_matches

        if matches_df is None or matches_df.empty or len(matches_df) < 4:
            continue

        number_of_overlapping_triangles = len(find_overlapping_triangles(matches_df, eval_map, base_map))
        if number_of_overlapping_triangles == 0:
            if matches_to_filter == 'superglue_matches' or matches_to_filter == 'ransac_filtered_matches':
                match_result.delaunay_filtered_matches = matches_df
            elif matches_to_filter == 'enhanced_matches'or matches_to_filter == 'enhanced_ransac_filtered_matches':
                match_result.enhanced_delaunay_filtered_matches = matches_df
        
        scores = matches_df['match_score'].values
        # Filter out matches with scores below the threshold
        valid_matches = scores > min_score_match
        matches_df = matches_df[valid_matches].reset_index(drop=True)
        
        # Initialize variables to track overlap reduction
        prev_number_of_overlapping_triangles = float('inf')
        decreasing = True
        iteration = 0
        
        while decreasing:
            # Filter matches using the Delaunay filtering function
            filtered_matches = filter_matches_with_delaunay(matches_df, similarity_threshold = similarity_threshold)
            
            # Count the number of overlapping triangles after filtering
            new_number_of_overlapping_triangles = len(find_overlapping_triangles(filtered_matches, eval_map, base_map))


            # Check if the number of overlapping triangles has diminished
            if new_number_of_overlapping_triangles < prev_number_of_overlapping_triangles:
                # Update matches_df and prev_number_of_overlapping_triangles for next iteration
                matches_df = filtered_matches
                prev_number_of_overlapping_triangles = new_number_of_overlapping_triangles
                iteration += 1
            else:
                # If the number stops diminishing, end the loop
                decreasing = False
        
        # only do this if theres more than 3 points
        if len(matches_df) > 3:
            number_of_overlapping_triangles = len(find_overlapping_triangles(matches_df, eval_map, base_map))
            final_filtered_matches = remove_remaining_overlaps(matches_df, eval_map, base_map)
            # Save the fully filtered matches
            if matches_to_filter == 'superglue_matches' or matches_to_filter == 'ransac_filtered_matches':
                match_result.delaunay_filtered_matches = final_filtered_matches
            elif matches_to_filter == 'enhanced_matches'or matches_to_filter == 'enhanced_ransac_filtered_matches': 
                match_result.enhanced_delaunay_filtered_matches = final_filtered_matches

    return eval_map


def filter_match_with_delaunay(eval_map, base_dataset, match_result, match_to_filter_df , similarity_threshold = 0.3, min_score_match = 0.1):
    """
    Filters matches based on Delaunay triangulation and similarity of triangles.
    we can choose to filter the matches from the initial matches, the ransac filtered matches, the enhanced matches or the enhanced ransac filtered matches.
    matches_to_filter can be 'superglue_matches', 'ransac_filtered_matches', 'enhanced_matches' or 'enhanced_ransac_filtered_matches'
    """
    
    base_map = next((map_obj for map_obj in base_dataset if map_obj.map_info.folder == match_result.base_folder), None)
    matches_df = match_to_filter_df

    if matches_df is None or matches_df.empty or len(matches_df) < 4:
        return matches_df

    number_of_overlapping_triangles = len(find_overlapping_triangles(matches_df, eval_map, base_map))
    if number_of_overlapping_triangles == 0:
        return matches_df

        
    scores = matches_df['match_score'].values
    # Filter out matches with scores below the threshold
    valid_matches = scores > min_score_match
    matches_df = matches_df[valid_matches].reset_index(drop=True)
    
    # Initialize variables to track overlap reduction
    prev_number_of_overlapping_triangles = float('inf')
    decreasing = True
    iteration = 0
        
    while decreasing:
        # Filter matches using the Delaunay filtering function
        filtered_matches = filter_matches_with_delaunay(matches_df, similarity_threshold = similarity_threshold)
        
        # Count the number of overlapping triangles after filtering
        if len(filtered_matches) < 4:
            return filtered_matches

        new_number_of_overlapping_triangles = len(find_overlapping_triangles(filtered_matches, eval_map, base_map))


        # Check if the number of overlapping triangles has diminished
        if new_number_of_overlapping_triangles < prev_number_of_overlapping_triangles:
            # Update matches_df and prev_number_of_overlapping_triangles for next iteration
            matches_df = filtered_matches
            prev_number_of_overlapping_triangles = new_number_of_overlapping_triangles
            iteration += 1
        else:
            # If the number stops diminishing, end the loop
            decreasing = False
    
    # only do this if theres more than 3 points
    if len(matches_df) > 3:
        number_of_overlapping_triangles = len(find_overlapping_triangles(matches_df, eval_map, base_map))
        final_filtered_matches = remove_remaining_overlaps(matches_df, eval_map, base_map)
    else:
        final_filtered_matches = matches_df

    return final_filtered_matches


def estimate_north_rotation(map_obj, best_match, min_match_score = 0.4, plot_transformation = False):
    best_map = best_match["base_map"]

    # Step 4: Ensure Both Maps Have SuperPoint Results
    if best_map.superpoint_results.keypoints is None or best_map.superpoint_results.descriptors is None:
        print(f"Best match {best_map.map_info.folder} does not have keypoints or descriptors.")
    # Step 6: Access Precomputed Matches from superglue_matches_df
    matches_df = best_match.get('superglue_matches_df', None)
    if matches_df is None or matches_df.empty:
        print(f"No matches found in superglue_matches_df for {best_map.map_info.folder}.")
        derived_north_rotation = 0.0  # Assign a default value or handle accordingly
    else:
        # Optional: Filter Matches Based on Match Score Threshold
        score_threshold = min_match_score  # Adjust based on your requirements
        filtered_matches_df = matches_df[matches_df['match_score'] >= score_threshold]
        num_matches = len(filtered_matches_df)
        #print(f"Number of matches after filtering: {num_matches}")
        
        if num_matches < 4:
            print(f"Not enough high-confidence matches between {map_obj.map_info.folder} and {best_map.map_info.folder}.")
            derived_north_rotation = 0.0  # Assign a default value or handle accordingly
        else:
            # Step 7: Extract Matched Keypoints
            src_pts = filtered_matches_df[['kp1_x', 'kp1_y']].values
            dst_pts = filtered_matches_df[['kp2_x', 'kp2_y']].values
            
            try:
                model_robust, inliers = ransac(
                    (src_pts, dst_pts),
                    AffineTransform,
                    min_samples=3,
                    residual_threshold=5,
                    max_trials=1000
                )
            except ValueError as e:
                print(f"RANSAC failed to find a valid affine transform: {e}")
                model_robust = None

            if model_robust is not None:
                # Step 9: Extract Rotation Angle from Affine Matrix
                # Access the transformation matrix via 'params' attribute
                theta_rad = np.arctan2(model_robust.params[1, 0], model_robust.params[0, 0])
                if map_obj.north_rotation_angle is not None:
                    derived_north_rotation = theta_rad + map_obj.north_rotation_angle
                else:
                    derived_north_rotation = theta_rad
                #print(f"Derived North Rotation (radians): {derived_north_rotation}")
            else:
                # Handle the Case Where Transformation Couldn't Be Estimated
                print(f"Transformation matrix could not be estimated between {map_obj.map_info.folder} and {best_map.map_info.folder}.")
                derived_north_rotation = 0.0  # Assign a default value

        map_obj.north_rotation_angle = derived_north_rotation

        if plot_transformation and model_robust is not None:
            # Step 1: Convert Rotation Angle from Radians to Degrees
            theta_deg = -np.degrees(derived_north_rotation)
            print(theta_deg)
            # Step 2: Get Image Dimensions
            image_height, image_width = map_obj.image.shape[:2]
            image_center = (image_width / 2, image_height / 2)

            # Step 3: Compute the Rotation Matrix
            rotation_matrix = cv2.getRotationMatrix2D(image_center, theta_deg, 1.0)  # Scale=1.0 for rigid rotation

            # Step 4: Calculate the sine and cosine of rotation angle
            #correct to account for possible negative values? like 260 degrees is problematic
            abs_cos = abs(rotation_matrix[0, 0])
            abs_sin = abs(rotation_matrix[0, 1])

            # Step 5: Compute the new bounding dimensions of the image
            new_width = int(image_height * abs_sin + image_width * abs_cos)
            new_height = int(image_height * abs_cos + image_width * abs_sin)

            # Step 6: Adjust the rotation matrix to account for translation
            rotation_matrix[0, 2] += new_width / 2 - image_center[0]
            rotation_matrix[1, 2] += new_height / 2 - image_center[1]

            # Step 7: Perform the rotation
            warped_image = cv2.warpAffine(map_obj.image, rotation_matrix, (new_width, new_height))

            # Step 8: Convert images from BGR to RGB for plotting
            if warped_image.ndim == 3:
                warped_image_rgb = cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)
            else:
                warped_image_rgb = warped_image.copy()

            if map_obj.image.ndim == 3:
                original_image_rgb = cv2.cvtColor(map_obj.image, cv2.COLOR_BGR2RGB)
            else:
                original_image_rgb = map_obj.image.copy()

            if best_map.image.ndim == 3:
                best_map_image_rgb = cv2.cvtColor(best_map.image, cv2.COLOR_BGR2RGB)
            else:
                best_map_image_rgb = best_map.image.copy()

            # Step 9: Plot Original, Warped, and Best Match Images Side by Side
            plt.figure(figsize=(20, 10))

            plt.subplot(1, 3, 1)
            plt.imshow(original_image_rgb)
            plt.title(f"{map_obj.map_info.folder} - Original")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(warped_image_rgb)
            plt.title(f"{map_obj.map_info.folder} - Rigidly Rotated to {best_map.map_info.folder}")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(best_map_image_rgb)
            plt.title(f"{best_map.map_info.folder} - Reference")
            plt.axis('off')

            plt.show()

        return derived_north_rotation
    

def find_north_from_best_match(target_map, base_dataset, min_match_score, plot=False):
    #check that we are adding the north rotation to the initial one todo
    rotation_angles = [np.pi / 2, np.pi, 3 * np.pi / 2]
    target_map.run_superpoint_pipeline()
    if plot:
        target_map.plot_superpoint_results()
    best_match = find_single_best_match(target_map, base_dataset, min_score=min_match_score)
    num_matches = len(best_match['superglue_matches_df'])
    if num_matches < 100:
        best_num_matches = num_matches
        best_angle_rad = None
        for angle_rad in rotation_angles:
            target_map.north_rotation_angle = angle_rad
            target_map.run_superpoint_pipeline()
            best_match_rotated = find_single_best_match(target_map, base_dataset, min_score=min_match_score)
            new_num_matches = len(best_match_rotated['superglue_matches_df'])
            if new_num_matches > best_num_matches and new_num_matches > 100:
                best_num_matches = new_num_matches
                best_angle_rad = angle_rad
                best_match = best_match_rotated
        if best_angle_rad is not None:
            target_map.north_rotation_angle = best_angle_rad
            #print(f"Imposed rotation of {np.degrees(best_angle_rad)} degrees improved the number of matches from {num_matches} to {best_num_matches}.")
        else:
            target_map.north_rotation_angle = 0.0
            #print(f"No imposed rotation improved the number of matches from {num_matches}.")
    else:
        #print(f"Number of matches: {num_matches}")
        rotation_angle = estimate_north_rotation(target_map, best_match, min_match_score=min_match_score, plot_transformation=False)
        target_map.north_rotation_angle = rotation_angle
    if target_map.north_rotation_angle is None:
        print(f"North rotation angle was not saved for {target_map.map_info.folder}.")
    target_map.run_superpoint_pipeline()
    if plot:
        target_map.plot_superpoint_results()




def remove_outliers_ransac(matches_df, ransac_threshold=0.1, ransac_max_trials=1000):
    """
    Removes outliers using RANSAC on the match coordinates.

    Parameters:
    - matches_df: DataFrame containing match coordinates 'kp1_x', 'kp1_y', 'kp2_x', 'kp2_y'.
    - ransac_threshold: Threshold for RANSAC in pixel units.
    - ransac_max_trials: Maximum number of RANSAC iterations.

    Returns:
    - filtered_matches_df: DataFrame with outliers removed.
    """
    
    # Extract keypoint coordinates
    kp1_coords = matches_df[['kp1_x', 'kp1_y']].values
    kp2_coords = matches_df[['kp2_x', 'kp2_y']].values

    # Perform RANSAC to remove outliers
    model, inliers = ransac((kp1_coords, kp2_coords), AffineTransform, min_samples=3, residual_threshold=ransac_threshold, max_trials=ransac_max_trials)

    # Filter matches based on inlier indices
    filtered_matches_df = matches_df.iloc[inliers].reset_index(drop=True)
    return filtered_matches_df
