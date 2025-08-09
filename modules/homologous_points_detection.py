from SuperGluePretrainedNetwork.models.superglue import SuperGlue
import torch
from typing import List
from modules.MapDataset import MapDataset
from modules.MatchingResult import MatchingResult
from scipy.spatial import Delaunay
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.strtree import STRtree
from collections import Counter
import gc  # Garbage collector for memory management
import cv2
from skimage.measure import ransac
from skimage.transform import AffineTransform
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from copy import deepcopy


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
        enhanced_delaunay_filtered_matches=None
    )


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


def filter_match_with_delaunay(
    eval_map,
    base_dataset,
    match_result,
    match_to_filter_df,
    similarity_threshold=0.3,
    min_score_match=0.1
):
    """
    Iteratively remove bad matches by:
      1) Building a Delaunay triangulation only on the first set of points (kp1_x, kp1_y).
      2) Imposing that same connectivity on the second set (kp2_x, kp2_y).
      3) Detecting any triangles in the second set that overlap each other.
      4) Checking side-length similarity between the first-set triangle and the second-set
         triangle (via filter_matches_with_delaunay).
      5) Removing offending points, repeating until stable.
      6) Finally calling remove_remaining_overlaps.

    Parameters
    ----------
    eval_map : object
        Map object for the "eval" side (unused except possibly inside remove_remaining_overlaps).
    base_dataset : list
        List of base map objects.
    match_result : object
        Contains match info, including match_result.base_folder to identify the base map.
    match_to_filter_df : pd.DataFrame
        Must have columns [kp1_x, kp1_y, kp2_x, kp2_y, match_score, ...].
    similarity_threshold : float
        Maximum allowed side-length ratio deviation. If a triangle in the second set
        differs too much from the first set’s shape, we remove it.
    min_score_match : float
        Minimum match score to keep before we even begin.

    Returns
    -------
    final_filtered_matches : pd.DataFrame
        The matches after iterative removal of overlapping or shape-mismatch triangles,
        plus a final pass of remove_remaining_overlaps.
    """

    # -------------------------------------------------------------------------
    # 1) Pre-filter by match_score
    # -------------------------------------------------------------------------
    base_map = next(
        (m for m in base_dataset if m.map_info.folder == match_result.base_folder),
        None
    )
    matches_df = match_to_filter_df
    if matches_df is None or matches_df.empty:
        return matches_df

    # Filter by min score
    matches_df = matches_df[matches_df["match_score"] > min_score_match].reset_index(drop=True)
    if len(matches_df) < 4:
        return matches_df  # Not enough points for Delaunay

    # -------------------------------------------------------------------------
    # 2) Iterative removal loop
    # -------------------------------------------------------------------------
    prev_num_removed = -1

    while True:
        if len(matches_df) < 4:
            break  # Not enough for Delaunay

        # 2a) Build a Delaunay triangulation on the *first* set (eval side)
        kp1_coords = matches_df[["kp1_x", "kp1_y"]].values
        tri_eval = Delaunay(kp1_coords)
        if len(tri_eval.simplices) < 1:
            break  # No triangles formed

        # 2b) Impose that connectivity on the second set (kp2_x, kp2_y),
        #     then find all pairs of overlapping triangles among themselves.
        kp2_coords = matches_df[["kp2_x", "kp2_y"]].values
        polygons_second = [
            ShapelyPolygon(kp2_coords[simplex]) for simplex in tri_eval.simplices
        ]
        if len(polygons_second) < 2:
            # No chance of overlap with <2 triangles
            break

        # Build STRtree for these "imposed" polygons
        spatial_index = STRtree(polygons_second)

        # Identify which triangles in the second set overlap each other
        overlapping_tri_indices = set()
        for i, poly_i in enumerate(polygons_second):
            if not poly_i.is_valid:
                continue
            # Query potential overlaps
            candidates = spatial_index.query(poly_i)
            for j in candidates:
                if j <= i:
                    continue
                poly_j = polygons_second[j]
                if poly_j.is_valid and poly_i.intersects(poly_j):
                    if poly_i.intersection(poly_j).area > 0:
                        overlapping_tri_indices.update([i, j])

        if not overlapping_tri_indices:
            # No overlaps among imposed triangles
            # We still want to do a side-length shape check below,
            # so we'll treat overlapping_tri_indices as empty but keep going.
            pass


        overlapping_tri_indices = sorted(list(overlapping_tri_indices))
        if not overlapping_tri_indices:
            # If we want to also do side-check for all triangles, we can do that.
            # For example:
            overlapping_tri_indices = list(range(len(tri_eval.simplices)))
            # This means we shape-check *all* triangles each iteration.

        # 2d) Gather the vertex indices from those triangles
        overlapping_vertices = set()
        for tri_idx in overlapping_tri_indices:
            if tri_idx < len(tri_eval.simplices):
                simplex = tri_eval.simplices[tri_idx]
                overlapping_vertices.update(simplex)

        # If there are no vertices to check, we are done
        if len(overlapping_vertices) < 3:
            break

        # 2e) Build a subset DataFrame with just these overlapping vertices
        overlapping_vertices = list(overlapping_vertices)
        subset_df = matches_df.iloc[overlapping_vertices].copy()
        # subset_df.index is not 0..N necessarily. Let's store the old index in a column:
        subset_df.reset_index(drop=False, inplace=True)
        # Now subset_df has columns ["index", "kp1_x", "kp1_y", "kp2_x", "kp2_y", ...]

        # Re-map each triangle from old indices -> local subset indices
        old_to_new = {old_i: new_i for new_i, old_i in enumerate(subset_df["index"])}
        remapped_simplices = []
        for tri_idx in overlapping_tri_indices:
            if tri_idx < len(tri_eval.simplices):
                simplex = tri_eval.simplices[tri_idx]
                new_simplex = []
                valid = True
                for pt_idx in simplex:
                    if pt_idx not in old_to_new:
                        valid = False
                        break
                    new_simplex.append(old_to_new[pt_idx])
                if valid:
                    remapped_simplices.append(new_simplex)

        # 2f) Apply the side-length similarity check
        #     (We compare each triangle in subset_df's kp1 vs kp2).
        filtered_subset_df = filter_matches_with_delaunay(
            matches_to_filter=subset_df,
            overlapping_simplices=remapped_simplices,
            similarity_threshold=similarity_threshold
        )

        # The removed points are those that no longer appear in filtered_subset_df
        removed_points = set(subset_df["index"]) - set(filtered_subset_df["index"])

        # If nothing was removed, we are stable
        if not removed_points:
            break

        # 2g) Drop them from matches_df
        matches_df = matches_df.drop(labels=removed_points, axis="index", errors="ignore")
        matches_df.reset_index(drop=True, inplace=True)

    # -------------------------------------------------------------------------
    # 3) Final pass: remove any lingering overlaps, if desired
    # -------------------------------------------------------------------------
    if len(matches_df) >= 4:
        final_filtered_matches = remove_remaining_overlaps(matches_df, eval_map, base_map)
    else:
        final_filtered_matches = matches_df

    return final_filtered_matches

def filter_matches_with_delaunay(matches_to_filter, overlapping_simplices, similarity_threshold=0.5, plot=False):
    """
    Given a subset of matches (matches_to_filter) and a list of overlapping triangles
    (overlapping_simplices), check if they pass the side-length similarity threshold.
    Triangles that fail are marked for removal.

    Parameters:
    - matches_to_filter: DataFrame with columns kp1_x, kp1_y, kp2_x, kp2_y, and an Index 0..N
    - overlapping_simplices: list of length-K, each is [i0, i1, i2] with i0,i1,i2 in [0..len(matches_to_filter)-1]
    - similarity_threshold: e.g. 0.5 or 1. The maximum allowed (max_ratio - min_ratio).
    - plot: if True, you could visualize outliers for debugging

    Returns:
    - filtered_matches: DataFrame after removing points that appear in "bad" triangles.
    """

    kp1_coords = matches_to_filter[["kp1_x", "kp1_y"]].values
    kp2_coords = matches_to_filter[["kp2_x", "kp2_y"]].values

    if len(matches_to_filter) < 4 or len(overlapping_simplices) == 0:
        # Nothing to filter
        return matches_to_filter

    # --- Compute side-lengths in the subset of triangles ---
    # For each simplex [a,b,c] in overlapping_simplices, we measure the sides
    # in the first image (kp1_coords) and the second (kp2_coords).
    #
    # Each triangle has 3 edges, so we measure 3 side lengths. We'll get shape Nx3.
    try:
        kp1_sides = np.array([
            [
                np.linalg.norm(kp1_coords[simplex[i]] - kp1_coords[simplex[j]])
                for i in range(3)
                for j in range(i+1, 3)
            ]
            for simplex in overlapping_simplices
        ])
        kp2_sides = np.array([
            [
                np.linalg.norm(kp2_coords[simplex[i]] - kp2_coords[simplex[j]])
                for i in range(3)
                for j in range(i+1, 3)
            ]
            for simplex in overlapping_simplices
        ])
    except IndexError:
        # Means one of the simplex indices was out-of-range.
        # Usually you avoid this by re-mapping in the caller function.
        return matches_to_filter

    # --- Normalize side lengths to compare shape similarity ---
    kp1_mean = kp1_sides.mean(axis=1, keepdims=True)
    kp2_mean = kp2_sides.mean(axis=1, keepdims=True)

    # Avoid division by zero for degenerate triangles
    kp1_sides_norm = np.divide(kp1_sides, kp1_mean, out=np.zeros_like(kp1_sides), where=(kp1_mean!=0))
    kp2_sides_norm = np.divide(kp2_sides, kp2_mean, out=np.zeros_like(kp2_sides), where=(kp2_mean!=0))

    # Ratio of side lengths
    side_ratios = np.divide(kp2_sides_norm, kp1_sides_norm, out=np.ones_like(kp2_sides_norm), where=(kp1_sides_norm!=0))

    max_ratios = side_ratios.max(axis=1)
    min_ratios = side_ratios.min(axis=1)
    similarity_deviation = max_ratios - min_ratios

    # Identify triangles that fail the similarity threshold
    bad_tri_indices = np.where(similarity_deviation > similarity_threshold)[0]
    if len(bad_tri_indices) == 0:
        # All good => no removal
        return matches_to_filter

    # Collect all the points in "bad" triangles
    bad_triangles = [overlapping_simplices[i] for i in bad_tri_indices]
    points_to_remove = set()
    for tri in bad_triangles:
        points_to_remove.update(tri)

    # Filter them out
    keep_mask = ~matches_to_filter.index.isin(points_to_remove)
    filtered_matches = matches_to_filter[keep_mask].reset_index(drop=True)

    return filtered_matches

def plot_triangle(kp1_coords, kp2_coords, simplex, similarity_deviation):
    """
    Helper function to plot a problematic triangle.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    labels = ['A', 'B', 'C']

    # Plot triangle in kp1
    triangle_kp1 = kp1_coords[simplex]
    axes[0].plot(triangle_kp1[:, 0], triangle_kp1[:, 1], 'o-', color='blue')
    axes[0].set_title(f"Triangle in kp1 (Deviation: {similarity_deviation:.2f})")
    for i, (x, y) in enumerate(triangle_kp1):
        axes[0].text(x, y, labels[i], fontsize=12, ha='right', color='blue')

    # Plot triangle in kp2
    triangle_kp2 = kp2_coords[simplex]
    axes[1].plot(triangle_kp2[:, 0], triangle_kp2[:, 1], 'o-', color='green')
    axes[1].set_title("Corresponding Triangle in kp2")
    for i, (x, y) in enumerate(triangle_kp2):
        axes[1].text(x, y, labels[i], fontsize=12, ha='right', color='green')

    plt.show()


def find_overlapping_triangles(matches_df):
    """
    Perform a Delaunay triangulation on the first set of points (kp1_x, kp1_y),
    then impose that same connectivity on the second set (kp2_x, kp2_y).
    Detect any triangles in the second set that overlap each other.

    Returns a list of indices (triangle IDs) that are involved in overlap.
    """
    kp1 = matches_df[["kp1_x","kp1_y"]].values
    kp2 = matches_df[["kp2_x","kp2_y"]].values

    if len(kp1) < 4:
        # Not enough for a Delaunay
        return []

    # 1) Delaunay on the first set
    tri = Delaunay(kp1)  
    # tri.simplices is shape (N, 3) => each row = [i0, i1, i2]

    if len(tri.simplices) < 1:
        return []

    # 2) Build polygons in the second set using the same connectivity
    polygons_imposed = [
        ShapelyPolygon(kp2[simplex]) for simplex in tri.simplices
    ]

    if len(polygons_imposed) < 2:
        # You can't overlap with fewer than 2 triangles
        return []

    # 3) Put these polygons in an STRtree to find overlapping pairs efficiently
    spatial_index = STRtree(polygons_imposed)

    # 4) Check each polygon for overlap with the others
    overlapping_indices = set()

    for i, poly_i in enumerate(polygons_imposed):
        if not poly_i.is_valid:
            continue
        # Query the tree: returns indices of polygons that might overlap
        candidate_indices = spatial_index.query(poly_i)

        for j in candidate_indices:
            if j <= i:
                continue  # Avoid double-counting or comparing the same poly

            poly_j = polygons_imposed[j]
            if not poly_j.is_valid:
                continue

            # Check actual intersection area
            if poly_i.intersects(poly_j):
                if poly_i.intersection(poly_j).area > 0:
                    # Mark them
                    overlapping_indices.add(i)
                    overlapping_indices.add(j)

    # Return the triangle IDs from tri.simplices that overlap
    return list(overlapping_indices)

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

def estimate_north_rotation(map_obj, best_match, min_match_score=0.4, plot_transformation=False):
    """
    Estimates the north rotation of the target map by leveraging the best matched anchor map.
    The estimated rotation accounts for:
    - The relative transformation estimated via RANSAC.
    - The absolute north orientation of the best match (anchor map).
    - The macro rotation that was previously applied to the target map.

    Parameters:
        - map_obj: The target map object.
        - best_match: The best match dictionary containing the anchor map and matches.
        - min_match_score: Minimum match score threshold for filtering.
        - plot_transformation: Whether to visualize the transformation.

    Returns:
        - derived_north_rotation: The estimated absolute north rotation of the target map.
    """
    
    best_map = best_match["base_map"]
    
    if best_map.superpoint_results.keypoints is None or best_map.superpoint_results.descriptors is None:
        print(f"Best match {best_map.map_info.folder} does not have keypoints or descriptors.")
        return 0.0

    matches_df = best_match.get('superglue_matches_df', None)
    if matches_df is None or matches_df.empty:
        print(f"No matches found in superglue_matches_df for {best_map.map_info.folder}.")
        return 0.0  # Default if no valid matches

    # Filter matches based on score
    filtered_matches_df = matches_df[matches_df['match_score'] >= min_match_score]
    if len(filtered_matches_df) < 4:
        print(f"Not enough high-confidence matches between {map_obj.map_info.folder} and {best_map.map_info.folder}.")
        return 0.0  # Default if not enough points
    # Extract matched keypoints from the *tensor* space
    src_pts_tensor = filtered_matches_df[['kp1_x', 'kp1_y']].values  # Keypoints from target map (tensor space)
    dst_pts_tensor = filtered_matches_df[['kp2_x', 'kp2_y']].values  # Keypoints from anchor map (tensor space)

    # Convert them to homogeneous coords for matrix multiplication
    src_pts_h = np.column_stack([src_pts_tensor, np.ones(len(src_pts_tensor))])
    dst_pts_h = np.column_stack([dst_pts_tensor, np.ones(len(dst_pts_tensor))])

    # Invert to original image space
    src_pts_orig = []
    dst_pts_orig = []

    for i in range(len(src_pts_h)):
        # Target map
        src_pt_img = map_obj.tensor_to_image_transform @ src_pts_h[i]
        src_pt_img /= src_pt_img[2]
        src_pts_orig.append(src_pt_img[:2])

        # Anchor map
        dst_pt_img = best_map.tensor_to_image_transform @ dst_pts_h[i]
        dst_pt_img /= dst_pt_img[2]
        dst_pts_orig.append(dst_pt_img[:2])

    src_pts_orig = np.array(src_pts_orig)
    dst_pts_orig = np.array(dst_pts_orig)

    # Now run RANSAC on these "original space" points
    try:
        model_robust, inliers = ransac(
            (src_pts_orig, dst_pts_orig),
            AffineTransform,
            min_samples=3,
            residual_threshold=5,
            max_trials=1000
        )
    except ValueError as e:
        print(f"RANSAC failed to find a valid affine transform: {e}")
        return 0.0

    if model_robust is not None:
        # Extract the relative rotation component from the affine transformation
        theta_rad = -np.arctan2(model_robust.params[1, 0], model_robust.params[0, 0])

        # Consider the absolute north rotation of the anchor map
        best_map_rotation = best_map.north_rotation_angle if best_map.north_rotation_angle is not None else 0.0

        # Consider the macro rotation already applied to the target map
        previous_target_rotation = map_obj.north_rotation_angle if map_obj.north_rotation_angle is not None else 0.0

        # Compute the final absolute north rotation of the target map
        derived_north_rotation =  (best_map_rotation or 0.0)-(theta_rad or 0.0) #+ (previous_target_rotation or 0.0)
        if derived_north_rotation is None:
            print(f"Warning: derived_north_rotation is None for {map_obj.folder}. Defaulting to 0.0.")
            derived_north_rotation = 0.0

    else:
        print(f"Transformation matrix could not be estimated between {map_obj.folder} and {best_map.map_info.folder}.")
        return 0.0

    # Assign the computed rotation to the target map
    map_obj.north_rotation_angle = derived_north_rotation

    # Optional: Visualization of rotation
    if plot_transformation:
        # Convert to degrees (ensure correct sign)
        theta_deg = np.degrees(derived_north_rotation)
        
        print(f"In degrees, Macro rotation: {np.degrees(previous_target_rotation)}, "
            f"Best map rotation: {np.degrees(best_map.north_rotation_angle)}, STORED target ROTATION: {np.degrees(map_obj.north_rotation_angle)} "
            f"Relative rotation: {np.degrees(theta_rad)}, Total rotation: {np.degrees(derived_north_rotation)}")

        image_height, image_width = map_obj.image.shape[:2]
        image_center = (image_width / 2, image_height / 2)

        # Apply rotation to the target map
        rotation_matrix = cv2.getRotationMatrix2D(image_center, -theta_deg, 1.0)  # Negative for correct OpenCV convention
        warped_image = cv2.warpAffine(map_obj.image, rotation_matrix, (image_width, image_height))

        # Function to plot an arrow for north direction
        def plot_north_arrow(ax, center, rotation, length_factor=0.5, color="red"):
            """
            Plots an arrow indicating north direction.

            Parameters:
                - ax: Matplotlib axis to plot on.
                - center: (x, y) coordinates for arrow origin.
                - rotation: Angle in radians.
                - length_factor: Scale factor for arrow length.
                - color: Arrow color.
            """
            x, y = center
            length = max(image_width, image_height) * length_factor  # Scale arrow size relative to image
            dx = length * np.sin(-rotation)  # Negative to align with image coordinates
            dy = -length * np.cos(-rotation)  

            ax.arrow(x, y, dx, dy, head_width=length * 0.05, head_length=length * 0.1,
                    fc=color, ec=color, linewidth=3)
            ax.text(x + dx * 1.2, y + dy * 1.2, "N", fontsize=14, color=color, weight='bold')

        # Plot results
        fig, axes = plt.subplots(1, 3, figsize=(20, 10))

        # Original target map
        axes[0].imshow(map_obj.image, cmap='gray')
        axes[0].set_title(f"{map_obj.folder} - Original")
        axes[0].axis('off')
        #plot_north_arrow(axes[0], image_center, derived_north_rotation, length_factor=0.3, color="blue")  # Initial north

        # Rotated target map
        axes[1].imshow(warped_image, cmap='gray')
        axes[1].set_title(f"{map_obj.folder} - Rotated to Align with North")
        axes[1].axis('off')
        plot_north_arrow(axes[1], image_center, 0.0, length_factor=0.3, color="red")  # Should be aligned north

        # Anchor map
        #tensor image 
        tensor = best_map.tensor
        # get image from tensor considering that the shape is 1,1,480,640
        image_tensor = tensor.squeeze().detach().cpu().numpy()

        axes[2].imshow(image_tensor, cmap='gray')
        axes[2].set_title(f"{best_map.map_info.folder} - Reference Anchor Map")
        axes[2].axis('off')
        plt.show()
    
    return derived_north_rotation

    


def find_north_from_best_match(target_map, base_dataset, min_match_score, plot=False):
    
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
    if matches_df is None:
        #print("No matches to filter.")
        return None
    
    kp1_coords = matches_df[['kp1_x', 'kp1_y']].values
    kp2_coords = matches_df[['kp2_x', 'kp2_y']].values
    
    if kp1_coords.shape[0] < 3 or kp2_coords.shape[0] < 3 or len(matches_df) < 3:
        #print("Not enough points for RANSAC.")
        return None
    
    # Perform RANSAC to remove outliers
    model, inliers = ransac((kp1_coords, kp2_coords), AffineTransform, min_samples=3, residual_threshold=ransac_threshold, max_trials=ransac_max_trials)

    # Filter matches based on inlier indices
    filtered_matches_df = matches_df.iloc[inliers].reset_index(drop=True)
    return filtered_matches_df


def process_target_map_with_rotation(map_obj, base_dataset, min_match_score=0.3, match_threshold=100, plot_transformation=False):
    """
    Processes a single target map object, running SuperPoint and handling rotations
    to improve matches and estimate the north rotation angle.

    Parameters:
    - map_obj: The target map object to process.
    - base_dataset: The dataset of base maps to match against.
    - min_match_score: Minimum match score for estimating north rotation (default: 0.3).
    - match_threshold: Minimum number of matches to skip rotation (default: 100).
    - plot_transformation: Whether to plot the transformation during north rotation estimation.

    Returns:
    - map_obj: The processed map object with updated `north_rotation_angle`.
    """
    rotation_angles = [np.pi / 2, np.pi, 3 * np.pi / 2]
    rotation_degrees = [90, 180, 270]

    # Run SuperPoint pipeline on the target map
    map_obj.run_superpoint_pipeline()
    best_match = find_single_best_match(map_obj, base_dataset)
    num_matches = len(best_match['superglue_matches_df'])

    if num_matches < match_threshold:
        best_num_matches = num_matches
        best_angle_rad = None

        # Attempt rotations to improve matches
        for angle_rad, angle_deg in zip(rotation_angles, rotation_degrees):
            map_obj.north_rotation_angle = angle_rad
            map_obj.run_superpoint_pipeline()

            best_match_rotated = find_single_best_match(map_obj, base_dataset)
            new_num_matches = len(best_match_rotated['superglue_matches_df'])

            if new_num_matches > best_num_matches:
                best_num_matches = new_num_matches
                best_angle_rad = angle_rad
                best_match = best_match_rotated

        if best_angle_rad is not None:
            map_obj.north_rotation_angle = best_angle_rad
            map_obj.run_superpoint_pipeline()
            best_match = best_match_rotated

        # Estimate north rotation based on best match
        derived_north_rotation = estimate_north_rotation(
            map_obj, best_match, min_match_score=min_match_score, plot_transformation=plot_transformation
        )
    else:
        # Directly estimate north rotation without imposed rotations
        derived_north_rotation = estimate_north_rotation(
            map_obj, best_match, min_match_score=min_match_score, plot_transformation=plot_transformation
        )

    # Update the map's north rotation angle and re-run the pipeline
    map_obj.north_rotation_angle = derived_north_rotation
    map_obj.run_superpoint_pipeline()

    return map_obj




###################






