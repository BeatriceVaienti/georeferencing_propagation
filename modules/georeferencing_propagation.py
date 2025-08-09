import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import RBFInterpolator
from pyproj import Transformer
from IPython.display import display
import matplotlib.pyplot as plt
import cv2
import pandas as pd

def meters_to_tensors(match_result,  base_dataset, distance_in_meters=100):
    """
    Convers a value in meters to the tensor space of the target and base maps.

    Parameters:
    - match_result: The MatchingResult object containing the filtered match result.
    - target_map: The MapDataset object for the target map.
    - base_dataset: List of MapDataset objects for base maps.
    - radius: The radius of the circle in meters.

    Returns:
    - distance_target_tensor: The radius of the search region in the tensor space of the target map. (kp1)
    - distance_base_tensor: The radius of the search region in the tensor space of the base map. (kp2)
    """
    base_map_folder = match_result.base_folder
    base_map = next((map_obj for map_obj in base_dataset if map_obj.map_info.folder == base_map_folder), None)

    gcps = base_map.gcps
    world_coords = gcps[['mapX', 'mapY']].values
    image_coords = gcps[['sourceX', 'sourceY']].values
    
    # Calculate the scaling factor for the radius to go from meters to image pixels
    # using a median
    num_points = len(world_coords) -1 
    distances = np.linalg.norm(world_coords[:num_points] - world_coords[1:num_points + 1], axis=1)
    scaling_factor = np.median(distances) / np.median(np.linalg.norm(image_coords[:num_points] - image_coords[1:num_points + 1], axis=1))
    distance_base_image = distance_in_meters / scaling_factor
    transform_tensor_to_image_base = base_map.tensor_to_image_transform
    # inverse the transform to get the image to tensor, the matrix is not square so we need to use the pseudo inverse

    distance_base_tensor = distance_base_image / transform_tensor_to_image_base[0, 0].item()

    # Extract inlier keypoints directly in tensor coordinates
    kp1_inliers = match_result.delaunay_filtered_matches[['kp1_x', 'kp1_y']].values # eval
    kp2_inliers = match_result.delaunay_filtered_matches[['kp2_x', 'kp2_y']].values # base

    # Ensure we don't exceed the number of available inliers
    num_inliers = len(kp1_inliers) - 1
    num_points = min(num_points, num_inliers)
    
    # Calculate scaling factor for the radius on the second image
    distances_kp1 = np.linalg.norm(kp1_inliers[:num_points] - kp1_inliers[1:num_points + 1], axis=1) # eval
    distances_kp2 = np.linalg.norm(kp2_inliers[:num_points] - kp2_inliers[1:num_points + 1], axis=1) #base
    scaling_factor = np.median(distances_kp1 / distances_kp2) if len(distances_kp2) > 0 else 1 # scaling factor from base to eval

    distance_target_tensor = distance_base_tensor * scaling_factor

    return distance_target_tensor, distance_base_tensor

def get_base_map(match_result, base_dataset):
    """
    Retrieves the base map corresponding to the match_result from the base_dataset.
    """
    return next((map_obj for map_obj in base_dataset if map_obj.map_info.folder == match_result.base_folder), None)

def load_base_gcps(base_map):
    """
    Loads and prepares GCP coordinates from the base map.
    """
    base_gcps = base_map.gcps
    if base_gcps is None or base_gcps.empty:
        return None
    gcp_pixel_coords_base = base_gcps[['sourceX', 'sourceY']].values
    gcp_pixel_coords_base[:, 1] *= -1  # Adjust Y-coordinates to match image system
    return gcp_pixel_coords_base, base_gcps

def get_transformation_matrices(target_map, base_map):
    """
    Retrieves the transformation matrices for target and base maps.
    """
    return (
        target_map.tensor_to_image_transform,
        base_map.tensor_to_image_transform,
        base_map.image_to_tensor_transform
    )

def get_keypoints(df_points_to_use):
    """
    Extracts keypoints from the DataFrame.
    """
    kp_target = df_points_to_use[['kp1_x', 'kp1_y']].values
    kp_base = df_points_to_use[['kp2_x', 'kp2_y']].values
    return kp_target, kp_base


def transform_gcps_to_tensor_space(gcp_pixel_coords_base, inverse_transformation_matrix_base):
    """
    Transforms GCP coordinates to tensor space using the inverse transformation matrix.
    """
    return cv2.perspectiveTransform(
        gcp_pixel_coords_base.reshape(-1, 1, 2), inverse_transformation_matrix_base
    ).reshape(-1, 2)


def compute_threshold_tensor_base(match_result, base_dataset, distance_threshold):
    """
    Computes the distance threshold in tensor units.
    """
    _, threshold_tensor_base = meters_to_tensors(match_result, base_dataset, distance_threshold)
    return threshold_tensor_base

def find_keypoints_close_to_gcps(kp_base, gcp_tensor_coords_base, threshold_tensor_base):
    """
    Finds keypoints in the base map that are within a certain distance to the GCPs.
    """
    tree_base = KDTree(kp_base)
    distances, indices = tree_base.query(gcp_tensor_coords_base, k=len(kp_base))
    filtered_indices = [ind[dist <= threshold_tensor_base] for dist, ind in zip(distances, indices)]
    unique_indices = set(np.concatenate(filtered_indices))
    return unique_indices


def create_rbf_interpolator(base_gcps, rbf_smoothing):
    """
    Creates an RBF interpolator using the base GCPs.
    """
    base_gcps_image = base_gcps[['sourceX', 'sourceY']].values
    base_gcps_world = base_gcps[['mapX', 'mapY']].values
    return RBFInterpolator(
        base_gcps_image, base_gcps_world, kernel='thin_plate_spline', smoothing=rbf_smoothing
    )


def compute_guessed_gcp(
    idx, kp_base, kp_target, transformation_matrix_base, transformation_matrix_target, rbf_interpolator
):
    """
    Computes the guessed GCP for a given index.
    """
    if idx >= len(kp_base) or idx >= len(kp_target):
        return None

    kp_base_tensor = kp_base[idx]
    kp_target_tensor = kp_target[idx]

    # Transform base keypoint to image space
    kp_base_image = cv2.perspectiveTransform(
        kp_base_tensor.reshape(-1, 1, 2), transformation_matrix_base
    ).reshape(-1, 2)
    kp_base_image[:, 1] *= -1  # Adjust Y-coordinate

    # Interpolate real-world coordinates
    mapX, mapY = rbf_interpolator(kp_base_image).flatten()

    # Transform target keypoint to image space
    kp_target_image = cv2.perspectiveTransform(
        kp_target_tensor.reshape(-1, 1, 2), transformation_matrix_target
    ).reshape(-1, 2)
    sourceX = kp_target_image[0, 0]
    sourceY = kp_target_image[0, 1]

    # Store guessed GCP
    return {
        'mapX': mapX,
        'mapY': mapY,
        'sourceX': sourceX,
        'sourceY': -sourceY
    }


def propagate_georeferencing(
    target_map, match_result, df_points_to_use, base_dataset, distance_threshold=20, rbf_smoothing=0.1):
    """
    Propagates georeferencing from the base map to the target map using matching keypoints.
    """
    # Retrieve base map
    base_map = get_base_map(match_result, base_dataset)
    if base_map is None:
        return pd.DataFrame()

    # Load and prepare base GCPs
    result = load_base_gcps(base_map)
    if result is None:
        return pd.DataFrame()
    gcp_pixel_coords_base, base_gcps = result

    # Get transformation matrices
    (
        transformation_matrix_target,
        transformation_matrix_base,
        inverse_transformation_matrix_base
    ) = get_transformation_matrices(target_map, base_map)

    # Get keypoints from matches
    kp_target, kp_base = get_keypoints(df_points_to_use)

    # Transform GCPs to tensor space
    gcp_tensor_coords_base = transform_gcps_to_tensor_space(
        gcp_pixel_coords_base, inverse_transformation_matrix_base
    )

    # Compute threshold in tensor units
    threshold_tensor_base = compute_threshold_tensor_base(
        match_result, base_dataset, distance_threshold
    )

    # Find keypoints close to GCPs
    unique_indices = find_keypoints_close_to_gcps(
        kp_base, gcp_tensor_coords_base, threshold_tensor_base
    )

    # Create RBF interpolator
    rbf_interpolator = create_rbf_interpolator(base_gcps, rbf_smoothing)

    # Compute guessed GCPs
    guessed_gcps = []
    for idx in unique_indices:
        guessed_gcp = compute_guessed_gcp(
            idx, kp_base, kp_target, transformation_matrix_base,
            transformation_matrix_target, rbf_interpolator
        )
        if guessed_gcp is not None:
            guessed_gcps.append(guessed_gcp)

    return pd.DataFrame(guessed_gcps)



def calculate_rmse_with_georeferencing(
    propagated_gcps,
    manual_gcps,
    smoothing=1e-5,
    plot=True,
    consider_closest_gcps=False,
    epsg_code="EPSG:28193",
    distance_threshold=20
):
    """
    Calculates the RMSE between the propagated and manual GCPs using georeferencing transformation.
    Optionally plots the points on a folium map with lines connecting manual GCPs and predicted locations.
    Only calculates the RMSE for manual GCPs that are within a specified distance threshold of the transformed predicted GCPs.

    Parameters:
    - propagated_gcps: DataFrame with propagated GCPs (must contain 'sourceX', 'sourceY', 'mapX', 'mapY').
    - manual_gcps: DataFrame with manual GCPs (must contain 'sourceX', 'sourceY', 'mapX', 'mapY').
    - smoothing: Smoothing factor for the RBF interpolator.
    - plot: Boolean flag to enable plotting on a folium map.
    - consider_closest_gcps: Boolean flag to consider only closest GCPs within the distance threshold.
    - epsg_code: EPSG code for coordinate transformation (default is "EPSG:28193").
    - distance_threshold: Distance threshold in meters for selecting manual GCPs close to transformed GCPs.

    Returns:
    - RMSE value (float) representing the root mean square error between the filtered transformed points.
    """

    if len(propagated_gcps) < 3:
        print("Insufficient propagated GCPs for transformation. Returning infinite RMSE.")
        return float('inf')

    propagated_coords_image = propagated_gcps[['sourceX', 'sourceY']].values
    propagated_coords_world = propagated_gcps[['mapX', 'mapY']].values

    # Create an RBF interpolator for transforming image coordinates to world coordinates
    rbf_interpolator = RBFInterpolator(
        propagated_coords_image,
        propagated_coords_world,
        kernel='thin_plate_spline',
        smoothing=smoothing
    )

    # Transform the source coordinates of the manual GCPs to world coordinates
    manual_coords_image = manual_gcps[['sourceX', 'sourceY']].values
    transformed_coords_world = rbf_interpolator(manual_coords_image)

    # Calculate distances between transformed coordinates and manual coordinates
    manual_coords_world = manual_gcps[['mapX', 'mapY']].values
    distances = np.linalg.norm(transformed_coords_world - manual_coords_world, axis=1)

    if consider_closest_gcps:
        # Filter manual GCPs and corresponding transformed points based on the distance threshold
        close_indices = np.where(distances <= distance_threshold)[0]
        if len(close_indices) == 0:
            print("No manual GCPs found within the distance threshold. Returning infinite RMSE.")
            return float('inf')
    else:
        close_indices = np.arange(len(distances))

    filtered_transformed_coords = transformed_coords_world[close_indices]
    filtered_manual_coords = manual_coords_world[close_indices]

    # Calculate RMSE between the filtered transformed coordinates and manual coordinates using mean
    error = np.linalg.norm(filtered_transformed_coords - filtered_manual_coords, axis=1)
    rmse = np.sqrt(np.mean(error**2))

    # Optional plotting on folium map
    if plot:
        # Convert coordinates from the given EPSG code to EPSG 4326 for visualization
        transformer = Transformer.from_crs(epsg_code, "EPSG:4326", always_xy=True)

        # Convert propagated GCPs and manual GCPs for plotting
        propagated_coords_converted = np.array([transformer.transform(x, y) for x, y in propagated_coords_world])
        manual_coords_converted = np.array([transformer.transform(x, y) for x, y in manual_coords_world])
        transformed_coords_converted = np.array([transformer.transform(x, y) for x, y in transformed_coords_world])
        filtered_transformed_coords_converted = np.array([transformer.transform(x, y) for x, y in filtered_transformed_coords])
        filtered_manual_coords_converted = np.array([transformer.transform(x, y) for x, y in filtered_manual_coords])

        folium_map = folium.Map(
            location=[manual_coords_converted[:, 1].mean(), manual_coords_converted[:, 0].mean()],
            zoom_start=12
        )

        # Plot manual GCPs
        for lon, lat in manual_coords_converted:
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color='green',
                fill=True,
                fill_opacity=0.6,
                tooltip='Manual GCP'
            ).add_to(folium_map)

        # Plot transformed predicted locations from manual GCPs
        for lon, lat in transformed_coords_converted:
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color='red',
                fill=True,
                fill_opacity=0.6,
                tooltip='Predicted GCP'
            ).add_to(folium_map)

        # Highlight filtered matches
        for lon, lat in filtered_transformed_coords_converted:
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color='purple',
                fill=True,
                fill_opacity=0.8,
                tooltip='Filtered Predicted GCP'
            ).add_to(folium_map)

        for lon, lat in filtered_manual_coords_converted:
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color='orange',
                fill=True,
                fill_opacity=0.8,
                tooltip='Filtered Manual GCP'
            ).add_to(folium_map)
        
        for lon, lat in propagated_coords_converted:
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color='blue',
                fill=True,
                fill_opacity=0.6,
                tooltip='Propagated GCP'
            ).add_to(folium_map)

        # Connect each filtered manual GCP to its filtered predicted location
        for manual, predicted in zip(filtered_manual_coords_converted, filtered_transformed_coords_converted):
            folium.PolyLine(
                locations=[(manual[1], manual[0]), (predicted[1], predicted[0])],
                color="black",
                weight=1,
                opacity=0.7
            ).add_to(folium_map)

        # Display the map
        display(folium_map)

    return rmse


def plot_match_with_map_and_image(match, eval_map, base_dataset, plot_map=True, plot_image=True, epsg_code="EPSG:28193"):
    """
    Plots the georeferenced points from a match on a folium map and an image.

    Parameters:
    - match: The MatchingResult object containing guessed GCPs.
    - eval_map: The evaluation MapDataset object containing image data.
    - base_dataset: List of MapDataset objects for base maps.
    - plot_map: Boolean flag to enable/disable plotting on a folium map.
    - plot_image: Boolean flag to enable/disable plotting on the map image.
    - epsg_code: EPSG code for coordinate transformation (default is "EPSG:28193").
    """

    # Determine the appropriate GCPs from the match
    if hasattr(match, 'guessed_gcps_enhanced_delaunay_filtered_matches') and len(match.guessed_gcps_enhanced_delaunay_filtered_matches) > 0:
        gcps = match.guessed_gcps_enhanced_delaunay_filtered_matches
        kp_target = match.enhanced_delaunay_filtered_matches[['kp1_x', 'kp1_y']].values
        kp_base = match.enhanced_delaunay_filtered_matches[['kp2_x', 'kp2_y']].values
        # find kp_target that were chosen as gcps considering the coordinates
    elif hasattr(match, 'guessed_gcps_delaunay_filtered_matches') and len(match.guessed_gcps_delaunay_filtered_matches) > 0:
        gcps = match.guessed_gcps_delaunay_filtered_matches
        kp_target = match.delaunay_filtered_matches[['kp1_x', 'kp1_y']].values
        kp_base = match.delaunay_filtered_matches[['kp2_x', 'kp2_y']].values
        # find kp_target that were chosen as gcps
    else:
        print("No valid GCPs found in the match.")
        return

    # Ensure the DataFrame contains the required columns
    required_columns = {'sourceX', 'sourceY', 'mapX', 'mapY'}
    if not isinstance(gcps, pd.DataFrame) or not required_columns.issubset(gcps.columns):
        print("Invalid GCPs DataFrame or missing required columns.")
        return

    # Plotting on the folium map
    if plot_map:
        # Convert coordinates from the given EPSG code to EPSG 4326 for visualization
        transformer = Transformer.from_crs(epsg_code, "EPSG:4326", always_xy=True)
        converted_coords = np.array([transformer.transform(x, y) for x, y in gcps[['mapX', 'mapY']].values])

        folium_map = folium.Map(location=[converted_coords[:, 1].mean(), converted_coords[:, 0].mean()], zoom_start=12)
        for lon, lat in converted_coords:
            folium.CircleMarker(location=[lat, lon], radius=5, color='blue', fill=True, fill_opacity=0.6, tooltip='GCP').add_to(folium_map)

        # Display the folium map in the notebook
        display(folium_map)

    # Plotting on the images
    if plot_image and hasattr(eval_map, 'image_path'):
        # Plot on eval_map image
        image = cv2.imread(eval_map.image_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct display

            plt.figure(figsize=(24, 16))
            plt.imshow(image)
            plt.scatter(gcps['sourceX'], -gcps['sourceY'], color='red', s=10, label='GCPs')
            plt.title(f"GCPs on Image: {eval_map.map_info.folder}")
            plt.xlabel("X (pixels)")
            plt.ylabel("Y (pixels)")
            plt.legend()
            plt.show()
        else:
            print(f"Failed to load image from path: {eval_map.image_path}")

        # Plot on base_map image
        base_map = next((map_obj for map_obj in base_dataset if map_obj.map_info.folder == match.base_folder), None)
        if base_map is None:
            print(f"Base map not found for folder: {match.base_folder}")
            return

        image = cv2.imread(base_map.image_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(24, 16))
            plt.imshow(image)
            transform_tensor_to_image_base = base_map.tensor_to_image_transform

            # Ensure kp_base has the correct shape
            if kp_base.shape[1] != 2:
                print(f"kp_base has incorrect shape: {kp_base.shape}")
                return

            # Transform kp_base using cv2.perspectiveTransform
            kp_base_transformed = cv2.perspectiveTransform(
                kp_base.reshape(-1, 1, 2).astype(np.float32),
                transform_tensor_to_image_base
            ).reshape(-1, 2)

            # Adjust Y-coordinate if necessary
            kp_base_transformed[:, 1] *= -1

            plt.scatter(kp_base_transformed[:, 0], -kp_base_transformed[:, 1], color='blue', s=10, label='Base Keypoints')
            plt.title(f"Keypoints on Image: {base_map.map_info.folder}")
            plt.xlabel("X (pixels)")
            plt.ylabel("Y (pixels)")
            plt.legend()
            plt.show()
        else:
            print(f"Failed to load image from path: {base_map.image_path}")


def combine_gcps_for_map(best_matches_results, evaluation_dataset, distance_threshold=5, transformation_threshold=20, smoothing=1e-5, plot=False):
    """
    Combines GCPs from multiple matches for a given evaluation map, ensuring that new GCPs are added only if they
    are consistent with the initial transformation model established from the first match.

    Parameters:
    - best_matches_results: List of MatchingResults to use for combining GCPs.
    - evaluation_dataset: List of evaluation maps (MapDataset objects).
    - distance_threshold: Minimum distance (in meters) for considering a new GCP.
    - transformation_threshold: Maximum allowed deviation (in meters) for a point to be considered consistent.
    - smoothing: Smoothing factor for the RBF interpolator.
    - plot: Boolean flag to show plots of points being combined.

    Returns:
    - combined_gcps_df: A DataFrame containing the combined GCPs for the evaluation map.
    """
    combined_gcps = []

    if len(best_matches_results) > 0:
        # Use the first match to establish an initial transformation model
        first_match = best_matches_results[0]
        if hasattr(first_match, 'guessed_gcps_enhanced_delaunay_filtered_matches') and len(first_match.guessed_gcps_enhanced_delaunay_filtered_matches) > 3:
            gcps = first_match.guessed_gcps_enhanced_delaunay_filtered_matches
        elif hasattr(first_match, 'guessed_gcps_delaunay_filtered_matches') and len(first_match.guessed_gcps_delaunay_filtered_matches) > 3:
            gcps = first_match.guessed_gcps_delaunay_filtered_matches
        else:
            print("No valid GCPs found in the first match.")
            return pd.DataFrame()  # No valid GCPs to use

        # Check if gcps is a DataFrame and has the required columns
        if not isinstance(gcps, pd.DataFrame):
            print("GCPs is not a DataFrame.")
            return pd.DataFrame()
        if 'sourceX' not in gcps.columns or 'sourceY' not in gcps.columns:
            print(f"Expected columns 'sourceX' and 'sourceY' not found in GCPs DataFrame. Columns found: {gcps.columns}")
            return pd.DataFrame()

        # Store the initial GCPs
        combined_gcps.extend(gcps.to_dict('records'))

        # Prepare the data for initial RBF interpolation
        propagated_coords_image = gcps[['sourceX', 'sourceY']].values
        propagated_coords_world = gcps[['mapX', 'mapY']].values
        rbf_interpolator = RBFInterpolator(propagated_coords_image, propagated_coords_world, kernel='thin_plate_spline', smoothing=smoothing)

        # Convert coordinates from EPSG 28193 to EPSG 4326 for visualization
        transformer = Transformer.from_crs("EPSG:28193", "EPSG:4326", always_xy=True)
        converted_coords = np.array([transformer.transform(x, y) for x, y in propagated_coords_world])

        # Plot the initial GCPs on a folium map if plot flag is enabled
        if plot:
            folium_map = folium.Map(location=[converted_coords[:, 1].mean(), converted_coords[:, 0].mean()], zoom_start=14)
            for lon, lat in converted_coords:
                folium.CircleMarker(location=[lat, lon], radius=5, color='blue', fill=True, fill_opacity=0.6, tooltip='Initial GCP').add_to(folium_map)
            display(folium_map)

        # Process remaining matches
        for match_result in best_matches_results[1:]:
            if hasattr(match_result, 'guessed_gcps_enhanced_delaunay_filtered_matches') and len(match_result.guessed_gcps_enhanced_delaunay_filtered_matches) > 3:
                new_gcps = match_result.guessed_gcps_enhanced_delaunay_filtered_matches
            elif hasattr(match_result, 'guessed_gcps_delaunay_filtered_matches') and len(match_result.guessed_gcps_delaunay_filtered_matches) > 3:
                new_gcps = match_result.guessed_gcps_delaunay_filtered_matches
            else:
                continue

            # Check if new_gcps is valid
            if not isinstance(new_gcps, pd.DataFrame) or new_gcps.empty:
                continue
            if 'sourceX' not in new_gcps.columns or 'sourceY' not in new_gcps.columns:
                print(f"Expected columns 'sourceX' and 'sourceY' not found in new GCPs DataFrame. Columns found: {new_gcps.columns}")
                continue

            # Create a KDTree for existing combined GCPs
            existing_coords = np.array([(gcp['mapX'], gcp['mapY']) for gcp in combined_gcps])
            tree = KDTree(existing_coords) if len(existing_coords) > 0 else None

            # Check each new GCP
            for _, gcp in new_gcps.iterrows():
                gcp_coords = np.array([[gcp['mapX'], gcp['mapY']]])

                # Distance check
                if tree is not None:
                    distances, _ = tree.query(gcp_coords, k=1)
                    if distances[0] <= distance_threshold:
                        continue  # Skip if too close to existing points

                # Consistency check with the initial transformation model
                predicted_coords_world = rbf_interpolator([[gcp['sourceX'], gcp['sourceY']]])
                deviation = np.linalg.norm(predicted_coords_world - gcp_coords)
                if deviation <= transformation_threshold:
                    combined_gcps.append(gcp.to_dict())
                    if plot:
                        lon, lat = transformer.transform(gcp['mapX'], gcp['mapY'])
                        folium.CircleMarker(location=[lat, lon], radius=5, color='red', fill=True, fill_opacity=0.6, tooltip='New GCP').add_to(folium_map)

        # Plot the corresponding image if enabled
        if plot:
            eval_folder = first_match.evaluation_folder
            map_object = next((map_obj for map_obj in evaluation_dataset if map_obj.map_info.folder == eval_folder), None)
            if map_object and hasattr(map_object, 'image_path'):
                image = cv2.imread(map_object.image_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for plotting
                    plt.figure(figsize=(12, 8))
                    plt.imshow(image)
                    plt.scatter(gcps['sourceX'], -gcps['sourceY'], color='yellow', edgecolors='black', s=100, label='GCPs')
                    plt.title(f"GCPs on Image: {eval_folder}")
                    plt.xlabel("X (pixels)")
                    plt.ylabel("Y (pixels)")
                    plt.legend()
                    plt.show()
                    plt.close()

    combined_gcps_df = pd.DataFrame(combined_gcps)
    return combined_gcps_df

