# Choose here the number of matches you'd like to retain either as the minimum number of matches
# or the number of best results to retain. These results will be saved in the pkl file and can be used for later analyses.
minimum_number_of_matches_to_retain = 100
number_of_best_results_to_retain = 3

import os
import sys
import json
import pickle
import gc
import logging
import warnings

import click
import pandas as pd
import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", message="invalid value encountered in intersection")
warnings.filterwarnings("ignore", message="libpng warning: iCCP: known incorrect sRGB profile")

# Add modules directory to system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.MapDataset import create_map_dataset, MapDataset
from modules.data_preparation import extract_epsg
from modules.homologous_points_detection import (
    process_target_map_with_rotation,
    find_best_matches,
    remove_outliers_ransac,
    filter_match_with_delaunay
)
from modules.enhancement import enhance_matches_with_patches
from modules.georeferencing_propagation import propagate_georeferencing



def select_best_anchor_match(best_matches):
    """
    Given a list of best_matches_result, pick the match that
    has the largest number of final matches (enhanced or not).
    """
    best_match_obj = None
    best_num_matches = 0

    for match in best_matches:
        # Check how many 'final' matches we have
        # If enhanced matches exist, prefer them; otherwise use normal
        if match.enhanced_delaunay_filtered_matches is not None:
            num = len(match.enhanced_delaunay_filtered_matches)
        elif match.delaunay_filtered_matches is not None:
            num = len(match.delaunay_filtered_matches)
        else:
            num = 0

        if num > best_num_matches:
            best_num_matches = num
            best_match_obj = match

    return best_match_obj

def create_and_save_gcp_csv(gcps_df: pd.DataFrame, output_dir: str):
    """
    Takes a DataFrame of guessed GCPs, renames columns to the desired format,
    and saves them as a CSV in the specified folder.
    If certain columns (like 'dX', 'dY', 'residual') are missing, they are skipped.
    """
    # Always add 'enable'
    gcps_df['enable'] = 1

    # Desired order if all columns exist
    desired_columns = ['mapX', 'mapY', 'sourceX', 'sourceY', 'enable', 'dX', 'dY', 'residual']

    # Filter out those that aren't present in gcps_df.columns
    final_columns = [col for col in desired_columns if col in gcps_df.columns]

    # Subset (only columns that exist)
    out_df = gcps_df[final_columns]

    csv_path = os.path.join(output_dir, "guessed_gcp.csv")
    out_df.to_csv(csv_path, index=False)
    logger.info(f"GCP CSV saved to {csv_path}")

@click.command()
@click.argument("config_file", type=click.Path(exists=True))
def run_pipeline(config_file):
    """
    Runs the map processing pipeline using parameters from a configuration file.

    The config file should contain:
      - target_image_paths (List[str])
      - target_image_mask_paths (List[str])
      - anchor_image_paths (List[str])
      - anchor_image_masks (List[str])
      - anchor_points_paths (List[str])
      - output_dirs (List[str])

    Example usage:
        python run_pipeline.py path/to/config.json
    """
    # Load configuration
    with open(config_file, 'r') as f:
        config = json.load(f)

    target_image_paths = config["target_image_paths"]
    target_image_masks = config["target_image_mask_paths"]
    anchor_image_paths = config["anchor_image_paths"]
    anchor_mask_paths = config["anchor_image_masks"]
    anchor_points_paths = config["anchor_points_paths"]
    output_dirs = config["output_dirs"]

    # Process anchor maps (load + SuperPoint pipeline)
    anchor_dataset = []
    for image_path, mask_path, points_path in tqdm.tqdm(
        zip(anchor_image_paths, anchor_mask_paths, anchor_points_paths),
        desc="Processing anchor maps"
    ):
        if not (os.path.exists(image_path) and os.path.exists(mask_path) and os.path.exists(points_path)):
            logger.warning(f"Skipping anchor map with missing files: {image_path}, {mask_path}, {points_path}")
            continue

        first_row, points = get_points_file(points_path)
        anchor_map_info = create_map_info(image_path, mask_path, first_row, points)
        if anchor_map_info is None:
            continue

        anchor_map = create_map_dataset(anchor_map_info)
        anchor_map.calculate_and_store_north_rotation()
        anchor_map.run_superpoint_pipeline()
        anchor_dataset.append(anchor_map)

    # Process target maps
    for target_image_path, target_image_mask, output_dir in tqdm.tqdm(
        zip(target_image_paths, target_image_masks, output_dirs),
        desc="Processing target maps"
    ):
        if not (os.path.exists(target_image_path) and os.path.exists(target_image_mask)):
            logger.error(f"Invalid target or mask path: {target_image_path}, {target_image_mask}")
            continue

        logger.info(f"Loading target map: {target_image_path}")
        target_map_info = create_map_info(target_image_path, target_image_mask)
        if target_map_info is None:
            continue

        target_map = create_map_dataset(target_map_info)

        # 1. Pairwise matching to align target map North
        logger.info("Running initial pairwise matching to align target map North...")
        process_target_map_with_rotation(
            target_map,
            anchor_dataset,
            min_match_score=0.3,
            match_threshold=100,
            plot_transformation=False
        )

        # 2. Re-match with the new transformation
        logger.info("Running pairwise matching with updated orientation...")
        find_best_matches(
            [target_map],
            anchor_dataset,
            threshold_number_matches=minimum_number_of_matches_to_retain,
            number_best_results=number_of_best_results_to_retain,
            evaluation=True,
            min_score=0.1
        )

        # 3. Improve matches by filtering + georeferencing propagation
        logger.info("Improving matches and propagating georeferencing...")
        for best_match in tqdm.tqdm(target_map.best_matches_result, desc="Processing matches"):
            logger.info(f"Number of initial SuperGlue matches: {len(best_match.superglue_matches)}")

            best_match.ransac_filtered_matches = remove_outliers_ransac(
                best_match.superglue_matches, ransac_threshold=20, ransac_max_trials=1000
            )
            if not intermediate_check(best_match.ransac_filtered_matches):
                continue

            logger.info(f"Number of RANSAC-filtered matches: {len(best_match.ransac_filtered_matches)}")

            best_match.delaunay_filtered_matches = filter_match_with_delaunay(
                target_map,
                anchor_dataset,
                best_match,
                best_match.ransac_filtered_matches,
                similarity_threshold=0.6,
                min_score_match=0.2
            )
            if not intermediate_check(best_match.delaunay_filtered_matches):
                continue

            logger.info(f"Number of Delaunay-filtered matches: {len(best_match.delaunay_filtered_matches)}")

            df_guessed_gcps_before = propagate_georeferencing(
                target_map,
                best_match,
                best_match.delaunay_filtered_matches,
                anchor_dataset,
                distance_threshold=80
            )
            best_match.gcp_propagated_before_enhancement = df_guessed_gcps_before

            # Patch-based enhancement
            best_match.enhanced_matches = enhance_matches_with_patches(
                best_match.delaunay_filtered_matches,
                best_match,
                target_map,
                anchor_dataset,
                patch_length=1600,
                overlap=200,
                plot=False
            )
            if not intermediate_check(best_match.enhanced_matches):
                continue

            logger.info(f"Number of enhanced matches: {len(best_match.enhanced_matches)}")

            best_match.enhanced_ransac_filtered_matches = remove_outliers_ransac(
                best_match.enhanced_matches, ransac_threshold=10, ransac_max_trials=1000
            )
            if not intermediate_check(best_match.enhanced_ransac_filtered_matches):
                continue

            logger.info(f"Number of enhanced RANSAC-filtered matches: {len(best_match.enhanced_ransac_filtered_matches)}")

            best_match.enhanced_delaunay_filtered_matches = filter_match_with_delaunay(
                target_map,
                anchor_dataset,
                best_match,
                best_match.enhanced_ransac_filtered_matches,
                similarity_threshold=0.6,
                min_score_match=0.2
            )
            if not intermediate_check(best_match.enhanced_delaunay_filtered_matches):
                continue

            logger.info(f"Number of enhanced Delaunay-filtered matches: {len(best_match.enhanced_delaunay_filtered_matches)}")

            # Final georeferencing propagation
            df_guessed_gcps_after = propagate_georeferencing(
                target_map,
                best_match,
                best_match.enhanced_delaunay_filtered_matches,
                anchor_dataset,
                distance_threshold=80
            )
            best_match.gcp_propagated_after_enhancement = df_guessed_gcps_after

            logger.info(f"GCPS before enhancement: {len(df_guessed_gcps_before)} | After: {len(df_guessed_gcps_after)}")

        best_anchor_match = select_best_anchor_match(target_map.best_matches_result)
        
        if best_anchor_match is not None:
            # We have a best anchor match
            df_gcps_before = best_anchor_match.gcp_propagated_before_enhancement
            df_gcps_after  = best_anchor_match.gcp_propagated_after_enhancement

            # Decide which set of GCPs you want to save: after enhancement if exist & not empty
            if (best_anchor_match.enhanced_delaunay_filtered_matches is not None 
                and len(best_anchor_match.enhanced_delaunay_filtered_matches) > 0
                and df_gcps_after is not None 
                and not df_gcps_after.empty
            ):
                final_gcp_df = df_gcps_after
                logger.info("Using the AFTER-enrichment GCP set.")
            else:
                final_gcp_df = df_gcps_before
                logger.info("Using the BEFORE-enrichment GCP set.")

            if final_gcp_df is not None and not final_gcp_df.empty:
                # Save as CSV
                create_and_save_gcp_csv(final_gcp_df, output_dir)
            else:
                logger.warning("No GCPs available to write for this best match.")
        else:
            logger.warning("No best match found for this target map.")

        # Save results
        logger.info(f"Saving results to {output_dir} ...")
        save_results(target_map, output_dir)

        # Clean up
        del target_map
        del target_map_info
        gc.collect()

def get_points_file(points_path):
    """
    Reads the .points file (if it exists) to extract EPSG and GCP data.

    Returns:
        tuple: (header_row, points_dataframe) or (None, None) if file not found.
    """
    if points_path is not None and os.path.exists(points_path):
        points_df = pd.read_csv(points_path, skiprows=1, delimiter=',')
        header_row = pd.read_csv(points_path, nrows=1, delimiter=',')
        return header_row, points_df
    else:
        logger.warning(f"Points file does not exist: {points_path}")
        return None, None

def create_map_info(image_path, mask_path, first_row=None, points=None):
    """
    Creates a dictionary of metadata for a map given its image path, mask path, and (optionally) GCP/CRS info.

    Args:
        image_path (str): Path to the map image.
        mask_path (str): Path to the mask for the map.
        first_row (pd.DataFrame, optional): The header row from the .points file (to extract EPSG).
        points (pd.DataFrame, optional): The rest of the .points file (actual GCP data).

    Returns:
        dict or None: Dictionary containing image, mask, EPSG, and point data (if valid), otherwise None.
    """
    if not os.path.exists(image_path):
        logger.warning(f"Image file does not exist: {image_path}")
        return None

    info = {
        'image_path': image_path,
        'folder': os.path.basename(os.path.normpath(image_path))
    }

    if os.path.exists(mask_path):
        info['mask_path'] = mask_path

    if first_row is not None and points is not None:
        info['points'] = points
        crs_info = str(first_row.columns)
        epsg_code = extract_epsg(crs_info)
        info['epsg'] = epsg_code

    return info

def save_results(target_map, output_dir):
    """
    Serializes the `target_map` object (containing match data, transformations, etc.) to a .pkl file.

    Args:
        target_map (MapDataset): The processed MapDataset object for the target map.
        output_dir (str): Path to the folder where the pickle file will be saved.
    """
    output_path = os.path.join(output_dir, 'target_map_all_matches.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(target_map, f)

def intermediate_check(result):
    """
    Simple utility to confirm we have enough matches to continue processing.

    Args:
        result (list or None): The current list of matches.

    Returns:
        bool: True if matches are present and length >= 4, otherwise False.
    """
    if result is None:
        logger.warning("No matches found. Stopping this step.")
        return False
    if len(result) < 4:
        logger.warning("Not enough matches to continue. Stopping this step.")
        return False
    return True

if __name__ == "__main__":
    run_pipeline()


