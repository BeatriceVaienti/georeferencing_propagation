
import click
import os
import logging
import sys
import pandas as pd
import json
import pickle
import tqdm
import warnings
import gc
warnings.filterwarnings("ignore", message="invalid value encountered in intersection")
warnings.filterwarnings("ignore", message="libpng warning: iCCP: known incorrect sRGB profile")

logging.basicConfig(level=logging.WARNING)
# Add modules directory to system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.MapDataset import MapDataset, create_map_dataset
from modules.data_preparation import extract_epsg
from modules.homologous_points_detection import (
    process_target_map_with_rotation,
    find_best_matches,
    remove_outliers_ransac,
    filter_match_with_delaunay
)
from modules.enhancement import enhance_matches_with_patches
from modules.georeferencing_propagation import propagate_georeferencing

@click.command()
@click.argument("config_file", type=click.Path(exists=True))
def run_pipeline(config_file):
    """
    Runs the map processing pipeline using parameters from a configuration file.

    Parameters:
    - config_file (str): Path to the JSON configuration file.
    """
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Extract parameters from the config file
    target_image_paths = config["target_image_paths"]
    target_image_masks = config["target_image_mask_paths"]
    anchor_image_paths = config["anchor_image_paths"]
    anchor_mask_paths = config["anchor_image_masks"]
    anchor_points_paths = config["anchor_points_paths"]
    output_dirs = config["output_dirs"]


    anchor_dataset = []
    for image_path, mask_path, points_path in tqdm.tqdm(zip(anchor_image_paths, anchor_mask_paths, anchor_points_paths), desc="Processing anchor maps: "):
        if os.path.exists(image_path) and os.path.exists(mask_path) and os.path.exists(points_path):
            #logging.info(f"Loading anchor map {image_path}...")
            first_row, points = get_points_file(points_path)
            anchor_map_info = create_map_info(image_path, mask_path, first_row, points)
            #logging.info(f"{anchor_map_info.get('points')}")
            anchor_map = create_map_dataset(anchor_map_info)
            anchor_map.calculate_and_store_north_rotation()
            anchor_map.run_superpoint_pipeline()
            anchor_dataset.append(anchor_map)
        else:
            logging.warning(f"Skipping anchor map with missing files: {image_path}, {mask_path}, {points_path}")

    for target_image_path, target_image_mask, output_dir in tqdm.tqdm(zip(target_image_paths, target_image_masks, output_dirs), desc="Processing target maps: "):
        logging.info("Loading target map...")
        # Check that the target map info is valid
        if not os.path.exists(target_image_path) or not os.path.exists(target_image_mask):
            logging.error("Invalid target image or mask path.")
            return

        target_map_info = create_map_info(target_image_path, target_image_mask)
        target_map = create_map_dataset(target_map_info)

        # First Pairwise Matching to align the target map North
        logging.info("Running pairwise matching to align the target map North...")
        process_target_map_with_rotation(
            target_map,
            anchor_dataset,
            min_match_score=0.3,
            match_threshold=100,
            plot_transformation=False,
        )

        # Repeating the Pairwise matching
        logging.info("Running pairwise matching with the new tensor...")

        # we're collecting all matches here, actually since we want GOOD matches, we can filter and take only matches with more than 100 matches
        find_best_matches(
            [target_map], 
            anchor_dataset,
            threshold_number_matches=100,
            number_best_results=3,
            evaluation=True,
            min_score=0.1
        )

        # Improving the matches
        logging.info("Improving the matches...")

        for best_match in tqdm.tqdm(target_map.best_matches_result, desc="Processing matches: "):
            logging.info("NUMBER OF INITIAL MATCHES: " + str(len(best_match.superglue_matches))) #newline )
            best_match.ransac_filtered_matches = remove_outliers_ransac(
                best_match.superglue_matches, ransac_threshold=20, ransac_max_trials=1000)
            
            if intermediate_check(best_match.ransac_filtered_matches) == False:
                continue

            logging.info("NUMBER OF RANSAC FILTERED MATCHES: " + str(len(best_match.ransac_filtered_matches))) #newline )

            best_match.delaunay_filtered_matches = filter_match_with_delaunay(
                target_map, anchor_dataset, best_match, best_match.ransac_filtered_matches, similarity_threshold=0.6, min_score_match=0.2)

            if intermediate_check(best_match.delaunay_filtered_matches) == False:
                continue
        
            logging.info("NUMBER OF DELAUNAY FILTERED MATCHES: " + str(len(best_match.delaunay_filtered_matches))) #newline )
            
            df_guessed_gcps_before = propagate_georeferencing(
                target_map, best_match, best_match.delaunay_filtered_matches, anchor_dataset, distance_threshold=80)
            best_match.gcp_propagated_before_enhancement = df_guessed_gcps_before

            if best_match.delaunay_filtered_matches is None:
                logging.warning("No matches, stopping the pipeline.")
                continue
            
            best_match.enhanced_matches = enhance_matches_with_patches(
                best_match.delaunay_filtered_matches, best_match, target_map, anchor_dataset, patch_length=1600, overlap=200, plot=False)

            if intermediate_check(best_match.enhanced_matches) == False:
                continue

            logging.info("NUMBER OF ENHANCED MATCHES: " + str(len(best_match.enhanced_matches))) #newline )
        

            best_match.enhanced_ransac_filtered_matches = remove_outliers_ransac(
                best_match.enhanced_matches, ransac_threshold=10, ransac_max_trials=1000)

            if intermediate_check(best_match.enhanced_ransac_filtered_matches) == False:    
                continue    

            logging.info("NUMBER OF ENHANCED RANSAC FILTERED MATCHES: " + str(len(best_match.enhanced_ransac_filtered_matches))) #newline )

            best_match.enhanced_delaunay_filtered_matches = filter_match_with_delaunay(
                target_map, anchor_dataset, best_match, best_match.enhanced_ransac_filtered_matches, similarity_threshold=0.6, min_score_match=0.2)


            if intermediate_check(best_match.enhanced_delaunay_filtered_matches) == False:  
                continue

            logging.info("NUMBER OF ENHANCED DELAUNAY FILTERED MATCHES: " + str(len(best_match.enhanced_delaunay_filtered_matches))) #newline )

            # Propagate the georeferencing using the matches
            df_guessed_gcps_after = propagate_georeferencing(
                target_map, best_match, best_match.enhanced_delaunay_filtered_matches, anchor_dataset, distance_threshold=80)
            best_match.gcp_propagated_after_enhancement = df_guessed_gcps_after
            #join the guessed gcps with the first_row

            print(f"NUMBER OF GUESSED BEFORE GCPS: {len(df_guessed_gcps_before)} - NUMBER OF GUESSED AFTER GCPS: {len(df_guessed_gcps_after)}")

        # Save the results
        logging.info(f"Saving the results in {output_dir}...")
        save_results(target_map, output_dir)
        # delete the target map with garbage collector
        del target_map
        del target_map_info
        gc.collect()






def get_points_file(points_path):
    if points_path is not None and os.path.exists(points_path):
        points = pd.read_csv(points_path, skiprows=1, delimiter=',') 
        first_row = pd.read_csv(points_path, nrows=1, delimiter=',')
        # remove the first row 
        return first_row , points
    else:
        logging.info(f"Points file {points_path} does not exist.")
        return None
def create_map_info(image_path, mask_path, first_row=None, points=None):
    if os.path.exists(image_path):
        objects_to_collect = {}
        objects_to_collect['image_path'] = image_path
        folder = os.path.basename(os.path.normpath(image_path))
        objects_to_collect['folder'] = folder
        if os.path.exists(mask_path):
            objects_to_collect['mask_path'] = mask_path
        if first_row is not None and points is not None:
            objects_to_collect['points'] = points
            crs_info = str(first_row.columns)
            epsg_code = extract_epsg(crs_info)
            objects_to_collect['epsg'] = epsg_code
        return objects_to_collect
    else:
        logging.info(f"Map folder {image_path} does not exist.")
        return None
    


def save_results(target_map, output_dir):
    with open(os.path.join(output_dir, 'target_map_all_matches.pkl'), 'wb') as f:
        pickle.dump(target_map, f)

def intermediate_check(result):
    if result is None:
        logging.warning("No matches, stopping the pipeline.")
        return False

    if len(result) < 4:
        logging.warning("Not enough matches to continue, stopping the pipeline.")
        return False

if __name__ == "__main__":
    run_pipeline()


