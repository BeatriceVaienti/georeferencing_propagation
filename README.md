# Georeferencing Historical Maps Using Image Matching and Delaunay Triangulation

This repository contains the **replication code** for the paper:

> **_Georeferencing Historical Maps Using Image Matching and Delaunay Triangulation._**

It demonstrates an end-to-end pipeline that takes a set of Anchor maps (with known georeferencing) and propagates their ground control points (GCPs) to new (“Target”) historical maps using robust feature matching techniques, outlier exclusion based on RANSAC and Delaunay triangulation and patch-based enhancement.

---

## Table of Contents
1. [Overview of the Pipeline](#overview-of-the-pipeline)
2. [Repository Structure](#repository-structure)
3. [Creating the `config.json` File](#creating-the-configjson-file)
4. [Running the Pipeline](#running-the-pipeline)
   - [From the Terminal](#from-the-terminal)
   - [From Jupyter Notebook](#from-jupyter-notebook)
5. [Output Files](#output-files)
6. [Citation](#citation)

---

## Overview of the Pipeline

1. **Map Dataset Preparation**: We load our “Anchor” maps (with known GCPs) and “Target” maps (to be georeferenced).
2. **Feature Extraction & Matching**: We use SuperPoint & SuperGlue for robust keypoint detection and matching.
3. **Filtering with RANSAC and Delaunay**: Matches are filtered to remove outliers, then refined using Delaunay triangulation.
4. **Patch-Based Enhancement**: Matches are further enhanced by local patch matching.
5. **Georeferencing Propagation**: The final set of stable matches is used to propagate ground control points onto the target maps.
6. **Output**: A pickle file containing the entire processed dataset, and a CSV with the final GCPs for each Target map.

---

## Repository Structure
```
georeferencing_propagation/
├── modules/
│   ├── data_preparation.py
│   ├── enhancement.py
│   ├── georeferencing_propagation.py
│   ├── homologous_points_detection.py
│   ├── MapDataset.py
│   ├── MatchingResult.py
│   ├── superpoint.py
│   └── visualization.py
├── scripts/
│   └── run_pipeline.py
├── input/
│   └── config.json          # Generated pipeline configuration file
├── input_maps/
│   ├── anchor/
│   │   ├── 1846_vandevelde/  # Maps we provide with known GCPs
│   │   └── ...
│   └── target/
│       ├── 1860_bartholomew/  # Maps to be georeferenced, GCPs are still provided for testing
│       └── ...
└── map_list_preparation.ipynb
```

- **`modules/`**: Core functionality for data loading, feature matching, refinement, and georeferencing.
- **`scripts/`**: Contains the main pipeline script (`run_pipeline.py`).
- **`input/`**: Folder where the configuration JSON (`config.json`) will be saved.
- **`input_maps/anchor` & `input_maps/target`**: Example subfolders containing images, masks, and `.points` files.
- **`map_list_preparation.ipynb`**: A Jupyter Notebook that helps you build `config.json` by scanning the `input_maps/` folders.


## Creating the `config.json` File

Before running the pipeline, you need a configuration file (`config.json`) describing:

- Paths to **target** maps & masks
- Paths to **anchor** maps & masks & GCP `.points` files
- Output directories where results will be stored

## Running the Pipeline

### From the Terminal

Assuming `run_pipeline.py` is in `scripts/` and your config file is `input/config.json`, run:

\`\`\`bash
python scripts/run_pipeline.py input/config.json
\`\`\`

This script will:
1. Load **anchor** maps and run the SuperPoint pipeline.
2. Load **target** maps.
3. Align, match, filter, enhance, and propagate georeferencing from anchors to targets.
4. Save output files in each target map’s `propagated_gcp_results/` directory.

### From Jupyter Notebook

In your `map_list_preparation.ipynb` (or any other notebook), you can use:

\`\`\`python
!python scripts/run_pipeline.py input/config.json
\`\`\`


## Output Files

1. **`target_map_all_matches.pkl`**  
   A pickle file stored in each target map’s `propagated_gcp_results` folder, containing the entire `MapDataset` object and intermediate matching results.

2. **`guessed_gcp.csv`**  
   A CSV file in the same folder with the final GCPs (columns can include `mapX, mapY, sourceX, sourceY, enable, dX, dY, residual`), depending on availability.

## Citation
