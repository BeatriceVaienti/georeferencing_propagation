from dataclasses import dataclass
from typing import Optional
import pandas as pd

""" MatchingResult class
    This class is used to store the results of the matching process.
    
    Attributes:
    - evaluation_folder: the evaluation map folder name.
    - base_folder: the base map folder name.
    - superglue_matches: the matches obtained from SuperGlue between the two maps, in the form of a DataFrame.
    - ransac_filtered_matches: the matches filtered by RANSAC, in the form of a DataFrame.
    - delaunay_filtered_matches: the matches filtered by Delaunay triangulation, in the form of a DataFrame.
    - ransac_first_model_robust: the first RANSAC model obtained when filtering the matches.
    - enhanced_matches: the matches obtained after enhancing the initial matches, in the form of a DataFrame.
    - enhanced_ransac_filtered_matches: the matches filtered by RANSAC after enhancing, in the form of a DataFrame.
    - ransac_second_model_robust: the second RANSAC model obtained when filtering the enhanced matches.
    - gcp_propagated: the matches propagated using Ground Control Points (GCPs), in the form of a DataFrame.

    """

@dataclass
class MatchingResult:
    evaluation_folder: str
    base_folder: str
    superglue_matches: pd.DataFrame
    ransac_filtered_matches: Optional[pd.DataFrame] = None
    delaunay_filtered_matches: Optional[pd.DataFrame] = None
    enhanced_matches: Optional[pd.DataFrame] = None
    enhanced_ransac_filtered_matches: Optional[pd.DataFrame] = None
    enhanced_delaunay_filtered_matches: Optional[pd.DataFrame] = None
    gcp_propagated_before_enhancement: Optional[pd.DataFrame] = None
    gcp_propagated_after_enhancement: Optional[pd.DataFrame] = None