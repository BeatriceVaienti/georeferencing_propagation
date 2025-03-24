import os
import logging
from typing import Optional, List
from pdf2image import convert_from_path
import cv2
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from typing import Dict, Any
from modules.data_preparation import preprocess_image
from modules.superpoint import run_superpoint_inference
import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in intersection")
warnings.filterwarnings("ignore", message="libpng warning: iCCP: known incorrect sRGB profile")
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SuperPointResults:
    """
    Data structure to hold SuperPoint feature extraction results.
    """
    keypoints: torch.Tensor
    descriptors: torch.Tensor
    scores: torch.Tensor


@dataclass
class MapInfo:
    """
    Data structure to store metadata and file paths related to a map.
    """
    image_path: str
    mask_path: Optional[str] = None
    folder: Optional[str] = None
    author: Optional[str] = None
    year: Optional[int] = None

from dataclasses import dataclass, field
from typing import Optional, List
import os
import cv2
import logging
import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

@dataclass
class MapInfo:
    """
    Data structure to store metadata and file paths related to a map.
    """
    image_path: str
    mask_path: Optional[str] = None
    folder: Optional[str] = None
    author: Optional[str] = None
    year: Optional[int] = None
    initial_image_path: Optional[str] = None
    initial_mask_path: Optional[str] = None


@dataclass
class MapDataset:
    """
    Dataset class that encapsulates the loading and management of map data.
    It relies on a MapInfo object for file paths and metadata.
    """
    map_info: MapInfo
    gcps: Optional[pd.DataFrame] = None
    north_rotation_angle: Optional[np.ndarray] = None
    tensor: Optional[torch.Tensor] = None
    tensor_to_image_transform: Optional[torch.Tensor] = None
    image_to_tensor_transform: Optional[torch.Tensor] = None
    superpoint_results: Optional[object] = None  # Replace with actual SuperPointResults type if defined
    initial_matches_stats: Optional[pd.DataFrame] = None

    mask: Optional[np.ndarray] = None
    image: Optional[np.ndarray] = None
    best_matches_result: List = field(default_factory=list, init=False)

    def __post_init__(self):
        """
        Post-initialization processing: load image and mask files if available.
        """
        if self.image is None:
            self._load_image()
        #else:
            #logger.info("Image already provided; skipping image loading.")

        if self.mask is None:
            self._load_mask()
        #else:
            #logger.info("Mask already provided; skipping mask loading.")

    def _load_image(self):
        """
        Loads the main image from the file path specified in map_info.
        If the image is too large, creates a downscaled copy and updates the image path.
        """
        image_path = self.map_info.image_path

        if not os.path.exists(image_path):
            logger.error("Image file not found: %s", image_path)
            raise FileNotFoundError(f"Image file not found: {image_path}")

        ext = os.path.splitext(image_path)[1].lower()
        # If the file is a PDF, convert it to an image:
        if ext == '.pdf':
            #logger.info("Converting PDF to image: %s", image_path)
            try:
                from pdf2image import convert_from_path
                pages = convert_from_path(image_path, dpi=200)
                if not pages:
                    raise ValueError(f"No pages found in PDF: {image_path}")
                pil_image = pages[0]
                self.image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                #logger.info("Converted PDF to image successfully.")
            except Exception as e:
                logger.error("Failed to convert PDF to image: %s", e)
                raise ValueError(f"Failed to convert PDF to image: {image_path}") from e
        else:
            supported_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
            if ext not in supported_extensions:
                logger.error("Unsupported image file format: %s", ext)
                raise ValueError(f"Unsupported image file format: {ext} (file: {image_path})")

            try:
                self.image = cv2.imread(image_path)
                if self.image is None:
                    raise ValueError(f"Failed to load image from path: {image_path}")
            except cv2.error as e:
                logger.error("OpenCV error while loading image: %s", e)
                raise ValueError(f"OpenCV error while loading image from path: {image_path}") from e

        # Check for downsizing due to pixel limits
        if self.image is None:
            raise ValueError(f"Failed to load image from path: {image_path}")

        num_pixels = self.image.shape[0] * self.image.shape[1]
        max_allowed_pixels = 1e11
        if num_pixels > max_allowed_pixels:
            logger.warning("Image at %s is extremely large (%d pixels).", image_path, num_pixels)
            scale_factor = (max_allowed_pixels / num_pixels) ** 0.5
            new_size = (int(self.image.shape[1] * scale_factor), int(self.image.shape[0] * scale_factor))
            self.image = cv2.resize(self.image, new_size, interpolation=cv2.INTER_AREA)
            downscaled_path = os.path.splitext(image_path)[0] + "_downscaled.png"
            cv2.imwrite(downscaled_path, self.image)
            self.map_info.image_path = downscaled_path
            #logger.info("Resized image saved to %s", downscaled_path)
        #else:
            #logger.info("Image size is within limits.")

    def _load_mask(self):
        """
        Loads the mask image (in grayscale) if a mask path is provided in map_info.
        Ensures that the mask matches the image dimensions and updates the mask path if resized.
        """
        mask_path = self.map_info.mask_path
        if mask_path:
            if not os.path.exists(mask_path):
                logger.error("Mask file not found: %s", mask_path)
                raise FileNotFoundError(f"Mask file not found: {mask_path}")

            try:
                self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if self.mask is None:
                    raise ValueError(f"Failed to load mask from path: {mask_path}")

                #logger.info("Loaded mask from %s", mask_path)

                if hasattr(self, 'image') and self.image is not None:
                    if self.mask.shape[:2] != self.image.shape[:2]:
                        logger.warning(
                            "Mask dimensions %s do not match image dimensions %s. Resizing mask.",
                            self.mask.shape[:2], self.image.shape[:2]
                        )
                        self.mask = cv2.resize(self.mask, (self.image.shape[1], self.image.shape[0]),
                                               interpolation=cv2.INTER_NEAREST)
                        downscaled_mask_path = os.path.splitext(mask_path)[0] + "_downscaled.png" 
                        cv2.imwrite(downscaled_mask_path, self.mask)
                        self.map_info.mask_path = downscaled_mask_path
                        #logger.info("Resized mask saved to %s", downscaled_mask_path)
            except cv2.error as e:
                logger.error("OpenCV error while loading mask: %s", e)
                raise ValueError(f"OpenCV error while loading mask from path: {mask_path}") from e
        #else:
            #logger.info("No mask path provided; skipping mask loading.")

    def run_superpoint_pipeline(self):
        """
        Runs the complete SuperPoint pipeline without any scaling adjustments for points.
        """
        self.generate_tensor()
        self.run_superpoint()
        self.remove_points_near_mask()

        #logger.info("SuperPoint pipeline completed.")

    def generate_tensor(self):
        """
        Generates a tensor from the image and mask paths and stores it in the object.
        """
            # Ensure the image is converted to uint8 if necessary
        if self.image.dtype == np.float64:
            # Normalize the image if necessary, then convert to uint8
            self.image = (255 * np.clip(self.image, 0, 1)).astype(np.uint8) if np.max(self.image) <= 1 else self.image.astype(np.uint8)

        greyscale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.tensor, self.image_to_tensor_transform, self.tensor_to_image_transform = preprocess_image(greyscale_image, self.mask, north_rotation_angle=self.north_rotation_angle)


    def run_superpoint(self):
        """
        Uses an external function to perform SuperPoint inference.
        """
        # Call the external run_superpoint_inference function and get results
        keypoints, descriptors, scores = run_superpoint_inference(self.tensor)
        # change the shape of the descriptors tensor from descriptor_dim x num_keypoints to num_keypoints x descriptor_dim

        descriptors = torch.stack(descriptors).squeeze(0).permute(1, 0)

        # back to list
        descriptors = [descriptors]
        # Store the results in self.superpoint_results
        self.superpoint_results = SuperPointResults(keypoints, descriptors, scores)

    def remove_points_near_mask(self, buffer: int = 20, plot: bool = False):
        """
        Removes keypoints that fall in masked-out regions using a specified buffer.
        Optionally, it plots the dilated mask overlayed on the map image with the filtered keypoints.
        
        The method performs the following steps:
            1. Checks if mask and keypoints are available.
            2. Dilates the mask using a square kernel of size (buffer x buffer).
            3. Applies the tensor-to-image transformation to map keypoints to image coordinates.
            4. Filters keypoints that fall on regions where the dilated mask equals zero.
            5. Updates the SuperPointResults with the filtered keypoints, descriptors, and scores.
            6. Optionally generates a plot showing the mask and keypoints.
        
        Parameters:
            buffer (int): Number of pixels by which to dilate the mask (default is 20).
            plot (bool): Whether to display a plot of the dilated mask and filtered keypoints (default False).
        
        Raises:
            ValueError: If there is a mismatch in the number of keypoints and descriptors.
        """
        # Check the presence of mask and keypoints.
        if (self.map_info.mask_path is None or not self.map_info.mask_path) and \
        (self.mask is None or self.mask.size == 0) or self.superpoint_results.keypoints is None:
            logger.warning("No mask or keypoints available for processing (Map: %s)", self.map_info.folder)
            return        

        image = self.image
        mask = self.mask
        if image is None or mask is None:
            logger.warning("Image or mask is not loaded for map: %s", self.map_info.folder)
            return

        # Convert the image to RGB for plotting if it is a color image.
        if image.ndim == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image.copy()

        # Dilate the mask.
        kernel = np.ones((buffer, buffer), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        #logger.info("Mask dilated with buffer: %d", buffer)

        # Retrieve the tensor_to_image_transform (assumed to be a 3x3 NumPy array).
        transform: np.ndarray = self.tensor_to_image_transform
        if transform is None or transform.shape != (3, 3):
            logger.error("Invalid tensor_to_image_transform: %s", transform)
            raise ValueError("Invalid tensor_to_image_transform.")

        # Retrieve keypoints, descriptors, and scores.
        keypoints_tensor = self.superpoint_results.keypoints[0]         # Expected shape: (num_keypoints, 2)
        descriptors_tensor = self.superpoint_results.descriptors[0]     # Expected shape: (descriptor_dim, num_keypoints)
        scores_tensor = self.superpoint_results.scores[0]               # Expected shape: (num_keypoints,)

        num_keypoints = keypoints_tensor.shape[0]
        if num_keypoints == 0:
            logger.warning("No keypoints to process for map: %s", self.map_info.folder)
            return

        if descriptors_tensor.shape[0] != num_keypoints:
            raise ValueError(
                f"Mismatch in keypoints and descriptors count: {num_keypoints} vs {descriptors_tensor.shape[0]}"
            )

        # Convert keypoints to homogeneous coordinates: shape (num_keypoints, 3)
        keypoints_homogeneous = torch.cat(
            [keypoints_tensor, torch.ones((num_keypoints, 1))],
            dim=1
        )

        # Convert to NumPy and perform transformation.
        keypoints_np = keypoints_homogeneous.numpy().T  # shape: (3, num_keypoints)
        transformed_keypoints = transform @ keypoints_np  # shape: (3, num_keypoints)
        # Normalize homogeneous coordinates.
        transformed_keypoints /= transformed_keypoints[2, :]
        transformed_x = transformed_keypoints[0, :].astype(int)
        transformed_y = transformed_keypoints[1, :].astype(int)

        # Create a mask for valid keypoints that are within image boundaries.
        valid_mask = (
            (transformed_x >= 0) & (transformed_x < dilated_mask.shape[1]) &
            (transformed_y >= 0) & (transformed_y < dilated_mask.shape[0])
        )
        transformed_x = transformed_x[valid_mask]
        transformed_y = transformed_y[valid_mask]
        keep_indices = np.where(valid_mask)[0]
        if len(keep_indices) == 0:
            logger.warning("All keypoints are outside image boundaries for map: %s", self.map_info.folder)
            self.superpoint_results.keypoints = None
            self.superpoint_results.descriptors = None
            self.superpoint_results.scores = None
            return

        # Identify keypoints falling on the masked-out regions.
        # We assume a mask value of 0 indicates a masked-out region.
        keypoints_on_mask = dilated_mask[transformed_y, transformed_x] == 0
        final_keep_indices = keep_indices[keypoints_on_mask]

        # Filter out keypoints, descriptors, and scores.
        if len(final_keep_indices) > 0:
            filtered_keypoints = keypoints_tensor[final_keep_indices]  # shape: (num_keep, 2)
            filtered_descriptors = descriptors_tensor[final_keep_indices]  # shape: (descriptor_dim, num_keep)
            filtered_scores = scores_tensor[final_keep_indices]  # shape: (num_keep,)

            self.superpoint_results.keypoints = [filtered_keypoints]
            self.superpoint_results.descriptors = [filtered_descriptors]
            self.superpoint_results.scores = [filtered_scores]
            #logger.info("Filtered keypoints: %d remaining after mask removal.", len(final_keep_indices))
        else:
            #logger.info("No keypoints remain after filtering for map: %s", self.map_info.folder)
            self.superpoint_results.keypoints = None
            self.superpoint_results.descriptors = None
            self.superpoint_results.scores = None

        # Optionally plot the blended image with filtered keypoints.
        if plot:
            if image_rgb.ndim == 2:
                overlay = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
            else:
                overlay = image_rgb.copy()

            # Create a red overlay for the dilated mask.
            red_mask = np.zeros_like(overlay)
            red_mask[dilated_mask > 0] = [255, 0, 0]
            blended_image = cv2.addWeighted(overlay, 0.7, red_mask, 0.3, 0)

            plt.figure(figsize=(10, 10))
            plt.imshow(blended_image)
            if self.superpoint_results.keypoints is not None:
                filtered_keypoints_tensor = self.superpoint_results.keypoints[0].numpy()  # shape: (num_keep, 2)
                # Convert to homogeneous coordinates.
                filtered_keypoints_homogeneous = np.hstack([filtered_keypoints_tensor, np.ones((filtered_keypoints_tensor.shape[0], 1))]).T
                transformed_filtered_keypoints = transform @ filtered_keypoints_homogeneous
                transformed_filtered_keypoints /= transformed_filtered_keypoints[2, :]
                transformed_filtered_x = transformed_filtered_keypoints[0, :]
                transformed_filtered_y = transformed_filtered_keypoints[1, :]

                plt.scatter(
                    transformed_filtered_x, transformed_filtered_y,
                    s=50, c='lime', marker='o', edgecolors='black', linewidths=0.5
                )

            plt.title(f"Dilated Mask Overlay with Filtered Keypoints: {self.map_info.folder}")
            plt.axis('off')
            plt.show()
        """
        Removes keypoints that are near masked-out regions using a buffer in pixels.
        Optionally plots the dilated mask over the image with the remaining keypoints.

        Parameters:
        - buffer: The number of pixels by which to dilate the mask to determine proximity (default is 20).
        - plot: If True, plots the dilated mask over the map image along with filtered keypoints (default is False).
        """
        if (self.map_info.mask_path is None or not self.map_info.mask_path) and \
        (self.mask is None or self.mask.size == 0) or \
        self.superpoint_results.keypoints is None:
            print(f"No mask or keypoints available for map: {self.map_info.folder}")
            return

        # Load the map image and mask
        image = self.image
        mask = self.mask

        if image is None or mask is None:
            print(f"Failed to load image or mask for map: {self.map_info.folder}")
            return

        # Convert image from BGR to RGB for plotting if it's color
        if image.ndim == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image.copy()

        # Dilate the mask
        kernel = np.ones((buffer, buffer), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)

        # Retrieve the tensor_to_image_transform (3x3 matrix)
        transform = self.tensor_to_image_transform  # Should be a 3x3 numpy array

        # Access the actual keypoints, descriptors, and scores tensors
        keypoints_tensor = self.superpoint_results.keypoints[0]        # Shape: (num_keypoints, 2)
        descriptors_tensor = self.superpoint_results.descriptors[0]    # Shape: (descriptor_dim, num_keypoints)
        scores_tensor = self.superpoint_results.scores[0]              # Shape: (num_keypoints,)

        num_keypoints = keypoints_tensor.shape[0]

        if num_keypoints == 0:
            print(f"No keypoints to process for map: {self.map_info.folder}")
            return

        # Ensure keypoints and descriptors have the same number of keypoints
        if descriptors_tensor.shape[0] != num_keypoints:
            raise ValueError(f"Mismatch in keypoints and descriptors count: {num_keypoints} vs {descriptors_tensor.shape[0]}")

        # Convert keypoints to homogeneous coordinates and apply the transformation
        keypoints_homogeneous = torch.cat([keypoints_tensor, torch.ones((num_keypoints, 1))], dim=1)  # Shape: (num_keypoints, 3)
        
        # Convert to NumPy for matrix multiplication
        keypoints_np = keypoints_homogeneous.numpy().T  # Shape: (3, num_keypoints)
        
        # Apply the transformation: image_space_keypoints = transform @ keypoints_np
        transformed_keypoints = transform @ keypoints_np    # Shape: (3, num_keypoints)
        
        # Normalize to get (x, y) coordinates
        transformed_keypoints /= transformed_keypoints[2, :]  # Divide by the homogeneous coordinate
        
        # Extract x and y coordinates
        transformed_x = transformed_keypoints[0, :].astype(int)
        transformed_y = transformed_keypoints[1, :].astype(int)
        
        # Create a mask to filter valid keypoints within image boundaries
        valid_mask = (
            (transformed_x >= 0) & (transformed_x < dilated_mask.shape[1]) &
            (transformed_y >= 0) & (transformed_y < dilated_mask.shape[0])
        )
        
        # Apply boundary mask
        transformed_x = transformed_x[valid_mask]
        transformed_y = transformed_y[valid_mask]
        keep_indices = np.where(valid_mask)[0]
        
        if len(keep_indices) == 0:
            print(f"All keypoints are outside the image boundaries for map: {self.map_info.folder}")
            self.superpoint_results.keypoints = None
            self.superpoint_results.descriptors = None
            self.superpoint_results.scores = None
            return
        
        # Check which keypoints fall on masked-out regions
        # Assuming mask value 0 indicates masked-out regions
        keypoints_on_mask = dilated_mask[transformed_y, transformed_x] == 0  # Boolean array
        
        # Indices of keypoints to keep (those on masked regions)
        final_keep_indices = keep_indices[keypoints_on_mask]
        
        # Filter keypoints, descriptors, and scores
        if len(final_keep_indices) > 0:
            filtered_keypoints = keypoints_tensor[final_keep_indices]          # Shape: (num_keep, 2)
            filtered_descriptors = descriptors_tensor[final_keep_indices]    # Shape: (descriptor_dim, num_keep)
            filtered_scores = scores_tensor[final_keep_indices]                # Shape: (num_keep,)
            
            # Update superpoint_results with filtered keypoints, descriptors, and scores
            self.superpoint_results.keypoints = [filtered_keypoints]
            self.superpoint_results.descriptors = [filtered_descriptors]
            self.superpoint_results.scores = [filtered_scores]
        else:
            # No keypoints remain after filtering
            self.superpoint_results.keypoints = None
            self.superpoint_results.descriptors = None
            self.superpoint_results.scores = None

        if plot:
            # Overlay the mask by blending it with the image
            if image_rgb.ndim == 2:
                # Grayscale image
                overlay = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
            else:
                overlay = image_rgb.copy()
            
            # Create a red mask overlay
            red_mask = np.zeros_like(overlay)
            red_mask[dilated_mask > 0] = [255, 0, 0]  # Red color for masked regions
            
            # Blend the original image with the red mask
            blended_image = cv2.addWeighted(overlay, 0.7, red_mask, 0.3, 0)
        
            # Plot the blended image
            plt.figure(figsize=(10, 10))
            plt.imshow(blended_image)
        
            # Plot the filtered keypoints
            if self.superpoint_results.keypoints:
                filtered_keypoints_tensor = self.superpoint_results.keypoints[0].numpy()  # Shape: (num_keep, 2)
                
                # Convert to homogeneous coordinates for transformation
                filtered_keypoints_homogeneous = np.hstack([filtered_keypoints_tensor, np.ones((filtered_keypoints_tensor.shape[0], 1))]).T  # Shape: (3, num_keep)
                
                # Apply transformation to get image space coordinates
                transformed_filtered_keypoints = transform @ filtered_keypoints_homogeneous  # Shape: (3, num_keep)
                transformed_filtered_keypoints /= transformed_filtered_keypoints[2, :]  # Normalize
                transformed_filtered_x = transformed_filtered_keypoints[0, :]
                transformed_filtered_y = transformed_filtered_keypoints[1, :]
                
                # Plot each filtered keypoint
                plt.scatter(transformed_filtered_x, transformed_filtered_y, s=50, c='lime', marker='o', edgecolors='black', linewidths=0.5)
        
            plt.title(f"Dilated Mask Overlay with Filtered Keypoints: {self.map_info.folder}")
            plt.axis('off')
            plt.show()

    def calculate_north_rotation(self):
        """
        Calculates the rotation angle to align the image north using all Ground Control Points (GCPs).
        The GCPs are used to determine the orientation of the image relative to the real-world north direction.

        Returns the computed rotation angle (in radians) from the GCPs.
        This method can only be used when the GCPs are available.
        """
        import numpy as np

        # Check for minimum number of GCPs
        if self.gcps is None or len(self.gcps) < 2:
            print("Not enough GCPs to calculate a north rotation angle.")
            return 0  # Default to 0 if no rotation can be computed

        # Extract image and world coordinates from GCPs as numpy arrays
        manual_coords_image = self.gcps[['sourceX', 'sourceY']].to_numpy()
        manual_coords_world = self.gcps[['mapX', 'mapY']].to_numpy()

        angles = []
        n = len(self.gcps)
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Vector in world and image coordinates
                vec_world = manual_coords_world[j] - manual_coords_world[i]
                vec_image = manual_coords_image[j] - manual_coords_image[i]

                # Compute angles
                angle_world = np.arctan2(vec_world[1], vec_world[0])
                angle_image = np.arctan2(vec_image[1], vec_image[0])

                angles.append(angle_image - angle_world)

        # Return the median of all pairwise angles (could also use mean)
        return np.median(angles)

    def calculate_and_store_north_rotation(self):
        """
        Calculates and then stores the north rotation angle (in radians) into self.north_rotation_angle
        using the Ground Control Points (GCPs).
        """
        rotation_angle = self.calculate_north_rotation()
        self.north_rotation_angle = rotation_angle
        # Optionally print or log:
        # print(f"Stored north rotation angle (radians): {self.north_rotation_angle}")
              
    def extract_patch(self, x_min, y_min, x_max, y_max):
        """
        Extracts a rectangular patch from the map's tensor image based on specified coordinates.
        This method assumes that the map tensor is a 2D or 3D tensor where the last two dimensions represent height and width.

        Parameters:
        - x_min, y_min: Top-left corner coordinates of the patch.
        - x_max, y_max: Bottom-right corner coordinates of the patch.

        Returns:
        - patch_map: A new MapDataset object representing the extracted patch.
        """
        # Ensure coordinates are integers and within image boundaries
        x_min = max(0, int(x_min))
        y_min = max(0, int(y_min))
        x_max = min(self.tensor.shape[-1], int(x_max))  # Width dimension
        y_max = min(self.tensor.shape[-2], int(y_max))  # Height dimension

        # Extract the patch from the image tensor
        patch_tensor = self.tensor[..., y_min:y_max, x_min:x_max]

        # Optionally, handle keypoints, descriptors, and scores if applicable
        keypoints = self.superpoint_results.keypoints[0]  # Assume keypoints are available
        descriptors = self.superpoint_results.descriptors[0]
        scores = self.superpoint_results.scores[0]

        # Filter keypoints to be within the patch
        mask = (
            (keypoints[:, 0] >= x_min) & (keypoints[:, 0] < x_max) &
            (keypoints[:, 1] >= y_min) & (keypoints[:, 1] < y_max)
        )
        filtered_keypoints = keypoints[mask]
        filtered_descriptors = descriptors[mask]
        filtered_scores = scores[mask]

        # Adjust keypoints' coordinates relative to the patch
        filtered_keypoints[:, 0] -= x_min
        filtered_keypoints[:, 1] -= y_min

        # Create a new MapDataset object for the patch (assuming you have a constructor for it)
        patch_map = MapDataset(
            tensor=patch_tensor,
            
            superpoint_results=SuperPointResults(
                keypoints=[filtered_keypoints],
                descriptors=[filtered_descriptors],
                scores=[filtered_scores]
            ),
            # Copy other necessary attributes if applicable
            map_info=self.map_info  # Optional: retain any other metadata as needed
        )

        return patch_map
    

    def extract_patch_from_image(self, x_min, y_min, x_max, y_max):
        """
        Extracts a rectangular patch from the original image based on specified coordinates.
        This function replaces `extract_patch` to work with original images instead of tensors.
        """
        # Ensure coordinates are integers and within image boundaries
        x_min = max(0, int(x_min))
        y_min = max(0, int(y_min))
        image = self.image
        x_max = min(image.shape[1], int(x_max))  # Width dimension
        y_max = min(image.shape[0], int(y_max))  # Height dimension

        # Extract the patch from the original image
        patch_image = image[y_min:y_max, x_min:x_max]


        # Optionally, handle keypoints, descriptors, and scores if applicable
        keypoints = self.superpoint_results.keypoints[0]  # Assuming keypoints are available
        descriptors = self.superpoint_results.descriptors[0]
        scores = self.superpoint_results.scores[0]

        # Filter keypoints to be within the patch
        mask = (
            (keypoints[:, 0] >= x_min) & (keypoints[:, 0] < x_max) &
            (keypoints[:, 1] >= y_min) & (keypoints[:, 1] < y_max)
        )
        filtered_keypoints = keypoints[mask]
        filtered_descriptors = descriptors[mask]
        filtered_scores = scores[mask]

        # Adjust keypoints' coordinates relative to the patch
        filtered_keypoints[:, 0] -= x_min
        filtered_keypoints[:, 1] -= y_min

        # Create a new MapDataset object for the patch (assuming you have a constructor for it)
        patch_map = MapDataset(
            map_info=self.map_info,
            image=patch_image,
            mask = self.mask,
            superpoint_results=SuperPointResults(
                keypoints=[filtered_keypoints],
                descriptors=[filtered_descriptors],
                scores=[filtered_scores]
            )
        )

        return patch_map
        
    def plot_keypoints(self):
        """
        Plots the keypoints on the map image using matplotlib, adjusting for transformations if necessary.
        Overlays the mask with transparency over the image. Keypoints are plotted as dots with
        size 5, and their color depends on the score.
        """
        if self.superpoint_results.keypoints is None or not self.image_path:
            print(f"No keypoints or image available for map: {self.map_info.folder}")
            return

        # Load the map image
        image = self.image
        
        if image is None:
            print(f"Failed to load image at: {self.image_path}")
            return

        # Convert image from BGR to RGB for plotting
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.mask is None and self.map_info.mask_path:
            self.mask = cv2.imread(self.map_info.mask_path, cv2.IMREAD_GRAYSCALE)

        mask = self.mask

        # Get the dimensions of the current image
        img_height, img_width, _ = image_rgb.shape

        # Plot the image
        plt.figure(figsize=(10, 10))
        plt.imshow(image_rgb)
        
        # Overlay the mask if available
        if mask is not None:
            # Resize the mask to match the image dimensions if needed
            if (mask.shape[0] != img_height) or (mask.shape[1] != img_width):
                mask = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
            
            # Create an RGBA version of the mask with transparency
            mask_rgba = np.zeros((img_height, img_width, 4), dtype=np.uint8)
            mask_rgba[..., 0] = 255  # Red channel
            mask_rgba[..., 3] = (mask > 0).astype(np.uint8) * 128  # Alpha channel (transparency)

            # Overlay the mask with transparency
            plt.imshow(mask_rgba, alpha=0.5)

        # Plot the keypoints, adjusting for transformations
        keypoints_tensor = self.superpoint_results.keypoints[0]  # Access the first tensor if self.superpoint_results.keypoints is a list
        scores_tensor = self.superpoint_results.scores[0]  # Access the scores if self.superpoint_results.scores is a list

        transform = self.tensor_to_image_transform  # Full affine transformation matrix (2x3)
        
        # Apply the transformation to the keypoints to map them back to the original image space
        keypoint_coords = []
        for kp in keypoints_tensor:
            point = np.array([kp[0].item(), kp[1].item(), 1])  # Convert to homogeneous coordinates
            transformed_point = transform @ point  # Apply the affine transformation
            keypoint_coords.append([transformed_point[0], transformed_point[1]])
        keypoint_coords = np.array(keypoint_coords)

        scores = scores_tensor.cpu().numpy()  # Convert scores to a numpy array

        # Use a color map (e.g., 'viridis') to color the keypoints based on scores
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=np.min(scores), vmax=np.max(scores))  # Normalize scores to [0, 1]

        # Plot each keypoint as a dot with color based on its score
        scatter = plt.scatter(keypoint_coords[:, 0], keypoint_coords[:, 1], s=5, c=scores, cmap=cmap, norm=norm)

        # Add a colorbar to indicate the score range
        plt.colorbar(scatter, label='Keypoint Score')

        plt.title(f"Keypoints on Map: {self.map_info.folder} (Score-Coded)")
        plt.axis('off')  # Hide the axes for a cleaner look
        plt.show()



def create_map_dataset(map_data: Dict[str, Any]) -> MapDataset:
    """
    Creates and initializes a MapDataset instance from a dictionary of map data.
    
    The dictionary is expected to contain keys such as 'folder', 'image_path',
    'mask_path', and 'points'. The 'folder' is assumed to contain both the year
    and author (separated by an underscore). If parts are missing, 'Unknown' is used
    as a default.
    
    Parameters:
        map_data (Dict[str, Any]): Dictionary with map information.
        
    Returns:
        MapDataset: An initialized MapDataset instance.
    """
    # Extract folder and determine the year and author based on a naming convention.
    folder = map_data.get('folder', '')
    #year = folder.split('_')[0]
    #author = folder.split('_')[1]
    
    # Try to convert year to integer if possible, otherwise set to None.
    #try:
    #    year_int = int(year)
    #except ValueError:
    #    logger.warning("Year extraction failed or year is not an integer. Received: %s", year)
    #    year_int = None

    # Extract the image and mask paths
    image_path = map_data.get('image_path', '')
    mask_path = map_data.get('mask_path', '')

    # Extract GCPs (Ground Control Points), defaulting to an empty DataFrame if not provided.
    gcps = map_data.get('points', pd.DataFrame())    
    # Construct the MapInfo object
    map_info = MapInfo(
        image_path=image_path,
        mask_path=mask_path if mask_path else None,
        folder=folder,
        author=None,
        year=None
        #author=None,
        #year=year_int
    )
    
    # Create and return a MapDataset instance
    dataset = MapDataset(
        map_info=map_info,
        gcps=gcps
    )
    #print(f"Created MapDataset for folder: {folder} with points: {len(gcps)}")
    
    
    return dataset
