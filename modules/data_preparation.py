import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoImageProcessor 
# Load the AutoImageProcessor for SuperPoint
processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")


def extract_epsg(crs_info_str):
    try:
        # Search for EPSG code in the crs_info string
        epsg_code_start = crs_info_str.find('ID["EPSG".6')
        epsg_code_end = crs_info_str.find(']]', epsg_code_start)
        epsg_code = crs_info_str[epsg_code_start:epsg_code_end].split("'")[2].strip()
        return epsg_code
    except Exception as e:
        print(f"Error extracting EPSG code: {e}, {crs_info_str}")
        return '28193'


def preprocess_image(image, mask=None, north_rotation_angle=None, plot_steps=False):
    """
    Function to preprocess an image with optional rotation based on the north direction,
    masking, resizing, normalization, and cropping. Integrates AutoImageProcessor for
    final image tensor preparation and returns the image tensor along with transformation matrices.

    Parameters:
    - image: The input image (grayscale or BGR as a numpy array).
    - mask: The corresponding mask (grayscale or BGR; black pixels indicate areas of interest).
    - north_rotation_angle: The angle (in radians) by which to rotate the image clockwise, if available.
    - plot_steps: Boolean flag to indicate if intermediate steps should be plotted.

    Returns:
    - image_tensor: Preprocessed image as a PyTorch tensor.
    - transformation_matrix: 3x3 affine transformation matrix for mapping points from original
                             image space to tensor space.
    - tensor_to_image_transform: 3x3 affine transformation matrix for mapping points from tensor
                                 space back to original image space.
    """

    initial_image = image.copy()
    def display_image(img, title="Image", cmap=None):
        if plot_steps:
            plt.figure(figsize=(6,6))
            plt.imshow(img, cmap=cmap)
            plt.title(title)
            plt.axis('off')
            plt.show()

    def display_point(img, point, title="Image with Point", color='red'):
        if plot_steps:
            plt.figure(figsize=(6,6))
            if img.ndim == 2:
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(img)
            plt.scatter(point[0], point[1], c=color, s=50)
            plt.title(title)
            plt.axis('off')
            plt.show()

    def invert_mask(mask):
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, inverted_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY_INV)
        return inverted_mask

    def rotate_image_and_mask(image, mask, angle):
        """
        Rotates the image and mask by the given angle without cropping by adjusting the canvas size.

        Parameters:
        - image: The input image (grayscale or BGR).
        - mask: The corresponding mask (grayscale or BGR).
        - angle: The rotation angle in radians. Positive values rotate the image clockwise.

        Returns:
        - rotated_image: The rotated image with adjusted canvas size.
        - rotated_mask: The rotated mask with adjusted canvas size.
        - rotation_transform: The 3x3 rotation and translation transformation matrix.
        """
        h, w = image.shape[:2]
        
        # Convert angle from radians to degrees and invert for clockwise rotation
        angle_degrees = -np.degrees(angle)
        
        # Compute the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle_degrees, 1.0)
        
        # Calculate the sine and cosine of rotation angle
        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])
        
        # Compute the new width and height bounds
        new_w = int(h * abs_sin + w * abs_cos)
        new_h = int(h * abs_cos + w * abs_sin)
        
        # Adjust the rotation matrix to account for translation
        rotation_matrix[0, 2] += (new_w / 2) - w / 2
        rotation_matrix[1, 2] += (new_h / 2) - h / 2
        
        # Perform the rotation with the new bounds
        rotated_image = cv2.warpAffine(
            image, rotation_matrix, (new_w, new_h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        rotated_mask = cv2.warpAffine(
            mask, rotation_matrix, (new_w, new_h),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        
        # Convert the 2x3 rotation matrix to a 3x3 matrix for homogeneous coordinates
        rotation_transform = np.vstack([rotation_matrix, [0, 0, 1]])
        
        return rotated_image, rotated_mask, rotation_transform

    def apply_mask(image, mask):
        # Updated apply_mask function
        if image.ndim == 2:
            # Grayscale image
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 3:
            # Color image
            if mask.ndim == 2:
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            raise ValueError("Unsupported image format!")
        
        # Ensure both image and mask have the same size
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"Image and mask sizes do not match: {image.shape[:2]} vs {mask.shape[:2]}")
        
        return cv2.bitwise_and(image, mask)

    def find_bounding_box(mask):
        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            return x, y, w, h
        else:
            return None

    def crop_image(image, x, y, w, h):
        return image[y:y+h, x:x+w]

    def process_image_with_auto_processor(image):
        cropped_image = image.astype(np.float32)
        if cropped_image.ndim == 2:
            cropped_image = cropped_image[:, :, np.newaxis]
        # Repeat channels to make it 3-channel if needed
        if cropped_image.shape[2] == 1:
            cropped_image = np.repeat(cropped_image, 3, axis=2)
        # Normalize image if required by the processor
        processed_inputs = processor(images=cropped_image, return_tensors="pt")
        processed_image = processed_inputs.pixel_values
        processed_image = processed_image.mean(dim=1, keepdim=True)  # Convert to single channel if necessary
        return processed_image

    # Initial display
    if image.ndim == 3:
        display_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), "Original Image")
    else:
        display_image(image, "Original Image")
    
    if mask is not None:
        if mask.ndim == 3:
            display_image(mask, "Original Mask")
        else:
            display_image(mask, "Original Mask", cmap='gray')

    if mask is None:
        if image.ndim == 2:
            mask = np.ones(image.shape, dtype=np.uint8) * 255
        else:
            mask = np.ones(image.shape[:2], dtype=np.uint8) * 255  # Single-channel mask

    mask = invert_mask(mask)
    display_image(mask, "Inverted Mask", cmap='gray')

    # Initialize the transformation matrix as identity
    transformation_matrix = np.eye(3, dtype=np.float32)

    # Plot a random initial point in the original image
    h, w = image.shape[:2]
    random_point_image = np.array([np.random.randint(0, w), np.random.randint(0, h), 1])  # Homogeneous coordinates
    display_point(image, random_point_image[:2], "Random Point in Original Image", color='red')

    if north_rotation_angle is not None:
        image, mask, rotation_transform = rotate_image_and_mask(image, mask, north_rotation_angle)
        if image.ndim == 3:
            display_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), "Rotated Image (Adjusted Size)")
        else:
            display_image(image, "Rotated Image (Adjusted Size)")
        if mask.ndim == 3:
            display_image(mask, "Rotated Mask (Adjusted Size)")
        else:
            display_image(mask, "Rotated Mask (Adjusted Size)", cmap='gray')
        transformation_matrix = rotation_transform @ transformation_matrix

        # Plot the transformed point after rotation
        transformed_point = rotation_transform @ random_point_image
        #transformed_point /= transformed_point[2]  # Normalize
        display_point(image, transformed_point[:2], "Random Point after Rotation", color='blue')
    else: 
        transformed_point = random_point_image

    # Apply mask
    image = apply_mask(image, mask)

    # Find bounding box
    bbox = find_bounding_box(mask)
    if bbox is not None:
        x, y, w_bbox, h_bbox = bbox
        if image.ndim == 3:
            image_with_bbox = image.copy()
            cv2.rectangle(image_with_bbox, (x, y), (x+w_bbox, y+h_bbox), (0, 255, 0), 2)
            display_image(cv2.cvtColor(image_with_bbox, cv2.COLOR_BGR2RGB), "Bounding Box on Image")
        else:
            image_with_bbox = image.copy()
            cv2.rectangle(image_with_bbox, (x, y), (x+w_bbox, y+h_bbox), (255), 2)
            display_image(image_with_bbox, "Bounding Box on Image", cmap='gray')
    else:
        return torch.zeros(1, 1, 256, 256), np.eye(3), np.eye(3)  # Changed to 3x3 identity

    # Crop image
    cropped_image = crop_image(image, x, y, w_bbox, h_bbox)
    if cropped_image.ndim == 3:
        display_image(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB), "Cropped Image")
    else:
        display_image(cropped_image, "Cropped Image", cmap='gray')

    # Translation matrix for cropping
    translation_matrix = np.array([
        [1, 0, -x],
        [0, 1, -y],
        [0, 0, 1]
    ], dtype=np.float32)
    transformation_matrix = translation_matrix @ transformation_matrix

    # Plot the transformed point after cropping
    transformed_point = translation_matrix @ transformed_point
    display_point(cropped_image, transformed_point[:2], "Random Point after Cropping", color='green')

    # Process image with AutoImageProcessor
    processed_image = process_image_with_auto_processor(cropped_image)
    image_tensor = processed_image

    # Scaling
    old_h, old_w = cropped_image.shape[:2]
    max_h = processed_image.shape[2]
    max_w = processed_image.shape[3]
    new_scale_x = max_w / old_w
    new_scale_y = max_h / old_h

    scaling_matrix = np.array([
        [new_scale_x, 0, 0],
        [0, new_scale_y, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    transformation_matrix = scaling_matrix @ transformation_matrix

    # Plot the transformed point after scaling
    transformed_point = scaling_matrix @ transformed_point
    tensor_img = image_tensor.squeeze().cpu().numpy()
    display_point(tensor_img, transformed_point[:2], "Random Point after Scaling", color='orange')

    # Define image-to-tensor transformation matrix by using transformation_matrix directly
    image_to_tensor_transform = transformation_matrix.copy()

    # Compute tensor_to_image_transform as the inverse of image_to_tensor_transform
    try:
        tensor_to_image_transform = np.linalg.inv(image_to_tensor_transform)
    except np.linalg.LinAlgError:
        raise ValueError("Transformation matrix is singular and cannot be inverted.")

    # Final validation: Transform a random point from image to tensor space and back
    transformed_to_tensor = image_to_tensor_transform @ random_point_image
    display_point(tensor_img, transformed_to_tensor[:2], "Point Transformed to Tensor Space", color='cyan')

    # Transform it back to image space using tensor_to_image_transform
    transformed_back_to_image = tensor_to_image_transform @ transformed_to_tensor
    display_point(initial_image, transformed_back_to_image[:2], "Point Transformed Back to Image Space", color='magenta')
    image_to_tensor_transform = transformation_matrix
    return image_tensor, image_to_tensor_transform, tensor_to_image_transform


def validate_transformations(image, image_tensor, image_to_tensor_transform, tensor_to_image_transform, plot_steps=False):
    """
    Function to validate the transformation matrices by plotting a point in the tensor space
    and transforming it back to the image space and vice versa.

    Parameters:
    - image: The original image after preprocessing (numpy array).
    - image_tensor: The processed image tensor (PyTorch tensor).
    - tensor_to_image_transform: The inverse transformation matrix (3x3 numpy array).
    - image_to_tensor_transform: The transformation matrix (3x3 numpy array).
    - plot_steps: Boolean flag to indicate if intermediate steps should be plotted.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    def display_point(image, point, title="Image with Point", color='red'):
        plt.figure(figsize=(6,6))
        if image.ndim == 2:
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(image)
        plt.scatter(point[0], point[1], c=color, s=50)
        plt.title(title)
        plt.axis('off')
        plt.show()

    # Ensure image_tensor is detached and converted to numpy
    image_tensor_np = image_tensor.squeeze().cpu().numpy()

    # Step 1: Plot a random point in the tensor space and transform it to image space
    tensor_h, tensor_w = image_tensor_np.shape[:2]
    random_point_tensor = np.array([np.random.randint(0, tensor_w), np.random.randint(0, tensor_h), 1])  # Homogeneous coordinates
    transformed_point_image_space = tensor_to_image_transform @ random_point_tensor
    transformed_point_image_space /= transformed_point_image_space[2]

    if plot_steps:
        # Plot the point in the tensor space
        display_point(image_tensor_np, random_point_tensor[:2], "Point in Tensor Space", color='blue')
        # Plot the corresponding point in the original image space
        display_point(image, transformed_point_image_space[:2], "Transformed Point in Image Space", color='green')

    # Step 2: Transform a point from image space to tensor space
    image_h, image_w = image.shape[:2]
    random_point_image = np.array([np.random.randint(0, image_w), np.random.randint(0, image_h), 1])  # Homogeneous coordinates
    transformed_point_tensor_space = image_to_tensor_transform @ random_point_image
    transformed_point_tensor_space /= transformed_point_tensor_space[2]

    if plot_steps:
        # Plot the point in the original image space
        display_point(image, random_point_image[:2], "Point in Image Space", color='red')
        # Plot the corresponding point in the tensor space
        display_point(image_tensor_np, transformed_point_tensor_space[:2], "Transformed Point in Tensor Space", color='orange')

    # Verification: Check if transformations are inverses
    identity_matrix = image_to_tensor_transform @ tensor_to_image_transform
    if not np.allclose(identity_matrix, np.eye(3), atol=1e-6):
        print("Warning: image_to_tensor_transform and tensor_to_image_transform are not exact inverses.")
        print("image_to_tensor_transform @ tensor_to_image_transform =")
        print(identity_matrix)
    else:
        print("Success: image_to_tensor_transform and tensor_to_image_transform are inverses.")
