from SuperGluePretrainedNetwork.models.superpoint import SuperPoint
import torch

# Define configuration dictionaries for SuperPoint and SuperGlue
superpoint_config = {
    'descriptor_dim': 256,
    'nms_radius': 4,
    'keypoint_threshold': 0.01,
    'max_keypoints': 2048
}

# Initialize and load the SuperPoint model with configuration
superpoint = SuperPoint(superpoint_config)
superpoint.load_state_dict(torch.load('./SuperPointPretrainedNetwork/superpoint_v1.pth'))
superpoint.eval()  # Set to evaluation mode


def run_superpoint_inference(tensor):
    """
    Runs SuperPoint inference on the given tensor file path and returns
    keypoints, descriptors, and scores.

    Parameters:
    - tensor_path: Path to the image tensor file.

    Returns:
    - keypoints, descriptors, scores: Results from SuperPoint.
    """
    # Load the image tensor
    image_tensor = tensor
    
    # Run SuperPoint inference
    with torch.no_grad():
        sp_data = {'image': image_tensor}
        output = superpoint(sp_data)
        
    # Extract the results
    keypoints = output['keypoints']
    descriptors = output['descriptors']
    scores = output['scores']
    
    return keypoints, descriptors, scores