import os
import cv2
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor

def extract_frames_and_save_masks(input_video_file, start_time, end_time, output_folder):
    """
    Extract frames from a video within the specified frame range and save them as images.

    Parameters:
    input_video_file (str): Path to the input video file.
    start_time (int): Starting time in seconds of the clip
    end_time (int): Ending time in seconds of the clip
    output_folder (str): Path to the folder where extracted masks will be saved.

    Returns:
    None
    """
    try:
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        #replace the path with your local model checkpoint file
        sam_p = sam_model_registry["vit_h"](checkpoint='/Users/rishabh_raj/Downloads/processing_folder/Grad/NYU_Masters_Computer_Science/spring-2024/csgy-6943/Project/sam_vit_h_4b8939.pth')
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        sam_p.to(device=device)
        predictor = SamPredictor(sam_p)
        # Initialize video capture
        cap = cv2.VideoCapture(input_video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Iterate through frames and save them as images
        frame_count = 0
        while cap.isOpened() and frame_count <= end_frame - start_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # Crop the frame(comment the line if the frame is already cropped)
            frame = frame[0:720, 250:1050]
            predictor.set_image(frame)

            # Define input box for mask generation (example)
            input_box = np.array([[20], [200], [300], [480]])

            # Perform prediction to get masks
            masks, _, _ = predictor.predict(point_coords=None, point_labels=None, box=input_box[None, :], multimask_output=False)

            # Scale the binary masks to 0-255 (for 8-bit grayscale image)
            mask_data = (masks * 255).astype(np.uint8)

            # Convert mask data to a PIL image
            mask_image = Image.fromarray(mask_data[0], mode='L')  # 'L' mode for grayscale image

            # Save the mask image to the output folder
            output_file = os.path.join(output_folder, f"frame_{str(frame_count).zfill(4)}.jpg")
            mask_image.save(output_file)

            frame_count += 1

        # Release video capture
        cap.release()

    except Exception as e:
        print(f"Error extracting frames: {e}")

