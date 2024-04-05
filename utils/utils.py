from pytube import YouTube
import os

import cv2
import numpy as np

def download_video(video_id, output_path="."):
    try:
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
        video = yt.streams.filter(file_extension='mp4', resolution='360p').first()
        video.download(output_path)
        print(f"Video downloaded successfully: {yt.title}")
    except Exception as e:
        print(f"Error downloading video: {e}")


def get_frames_from_video(video_path):

    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # Lists to store preprocessed frames
    frames = []

    # Iterate through the frames
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Check if the video has ended
        if not ret:
            break

        # Append the preprocessed frame to the list
        frames.append(frame)
        break

    # Release the VideoCapture object
    cap.release()

    # Returnlist of frames converted to a NumPy array
    return np.array(frames)