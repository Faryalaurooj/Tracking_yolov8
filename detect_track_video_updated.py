import cv2
import time
import warnings
from ultralytics import YOLO
from collections import defaultdict
import os

# Suppress specific FutureWarning
warnings.simplefilter("ignore", category=FutureWarning)

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Ask user for the input video file
input_video_path = input("Please enter the input video file path: ")

# Open the video file
cap = cv2.VideoCapture(input_video_path)

# Check if the video was successfully opened
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()  # Exit the program if the video is not found or cannot be opened

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

# Generate a unique filename using timestamp
output_video_path = f"output/annotated_video_{int(time.time())}.mp4"

# Initialize the VideoWriter to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

# Initialize variables for calculating FPS
prev_time = 0
fps = 0

# Initialize dictionaries to store performance metrics
object_tracks = defaultdict(list)  # Store the object IDs for MOTA/IDF1

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    
    if not success:
        print("Error: Failed to read frame.")
        break  # Exit the loop if a frame cannot be read

    # Start timing for FPS calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display FPS on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 255, 0)  # Green color
    thickness = 2
    position = (10, 30)  # Position of the text on the frame

    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", position, font, font_scale, font_color, thickness)

    # Save the annotated frame to the output video
    out.write(annotated_frame)

    # Display the annotated frame with FPS
    cv2.imshow("YOLOv8 Tracking", annotated_frame)

    # Record object tracking IDs for performance metrics
    for track in results[0].boxes:
        # Check if track.id is not None before processing it
        if track.id is not None:
            track_id = int(track.id)  # Get object ID from YOLO tracking results
            object_tracks[track_id].append(track.xyxy.cpu().numpy())  # Store the bounding box coordinates

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Print final message after processing
print(f"Output video saved as {output_video_path}")

