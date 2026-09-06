# garbage_detection.py
# A beginner-friendly script that uses YOLOv8 + OpenCV to detect litter/garbage
# objects (bottles, cups, cans, boxes, backpacks, etc.) in a video file,
# draws bounding boxes on each frame, and saves the result as a new video.

# ------------------------------------------------------------------
# STEP 1: Import the libraries we need
# ------------------------------------------------------------------
import cv2                     # OpenCV: for reading/writing video and drawing on frames
from ultralytics import YOLO   # Ultralytics: gives us the YOLOv8 model

# ------------------------------------------------------------------
# STEP 2: Load the pre-trained YOLOv8 model
# ------------------------------------------------------------------
# IMPORTANT: Any "yolov8X.pt" model (n/s/m/l/x) without "-oiv7" is trained on
# COCO, which only has 80 classes. COCO has NO "can" class and NO "plate" class.
# Using a bigger COCO model (like yolov8m.pt) improves accuracy and box quality,
# but it CANNOT fix this - it still only knows the same 80 COCO categories, so
# soda cans will still get labeled "bottle" no matter how large the model is.
#
# The fix is to use a different TRAINING DATASET, not just a bigger model.
# "yolov8m-oiv7.pt" is the medium-sized YOLOv8 model pre-trained on Open Images
# V7, which has 600+ classes, including separate classes for "Plate", "Bowl",
# "Tin can", and "Bottle". This keeps the accuracy benefits of the "m" (medium)
# size while actually knowing what a can looks like.
#
# Ultralytics will automatically download this file the first time you run the
# script if it isn't already present in your project folder (it's larger than
# the nano version, so the first run and each frame may be a bit slower).
model = YOLO("yolov8m-oiv7.pt")

# Uncomment the line below once to see every class name this model knows,
# so you can confirm the exact spelling/capitalization used for each item:
# print(model.names)

# ------------------------------------------------------------------
# STEP 3: Define which classes count as "garbage/litter"
# ------------------------------------------------------------------
# We list the items we care about in lowercase, simple keywords. The matching
# logic below checks each detected class name (lowercased) against this list,
# so it doesn't matter if the model's internal name is "Tin can", "tin can",
# or just "can" - it will still match correctly.
GARBAGE_KEYWORDS = {
    "bottle",
    "can",          # matches "Tin can" / "can" / "Can"
    "cup",
    "plate",        # now actually detectable with the OIV7 model
    "bowl",         # now actually detectable with the OIV7 model
    "backpack",
    "handbag",
    "suitcase",
    "box",
    "waste container",
    "plastic bag",
}
# Set this to False if you'd rather see ALL detected objects instead of just
# this litter/tableware subset.
FILTER_GARBAGE_ONLY = True

# ------------------------------------------------------------------
# STEP 4: Open the input video file
# ------------------------------------------------------------------
input_video_path = "street_video.mp4"
cap = cv2.VideoCapture(input_video_path)

# Check that the video actually opened correctly before continuing.
if not cap.isOpened():
    print(f"Error: Could not open video file '{input_video_path}'.")
    exit()  # Stop the script here if the file can't be opened.

# ------------------------------------------------------------------
# STEP 5: Extract the video's properties (width, height, fps)
# ------------------------------------------------------------------
# We need these to create an output video that matches the input video's size and speed.
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Input video properties -> Width: {frame_width}, Height: {frame_height}, FPS: {fps}")

# ------------------------------------------------------------------
# STEP 6: Set up the VideoWriter to save the processed output video
# ------------------------------------------------------------------
output_video_path = "output_garbage_detected.mp4"

# "mp4v" is a widely compatible codec for saving .mp4 files.
# (You could swap this for "XVID" if you want to save as an .avi file instead.)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# Create the VideoWriter object: it needs the output filename, codec, fps,
# and frame size (width, height) as a tuple.
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# ------------------------------------------------------------------
# STEP 7: Process the video frame-by-frame
# ------------------------------------------------------------------
print("Processing video... Press 'q' in the preview window to stop early.")

while True:
    # Read one frame from the video. "ret" is True if a frame was successfully read.
    ret, frame = cap.read()

    # If ret is False, we've reached the end of the video (or hit an error), so stop.
    if not ret:
        print("Finished processing (end of video reached).")
        break

    # ------------------------------------------------------------------
    # STEP 8: Run YOLOv8 detection on the current frame
    # ------------------------------------------------------------------
    # model(frame) runs the detector and returns a list of "Results" objects.
    # verbose=False stops YOLO from printing detection info to the console every frame.
    results = model(frame, verbose=False)

    # results[0] holds the detections for our single frame.
    # boxes contains one entry per detected object (its coordinates, class, confidence).
    boxes = results[0].boxes

    # Loop through every detected object in this frame.
    for box in boxes:
        # Get the class ID (a number) and convert it to the human-readable class name.
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        # Get the confidence score (how sure the model is), rounded to 2 decimals.
        confidence = float(box.conf[0])

        # If we're filtering for garbage-only objects, check whether any of our
        # keywords appears inside this class name (case-insensitive match).
        # e.g. "Tin can" contains "can", "Plastic bag" contains "plastic bag".
        name_lower = class_name.lower()
        is_garbage_item = any(keyword in name_lower for keyword in GARBAGE_KEYWORDS)

        if FILTER_GARBAGE_ONLY and not is_garbage_item:
            continue

        # Get the bounding box coordinates (top-left and bottom-right corners).
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Draw a rectangle around the detected object on the frame.
        # Color is in BGR format; (0, 255, 0) = green. Thickness = 2 pixels.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Create a text label showing the class name and confidence score.
        label = f"{class_name} {confidence:.2f}"

        # Draw the label text just above the top-left corner of the box.
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 0)),        # Position text slightly above the box (avoid negative y)
            cv2.FONT_HERSHEY_SIMPLEX,     # Font style
            0.6,                          # Font size
            (0, 255, 0),                  # Text color (green, matches the box)
            2                             # Text thickness
        )

    # ------------------------------------------------------------------
    # STEP 9: Write the processed frame to the output video file
    # ------------------------------------------------------------------
    out.write(frame)

    # ------------------------------------------------------------------
    # STEP 10: Show a live preview window of the processing
    # ------------------------------------------------------------------
    cv2.imshow("Garbage Detection - Press 'q' to quit", frame)

    # Wait 1 millisecond for a key press. If the key is 'q', stop the loop early.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Stopped early by user (pressed 'q').")
        break

# ------------------------------------------------------------------
# STEP 11: Clean up - release resources and close windows
# ------------------------------------------------------------------
cap.release()          # Release the input video file
out.release()          # Finish writing and close the output video file
cv2.destroyAllWindows()  # Close any OpenCV preview windows

print(f"Done! Processed video saved as '{output_video_path}'.")
