# Litter & Garbage Detection with YOLOv8

A Python script that scans a video for common litter and garbage items — bottles, cans, cups, plates, bowls, bags, and more — draws bounding boxes with labels and confidence scores, and saves the annotated result as a new video.

Built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) and [OpenCV](https://opencv.org/).

## Features

- 🎥 Reads any input video and preserves its resolution and frame rate in the output
- 🧠 Uses a YOLOv8 model pre-trained on the **Open Images V7** dataset (600+ classes) instead of the standard COCO model (80 classes), so it can actually recognize items like tin cans and plates that COCO-trained models can't
- 🏷️ Draws bounding boxes, class labels, and confidence scores on every frame
- 🔍 Filters detections down to a configurable list of litter/tableware keywords (or can show all detected objects)
- 👀 Live preview window while processing, with a "press Q to quit" early-exit option
- 💾 Saves the fully annotated video to disk

## Why Open Images V7 instead of standard COCO?

Most YOLOv8 tutorials use `yolov8n.pt` / `yolov8m.pt`, which are trained on the COCO dataset. COCO only has 80 object classes and is missing categories like **can** and **plate** entirely — so a COCO model will always mislabel a soda can as a "bottle," no matter how large the model is. Bumping up model size (n → s → m → l → x) improves accuracy, but it can't add categories the model was never taught.

This project instead uses the `-oiv7` model variants, which are the same YOLOv8 architectures pre-trained on **Open Images V7** — a dataset with 600+ classes, including distinct classes for `Tin can`, `Plate`, `Bowl`, and `Bottle`. This lets the detector actually tell these items apart.

## Requirements

```bash
pip install ultralytics opencv-python
```

## Usage

1. Place your input video in the project folder and name it `street_video.mp4` (or update `input_video_path` in the script).
2. Run the script:

```bash
python garbage_detection.py
```

3. Watch the live preview window as it processes (press **Q** to stop early).
4. Find the annotated output at `output_garbage_detected.mp4` once processing finishes.

> **Note:** The first run will automatically download the YOLOv8 model weights (`yolov8m-oiv7.pt`), so make sure you have an internet connection.

## Configuration

At the top of the script you can tweak:

| Setting | Description |
|---|---|
| `model = YOLO("yolov8m-oiv7.pt")` | Swap for `yolov8n-oiv7.pt` (faster, less accurate) or `yolov8l-oiv7.pt` / `yolov8x-oiv7.pt` (slower, more accurate) |
| `GARBAGE_KEYWORDS` | The set of keywords used to filter detections (matched case-insensitively as substrings, e.g. `"can"` matches `"Tin can"`) |
| `FILTER_GARBAGE_ONLY` | Set to `False` to draw boxes on **all** detected objects instead of just the litter/tableware subset |

## Example class keywords included by default

`bottle`, `can`, `cup`, `plate`, `bowl`, `backpack`, `handbag`, `suitcase`, `box`, `waste container`, `plastic bag`

## How it works

1. Loads the YOLOv8 model and opens the input video with OpenCV.
2. Reads the video's width, height, and FPS, and sets up a matching `VideoWriter`.
3. Loops through the video frame-by-frame, running YOLOv8 inference on each frame.
4. For each detected object, checks whether its class name matches one of the configured keywords.
5. Draws a bounding box and a `class_name confidence` label for each matching detection.
6. Writes the annotated frame to the output video and shows it in a live preview window.
7. Releases all resources and closes windows when the video ends (or when you press Q).

## License

Feel free to use and modify this project for personal or educational purposes.
