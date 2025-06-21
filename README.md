# Object Detection & Counting Streamlit App (MobileNet SSD + OpenCV)

This is a Streamlit application for real-time object detection and counting using a pre-trained MobileNet SSD deep learning model with OpenCV's DNN module. It supports both images and videos.

## 🚀 Features

- Image Upload:
  - Detects objects (e.g., person, car, dog) using MobileNet SSD.
  - Displays bounding boxes, class names, confidence scores, and per-class counts.

- Video Upload:
  - Frame-by-frame object detection using MobileNet SSD.
  - Annotates video with bounding boxes and labels.
  - Allows video download after processing.

- Predefined Settings:
  - Minimum object area filtering.
  - Confidence threshold: 0.10
  - Non-Max Suppression (NMS) threshold: 0.10
  - Blob input size: 600

## 🧠 Deep Learning Model Used

- Model: MobileNet SSD (Single Shot MultiBox Detector)
- Framework: Caffe
- Model Files Required:
  - MobileNetSSD_deploy.prototxt
  - MobileNetSSD_deploy.caffemodel

⚠️ Place both files in the same directory as `app.py`.

## ⚙️ How to Run

1. Install dependencies:
   `pip install -r requirements.txt`

2. Run the Streamlit app:
   `streamlit run app.py`

3. Open the app in your browser (usually at http://localhost:8501).

## 📂 Supported File Formats

- Images: .jpg, .jpeg, .png
- Videos: .mp4, .avi, .mov

## 🧪 Usage Instructions

1. Choose between Image or Video in the sidebar.
2. Select which object classes to detect (e.g., person, car).
3. Upload your media file.
4. View results:
   - For images: annotated image and class-wise count.
   - For videos: downloadable annotated video.

## 📌 Notes

- This app uses Deep Learning via OpenCV’s DNN module.
- Model is pre-trained; no training is required.
- Does not use classical methods like HOG, background subtraction, or trackers.
