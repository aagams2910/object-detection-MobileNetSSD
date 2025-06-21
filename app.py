import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from datetime import datetime

# Streamlit app config
st.set_page_config(page_title="Object Detection & Counting", layout="centered")
st.title("🔍 Object Detection & Counting (MobileNet SSD + OpenCV)")

# Sidebar settings
st.sidebar.header("⚙️ Settings")
media_type = st.sidebar.radio("Select input type:", ["Image", "Video"])
available_classes = ["person", "car", "bus", "bicycle", "motorbike", "truck", "dog", "cat"]
selected_classes = st.sidebar.multiselect("Select object classes to detect", available_classes, default=["person", "car"])

# Predefined parameters (fixed)
MIN_AREA = 100               # Minimum object area for filtering
CONF_THRESHOLD = 0.10        # Confidence threshold for detections
NMS_THRESHOLD = 0.10         # Non-maximum suppression threshold
BLOB_SIZE = 600              # Blob input size for DNN

uploaded_file = st.file_uploader("Upload Image/Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

# Model files
prototxt_path = "MobileNetSSD_deploy.prototxt"
caffemodel_path = "MobileNetSSD_deploy.caffemodel"
if not os.path.isfile(prototxt_path) or not os.path.isfile(caffemodel_path):
    st.error("❌ Model files not found. Ensure prototxt and caffemodel are in current directory.")
    st.stop()

# Load MobileNet SSD
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]
selected_class_ids = {cls: CLASSES.index(cls) for cls in selected_classes if cls in CLASSES}

# Detection helper
def detect_objects(frame, conf_thresh, nms_thresh, class_ids, blob_size):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (blob_size, blob_size)), 0.007843, (blob_size, blob_size), 127.5)
    net.setInput(blob)
    detections = net.forward()
    boxes, confidences, labels = [], [], []
    for i in range(detections.shape[2]):
        score = float(detections[0, 0, i, 2])
        cid = int(detections[0, 0, i, 1])
        if cid in class_ids.values() and score >= conf_thresh:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype('int')
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(score)
            labels.append(cid)
    if not boxes:
        return []
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)
    results = []
    for idx in indices:
        i = idx[0] if isinstance(idx, (list, tuple, np.ndarray)) else idx
        results.append((boxes[i], labels[i], confidences[i]))
    return results

# Main logic
if uploaded_file:
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(uploaded_file.read())
    path = tmp.name

    if media_type == "Image":
        img = cv2.imread(path)
        dets = detect_objects(img, CONF_THRESHOLD, NMS_THRESHOLD, selected_class_ids, BLOB_SIZE)
        counts = {cls: 0 for cls in selected_class_ids}
        for (box, cid, conf) in dets:
            x, y, w, h = box
            if w * h >= MIN_AREA:
                label = CLASSES[cid]
                counts[label] += 1
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(img, f"{label}:{counts[label]} ({conf:.2f})", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        st.subheader("Detected Output")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB")
        st.success(f"Counts: {counts}")
    else:
        cap = cv2.VideoCapture(path)
        w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        out_file = f"output_{datetime.now():%Y%m%d_%H%M%S}.avi"
        writer = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'XVID'), fps, (w_frame, h_frame))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            dets = detect_objects(frame, CONF_THRESHOLD, NMS_THRESHOLD, selected_class_ids, BLOB_SIZE)
            for (box, cid, conf) in dets:
                x, y, w, h = box
                if w * h >= MIN_AREA:
                    label = CLASSES[cid]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, f"{label} ({conf:.2f})", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            writer.write(frame)
        cap.release(), writer.release()

        st.subheader("Processed Video Output")
        st.video(open(out_file, 'rb').read())
        st.download_button("Download Video", data=open(out_file, 'rb').read(), file_name="processed.avi")
        os.remove(path); os.remove(out_file)