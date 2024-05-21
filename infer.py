import cv2
import time
import yaml
import torch
from openvino.runtime import Core
# https://github.com/zhiqwang/yolov5-rt-stack
from yolort.v5 import non_max_suppression, scale_coords

# Load COCO Label from yolov5/data/coco.yaml
with open('./data/coco.yaml', 'r', encoding='utf-8') as f:
    result = yaml.load(f.read(), Loader=yaml.FullLoader)
class_list = result['names']

# Step1: Create OpenVINO Runtime Core
core = Core()

# Step2: Compile the Model, using dGPU
net = core.compile_model("yolov5_openvino/yolov5s.xml", "GPU.1")
output_node = net.outputs[0]

# Color palette
colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

# Import the letterbox for preprocess the frame
from utils.augmentations import letterbox

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    start = time.time()  # total execution time = preprocess + infer + postprocess

    # Preprocess frame by letterbox
    letterbox_img, _, _ = letterbox(frame, auto=False)

    # Normalization + Swap RB + Layout from HWC to NCHW
    blob = cv2.dnn.blobFromImage(letterbox_img, 1/255.0, swapRB=True)

    # Step 3: Do the inference
    outs = torch.tensor(net([blob])[output_node])

    # Postprocess of YOLOv5: NMS
    dets = non_max_suppression(outs)[0].numpy()
    bboxes, scores, class_ids = dets[:, :4], dets[:, 4], dets[:, 5]

    # Rescale the coordinates
    bboxes = scale_coords(letterbox_img.shape[:-1], bboxes, frame.shape[:-1]).astype(int)
    end = time.time()

    # Show bbox
    for bbox, score, class_id in zip(bboxes, scores, class_ids):
        if int(class_id) < len(class_list):
            color = colors[int(class_id) % len(colors)]
            # Draw bounding box
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            # Draw filled rectangle for class label
            cv2.rectangle(frame, (bbox[0], bbox[1] - 20), (bbox[2], bbox[1]), color, -1)
            # Add class label
            cv2.putText(frame, class_list[int(class_id)], (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
            # Calculate and draw center coordinates
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)
            cv2.circle(frame, (center_x, center_y), 3, color, -1)
            # Add text with center coordinates
            cv2.putText(frame, f"({center_x}, {center_y})", (center_x + 5, center_y - 5), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))

    # Show FPS
    fps = (1 / (end - start))
    fps_label = "FPS: %.2f" % fps
    cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    print(fps_label + "; Detections: " + str(len(class_ids)))

    # Display the output frame
    cv2.imshow("output", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
