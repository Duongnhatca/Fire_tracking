import cv2

net = cv2.dnn.readNet("F:/Final_Project/yolov4-tiny/yolov4-tiny-custom_final.weights",
                      "F:/Final_Project/yolov4-tiny/yolov4-tiny-custom.cfg")
model = cv2.dnn_DetectionModel(net)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    rows, cols, *_ = frame.shape
    x_medium = int(cols / 2)
    center_x = int(cols / 2)
    y_medium = int(rows/2)
    center_y = int(rows/2)
    (class_ids, scores, bboxes) = model.detect(frame)
    for class_id, score, bbox in zip(class_ids, scores,bboxes):
        (x,y,w,h)=bbox
        x_medium = int((x + x + w) / 2)
        y_medium = int((y + y + h) / 2)
        break
    cv2.line(frame, (x_medium, 0), (x_medium, rows), (0, 255, 0), 2)
    cv2.line(frame, (0, y_medium), (cols, y_medium), (0, 255, 0), 2)
    cv2.imshow("frame",frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()




