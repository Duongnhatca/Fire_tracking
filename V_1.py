import cv2
import numpy as np
from adafruit_servokit import ServoKit
import serial
import time
import RPi.GPIO as GPIO
# Tải mô hình đã train
net = cv2.dnn.readNet("F:/Final_Project/yolov4-tiny/yolov4-tiny-custom_final.weights",
                      "F:/Final_Project/yolov4-tiny/yolov4-tiny-custom.cfg")
model = cv2.dnn_DetectionModel(net)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
# Giao tiếp với arduino để nhận dữ liệu từ cảm biến bằng giao thức uart
ser = serial.Serial('/dev/ttyACM0', 9600)
data = ser.readline()
data_1 = data.decode()
data_2 = data_1.rstrip()
print(data_2)
# Khai báo chân cho servo và setup các góc ban đầu
nbPCAServo = 16
pca = ServoKit(channels=16)
pca.servo[0].angle = 90
time.sleep(1)
pca.servo[1].angle = 30
time.sleep(1)
# Mở camera và tiến hành lấy các thông số của khung hình
cap = cv2.VideoCapture(0)
_, frame = cap.read()
rows, cols, _ = frame.shape
x_medium = int(cols / 2)
center_x = int(cols / 2)
y_medium = int(rows/2)
center_y = int(rows/2)
# Setup các điểm ban đầu theo góc servo
position_x = 90  # degrees
position_y = 30  # degrees
# Setup các chân gpio cho rasp
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)
GPIO.setup(22, GPIO.OUT)
GPIO.setup(27, GPIO.IN)
last_position = 0

while True:
    if int(data_2) == 10:
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

       # Move servo motor
        if x_medium < center_x - 30:
            position_x += 1
        elif x_medium > center_x + 30:
            position_x -= 1
        pca.servo[0].angle = position_x
        time.sleep(0.5)
        if y_medium < center_y - 30:
            position_y += 1
        elif y_medium > center_y + 30:
            position_y -= 1
        pca.servo[1].angle = position_y
        time.sleep(0.5)
        
        if x_medium == center_x:
            pca.servo[0].angle = None
            time.sleep(0.5)
        if y_medium == center_y:
            pca.servo[1].angle = None
            time.sleep(0.5)

        print(position_x)
        print(position_y)
        if last_position != position_x:
            last_position = position_x
        else:
            GPIO.output(17, 1)
            if GPIO.input(27) == 0:
                GPIO.output(22, 1)
GPIO.output(17, 0)
GPIO.output(22, 0)
cap.release()
cv2.destroyAllWindows()
