import cv2
from camera_utils import draw_bounding_box

def load_face_detection_model(face_proto, face_model):
    """Загрузка модели детектирования лиц"""
    net = cv2.dnn.readNet(face_model, face_proto)
    return net

def detect_faces(net, frame, conf_threshold=0.7):
    """Детектирование лиц на кадре"""
    frame_opencv_dnn = frame.copy()
    frame_height = frame_opencv_dnn.shape[0]
    frame_width = frame_opencv_dnn.shape[1]

    blob = cv2.dnn.blobFromImage(frame_opencv_dnn, 1.0, (300, 300), 
                               [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    face_boxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            w, h = x2 - x1, y2 - y1
            face_boxes.append([x1, y1, x2, y2])
            
            draw_bounding_box(frame_opencv_dnn, (x1, y1, w, h), f"{confidence:.2f}", (0, 255, 0))

    return frame_opencv_dnn, face_boxes
