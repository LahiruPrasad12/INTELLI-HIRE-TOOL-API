import cv2
import mediapipe as mp
import numpy as np
import time
import torch
from mediapipe.python import Matrix


class EthicalBenchMarkDetector:
    def __init__(self):
        self.model = self.load_model()
        self.classes = self.model.names
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    def plot_boxes(self, result, frame, looking):
        # Plots bounding boxes and labels on the frame
        labels, cord = result
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                    row[3] * y_shape)
                bgr = (0, 255, 0)
                percentage = round(float(row[4]) * 100, 2)
                label = self.class_to_label(labels[i])
                text = f"{label}: {percentage}%"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                cv2.putText(frame, looking, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame

    # takes an input frame, applies the YOLOv5 model to it, and returns the predicted labels and coordinates of the
    # detected objects.
    def score_frame(self, frame):
        # Scores the frame using the YOLOv5 model
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def load_model(self):
        # Loads the YOLOv5 model
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def class_to_label(self, x):
        # Converts a numeric label to its corresponding string label
        return self.classes[int(x)]

    def process_frame(self, image):
        # Processes a single frame
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.face_mesh.process(image)
        image.flags.writeable = True
        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)
                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]])
                dis_matrix = np.zeros((4, 1), dtype=np.float64)
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dis_matrix)
                rmat, jsc = cv2.Rodrigues(rot_vec)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360
                if y < -5:
                    text = "Looking Left"
                elif y > 5:
                    text = "Looking Right"
                elif x < -5:
                    text = "Looking Down"
                elif x > 5:
                    text = "Looking Up"
                else:
                    text = "Looking forward"
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec
                )
                cv2.line(image, p1, p2, (255, 0, 0), 3)
                results = self.score_frame(image)
                self.plot_boxes(results, image, text)
        return image

    def run(self):
        # Runs the head pose detection on the video capture
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, image = cap.read()
            start = time.time()
            processed_image = self.process_frame(image)
            # cv2.imshow('Ethical Benchmark', processed_image)
            _, buffer = cv2.imencode('.jpg', processed_image)
            frame_bytes = buffer.tobytes()
            # frame_bytes = processed_image.tobytes()
            yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
