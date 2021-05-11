import cv2
import dlib
from utils_yolo_face import *

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('D:/Projets python/Face-Morphing/code/utils/shape_predictor_68_face_landmarks.dat')

yolo_face_model = 'C:/Users/prm/PycharmProjects/faceidentity/yolo_cfg/yolov3-face.cfg'
yolo_face_weights = 'C:/Users/prm/PycharmProjects/faceidentity/yolo_model-weights/yolov3-wider_16000.weights'

net = cv2.dnn.readNetFromDarknet(yolo_face_model, yolo_face_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


toggle_landmarks = True
toggle_yolo_face = True

while True:
    ret, frame = cap.read()

    if toggle_landmarks:
        dets = detector(frame, 1)
        if len(dets) == 0:
            frame = cv2.putText(frame, "no face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                     0.6, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            for k, rect in enumerate(dets):
                shape = predictor(frame, rect)

                for i in range(0, 68):
                    frame = cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), 2)


    if cv2.waitKey(10) & 0xFF == ord('l'):
        print("Toggle landmark detection to " + str(toggle_landmarks))
        toggle_landmarks = not toggle_landmarks

    if cv2.waitKey(10) & 0xFF == ord('y'):
        print("Toggle to Yolo face detection")

        toggle_yolo_face = not toggle_yolo_face

    if toggle_yolo_face:
        # Create a 4D blob from a frame.
        IMG_HEIGHT, IMG_WIDTH , _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                     [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(get_outputs_names(net))

        # Remove the bounding boxes with low confidence
        faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
        # print('[i] ==> # detected faces: {}'.format(len(faces)))
        # print('#' * 60)

        # initialize the set of information we'll displaying on the frame
        info = [
            ('number of faces detected', '{}'.format(len(faces)))
        ]

        for (i, (txt, val)) in enumerate(info):
            text = '{}: {}'.format(txt, val)
            cv2.putText(frame, text, (10, (i * 20) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)


    cv2.imshow("webcam", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

