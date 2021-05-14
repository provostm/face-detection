import cv2
import dlib
from utils_yolo_face import *
import time

cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
print("Frames per second using cap.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('D:/Projets python/Face-Morphing/code/utils/shape_predictor_68_face_landmarks.dat')

yolo_face_model = '../yolo_cfg/yolov3-face.cfg'
yolo_face_weights = '../yolo_model-weights/yolov3-wider_16000.weights'

net = cv2.dnn.readNetFromDarknet(yolo_face_model, yolo_face_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

toggle_landmarks = True
toggle_yolo_face = False

fps_instance = FPS().start()

# Real time FPS: https://www.geeksforgeeks.org/python-displaying-real-time-fps-at-which-webcam-video-file-is-processed-using-opencv/
# used to record the time when we processed last frame
prev_frame_time = 0
# used to record the time at which we processed current frame
new_frame_time = 0

while True:
    ret, frame = cap.read()
    IMG_HEIGHT, IMG_WIDTH, _ = frame.shape

    # time when we finish processing for this frame
    new_frame_time = time.time()

    # Calculating the fps
    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = round(1 / (new_frame_time - prev_frame_time), 2)
    prev_frame_time = new_frame_time

    # Putting the FPS count on the frame
    cv2.putText(frame, str(fps), (IMG_WIDTH - 80, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)

    if toggle_landmarks:
        dets = detector(frame, 1)
        if len(dets) == 0:
            frame = cv2.putText(frame, "no face detected", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                     0.6, (0, 255, 0), 2, cv2.LINE_AA)

        else:
            for k, rect in enumerate(dets):
                shape = predictor(frame, rect)

                for i in range(0, 68):
                    frame = cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), 2)
        frame = cv2.putText(frame, "Face landmarks activated", (IMG_WIDTH - 170, IMG_HEIGHT - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)

    if cv2.waitKey(2) & 0xFF == ord('l'):
        toggle_landmarks = not toggle_landmarks
        print("Toggle landmark detection to " + str(toggle_landmarks))

    if cv2.waitKey(2) & 0xFF == ord('y'):
        toggle_yolo_face = not toggle_yolo_face
        print("Toggle Yolo face detection to " + str(toggle_yolo_face))

    if toggle_yolo_face:
        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                     [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(get_outputs_names(net))

        # Remove the bounding boxes with low confidence
        faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)

        # initialize the set of information we'll displaying on the frame
        info = [
            ('number of faces detected', '{}'.format(len(faces)))
        ]

        for (i, (txt, val)) in enumerate(info):
            text = '{}: {}'.format(txt, val)
            cv2.putText(frame, text, (10, (i * 20) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GREEN, 2)
        frame = cv2.putText(frame, "Yolo face activated", (IMG_WIDTH - 127, IMG_HEIGHT - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
    fps_instance.update()
    cv2.imshow("webcam", frame)

    if (cv2.waitKey(2) & 0xFF == ord('q')) or cv2.waitKey(2) == 27:
        break
fps_instance.stop()
print("[INFO] elasped time: {:.2f}".format(fps_instance.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps_instance.fps()))

cap.release()
cv2.destroyAllWindows()

