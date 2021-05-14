from threading import Thread, Lock
import cv2
import dlib
from utils_yolo_face import *


def face_detection(frame, detector, predictor, net):
    global toggle_landmarks
    global toggle_yolo_face
    global exit_bool
    IMG_HEIGHT, IMG_WIDTH, _ = frame.shape

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

    return frame


def toggle_buttons_function(interval=1):
    global toggle_landmarks
    global toggle_yolo_face
    global exit_bool
    # print("thread buttons is running")

    if cv2.waitKey(interval) & 0xFF == ord('l'):
        toggle_landmarks = not toggle_landmarks
        print("Toggle landmark detection to " + str(toggle_landmarks))

    if cv2.waitKey(interval) & 0xFF == ord('y'):
        toggle_yolo_face = not toggle_yolo_face
        print("Toggle Yolo face detection to " + str(toggle_yolo_face))

    if cv2.waitKey(interval) & 0xFF == ord('q'):
        exit_bool = True
        print("Toggle exit to " + str(exit_bool))

    # return toggle_landmarks, toggle_yolo_face, exit_bool


class WebcamVideoStream:
    """""
    Based on https://gist.github.com/allskyee/7749b9318e914ca45eb0a1000a81bf56
    to implement the multi-thread with cv2.imshow()
    """""

    def __init__(self, src=0, width=640, height=480):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
             'D:/Projets python/Face-Morphing/code/utils/shape_predictor_68_face_landmarks.dat')

        yolo_face_model = '../yolo_cfg/yolov3-face.cfg'
        yolo_face_weights = '../yolo_model-weights/yolov3-wider_16000.weights'
        net = cv2.dnn.readNetFromDarknet(yolo_face_model, yolo_face_weights)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.net = net

        self.read_lock = Lock()

    def start(self):
        if self.started:
            print("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            (grabbed, frame) = self.stream.read()
            frame = face_detection(frame, self.detector, self.predictor, self.net)
            frame = cv2.resize(frame, (960, 720), interpolation=cv2.INTER_AREA)
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self):
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self):
        print("STOP: release cam")
        self.started = False
        self.stream.release()
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback):
        print("release cam")
        self.stream.release()


class ToggleButtonStream:
    def __init__(self):
        global toggle_landmarks
        global toggle_yolo_face
        global exit_bool
        self.started = False
        self.landmark = toggle_landmarks
        self.yolo_face = toggle_yolo_face
        self.exit_bool = exit_bool
        self.read_lock = Lock()

    def start(self):
        if self.started:
            print("Toggle button stream already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            print("update " + str(self.started))
            toggle_buttons_function()

    def read(self):
        self.read_lock.acquire()
        landmark = self.landmark
        yolo_face = self.yolo_face
        exit_bool = self.exit_bool
        self.read_lock.release()
        return landmark, yolo_face, exit_bool

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()


if __name__ == "__main__":
    global toggle_landmarks
    global toggle_yolo_face
    global exit_bool
    toggle_landmarks = True
    toggle_yolo_face = True
    exit_bool = False

    vs = WebcamVideoStream().start()
    # vs_button = ToggleButtonStream().start()
    fps = FPS().start()

    while True:
        frame = vs.read()
        toggle_buttons_function()
        fps.update()
        cv2.imshow('webcam', frame)
        if cv2.waitKey(1) == 27 or exit_bool:
            break

    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    vs.stop()
    # vs_button.stop()
    # t1.join()
    cv2.destroyAllWindows()

