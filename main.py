import cv2
import dlib

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('D:/Projets python/Face-Morphing/code/utils/shape_predictor_68_face_landmarks.dat')

while True:
    ret, frame = cap.read()

    dets = detector(frame, 1)

    if len(dets) == 0:
        img_circle = cv2.putText(frame, "no face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        for k, rect in enumerate(dets):
            shape = predictor(frame, rect)

            for i in range(0, 68):
                img_circle = cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), 2)
                #img_circle = cv2.putText(img_circle, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_SIMPLEX,
                #                         0.3, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("webcam", img_circle)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
