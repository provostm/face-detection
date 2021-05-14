# Face Detection

# Requirements
- Python
- dlib  (pip install dlib)
- openCV (pip install opencv-python)
- Download "shape_predictor_68_face_landmarks.dat"

# Description
The purpose of this project is to gather into one app different models regarding faces.
- Face landmarks
- Face detection (yolo face CPU and GPU)
- Gender Age detection (not implemented yet)
- Identity (not implemented yet)

# How to execute
- Modify the path where you store "shape_predictor_68_face_landmarks.dat" model.
- Run `python main.py` for straight forward script with one For loop or run `python multithread.py` for a multithreaded script (better FPS)
- Quit by pressing "Q" or "Esc"
- Toggle landmarks by pressing "L"
- Toggle Yolo face detection (CPU) by pressing "Y" 
- Toggle Yolo face detection (GPU) by pressing "G"
- Toggle Gender and Age detection by pressing "A"
- Toggle Identity detection by pressing "I"

*Note that you need to press a certain amount of time on the key as the script has to go through the rest of the script before going back into the cv2.waitKey part. I need to make a dedicated thread for that but it seems not possible to have interactions outside of the main thread.*

Do not activate all at the same time or it will be very laggy.

# Screenshots
- When a face or more is detected
![face detected](https://user-images.githubusercontent.com/24222091/117327830-51290380-ae93-11eb-8752-16742be303c2.png)


- When no face detected
![no face detected](https://user-images.githubusercontent.com/24222091/117327856-571ee480-ae93-11eb-9bf7-4077053e9e6c.png)

# Show FPS
This topic is tricky as there is no official OpenCV method to show FPS. 
So I tried multiple techniques:
- utils FPS Class (FPS available when you stop the program)
- Calculation of time between 2 frames to deduct the FPS (possible to show it in real time)
- Use OpenCV get(cv2.CAP_PROP_FPS) (not efficient)
