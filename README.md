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
- Run main.py for a 1 thread script or run multithread.py for a multithread script (better FPS)
- Quit by pressing "Q"
- Toggle landmarks by pressing "L"
- Toggle Yolo face detection (CPU) by pressing "Y" 
- Toggle Yolo face detection (GPU) by pressing "G"
- Toggle Gender and Age detection by pressing "A"
- Toggle Identity detection by pressing "I"

Do not activate all at the same time or it will be very laggy.

# Screenshots
- When a face or more is detected
![face detected](https://user-images.githubusercontent.com/24222091/117327830-51290380-ae93-11eb-8752-16742be303c2.png)


- When no face detected
![no face detected](https://user-images.githubusercontent.com/24222091/117327856-571ee480-ae93-11eb-9bf7-4077053e9e6c.png)

