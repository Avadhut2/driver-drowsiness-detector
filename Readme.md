# 🛑 Driver Drowsiness Detection System

A Python project that uses facial landmarks and eye aspect ratio (EAR) to detect when a vehicle driver is becoming drowsy.

## 🚀 Features

- Real-time face and eye tracking
- EAR-based drowsiness detection
- Saves photo when drowsy
- Logs session details
- Portable (relative paths)
- Keyboard control: `q` to quit, `r` to reset, `s` to save frame

## 📂 Folder Structure



```
driver-drowsiness-detector/
│
├── main.py                          # Main script to run the detector
├── drowsiness_detector.log         # Log file to track drowsiness sessions
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── drowsy_photos/                  # Folder to save captured drowsy images
└── shape_predictor_68_face_landmarks.dat   # Facial landmark model (Not included in repo)
```


## 🔧 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/driver-drowsiness-detector.git

2. Install dependencies:
    pip install -r requirements.txt

3. Download the model file:

Download shape_predictor_68_face_landmarks.dat
[Download from here](https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

Extract it and place it in the project directory

4. ▶️ Run
    python main.py
Or with arguments:
    python main.py --camera 0 --ear-threshold 0.25 --drowsy-time 2.0


5. ## 🙌 Credits

Built with ❤️ by:

- [Avadhut Satpute](https://github.com/Avadhut2)
- [Atulya Sahu](https://github.com/devatulya) 