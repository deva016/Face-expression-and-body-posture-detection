
# **Face Expression and Body Posture Detection**

A Python-based machine learning project for detecting facial expressions and body postures using deep learning and computer vision techniques. This system combines a convolutional neural network (CNN) for emotion recognition and MediaPipe Pose for body posture detection to generate actionable insights.

---
## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Setup Instructions](#Setup-Instructions)
- [Usage](#usage)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

---
## **Features**

- Detects facial expressions from live video feeds or images.
- Recognizes the following emotions:
  - Angry
  - Happy
  - Sad
  - Neutral
  - Surprised
  - Disgusted
  - Fearful
- Identifies specific body postures using MediaPipe Pose landmarks.
- Calculates scores based on detected emotions and poses.
- Captures and saves selfies based on predefined conditions.

---

## **Project Structure**

```
Face_Expression_Detection/
├── data/                            # Datasets
│   ├── train/                       # Training dataset
│   └── validation/                  # Validation dataset
│
├── src/                             # Source code
│   ├── models/                      # Pre-trained models and Haar cascade
│   │   ├── expressiondata1.h5       # Pre-trained emotion recognition model
│   │   └── haarcascade_frontalface_default.xml
│   │
│   └── scripts/                     # Main scripts
│       └── work_modal_count_and_emotion.py
├── output/                          # Output directory
│   ├── selfies/                     # Saved selfie images
│   ├── logs/                        # Logs for debugging
│   └── results/                     # Final results
│
├── tests/                           # Test scripts
│   └── model.py
│
├── README.md                        # Project documentation
├── requirements.txt                 # Dependencies
└── .gitignore                       # Ignored files (e.g., virtual environment)
```

## **Setup Instructions**

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-repo/Face_Expression_Detection.git
   cd Face_Expression_Detection
   ```

2. **Create and Activate a Virtual Environment**
   ```bash
   python -m venv myenv
   # On Windows
   myenv\Scripts\activate
   # On macOS/Linux
   source myenv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the Dataset**
   - Place training and validation images in `data/train/` and `data/validation/` folders respectively.
   - Images should be categorized into subdirectories for each class (e.g., `happy`, `angry`).

5. **Run the Project**
   ```bash
   python src/scripts/work_modal_count_and_emotion.py
   ```

---

## **Technologies Used**

- **Libraries:**
  - TensorFlow/Keras: Deep learning for facial expression recognition.
  - OpenCV: Image processing and face detection.
  - MediaPipe: Body posture detection.
  - NumPy: Numerical computations.
  - dlib for facial landmark detection
- **Models:**
  - Custom CNN trained on grayscale 48x48 facial expression images.
  - Haar cascade for face detection.
- **Datasets**: Utilizes datasets such as FER2013 for training the emotion recognition model.

---

## **How It Works**

1. **Face Expression Detection:**
   - The Haar cascade detects faces in a frame.
   - Detected faces are passed to the pre-trained CNN (`expressiondata1.h5`) to classify emotions.
   - Emotion scores are assigned based on class labels.

2. **Body Posture Detection:**
   - MediaPipe Pose identifies key landmarks on the body (e.g., shoulders, elbows, wrists).
   - Angles between joints are calculated using trigonometry.
   - Specific poses trigger actions (e.g., saving selfies).

3. **Scoring Mechanism:**
   - Each emotion and pose contributes a score.
   - When scores exceed a threshold, selfies are saved to the `output/selfies/` directory.

---

## **Usage**

- **Run the script to start detection:**
  ```bash
  python src/scripts/work_modal_count_and_emotion.py
  ```
- **Output:**
  - Live video feed with detected emotions and body landmarks displayed.
  - Selfies saved to `output/selfies/` based on conditions.

---

## **Results**

- **Emotion Detection Accuracy:** ~85% on the validation dataset.
- **Pose Detection Robustness:** Accurate for common poses under well-lit conditions.
- **Sample Output:**
  - Emotion detection: `happy`, `sad`, etc., displayed over detected faces.
  - Body angles shown near corresponding joints in real-time.

---


### Review Outputs:
Check the `output/` directory for:
- Saved selfies
- Logs for performance evaluation
- Results of the emotion detection

## Contributing
Contributions to enhance the functionality of the Face Expression Detection project are welcome. Please open an issue or submit a pull request for improvements or new features.


## Acknowledgements
We would like to thank the faculty and peers at K L University Hyderabad for their guidance and support throughout the development of this project. Special thanks to Dr. P. Sudharsana Rao for supervision and encouragement.
