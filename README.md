# 🐍 Sign2Text

_Translate sign language into written words using machine learning._ 

---

## 📖 Overview

**Sign2Text** is a project designed to **build a complete pipeline** for training and deploying a model that **translates sign language gestures into text.**  

**This repository aims to:**
- Provide a clean training pipeline for sign language recognition.  
- Offer tools for data preprocessing, training, and evaluation.  
- Demonstrate a minimal prototype suitable for training a sign2text model

The system is structured into **four main components**:

1. **Video**: captures video from camera and extracts landmarks from mediapipe  
2. **Generate Dataset**: orchestrates collection and labeling of gesture samples
3. **Training**: trains a model on the generated dataset
4. **Inference**: runs real-time sign recognition from live camera feed

---

## ⚙️ Installation

**Prerequisites:**  
- Python 3.9
- [uv](https://github.com/astral-sh/uv) (for dependency management)

Install dependencies
```
uv sync
```

## Usage

1. **Plan your signs:** Decide the set of signs (classes) you want to record and train the model with.
2. **Configure the dataset parameters** in [data_config.py](./src/generate_dataset/data_config.py). This are the mandatory ones:
    - `ACTIONS`: np.array with the list of signs you want to record.
    - `NUM_VIDEOS`: number of videos to record per sign.
    - `VIDEO_FRAME_LENGTH`: length of each video (defines the first dimension of each LSTM sample).

3. **Run the main components** in [main_workflow.py](./main_workflow.py):
    - `video_with_landmarks()`: starts the camera and visualizes landmarks in real time (sanity check)
    - `generate_dataset()`: records gestures, extracts landmarks, and organizes them into labeled folders.
    - `training_pipeline_workflow(model_filename)`: trains the model using the prepared dataset and saves it as an .h5 file with model_filename.
    - `inference_pipeline_workflow(model_filename)`: runs real-time sign language recognition using the trained model and live camera feed.

> **Tips:**
> - **Run one component at a time** to test that it’s working before chaining the full workflow.
> - Start with a **small set of signs to validate the pipeline** before scaling
> - **Keep gestures consistent between recording** sessions to improve accuracy
> - **Ensure your camera is well-lit and positioned** for stable landmark detection

## Credits

Credits are to mention the people who contributed to the project.

| GitHub                                                                 | LinkedIn                                                                                          |
|-------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| [Almudena Zhou Ramirez](https://almudenazhou.github.io/)               | [linkedin.com/in/almudena-zhou-ramirez-lopez](https://www.linkedin.com/in/almudena-zhou-ramirez-lopez/) |
| [Javier Chico García](https://github.com/JavierChicoOfc)               | [linkedin.com/in/javier-chico-garcía-ofc](https://www.linkedin.com/in/javier-chico-garc%C3%ADa-ofc/) |
| [Jose Manuel Pinto Lozano](https://github.com/JoseManuelPintoLozano)   | [linkedin.com/in/josemanuelpintolozano](https://www.linkedin.com/in/josemanuelpintolozano/)         |


This project was inspired by [nicknochnack repository](https://github.com/nicknochnack/ActionDetectionforSignLanguage/tree/main)

