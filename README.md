# Real-Time Fatigue Detection System

A computer vision-based system that detects driver/user fatigue in real-time using facial landmark analysis. The system monitors eye closure, yawning, and head posture to determine fatigue levels without requiring ML model training.

## ✨ Features

- **Real-time face mesh detection** using MediaPipe
- **Side-by-side visualization**: Normal view + Analyzed mesh view
- **Multi-metric fatigue analysis**:
  - **EAR (Eye Aspect Ratio)**: Detects eye closure
  - **MAR (Mouth Aspect Ratio)**: Detects yawning
  - **Head Pose**: Monitors head tilting/drooping
- **Color-coded feedback**: Green = Normal, Red = Fatigue indicators
- **Fatigue level meter**: 0-100% visual indicator
- **No training required**: Rules-based detection

# Fatigue Detection System - Before & After

## System in Action

|  NORMAL STATE | FATIGUE STATE |
|:---------------:|:----------------:|
| ![Normal](images/normal_state.jpg) | ![Fatigue](images/fatigue_state.jpg) |
| **EAR:** 0.32 (Open) | **EAR:** 0.18 (Closed) |
| **MAR:** 0.41 (Normal) | **MAR:** 0.72 (Yawning) |
| **Head Tilt:** 5° | **Head Tilt:** 22° |
| **Fatigue:** 15% | **Fatigue:** 85% |
| **Status:**  AWAKE | **Status:** DROWSY |

---

##  Live Visual

### Normal State
[![](https://github.com/Sonu-Hansda/Face-Fatigue-Detection/blob/main/assets/fatigue.png)](https://github.com/Sonu-Hansda/Face-Fatigue-Detection/blob/main/assets/fatigue.png)
### Fatigue State 
![Normal State](images/https://github.com/Sonu-Hansda/Face-Fatigue-Detection/blob/main/assets/normal.png) 