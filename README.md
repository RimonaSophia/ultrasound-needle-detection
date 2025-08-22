#  Needle Detection in Ultrasound Images

##  Project Overview

This project focuses on detecting **needles in ultrasound images** using a **YOLOv11m** pretrained model. It demonstrates how to adapt pretrained object detection models for small, domain-specific datasets.

The dataset was pre-labeled and exported from **Roboflow** in YOLO format with an accompanying `data.yaml` file. All model training and evaluation were done in **Google Colab**.

---

##  Dataset Setup & EDA

* **Classes**: 1 class → `needle`
* **Images**: 314 (train), 90 (val), 43 (test)
* **Labels**: Checked and cleaned

  * Removed 3 images with **empty label files**
* **Image Format**: JPEG
* **Visualized**: Sample bounding boxes to verify data integrity

This step ensured the model would not learn from invalid examples and that training would run smoothly.

---

## 🛠️ Model Training

* **Model**: `yolov11m.pt` (pretrained on COCO)
* **Training config**:

  * Epochs: 50
  * Image size: 640
  * Batch size: 8
  * Device: GPU (Colab)
  * Validation split: 90/412 (\~22%)

### 🧪 Training Summary

* Used Roboflow's export with `data.yaml`
* All training was logged under `runs/detect/needle_yolov11m/`
* Evaluation metrics: mAP, precision, recall (available in results CSV)

---

## 🔍 Inference & Predictions

* **Loaded model** from best checkpoint:
  `model = YOLO('runs/detect/needle_yolov11m/weights/best.pt')`

* **Predicted on test set**:

  ```python
  model.predict(source='test/images', save=True, save_txt=True, conf=0.25)
  ```

* **Verified Predictions**:

  * Visual inspection confirmed high accuracy
  * Saved annotated images to `runs/detect/predict/`
  * Confirmed most predictions matched labeled needle locations

---

##  Model Performance (Coming Soon)

* Will include:

  * Precision / Recall / mAP
  * Confusion matrix
  * Confidence threshold tuning results

---

## 🧪 EDA Sample Markdown Cell

```markdown
### Exploratory Data Analysis (EDA)

Before training our object detection model, we want to:
- Understand how many images are in each dataset split
- Check if any label files are empty
- Visualize a sample with a bounding boxes
- Look at image sizes to confirm consistency

I removed 3 empty label files and their corresponding images from train/valid folders.
This ensures the model trains only on valid examples.

The data cleaning step is repeatable and safe*to rerun as part of the notebook pipeline. It’s designed so that even if the dataset is already clean, it will make no changes.
```

---

## Next Steps

* Hyperparameter tuning for better generalization
* Try YOLOv8 or fine-tune different backbones
* Evaluate false positives/negatives visually
* Compare with other detection models (e.g., SSD, Faster R-CNN)
* Familiarize more with real **ultrasound imaging** to interpret edge cases and assess model robustness

---

##  Project Highlights (for Resume)

**Needle Detection in Ultrasound Images**
Trained and evaluated a YOLOv11m object detection model on a labeled medical image dataset (Roboflow). Performed exploratory data analysis (EDA), data cleaning, and bounding box visualization. Tuned training hyperparameters in Google Colab and validated performance with visual inspection and evaluation metrics.

