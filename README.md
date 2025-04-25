# CT Scan-Based COVID-19 Detection Using CNN-Based Feature Extraction and Stack-Based Ensemble Techniques
This project uses deep learning to detect COVID-19 from CT scan images by combining CNN-based feature extraction (InceptionV3 and ResNet50) with a stacked ensemble approach for improved classification accuracy.
**Project Overview**
This project presents an AI-powered diagnostic system to detect COVID-19 from CT scan images. It integrates deep learning-based feature extraction using CNNs (ResNet50, InceptionV3) with machine learning classifiers (SVM, ANN) and ensemble methods (XGBoost, LightGBM, CatBoost, AdaBoost, Gradient Boosting) to achieve accurate and reliable results.
**Key Features**
Deep Feature Extraction using ResNet50 and InceptionV3

Handcrafted Features (HOG, LBP, GLCM) for texture analysis

Ensemble Learning with boosting algorithms

Explainable AI through visualization of CNN feature maps

Performance Metrics: Accuracy, Precision, Recall, F1-score, AUC-ROC

**Dataset**
CT scan images of COVID-19 positive and negative cases

Image format: PNG/JPEG

Resolution standardized to 224×224 pixels

Verified annotations from medical professionals

**Preprocessing Techniques**
Grayscale conversion

Image resizing and normalization

Noise reduction (Gaussian, median filtering)

Contrast enhancement (CLAHE, histogram equalization)

Data augmentation 
**Models Used**

**Technique	Models**
CNN Feature Extractors	ResNet50, InceptionV3
Classifiers	SVM, ANN
Boosting	XGBoost, LightGBM, AdaBoost, CatBoost, Gradient Boosting
⚙️ Technologies & Tools
Programming Language: Python

Deep Learning: TensorFlow, Keras, PyTorch

Machine Learning: scikit-learn, XGBoost, LightGBM

Visualization: Matplotlib, Seaborn

Image Processing: OpenCV, PIL

IDE: VS Code / Jupyter Notebook

✅**Results**
LightGBM achieved the highest accuracy: 94%

XGBoost followed closely with 93%

High F1-scores (0.94–0.95) for both COVID and non-COVID classes

Ensemble models demonstrated superior generalization and robustness

📌 **Conclusion**
This hybrid deep learning + ensemble learning system is a reliable and scalable solution for automated COVID-19 detection using CT images. It paves the way for AI-assisted diagnostics and can be extended for detecting other pulmonary conditions.

👥 **Contributors**
Nayeem Khan (221FA04139)

Lakshmi (221FA04639)

Siva Rama Krishna (221FA04153)

Harendra Kumar (221FA04741)
Guide: Mr. Sourav Mondal (Assistant Professor, CSE Department)





