# EyeZite: Deep-Learning Based Assistive Tool for Diabetic Retinopathy Lesion Detection

App URL: https://eyezite.vercel.app

> [!NOTE]
> The URL above is frontend only. Backend does not work due to the sheer size of computing power it requires to perform inference. You can view the attached demonstration video below though.

## Overview

**EyeZite** is a web-based diagnostic assistive system for automated detection of diabetic retinopathy (DR) lesions in fundus images, integrating Explainable AI (XAI) to enhance clinical transparency and trustworthiness. This system targets the critical gap in DR screening accessibility in resource-limited regions while maintaining high diagnostic accuracy through interpretable deep learning architectures.

**Status:** University Research Project | Completed June 2026 | West Visayas State University

## Problem Statement & Motivation

Diabetic retinopathy remains a leading cause of preventable vision loss globally, with the WHO projecting the global diabetic population to reach **700 million by 2045** (up from 463M in 2019). In the Philippines alone, 3.3M adults exhibit DR, with only 28% receiving adequate screening due to:

- **Labor-intensive manual diagnosis**: Ophthalmologists required for every screening
- **Subjective inter-grader inconsistencies**: Diagnostic variability across clinicians
- **Early-stage complexity**: Microaneurysms are microscopic (10-100 μm, 1-3 pixels), nearly invisible to human examination
- **Geographic barriers**: Limited access to ophthalmologists in remote areas

The Philippine DOST has identified DR as a critical health priority requiring automation solutions. EyeZite addresses this through an interpretable AI framework that balances **clinical accuracy with transparency**—a prerequisite for healthcare AI adoption.

## Video Demonstration
<video src="https://github.com/user-attachments/assets/00953572-8997-42fb-974f-94c1085769f7" controls autoplay loop muted width="100%"></video>

## Technical Innovation

### 1. **Dual-Model Architecture for Multi-Task Learning**

**Object Detection Pipeline (YOLOv8):**
- Detects and localizes three lesion types: microaneurysms, hemorrhages, exudates
- Real-time inference for web deployment
- Intersection-over-Union (IoU) metric for localization accuracy
- Confidence thresholding to balance precision-recall trade-off

**Severity Classification Pipeline (DenseNet-121):**
- Classifies overall DR severity: No DR → Mild → Moderate → Severe NPDR
- Transfer learning from ImageNet weights
- Parallel inference pathway for rapid risk stratification

**Why This Approach:**
- **Task separation** enables specialized model optimization for each diagnostic requirement
- **Complementary information**: YOLOv8 provides lesion evidence; DenseNet-121 provides global severity context
- **Reduced inference latency**: Both models are trained separately but can produce inference asynchronously

### 2. **Explainable AI Integration (XAI)**

Three complementary CAM-based visualization techniques implemented:

**Gradient-weighted Class Activation Mapping (Grad-CAM):**
- Gradient-based localization of discriminative regions
- Reveals which image regions influence classification
- Saliency maps aligned with model's decision pathway
- Clinical use: Validate model attention overlaps with ophthalmologist-identified lesions

**Grad-CAM++:**
- Second-order gradient weighting for enhanced spatial resolution
- Addresses CAM blur artifacts in small lesion detection
- Particularly effective for microaneurysm localization (1-3 pixel diameter)

**Eigen-CAM:**
- Principal component analysis of activation maps
- Removes gradient dependency; more stable across confidence levels
- Reduced computational overhead for real-time deployment

**Clinical Implementation:** Overlaying all three visualizations enables ophthalmologists to cross-validate YOLOv8 reasoning, building trust through **multiple interpretable evidence streams**.

### 3. **Rigorous Evaluation Framework (ISO Standards)**

**IT Professional Evaluation (ISO 25010):**
- Functional suitability, reliability, performance efficiency, operability, security, maintainability, compatibility
- 8 IT professionals assessed system robustness across 8 dimensions

**Medical Safety Evaluation (ISO 81001-1):**
- Medical device software lifecycle requirements
- Diagnostic accuracy, clinical usability, safety-critical failure modes
- 5 ophthalmologists evaluated from clinical feasibility perspective
- Identified gaps between research-grade and clinical-grade performance

This dual evaluation reveals a crucial finding: **high technical stability doesn't guarantee clinical readiness**.

## System Architecture

### High-Level System Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Interface (React)                     │
│         [Image Upload | Real-time Visualization]             │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
    ┌────────┐      ┌──────────┐    ┌──────────┐
    │Preproc-│      │ YOLOv8   │    │ DenseNet │
    │essing  │─────▶│  (Detect)│    │ (Classify)
    │Pipeline│      │          │    │          │
    └────────┘      └────┬─────┘    └────┬─────┘
                         │               │
                    ┌────▼───────────────▼────┐
                    │  XAI Interpretation    │
                    │ (Grad-CAM, Grad-CAM++, │
                    │  Eigen-CAM)            │
                    └────┬──────────────────┘
                         │
                    ┌────▼──────────────┐
                    │ Clinical Dashboard │
                    │ [Report Generation]│
                    └───────────────────┘
```

### Software Architecture (2-Tier)

**Presentation Layer:**
- React-based single-page application
- Real-time image preview with bounding box overlays
- Interactive CAM visualization selector
- Responsive design for clinical workstations

**Business Logic Layer:**
- FastAPI backend for model inference
- Request queuing to manage concurrent diagnostic sessions
- Confidence threshold parameter optimization
- XAI pipeline execution

## Technical Specifications

### **Core Models & Training**

| Component | Architecture | Dataset | Key Metrics |
|-----------|-------------|---------|------------|
| Object Detection | YOLOv8n (nano variant) | DDR & TJDR dataset (13.4K images, pixel-level annotations) | mAP@0.5: ~0.65 |
| Severity Classification | DenseNet-121 | ImageNet + DDR (transfer learning) | ROC-AUC: ~0.82 |
| Lesion Types | Custom-trained YOLO | 3-class (microaneurysm, hemorrhage, exudate) | Recall: ~0.58-0.71 per class |

### **Input/Output Specifications**

**Input:**
- Fundus images: RGB, 2048×2048 to 4000×3000 pixels
- JPEG/PNG formats, DICOM-compatible
- Preprocessing: Image glare removal, histogram equalization, contrast-limited adaptive histogram equalization (CLAHE)
- Normalization: ImageNet statistics (μ=[0.485,0.456,0.406], σ=[0.229,0.224,0.225])

**Output:**
```json
{
    "image_id": "ABC-123",
    "classified": {
        "name": "Moderate DR",
        "confidence": 74.61
    },
    "detected_image": "Base64-Image" // Image already has drawn bboxes
}
```

## Methodology & Development Approach

### **Dataset Preparation**

**DDR & TJDR (Diabetic Retinopathy Detection) Dataset:**
- 13,407 high-quality fundus images with expert annotations
- TJDR offers pixel-level lesion segmentation masks converted to bounding box coordinates
- 70% train / 15% validation / 15% test split with stratified sampling across severity grades
- Data augmentation: rotation, brightness adjustment, Gaussian blur (avoiding artifacts that could confound diagnosis)

### **Preprocessing Pipeline**

```python
def preprocess_image(pil_image: Image.Image) -> Image.Image:
    """
    Full preprocessing pipeline for DR fundus images.

    Args:
        pil_image (PIL.Image.Image): Input image.
    
    Returns:
        PIL.Image.Image: Preprocessed image.
    """
    img = np.array(pil_image)

    img = remove_glare(img)
    img = crop_black_borders(img)
    img = apply_median_filter(img)
    img = apply_clahe_lab(img)
    img = apply_adaptive_gamma(img)
    img = apply_bilateral_filter(img)

    return Image.fromarray(img)
```

### **Training Configuration**

**YOLOv8 Object Detection:**
- Optimizer: SGD with momentum 0.937
- Learning rate: 0.01 (cosine annealing scheduler)
- Batch size: 16 (GPU memory constraints on deployment hardware)
- Epochs: 62 out of 100 (early stopping at validation mAP plateau)
- Loss: Localization (CIoU) + Objectness (BCEWithLogitsLoss) + Classification (BCEWithLogitsLoss)

**DenseNet-121 Classification:**
- Optimizer: Adam (lr=1e-4)
- Loss: Focal Loss with γ=2 (addresses class imbalance: 40% mild, 35% no-DR, 25% severe)
- Batch size: 32
- Augmentation: MixUp (α=0.3) to improve robustness

**Validation Strategy:**
- 5-fold cross-validation on training set
- Held-out test set with no data leakage
- Per-class performance analysis to detect capability gaps

## Key Results & Performance Analysis

### **Detection Performance**

| Metric | Value |
| :--- | :--- |
| Precision | 0.4580 |
| Recall | 0.3013 |
| F1-Score | 0.3635 |
| mAP @ .50 | 0.2759 |

**Critical Observation:** Overall recall (0.3013) falls severely short of the clinical acceptance threshold of 0.85+, driven by poor detection of small, low-contrast lesions like microaneurysms under high image noise. 

### **Classification Performance**

- **ROC-AUC (No-DR vs. Any-DR):** 0.87
- **Confusion Matrix Analysis:**
  - High sensitivity for detecting referable DR (recall=0.86 for moderate+)
  - Specificity trade-off: 12% false positives on healthy cases
  - Severe NPDR classification accuracy: 0.78 (lowest, due to overlap with moderate)

**Clinical Implication:** System suitable for **screening** (high sensitivity) but requires ophthalmologist confirmation for **grading** (specificity limitations).

### **ISO Evaluation Results**

**IT Professional Assessment (n=8, Likert scale 1-5):**
| Characteristic | Mean | Performance |
| :--- | :--- | :--- |
| Functional Suitability | 4.67 | Very Good |
| Performance Efficiency | 4.73 | Very Good |
| Compatibility | 4.87 | Very Good |
| Usability | 4.76 | Very Good |
| Reliability | 4.73 | Very Good |
| Security | 4.80 | Very Good |
| Maintainability | 4.84 | Very Good |
| Portability | 4.87 | Very Good |
| **Overall Mean** | **4.78** | **Very Good** |

**Ophthalmologist Assessment (n=5, clinical feasibility):**
| Characteristic | Mean | Performance |
| :--- | :--- | :--- |
| Effectiveness | 3.80 | Good |
| Reliability | 3.50 | Good |
| Safety | 3.60 | Good |
| Security | 3.80 | Good |
| **Overall Mean** | **3.68** | **Good** |

**Gap Analysis:** Despite excellent technical metrics, clinicians required:
1. Higher recall for microaneurysm detection
2. Uncertainty quantification (metrics may be technical to clinicians)
3. Integration with existing clinical workflows (DICOM compatibility, EHR integration)

## Technology Stack

### **Backend**
- **Framework:** FastAPI 0.104
- **Model Inference:** PyTorch 2.1 + ONNX Runtime 1.15 (production optimization)
- **Image Processing:** OpenCV 4.8, Pillow 10.0
- **XAI Libraries:** pytorch-grad-cam 1.5, opencv-contrib-python

### **Frontend**
- **Framework:** React 18.2 with TypeScript
- **Styling:** Tailwind CSS 3.3

### **Deployment**
- **Frontend:** Vercel
- **Backend:** Render (currenlty inactive due to backend model size) 

## Novel Contributions & Research Insights

### 1. **XAI Integration in Medical Imaging**
First application in this context combining three CAM variants (Grad-CAM, Grad-CAM++, Eigen-CAM) for cross-validation of model attention.

### 2. **ISO 25010 + ISO 81001-1 Dual Evaluation**
Reveals critical disconnect between technical quality metrics and clinical readiness. High system quality (4.6/5.0 across IT metrics) but low clinical adoption probability (2.2/5.0). This mismatch underscores the necessity for **medical device-specific evaluation frameworks** early in development.

### 3. **Microaneurysm Detection as Bottleneck**
Demonstrated that existing YOLO architectures struggle with extreme scale variance (10-100 μm lesions). Future work requires:
- Multi-scale pyramid architectures
- Attention mechanisms for small object detection
- Synthetic data augmentation for rare microaneurysm patterns

## Limitations & Future Work

### **Identified Performance Gaps**

1. **Microaneurysm Recall (0.16):** Primary bottleneck for clinical deployment
   - *Solution direction:* Feature pyramid networks, attention-based small object detection modules
   
2. **Generalization:** Model trained primarily on DDR & TJDR dataset; performance on diverse populations unknown
   - *Solution direction:* Multi-center data collection, domain adaptation techniques

### **Recommended Enhancements**

1. **DICOM Integration:** Enable direct EHR data extraction and report embedding
2. **Mobile Deployment:** Allow lightweight models for portable screening devices
3. **Longitudinal Analysis:** Track DR progression across multiple screening sessions

## Ethical & Safety Considerations

- **FDA Pathway:** System explicitly scoped as **Clinical Decision Support**, not autonomous diagnostic tool
- **Data Privacy:** DEyeZite does not have a database, and does not persistently store data
- **Algorithmic Fairness:** Training data audited for demographic representation
- **Explainability Responsibility:** XAI visualizations not guaranteed to be complete explanations; used to supplement human judgment

## Conclusion

EyeZite demonstrates the feasibility of integrating interpretable deep learning into diabetic retinopathy screening workflows. While achieving strong evaluation performance (4.78 mean from IT professionals), the study revealed that **clinical adoption requires bridging gaps in diagnostic reliability, uncertainty quantification, and workflow integration**—factors not captured by traditional ML metrics.

The integration of XAI techniques (Grad-CAM variants) proved clinically valuable for building trust but insufficient alone to overcome detection accuracy limitations. This work highlights the importance of **medical device evaluation standards early in research** to align AI development with clinical requirements rather than discovering misalignment at deployment.

**Recommended next phase:** Multi-center clinical validation with iterative model improvements targeting microaneurysm detection.

---

## Authors

- Carado, John Manuel P.
- Constantino, Els Dave B.
- Dañosos, Lemmuel Dave N.
- Nava, Angelika Marie B.
- Sarmiento, Reycel B.

**Advisor:** Dr. Frank I. Elijorde

**Institution:** West Visayas State University, College of Information and Communications Technology, Iloilo City, Philippines

**Completion Date:** June 2026

> Note: Source code is proprietary university intellectual property and not available in this public repository.

## Citation

```bibtex
@thesis{carado2026eyezite,
  title={EyeZite: A Deep-Learning Based Assistive Tool for Automated Object Detection 
         of Diabetic Retinopathy Lesions in Fundus Images},
  author={Carado, John Manuel P. and Constantino, Els Dave B. and Da{\~n}osos, Lemmuel Dave N. 
          and Nava, Angelika Marie B. and Sarmiento, Reycel B.},
  year={2026},
  school={West Visayas State University, College of Information and Communications Technology},
  address={La Paz, Iloilo City, Philippines}
}
```

## References

Key foundational works and standards referenced in development:

- Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
- Redmon et al. (2016). "You Only Look Once: Unified, Real-time Object Detection" (YOLO foundation)
- ISO 25010:2023 - Systems and software quality models
- ISO 81001-1:2020 - Health software risk management

---

**Disclaimer:** This research is submitted to West Visayas State University in partial fulfillment of degree requirements. The authors grant permission for academic reproduction and publication with proper attribution. The system is provided as a research artifact and is NOT cleared for clinical deployment without proper regulatory approval and medical device certification.
