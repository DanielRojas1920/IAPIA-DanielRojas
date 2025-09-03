
# 📦 Train Model Branch

## 🧠 Chest X-Ray DenseNet

![Dataset Images](https://github.com/user-attachments/assets/0ce2ca7a-2438-42db-9ae1-d3287a2bc53b)

The dataset contains **112,120 frontal-view X-ray images** from **30,805 unique patients**, each image annotated with multiple thoracic pathologies:

- Atelectasis  
- Consolidation  
- Infiltration  
- Pneumothorax  
- Edema  
- Emphysema  
- Fibrosis  
- Effusion  
- Pneumonia  
- Pleural Thickening  
- Cardiomegaly  
- Nodule  
- Mass  
- Hernia  

> 🎯 **Goal**: Train a multi-label classification model using **DenseNet** as the backbone and fine-tune it for optimal performance.

---

## 🔄 Data Preprocessing

### 1. Label Mapping  
- Convert label strings to one-hot vectors.  
- Add binary columns for each pathology.  
- Show sample rows and frequency stats.

---

### 2. Undersampling  
- Address class imbalance (e.g., "No Finding" over-represented).

#### 📊 Original Distribution  
![Original Distribution](https://github.com/user-attachments/assets/aec3846f-7e36-47b0-b984-d887fd269c1e)

#### 📉 Final Distribution  
![Final Distribution](https://github.com/user-attachments/assets/8cdbebac-54f1-46d4-a2c4-05d29f67a4a8)

---

### 3. Disease Vector Creation  
- Remove "No Finding" from labels.  
- Generate `disease_vec`: binary vector per image.

---

### 4. Tabular Patient Data  
- Normalize numerical/categorical features (age, sex, view, etc.).

---

### 5. Final Dataset Preview  
- View preprocessed dataset with image paths, labels, vectors, and tabular features.

---

### 6. Dataset Split  
- Split into:
  - **Training**: 72%  
  - **Validation**: 8%  
  - **Test**: 20%  
- Stratified sampling by diagnosis.

---

### 7. Training-Only Preprocessing  
- Apply cleaning steps only on the training set to prevent leakage.

---

### 8. Save Processed Dataset  
- Save to Google Drive and sync to Hugging Face:  
  🔗 [Dataset (14.5GB)](https://huggingface.co/datasets/Carlos56g/ChestXRay15GB)

---

### 9. Data Augmentation & Transforms

#### 📌 Train Transform  
- ToTensor, RandomCrop (90%), Resize, RandomAffine (±7°, ±6%), GaussianBlur, Normalize.

![Train Augmentation](https://github.com/user-attachments/assets/65d1a241-b619-495b-9458-11578a6d9f98)

#### 📌 Validation/Test Transform  
- ToTensor, RandomCrop (90%), Resize, Normalize.

![Test Augmentation](https://github.com/user-attachments/assets/a4347789-aa2b-4382-9407-ad6d3e105060)

---

### 10. DataLoaders  
- Use `ChestXRayDataset` with batch loading and transforms.  
- Shuffle only training data.

---

## 🧬 Model: Multimodal Classifier with FPN

### Overview  
Combines **DenseNet121 + FPN + tabular features** to classify 14 diseases.

---

### 🔹 Components

#### 1. **Backbone: DenseNet121**
- Extract intermediate layers:
  - `denseblock2` → c3 (512)
  - `denseblock3` → c4 (1024)
  - `denseblock4` → c5 (1024)

#### 2. **FPN**
- Processes c3–c5 with lateral + smoothing convolutions.

#### 3. **Image Feature Head**
- Pool each FPN output, concatenate to `[B, 768]` → FC → `[B, 256]`

#### 4. **Tabular Head**
- Tabular input `[B, 3]` → FC → `[B, 8]`

#### 5. **Fusion Head**
- Combine image + tabular: `[B, 264]`  
- FC layers → final output `[B, 14]`

---

### 🔁 Input/Output

- **Input**:
  - `x_image`: `[B, 3, H, W]`
  - `x_tabular`: `[B, 3]`
- **Output**:
  - `[B, 14]` logits

---

## 🏋️‍♂️ Training Process

### ✅ Weighted Loss  
Helps balance rare vs. frequent classes using `BCEWithLogitsLoss` with `pos_weight`.

---

### 🎯 Threshold Functions  
Learns **class-specific thresholds** to convert logits into binary predictions (optimize F1/Precision).

---

### 🔁 Training Logic

1. **Setup**:
   - Load model, optimizer, scheduler.
   - Set loss weights with `pos_weight`.

2. **Loop per Epoch**:
   - Train:
     - Forward → loss → backward
     - Log loss and ETA
   - Validate:
     - No gradients
     - Predict + apply dynamic thresholds
   - Log precision, recall, F1, AUC

3. **Checkpointing**:
   - Save model based on:
     - Best AUC
     - Best Macro F1
     - Best Precision

4. **Scheduler**:
   - Reduce LR on plateau with `patience=2`.

5. **Timing**:
   - Print training time.

---

### 🔧 Key Functions

- `format_time(seconds)`  
- `find_optimal_thresholds_with_dynamic_adjustment()`  
- `classification_report()`  
- `BCEWithLogitsLoss`

---

## 🧪 Evaluation Summary

### 🔍 Predictions  
Examples from test set:

![Example 1](https://github.com/user-attachments/assets/b8d20d03-8c1e-4077-8785-45b1c50b1c39)
![Example 2](https://github.com/user-attachments/assets/19bdbe08-c85a-4c35-b7e6-fd4c472a3fca)
![Example 3](https://github.com/user-attachments/assets/2e66779f-be29-4cae-ba44-c99614fba2f8)
![Example 4](https://github.com/user-attachments/assets/c499f9db-2f80-4166-b692-6c052f11d29d)
![Example 5](https://github.com/user-attachments/assets/f47245f7-3d33-4b25-a0d4-b81612c53966)
![Example 6](https://github.com/user-attachments/assets/95ad83d8-0adb-44c1-9809-f3e63da8624f)
![Example 7](https://github.com/user-attachments/assets/b72d68f9-0c81-4824-9f82-19118bd2bbe3)
![Example 8](https://github.com/user-attachments/assets/6de48bc4-e12a-470c-8cc9-46a8e39d624e)
![Example 9](https://github.com/user-attachments/assets/64836da5-cbf1-4f5e-ab4c-04e81a3a780c)
![Example 10](https://github.com/user-attachments/assets/576bdd8a-4d65-47d7-9ef5-84c75e96095e)

---

### 📈 Strongest Classes

| Pathology     | F1 Score | AUC   | Support | Notes                                        |
|---------------|----------|-------|---------|----------------------------------------------|
| Effusion      | 0.602    | 0.752 | 2036    | Balanced precision/recall                    |
| Cardiomegaly  | 0.547    | 0.773 | 502     | High AUC, moderate prevalence                |
| Hernia        | 0.424    | 0.788 | 43      | Rare class, solid AUC with high threshold    |
| Emphysema     | 0.442    | 0.668 | 446     | Good recall, moderate precision              |

---

### 📉 Weakest Classes

| Pathology          | F1 Score | AUC   | Support | Notes                             |
|--------------------|----------|-------|---------|-----------------------------------|
| Consolidation      | 0.026    | 0.505 | 813     | Model fails to detect             |
| Pneumonia          | 0.033    | 0.506 | 248     | AUC near random                   |
| Pleural Thickening | 0.257    | 0.645 | 645     | High false positive rate          |

---

### 📊 Overall Metrics

- **Exact Match Ratio**: 0.1421  
- **Hamming Loss**: 0.1326  
- **Recall (Macro / Micro)**: 0.376 / 0.4346  
- **Label Ranking Avg Precision (LRAP)**: 0.6375  
- **Per-label Accuracy**: 0.8674  
- **Mean AUC**: 0.6448  

![ROC Curve by Class](https://github.com/user-attachments/assets/d22a53e5-5e7e-4f42-8ba8-a32512d4944b)


Global results indicate high per-label accuracy and moderate ranking precision, but a low Exact Match Ratio (expected in a 14-label setting). The Hamming Loss shows that on average only 13% of labels are incorrectly predicted.

### Oficial Documentation
- [Latex Article - English](https://drive.google.com/file/d/1azeHtvQ2CHdJH-t13q388mX3OuhunvvT/view?usp=sharing)
- [Article - Spanish](https://drive.google.com/file/d/1fMa_XvzvOPixa2HP7KKS1J320gpEWK8T/view?usp=sharing)

