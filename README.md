# Train Model Branch

This branch is dedicated to training a YOLO model on a Chest X-ray dataset using Python, TensorFlow and Keras on Google Colab. Currently, we are using YOLOv11 with plans to possibly explore additional models in the future.

The NIH Chest X-ray dataset comprises 112,120 frontal-view X-ray images from 30,805 unique patients. Each image may have multi-label annotations for the following fourteen thoracic pathologies, identified via natural language processing of associated radiological reports:

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
