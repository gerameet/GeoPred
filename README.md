# Task: Region ID Prediction

**Model(s) Used**:  
An ensemble of multiple pre-trained CNN architectures:  
- ConvNeXt Tiny  
- DenseNet  
- EfficientNet  
- ResNet  
- MobileNet  

All models were initialized with ImageNet-pretrained weights and fine-tuned on the provided region ID labels.

**Training Details**:
- **Loss Function**: `CrossEntropyLoss` with `label_smoothing=0.1`
- **Optimizer**: `AdamW` with `lr=1e-3`, `weight_decay=1e-4`
- **Scheduler**: `CosineAnnealingWarmRestarts` (`T_0=5`, `T_mult=2`)

**Pre-processing Techniques**:
- Resized input images to model-specific input sizes
- Normalized using ImageNet mean and std
- Augmentations: random horizontal/vertical flip, color jitter, random crop

**Innovative Ideas**:
- Used an ensemble voting mechanism to combine predictions across different CNNs for improved accuracy.
- Leveraged model diversity to reduce overfitting.

**Model Weights**:  
https://www.kaggle.com/models/meetgera/regionid_models [Kaggle Link]


# Task: Latitude-Longitude Regression

**Model(s) Used**:  
Two-stage architecture with EfficientNet-B0:
1. A **frozen** EfficientNet-B0 that extracts **softmax region ID embeddings** (14D).
2. A **trainable** EfficientNet-B0 that produces an image embedding.
3. Both embeddings are concatenated and passed to a regression head to predict latitude and longitude.

**Training Details**:
- **Loss Function**: `0.4 * MSE(Lat) + 0.6 * MSE(Long)`
- **Optimizer**: `Adam`

**Pre-processing Techniques**:
- Resized images to EfficientNet-B0 input size (224x224)
- Standard ImageNet normalization
- Augmentations: random flip, rotation, brightness, contrast

**Post-Processing**:
- Used a **KD-Tree** to find the **nearest training point** to each predicted coordinate.
- Added small **Gaussian jitter** to both predictions and training data during KD-Tree search to improve robustness.

**Innovative Ideas**:
- Weighted MSE loss to handle asymmetric error distribution between latitude and longitude.
- Region ID soft embeddings helped provide high-level location context.
- KD-Tree refinement post-prediction improved final accuracy and generalization.

**Model Weights**:  
https://www.kaggle.com/models/meetgera/latlong_models


# Task: Direction Estimation

**Model Used**:  
EfficientNet-B3, fine-tuned end-to-end on the task of predicting direction.

**Label Transformation**:
- Converted angle labels into `(sin(θ), cos(θ))` format to make training smoother over the circular angle space.

**Training Details**:
- **Loss Function**: `MSELoss` on predicted `(sin, cos)` vs ground truth
- **Optimizer**: `Adam` with `lr=1e-4`

**Pre-processing Techniques**:
- Resize to 300x300 (EfficientNet-B3 input size)
- Normalize to ImageNet statistics
- Data augmentations: rotation, color jitter

**Prediction & Post-Processing**:
- Predicted sin and cos values are normalized and converted back to angles using `atan2`.

**Innovative Ideas**:
- Sin-Cos encoding avoids angle discontinuity and improves training stability.
- Applied label-consistent image augmentation for rotation-sensitive learning.

**Model Weights**:  
Angle model link : https://www.kaggle.com/models/meetgera/angle_models_efficientnet


##  Results

| Task       | Metric           | Score     |
|------------|------------------|-----------|
| Region ID  | Accuracy          | 97.5%     |
| Lat-Long   | MSE          |   57.4k km²  |
| Direction  | Mean Angle Error  | 22°      |