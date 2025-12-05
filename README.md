# vignette-cnn

Vignette on building a Convolutional Neural Network (CNN) model for brain tumor medical image classification. This is a group project for PSTAT197A in Fall 2025 under Dr. Coburn's supervision.

## Contributors

Lucas Childs

Sophie Lian

Kaeya Mehta

Janice Jiang

## Vignette Abstract

This vignette introduces the basic knowledge of Convolutional Neural Networks (CNNs) and demonstrates their application for medical image classification using `R` and `Keras`. Using a brain tumor X-ray image dataset from [Kaggle](https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset), the vignette explains the hierarchical feature learning architecture of CNNs, outlines the complete training pipeline from data preprocessing to model evaluation, and visualizes classification performance through confusion matrices and ROC curves. The main document [vignette-cnn.qmd](https://github.com/pstat197/vignette-cnn/blob/main/vignette-cnn.qmd) shows the full pipeline from processing data, splitting data, training models, and validate on test set to save the model. This practical guide demonstrates how CNNs can be effectively applied to medical imaging tasks with limited computational resources while achieving clinically relevant performance.

## Repository Contents

```plaintext
vignette-cnn/
|-- data
    |-- images   # raw image from Kaggle
    |-- train_images   # training images after data splitting
    |-- test_images   # testing images after data splitting
|-- scripts
    |-- drafts
        |-- vignette-script.R
|-- results
    |-- cnn_brain_tumor_model.keras   # saved final model
|-- img
    |-- confusion_matrix.png
    |-- roc_curve.png
|-- vignette-cnn.qmd   # report
|-- vignette-cnn.html  # rendered html report
|-- README.md
```

## Reference

