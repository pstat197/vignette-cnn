# vignette-cnn

Vignette on building a Convolutional Neural Network (CNN) model for brain tumor medical image classification. This is a group project for PSTAT197A in Fall 2025 under Dr. Coburn's supervision.

## Contributors

Lucas Childs, Sophie Lian, Kaeya Mehta, Janice Jiang

## Vignette Abstract

This vignette introduces the basic knowledge of Convolutional Neural Networks (CNNs) and demonstrates their application for medical image classification using `R` and `Keras`. Using a brain tumor X-ray image dataset from [Kaggle](https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset), the vignette explains the hierarchical feature learning architecture of CNNs, outlines the complete training pipeline from data preprocessing to model evaluation, and visualizes classification performance through confusion matrices and ROC curves. The main document [vignette-cnn.qmd](https://github.com/pstat197/vignette-cnn/blob/main/vignette-cnn.qmd) explains the core concepts behind CNN architecture and shows the full pipeline from processing data, splitting data, training models, and validate on test set to save the model. This practical guide demonstrates how CNNs can be effectively applied to medical imaging tasks with limited computational resources while achieving clinically relevant performance.

## Repository Contents

### Structure

```plaintext
vignette-cnn/
  |-- data
      |-- images       
      |-- train_images  
      |-- test_images   
      |-- metadata.csv.xls
  |-- scripts
      |-- drafts
      |-- vignette-script.R
  |-- results
      |-- cnn_brain_tumor_model.keras   
  |-- img
      |-- confusion_matrix.png
      |-- roc_curve.png
  |-- vignette-cnn.qmd   
  |-- vignette-cnn.html 
  |-- README.md
```

### Key Files:

1. **Data**
    - `images/` - Contains all raw images from the Kaggle dataset.
    - `train_images/` - Contains the training images after preprocessing steps.
    - `test_images/` - Contains the testing images after preprocessing.
    - `metadata.csv.xls` - Includes the full dataset, listing each image name, class (tumor or normal), file format, mode, and shape.

2. **Scripts**
    - `drafts/` - Contains individual contributor drafts of the model throughout development.
    - `vignette_script.R` - The final script covering data upload, preprocessing, model training, and performance analysis.

3. **Results**
    - `cnn_brain_tumor_model.keras` - The saved final CNN model.

4. **`vignette-cnn.qmd`** - The final document explaining how CNNs work, describing the dataset, and providing a walkthrough of each part of the code.

## Reference

1. LeCun, Yann, Léon Bottou, Yoshua Bengio, and Patrick Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE 86, no. 11 (2002): 2278-2324.
2. Yse, Diego Lopez. Computer Vision | Image Classification using Convolutional Neural Networks (CNNs), October 30, 2024. https://medium.com/@lopezyse/computer-vision-image-classification-using-python-913cf7156812. 
