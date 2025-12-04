# -----------------------
# 1. Libraries
# -----------------------
# load required libraries

library(keras)
library(tensorflow)
library(ggplot2)
library(caret)
library(dplyr)
library(reticulate)
library(pROC)
library(readr)
library(fs)
library(rsample)
library(tidyverse)



# load the data
metadata <- read_csv("data/metadata.csv.xls")

# preview the data and column labels
head(metadata)
colnames(metadata)

# setting a seed for reproducibility
set.seed(11272025)

# split data with 80% of the data in the training set and 20% in test set
train_index <- createDataPartition(metadata$class, p = 0.8, list = FALSE)

train_data <- metadata[train_index, ]
test_data  <- metadata[-train_index, ]

# Check sizes
nrow(train_data)
nrow(test_data)

# Create folders for test and training images
dir_create("data/images", showWarnings = FALSE)
dir_create("data/train_images")
dir_create("data/test_images")

# Copy each file based on split into training and test sets
file_copy(
  path = paste0("data/images/", train_data$image),
  new_path = paste0("data/train_images/", train_data$image)
)

file_copy(
  path = paste0("data/images/", test_data$image),
  new_path = paste0("data/test_images/", test_data$image)
)





# CNN Image Generators

# Sets the target image size after resizing (224×224)
# batch_size = number of images processed in each training step
img_height <- 224
img_width  <- 224
batch_size <- 32

# training generator that loads images and applies augmentation
# this augmentation prevents overfitting by generating diverse images
train_gen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 10,
  width_shift_range = 0.1,
  height_shift_range = 0.1,
  horizontal_flip = TRUE
)

# create testing generator with no augmentation of images
test_gen <- image_data_generator(rescale = 1/255)

# flow images from training dataframe
# read images in batches from the folder train_images
train_flow <- flow_images_from_dataframe(
  dataframe = train_data,
  directory = "data/train_images",
  x_col = "image",
  y_col = "class",
  generator = train_gen,
  target_size = c(img_height, img_width),
  batch_size = batch_size,
  class_mode = "binary"
)

# flow images from the test dataframe
test_flow <- flow_images_from_dataframe(
  dataframe = test_data,
  directory = "data/test_images",
  x_col = "image",
  y_col = "class",
  generator = test_gen,
  target_size = c(img_height, img_width),
  batch_size = batch_size,
  class_mode = "binary",
  shuffle = FALSE
)




# Build CNN Model

model <- keras_model_sequential(list(
  # 32 convolution filters of size 3x3
  layer_conv_2d(filters = 32, kernel_size = 3, activation = "relu",
                input_shape = c(img_height, img_width, 3)),
  layer_max_pooling_2d(pool_size = 2),
  
  # 64 filters
  layer_conv_2d(filters = 64, kernel_size = 3, activation = "relu"),
  layer_max_pooling_2d(pool_size = 2),
  
  # 128 filters
  layer_conv_2d(filters = 128, kernel_size = 3, activation = "relu"),
  layer_max_pooling_2d(pool_size = 2),
  
  # Flatten and dense layers
  layer_flatten(),
  layer_dense(units = 128, activation = "relu"),
  layer_dropout(rate = 0.4),
  
  # Output layer
  layer_dense(units = 1, activation = "sigmoid")
))

# Compile the model
model$compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = list("accuracy")
)

model





# Train CNN
epochs <- 15

history <- model$fit(
  train_flow,
  steps_per_epoch = r_to_py(as.integer(ceiling(nrow(train_data) / batch_size))),
  validation_data = test_flow,
  validation_steps = r_to_py(as.integer(ceiling(nrow(test_data) / batch_size))),
  epochs = r_to_py(as.integer(epochs))
)


# Convert Python history to R list
history_values <- py_to_r(history$history)

# Plot loss
plot(1:epochs, history_values$loss, type = "l", col = "blue", lwd = 2,
     xlab = "Epoch", ylab = "Loss", ylim = range(c(history_values$loss, history_values$val_loss)))
lines(1:epochs, history_values$val_loss, col = "red", lwd = 2)
legend("topright", legend = c("Training Loss", "Validation Loss"),
       col = c("blue", "red"), lwd = 2)

# Plot accuracy
plot(1:epochs, history_values$accuracy, type = "l", col = "blue", lwd = 2,
     xlab = "Epoch", ylab = "Accuracy", ylim = range(c(history_values$accuracy, history_values$val_accuracy)))
lines(1:epochs, history_values$val_accuracy, col = "red", lwd = 2)
legend("bottomright", legend = c("Training Accuracy", "Validation Accuracy"),
       col = c("blue", "red"), lwd = 2)





# Evaluate
scores <- model$evaluate(test_flow)

# Convert Python evaluation output to R list
scores_r <- py_to_r(scores)
test_loss <- scores_r[[1]]
test_accuracy <- scores_r[[2]]

cat("Test loss:", test_loss, "\n")
# 0.2428377
cat("Test accuracy:", test_accuracy, "\n")
# 0.9075081



# Confusion Matrix

# Get predicted probabilities
pred_probs <- model$predict(test_flow)

# Convert probabilities to class labels (0 or 1)
pred_labels <- ifelse(pred_probs >= 0.5, 1, 0)

true_labels <- test_flow$classes  # returns the true class indices



# Use caret to show detailed confusion metrics (requires factor inputs)
cm <- caret::confusionMatrix(factor(pred_labels), factor(true_labels), 
                             positive = "1")

# Plot confusion matrix heatmap using ggplot
cm_df <- as.data.frame(cm_table)
colnames(cm_df) <- c("Predicted", "Actual", "Freq")


p_cm <- ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 6) +
  labs(title = "Confusion Matrix", x = "Actual", y = "Predicted") +
  theme_minimal()
p_cm


ggsave("img/confusion_matrix.png", plot = p_cm, width = 6, height = 5)






# ROC and AUC 

roc_obj <- roc(response = true_labels, predictor = as.numeric(pred_probs))
auc_val <- auc(roc_obj)
cat("AUC:", auc_val, "\n")
# 0.9750686

png("img/roc_curve.png", width = 900, height = 700)  # open PNG device
plot(roc_obj, col = "#2C3E50", lwd = 4, 
     main = paste0("ROC Curve (AUC = ", round(auc_val, 3), ")"))
grid()  # optional grid lines
dev.off()  # close device




# Save model

dir_create("results")
model$save("results/cnn_brain_tumor_model.keras")

cat("Model saved to results/cnn_brain_tumor_model.h5\n")




