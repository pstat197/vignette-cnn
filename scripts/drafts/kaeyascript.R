
# import library
library(keras)
library(tensorflow)
library(readr)
library(dplyr)   

data <- read_csv("/Users/kaeya/PSTAT 197A/vignette-cnn/data/metadata.csv")

n <- nrow(data)
train_idx <- sample(seq_len(n), size = 0.8 * n)  # 80% train
train_data <- data[train_idx, ]
test_data  <- data[-train_idx, ]

train_gen <- flow_images_from_dataframe(
  dataframe   = train_data,
  directory   = "/Users/kaeya/PSTAT 197A/vignette-cnn/data/Brain Tumor Data Set",
  x_col       = "image",
  y_col       = "class",
  target_size = c(224, 224),
  color_mode  = "rgb",
  class_mode  = "binary",
  batch_size  = 32,
  shuffle     = TRUE,
  seed        = 123
)

test_gen <- flow_images_from_dataframe(
  dataframe   = test_data,
  directory   = "/Users/kaeya/PSTAT 197A/vignette-cnn/data/Brain Tumor Data Set",
  x_col       = "image",
  y_col       = "class",
  target_size = c(224, 224),
  color_mode  = "rgb",
  class_mode  = "binary",
  batch_size  = 32,
  shuffle     = FALSE
)

model <- keras_model_sequential() %>%
  
  # 1st convolutional block
  layer_conv_2d(
    filters = 32, kernel_size = c(3,3), activation = "relu",
    input_shape = c(224, 224, 3)
  ) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  # 2nd convolutional block
  layer_conv_2d(
    filters = 64, kernel_size = c(3,3), activation = "relu"
  ) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  # Flatten + dense layers
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu") %>%
  
  # Output layer (binary classification)
  layer_dense(units = 1, activation = "sigmoid")


model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = "accuracy"
)

history <- model %>% fit(
  train_gen,
  epochs = 10,
  validation_data = test_gen
)

model %>% evaluate(test_gen)


