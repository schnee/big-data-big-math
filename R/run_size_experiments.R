
library(keras)
library(ggplot2)
library(purrr)
library(readr)
library(tibble)
library(dplyr)
library(magrittr)

devtools::load_all(here::here("packages/testbench"))


mnist <- keras::dataset_fashion_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

#
tib <- c(0.05) %>%
tib <- c(1:9 / 100, 1:9 / 10, 91:100 / 100) %>%
  sort() %>%
  map_dfr(run_size_exp, x_train, y_train, x_test, y_test)

 tib %>% write_csv("./fashion-data-size-results.csv")

tib <- read_csv("./fashion-data-size-results.csv")

ggplot(tib, aes(x=frac, y=acc, color=exp_name)) +
  geom_line(size=1) + geom_point(color="white", size = 0.2) +
  ggthemes::scale_color_few("Model Type", palette = "Dark") +
  ggthemes::theme_few() +
  labs(
    title = "Model Architectures and Training Batch Size",
    subtitle = "Fashion MNIST Dataset",
    x = "Fraction of Training Data\n(60,000 * x = # of samples)",
    y = "Inference Accuracy (OVA)"
  )

ggplot(tib, aes(x=frac, y=auc, color=exp_name)) +
   geom_line(size=1) + geom_point(color="white", size = 0.2) +
   ggthemes::scale_color_few("Model Type", palette = "Dark") +
   ggthemes::theme_few() +
   labs(
     title = "Model Architectures and Training Batch Size",
     subtitle = "Fashion MNIST Dataset",
     x = "Fraction of Training Data\n(60,000 * x = # of samples)",
     y = "AUC (OVA)"
   )
