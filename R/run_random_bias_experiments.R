
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

#damage_tib <- c(0.01) %>%
damage_tib <- c(0:9 / 100, 1:9 / 10, 91:99 / 100) %>%
  sort() %>%
  map_dfr(run_random_damage_exp, x_train, y_train, x_test, y_test)

damage_tib %>% write_csv("fashion-mnist-damage-results.csv")

damage_tib <- read_csv("fashion-mnist-damage-results.csv") %>%
  mutate(unbiased = 1-frac)

ggplot(damage_tib, aes(x=unbiased, y=acc, color=exp_name)) +
  geom_line(size=1) + geom_point(color="white", size = 0.2) +
  ggthemes::scale_color_few("Model Type", palette = "Dark") +
  ggthemes::theme_few() +
  scale_x_continuous(labels = scales::percent) +
  labs(
    title = "Model Architectures and Random Bias",
    subtitle = "Fashion MNIST Dataset",
    x = "Correctly labeled training data\n(percent of 60,000 obs)",
    y = "Accuracy (OVA)"
  )

ggsave(filename=here::here("plot/acc-rand-bias.png"), width = 12, height =7, dpi=200)

ggplot(damage_tib, aes(x=unbiased, y=auc, color=exp_name)) +
  geom_line(size=1) + geom_point(color="white", size = 0.2) +
  ggthemes::scale_color_few("Model Type", palette = "Dark") +
  ggthemes::theme_few() +
  scale_x_continuous(labels = scales::percent) +
  labs(
    title = "Model Architectures and Random Bias",
    subtitle = "Fashion MNIST Dataset",
    x = "Correctly labeled training data\n(percent of 60,000 obs)",
    y = "AUC (OVA)"
  )

ggsave(filename=here::here("plot/auc-rand-bias.png"), width = 12, height =7, dpi=200)
