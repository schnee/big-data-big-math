
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

damage_frac <- c(0:5 / 10) %>%
  sort()

sample_frac <- c(1:9/100, 1:9 / 10, 91:100 / 100) %>%
  sort()

cross_prod <- cross_df(list(d=damage_frac, s=sample_frac))

results_tib <-
  map2_dfr(cross_prod$d, cross_prod$s, run_cnn_damage_exp, x_train, y_train, x_test, y_test)

results_tib %>% write_csv("fashion-mnist-size_damage-results.csv")

results_tib <- read_csv("fashion-mnist-size_damage-results.csv") %>%
  mutate(unbiased = 1-frac_damage,
         frac_damage = as.factor(frac_damage))

ggplot(results_tib, aes(x=frac_sample, y=acc, color=frac_damage, group=frac_damage)) +
  geom_line(size=1) + geom_point(color="white", size = 0.2) +
  ggthemes::scale_color_few("Bias Fraction", palette = "Dark") +
  ggthemes::theme_few() +
  scale_x_continuous(labels = scales::percent) +
  labs(
    title = "CNN Architecture and Bias",
    subtitle = "Fashion MNIST Dataset",
    x = "Number of Samples\n(percent of 60,000 obs)",
    y = "Accuracy (OVA)"
  )

ggsave(filename=here::here("plot/acc-cnn-bias.png"),
       width = 16 * (1/3),
       height = 9 * (1/3),
       dpi = 300)

ggplot(results_tib, aes(x=frac_sample, y=auc, color=frac_damage, group=frac_damage)) +
  geom_line(size=1) + geom_point(color="white", size = 0.2) +
  ggthemes::scale_color_few("Bias Fraction", palette = "Dark") +
  ggthemes::theme_few() +
  scale_x_continuous(labels = scales::percent) +
  labs(
    title = "CNN Architecture and Bias",
    subtitle = "Fashion MNIST Dataset",
    x = "Number of Samples\n(percent of 60,000 obs)",
    y = "AUC (OVA)"
  )

ggsave(filename=here::here("plot/auc-cnn-bias.png"),
       width = 16 * (1/3),
       height = 9 * (1/3),
       dpi = 300)

results_tib %>%
  filter(auc > .98) %>%
  group_by(unbiased) %>%
  top_n(1, -auc) %>% ungroup() %>%
  mutate(num_obs = frac_sample * 60000) %>%
  mutate(delta = round((num_obs - lag(num_obs)) / lag(num_obs), digits=2)) %>%
  mutate(auc = round(auc, digits =3)) %>%
  select(unbiased, auc, frac_sample, num_obs, delta) %>% knitr::kable()
