
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
  #ggthemes::scale_color_few("Model Type", palette = "Dark") +
  ggthemes::theme_few() +
  scale_color_manual("Model Type", values = alegion_pal) +
  theme(plot.background = element_rect(fill="#000000"),
        panel.background = element_rect(fill="#000000"),
        plot.title = element_text(colour="#ffffff"),
        plot.subtitle = element_text(colour="#ffffff"),
        axis.title = element_text(color="#ffffff"),
        axis.text = element_text(color="#ffffff"),
        legend.background = element_rect(fill="#000000"),
        legend.title = element_text(color="#ffffff"),
        legend.text = element_text(color="#ffffff"),
        legend.key = element_rect(fill="#000000"))+
  labs(
    title = "Model Architectures and Training Batch Size",
    subtitle = "Fashion MNIST Dataset",
    x = "Fraction of Training Data\n(60,000 * x = # of samples)",
    y = "Inference Accuracy (OVA)"
  )

ggsave(filename=here::here("plot/black-acc-size.png"),
       width = 16 * (1/3),
       height = 9 * (1/3),
       dpi = 300)

ggplot(tib, aes(x=frac, y=auc, color=exp_name)) +
   geom_line(size=1) + geom_point(color="white", size = 0.2) +
   #ggthemes::scale_color_few("Model Type", palette = "Dark") +
   ggthemes::theme_few() +
  scale_color_manual("Model Type", values = alegion_pal) +
  theme(plot.background = element_rect(fill="#000000"),
        panel.background = element_rect(fill="#000000"),
        plot.title = element_text(colour="#ffffff"),
        plot.subtitle = element_text(colour="#ffffff"),
        axis.title = element_text(color="#ffffff"),
        axis.text = element_text(color="#ffffff"),
        legend.background = element_rect(fill="#000000"),
        legend.title = element_text(color="#ffffff"),
        legend.text = element_text(color="#ffffff"),
        legend.key = element_rect(fill="#000000"))+
   labs(
     title = "Model Architectures and Training Batch Size",
     subtitle = "Fashion MNIST Dataset",
     x = "Fraction of Training Data\n(60,000 * x = # of samples)",
     y = "AUC (OVA)"
   )

ggsave(filename=here::here("plot/black-auc-size.png"),
       width = 16 * (1/3),
       height = 9 * (1/3),
       dpi = 300)
