library(ggplot2)
library(ggrepel)

mnist <- keras::dataset_fashion_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

the_vector_to_damage <- y_train
damage_fraction <- 0.3

random_damage <- function(damage_fraction, the_vector_to_damage) {
  damaged_tib <- the_vector_to_damage %>%
    enframe() %>%
    mutate(damaged = value)

  bad_labels <- damaged_tib %>%
    group_by(value) %>%
    sample_frac(damage_fraction) %>%
    ungroup() %>%
    mutate(damaged = value + sample(1:9, nrow(.), replace = TRUE)) %>%
    mutate(damaged = damaged %% 10)

  damaged_tib %>%
    filter(!(name %in% bad_labels$name)) %>%
    bind_rows(bad_labels) %>%
    arrange(name)
}

const_damage <- function(damage_fraction, the_vector_to_damage) {
  damaged_tib <- the_vector_to_damage %>%
    enframe() %>%
    mutate(damaged = value)

  bad_labels <- damaged_tib %>%
    group_by(value) %>%
    sample_frac(damage_fraction) %>%
    ungroup() %>%
    mutate(damaged = value + 3) %>%
    mutate(damaged = damaged %% 10)

  damaged_tib %>%
    filter(!(name %in% bad_labels$name)) %>%
    bind_rows(bad_labels) %>%
    arrange(name)
}

blank <- tibble(damaged = c(0:9),
                n = 0)

random_damage(0.3, y_train) %>%
  filter(value == 5) %>%
  group_by(damaged) %>% tally() %>%
  ggplot(aes(x=factor(damaged), y = n)) + geom_col() +
  ggthemes::theme_few() +
  geom_label(aes(label = n)) +
  labs(
    title = "Randomly Biased Labels",
    subtitle = "True Label = 5",
    caption = "30% Damage",
    x = "Damaged Label",
    y = "Count"
  )

ggsave(
  here::here("plot/random-bias.png"),
  width = 16 * (2/3),
  height = 9 * (2/3),
  dpi = 300
)


const_damage(0.3, y_train) %>%
  filter(value == 5) %>%
  group_by(damaged) %>% tally()  %>%
  ggplot(aes(x=factor(damaged), y = n)) + geom_col() +
  ggthemes::theme_few() +
  geom_label(aes(label = n)) +
  labs(
    title = "Constant Biased Labels",
    subtitle = "True Label = 5",
    caption = "30% Damage",
    x = "Damaged Label",
    y = "Count"
  )

ggsave(
  here::here("plot/constant-bias.png"),
  width = 16 * (2/3),
  height = 9 * (2/3),
  dpi = 300
)

