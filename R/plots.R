library(keras)
library(dplyr)
library(tibble)
library(tidyr)
library(stringr)

make_mnist_df <- function(the_index) {
  mnist$test$x[the_index, , ] %>%
    as.matrix(nrow = 28, ncol = 28) %>%
    as_tibble() %>%
    mutate(label = mnist$test$y[the_index]) %>%
    mutate(index = the_index) %>%
    rownames_to_column(var = 'y') %>%
    gather(x, val, V1:V28) %>%
    mutate(x = str_replace(x, 'V', '')) %>%
    mutate(x = as.numeric(x),
           y = as.numeric(y)) %>%
    mutate(y = 28 - y)
}



mnist <- keras::dataset_mnist()

the_index <- mnist$test$y %>% enframe() %>% group_by(value) %>%
  sample_n(20) %>% ungroup() %>% pull(name)

dfs <- the_index %>% map(make_mnist_df) %>% bind_rows()

ggplot(dfs, aes(x, y)) +
  geom_tile(aes(fill = val + 1)) +
  coord_fixed() +
  scale_fill_gradient2(
    low = "white",
    high = "black",
    mid = "gray",
    midpoint = 127.5
  ) +
  facet_wrap(vars(label, index), ncol = 20, nrow = 10) +
  theme_void() +
  theme(legend.position = "none") + theme(strip.background = element_blank(),
                                          strip.text.x = element_blank())
ggsave(
  here::here("plot/normal-digits-data.png"),
  width = 16 * (2/3),
  height = 9 * (2/3),
  dpi = 300
)
