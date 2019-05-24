
frac_bad <- 0.0

apply_damage <- function(damage_fraction, the_vector_to_damage) {

  bad_labels <- the_vector_to_damage %>%
    enframe() %>%
    group_by(value) %>%
    sample_frac(damage_fraction) %>%
    ungroup() %>%
    mutate(value = value + sample(1:9, nrow(.), replace=TRUE)) %>%
    mutate(value = value %% 10)

  the_vector_to_damage %>% enframe() %>% filter(!(name %in% bad_labels$name)) %>%
    bind_rows(bad_labels) %>% arrange(name) %>% pull(value)

}

apply_damage(0.3, y_train)

