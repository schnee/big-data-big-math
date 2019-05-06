library(purrr)

a_fn <- function(x){
  dim(x) = eval(c(1))
  x + 1
}

b_fn <- function(x) {
  x + 2
}

c_fn <- function(x) {
  x + 3
}

args <- list(x = 1)

c(a_fn, b_fn, c_fn) %>%
  map_dbl(exec, !!!args)

frac <- 0.1
run_exp2 <- function(frac, x_train, y_train, x_test, y_test) {

  samples <- sample(nrow(x_train), floor(nrow(x_train) * frac), replace=FALSE)

  x_t <- x_train[samples,,]
  y_t <- y_train[samples]

  args <- list(x_train = x_t,
               y_train = y_t,
               x_test = x_test,
               y_test = y_test)

  experiments <- c(rf_exp, xgb_exp, dnn_exp, cnn_exp)

  preds <- experiments %>%
    map(exec, !!!args)

  accs <- preds %>%
    map(table, y_test) %>%
    map(diag) %>%
    map(sum) %>%
    map_dbl(.f = function(x){x/nrow(y_test)})

  tibble(frac = frac,
         exp_name = c("rf", "xgb", "dnn", "cnn"),
         acc = accs)
}


c(0.1, 0.2) %>% map_df(run_exp2, x_train, y_train, x_test, y_test)
