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


rf2_exp <- function(x_train, y_train, x_test, y_test) {
  num_trees <- 25
  dim(x_train) <- c(nrow(x_train), 784)
  dim(x_test) <- c(nrow(x_test), 784)
  x_train <- x_train / 255
  x_test <- x_test / 255

  rf <- randomForest(x_train, as.factor(y_train),
                     ntree=num_trees)

  # random forest returns a factor. that's cool and all, but
  # I just want the raw preditions.
  predict(rf, x_test, type="prob")
}

dnn2_exp <- function(x_train, y_train, x_test, y_test) {
  # reshape
  x_train <- array_reshape(x_train, c(nrow(x_train), 784))
  x_test <- array_reshape(x_test, c(nrow(x_test), 784))
  # rescale
  x_train <- x_train / 255
  x_test <- x_test / 255

  y_train <- to_categorical(y_train, 10)
  y_test <- to_categorical(y_test, 10)

  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>%
    layer_dropout(rate = 0.4) %>%
    layer_dense(units = 128, activation = 'relu') %>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units = 10, activation = 'softmax')

  model %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy')
  )

  history <- model %>% fit(
    x_train, y_train,
    epochs = 30, batch_size = 128,
    validation_split = 0.2,
    verbose=0
  )

  model %>% predict_proba(x_test)
}




library(caret)


mnist <- keras::dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y
frac <- 0.1
run_exp2 <- function(frac, x_train, y_train, x_test, y_test) {

  samples <- sample(nrow(x_train), floor(nrow(x_train) * frac), replace=FALSE)

  x_t <- x_train[samples,,]
  y_t <- y_train[samples]

  args <- list(x_train = x_t,
               y_train = y_t,
               x_test = x_test,
               y_test = y_test)

  experiments <- c(rf2_exp, dnn2_exp)

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

####
# wow
###
tibs <- preds %>%
  map(~ as_tibble(., .name_repair = "universal")) %>%
  map( ~ {
    set_colnames(.x, levels(factor(y_test)))
  }) %>%
  map(. %>%
        mutate('pred' = names(.)[apply(., 1, which.max)])) %>%
  map( ~ {
    cbind(., obs = factor(y_test))
  })

df <- cbind(obs=factor(y_test),
            preds[[1]] %>%
              as_tibble() %>%
              mutate('pred'=names(.)[apply(., 1, which.max)]))

df$pred = factor(df$pred)

mcs <- multiClassSummary(df,lev=levels(df$obs))

