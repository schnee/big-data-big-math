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

cnn2_exp <- function(x_train, y_train, x_test, y_test) {

  batch_size <- 128
  num_classes <- 10
  epochs <- 12

  # Input image dimensions
  img_rows <- 28
  img_cols <- 28

  # Redefine  dimension of train/test inputs
  x_train <- array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, 1))
  x_test <- array_reshape(x_test, c(nrow(x_test), img_rows, img_cols, 1))
  input_shape <- c(img_rows, img_cols, 1)

  # Transform RGB values into [0,1] range
  x_train <- x_train / 255
  x_test <- x_test / 255

  # Convert class vectors to binary class matrices
  y_train <- to_categorical(y_train, num_classes)
  y_test <- to_categorical(y_test, num_classes)

  # Define Model -----------------------------------------------------------

  # Define model
  model <- keras_model_sequential() %>%
    layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                  input_shape = input_shape) %>%
    layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_dropout(rate = 0.25) %>%
    layer_flatten() %>%
    layer_dense(units = 128, activation = 'relu') %>%
    layer_dropout(rate = 0.5) %>%
    layer_dense(units = num_classes, activation = 'softmax')

  # Compile model
  model %>% compile(
    loss = loss_categorical_crossentropy,
    optimizer = optimizer_adadelta(),
    metrics = c('accuracy')
  )

  # Train model
  model %>% fit(
    x_train, y_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_split = 0.2,
    verbose=0
  )

  model %>% predict_proba(x_test)
}

xgb2_exp <- function(x_train, y_train, x_test, y_test) {
  dim(x_train) <- c(nrow(x_train), 784)
  dim(x_test) <- c(nrow(x_test), 784)

  x_train <- x_train / 255
  x_test <- x_test / 255

  dtrain <- xgb.DMatrix(x_train, label = y_train)
  dtest <- xgb.DMatrix(x_test, label = y_test)
  train.gdbt<-xgb.train(params=list(objective="multi:softprob",
                                    num_class=10, eval_metric="mlogloss",
                                    eta=0.2, max_depth=5,
                                    subsample=1, colsample_bytree=0.5),
                        data=dtrain,
                        nrounds=150)

  p<-predict(train.gdbt, newdata = dtest)

  matrix(p, nrow=nrow(x_test), byrow=TRUE)

}


library(caret)


mnist <- keras::dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y
frac <- 0.05
run_exp2 <- function(frac, x_train, y_train, x_test, y_test) {

  samples <- sample(nrow(x_train), floor(nrow(x_train) * frac), replace=FALSE)

  x_t <- x_train[samples,,]
  y_t <- y_train[samples]

  args <- list(x_train = x_t,
               y_train = y_t,
               x_test = x_test,
               y_test = y_test)

  experiments <- c(rf2_exp, xgb2_exp, dnn2_exp, cnn2_exp)

  preds <- experiments %>%
    map(exec, !!!args)

  # transform each matrix into a tibble, appending two
  # new columns: pred and obs
  tibs <- preds %>%
    map(~ as_tibble(., .name_repair = "universal")) %>%
    map( ~ {
      set_colnames(.x, y_test %>% factor %>% levels %>% sort)
    }) %>%
    map(. %>%
          mutate('pred' = names(.)[apply(., 1, which.max)])) %>%
    map( ~ {
      cbind(., obs = factor(y_test))
    })

  accs <- tibs %>%
    map(~{pull(.,pred)}) %>%
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
    set_colnames(.x, y_test %>% factor %>% levels %>% sort)
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

