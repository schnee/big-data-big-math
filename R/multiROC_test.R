library(multiROC)
library(dplyr)
library(ggplot2)

data(iris)
head(iris)

set.seed(123456)
total_number <- nrow(iris)
train_idx <- sample(total_number, round(total_number*0.6))
train_df <- iris[train_idx, ]
test_df <- iris[-train_idx, ]

rf_res <- randomForest::randomForest(Species~., data = train_df, ntree = 100)
rf_pred <- predict(rf_res, test_df, type = 'prob')
rf_pred <- data.frame(rf_pred)
colnames(rf_pred) <- paste(colnames(rf_pred), "_pred_RF")

mn_res <- nnet::multinom(Species ~., data = train_df)
mn_pred <- predict(mn_res, test_df, type = 'prob')
mn_pred <- data.frame(mn_pred)
colnames(mn_pred) <- paste(colnames(mn_pred), "_pred_MN")

true_label <- dummies::dummy(test_df$Species)
true_label <- data.frame(true_label)
colnames(true_label) <- gsub(".*?\\.", "", colnames(true_label))
colnames(true_label) <- paste(colnames(true_label), "_true")
final_df <- cbind(true_label, rf_pred, mn_pred)

roc_res <- multi_roc(final_df, force_diag=T)
pr_res <- multi_pr(final_df, force_diag=T)

