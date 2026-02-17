
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Libraries ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

library(caret)
library(ggplot2)
library(viridis)
library(patchwork)

# ~~~~~~~~~~~~~~~~~~~~ Source Useful Functions ~~~~~~~~~~~~~~~~~~~~~~~~~

source("Functions//stylometryfunctions.R")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Load Data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

rosie_wd <- "~/University/Year 4/Statistical Case Studies/SCS-Sem2-Project/Data/FunctionWords/"
ella_wd <- "C:/Users/Ella Park/Desktop/Year 4/Sem 1/Stats Case Study/A3/SCS-Sem2-Project/Data/FunctionWords/"
kieran_wd <- "~/SCS-Sem2-Project/Data/FunctionWords/"


words <- loadCorpus(rosie_wd) # only run if necessary - it takes forever!!!


# ~~~~~~~~~~~~~~~~~~~~~~~~~~ Classification Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~


discriminantCorpus <- function(traindata, testdata) {
  thetas <- NULL
  preds <- NULL
  
  #first learn the model for each author
  for (i in 1:length(traindata)) {
    words <- apply(traindata[[i]],2,sum)
    
    #some words might never occur. This will be a problem since it will mean the theta for this word is 0, which means the likelihood will be 0 if this word occurs in the training set. So, we force each word to occur at leats once
    inds <- which(words==0) 
    if (length(inds) > 0) {words[inds] <- 1}
    thetas <- rbind(thetas, words/sum(words))
  }
  
  #now classify
  for (i in 1:nrow(testdata)) {
    probs <- NULL
    for (j in 1:nrow(thetas)) {
      probs <- c(probs, dmultinom(testdata[i,],prob=thetas[j,],log=TRUE))
    }
    preds <- c(preds, which.max(probs))
  }
  return(preds)
}


KNNCorpus <- function(traindata, testdata) {
  train <- NULL
  
  # combine data into profile
  for (i in 1:length(traindata)) {
    train <- rbind(train, apply(traindata[[i]],2,sum))
  }
  
  # convert train data into proportions
  for (i in 1:nrow(train)) {
    train[i,] <- train[i,]/sum(train[i,])
  }
  
  # convert test data into proportions 
  for (i in 1:nrow(testdata)) {
    testdata[i,] <- testdata[i,]/sum(testdata[i,])
  }
  
  # run knn 
  trainlabels <-  1:nrow(train)
  myKNN(train, testdata, trainlabels,k=1)
}

myKNN <- function(traindata, testdata, trainlabels, k=1) {
 
  # convert data to matrices if not already 
  if (mode(traindata) == 'numeric' && !is.matrix(traindata)) {
    traindata <- matrix(traindata,nrow=1)
  }
  if (mode(testdata) == 'numeric' && !is.matrix(testdata)) {
    testdata <- matrix(testdata,nrow=1)
  }
  
  # standardise data
  mus <- apply(traindata,2,mean)
  sigmas <- apply(traindata,2,sd)

  for (i in 1:ncol(traindata)) {
    traindata[,i] <- (traindata[,i] - mus[i])/sigmas[i]
  }

  for (i in 1:ncol(testdata)) {
    testdata[,i] <- (testdata[,i]-mus[i])/sigmas[i]
  }
  
  # predict using knn
  preds <- knn(traindata, testdata, trainlabels, k)
  return(preds)
}


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Cross Validation Functions ~~~~~~~~~~~~~~~~~~~~~~~~~


kfold_cv <- function(data, func, K = 5, R = 1, seed = 0){
  set.seed(seed)
  
  # initialise to store results
  predictions <- NULL
  truth <- NULL
  C <- length(data)  # number of classes
  fold_accuracies <- c()
  
  # combine all data into one matrix
  all_data <- do.call(rbind, data)
  print(nrow(data[[1]]))
  
  # all_labels as factor with levels 1:C
  all_labels <- factor(unlist(lapply(1:C, function(i) rep(i, nrow(data[[i]])))), levels = 1:C)
  N <- nrow(all_data)
  
  # repeat K-fold R times - usually just once
  for (r in 1:R){
    
    # create stratified folds on the full label vector
    folds <- createFolds(all_labels, k = K, list = TRUE, returnTrain = FALSE)
    
    # iterate over folds
    for (k in 1:K){
      
      # indexes for train and test data
      test_idx <- folds[[k]]
      train_idx <- setdiff(1:N, test_idx)
      
      # rebuild train_data as list of classes
      train_data <- vector("list", C)
      for (i in 1:C){
        class_rows <- train_idx[all_labels[train_idx] == i]
        train_data[[i]] <- all_data[class_rows, , drop = FALSE]
      }
      
      # get test data and corresponding labels
      test_data <- all_data[test_idx, , drop = FALSE]
      test_labels <- factor(all_labels[test_idx], levels = levels(all_labels))  # preserve all levels
      
      # call function on all test data at once
      results <- func(train_data, test_data)
      
      # compute fold accuracies 
      fold_acc <- mean(results == test_labels)
      fold_accuracies <- c(fold_accuracies, fold_acc)
      
      # store results
      predictions <- c(predictions, results)
      truth <- c(truth, test_labels)
    }
  }
  
  return(list(
    predictions = predictions,
    truth = truth,
    mean_accuracy = mean(fold_accuracies),
    sd_accuracy = sd(fold_accuracies),
    fold_accuracies = fold_accuracies
  ))
}


# ~~~~~~~~~~~~~~~~~~~~~~~~ KNN Multiclass Classification ~~~~~~~~~~~~~~~~~~~~~~~

features <- words$features
authors <- words$authornames  

# apply cross validation to KNNCorpus - ungrouped
knn_cv <- kfold_cv(features, KNNCorpus, K=10, seed=0)

# make confusion matrix
knn_predictions <- knn_cv[[1]]
knn_truth <- knn_cv[[2]]
knn_cm <- confusionMatrix(as.factor(knn_predictions), as.factor(knn_truth))
knn_cm

# compute standard deviation
knn_sd <-  knn_cv[[4]]
knn_sd

# group results to compare AI vs Human
knn_predictions_grouped <- ifelse(knn_predictions %in% c(1, 2, 4), "AI", "Human")
knn_truth_grouped <- ifelse(knn_truth %in% c(1, 2, 4), "AI", "Human")
knn_cm_grouped <- confusionMatrix(as.factor(knn_predictions_grouped), as.factor(knn_truth_grouped))
knn_cm_grouped

# ~~~~~~~~~~~~~~~ Discriminant Analysis Multiclass Classification ~~~~~~~~~~~~~~

# apply cross validation to discriminant analysis - ungrouped
discriminant_cv <- kfold_cv(features, discriminantCorpus, K=10)

# make confusion matrix
discriminant_predictions <- discriminant_cv[[1]]
discriminant_truth <- discriminant_cv[[2]]
discriminant_cm <- confusionMatrix(as.factor(discriminant_predictions), as.factor(discriminant_truth))
discriminant_cm

# compute standard deviation
discriminant_sd <- discriminant_cv[[4]]
discriminant_sd

# compute precision matrix
discriminant_pm <- prop.table(discriminant_cm$table, margin = 1)*100
discriminant_pm

# compute recall matrix 
discriminant_rm <- prop.table(discriminant_cm$table, margin = 2)*100
discriminant_rm

# store precision matrix in dataframe for plotting 
discriminant_precision_df <- 
  as.data.frame(as.table(discriminant_pm))

colnames(discriminant_precision_df) <- c("Prediction", "Truth", "Proportion") # rename headings

# make numbers match corresponding author name
lookup <- c(
  "1" = "Gemini",
  "2" = "GPT",
  "3" = "Human",
  "4" = "Llama"
)

discriminant_precision_df$Prediction <- lookup[as.character(discriminant_precision_df$Prediction)]
discriminant_precision_df$Truth <- lookup[as.character(discriminant_precision_df$Truth)]

# store recall matrix in dataframe for plotting 
discriminant_recall_df <- as.data.frame(as.table(discriminant_rm))

colnames(discriminant_recall_df) <- c("Prediction", "Truth", "Proportion")

discriminant_recall_df$Prediction <- lookup[as.character(discriminant_recall_df$Prediction)]
discriminant_recall_df$Truth <- lookup[as.character(discriminant_recall_df$Truth)]
discriminant_recall_df


# Precision heatmap
p1 <- ggplot(discriminant_precision_df,
             aes(x = Truth, y = Prediction, fill = Proportion)) +
  geom_tile(color = "white", alpha = 0.9) +
  geom_text(aes(label = sprintf("%.1f", Proportion)),
            size = 4, colour = "white", fontface = "bold") +
  scale_fill_viridis_c(option = "D", begin = 0.1, end = 0.9) +
  labs(title = "Precision Matrix", x = "True Class", y = "Predicted Class") +
  theme_minimal() +
  theme(panel.grid = element_blank())

# Recall heatmap 
p2 <- ggplot(discriminant_recall_df,
             aes(x = Truth, y = Prediction, fill = Proportion)) +
  geom_tile(color = "white", alpha = 0.9) +
  geom_text(aes(label = sprintf("%.1f", Proportion)),
            size = 4, colour = "white", fontface = "bold") +
  scale_fill_viridis_c(option = "D", begin = 0.1, end = 0.9, guide = "none") +  # hide legend
  labs(title = "Recall Matrix", x = "True Class", y = "Predicted Class") +
  theme_minimal() +
  theme(panel.grid = element_blank())

# Combine side by side
p1 + p2 + plot_layout(guides = "collect")  # collects one common legend


# group confusion matrix for AI vs Human comparison
discriminant_predictions_grouped <- ifelse(discriminant_predictions %in% c(1, 2, 4), "AI", "Human")
discriminant_truth_grouped <- ifelse(discriminant_truth %in% c(1, 2, 4), "AI", "Human")
discriminant_cm_grouped <- confusionMatrix(as.factor(discriminant_predictions_grouped), as.factor(discriminant_truth_grouped))
discriminant_cm_grouped

# group precision and recall matrices for AI vs Human comparison
discriminant_pm_grouped <- prop.table(discriminant_cm_grouped$table, margin = 1)*100
discriminant_rm_grouped <- prop.table(discriminant_cm_grouped$table, margin = 2)*100

# precision df 
grouped_precision_df <- 
  as.data.frame(as.table(discriminant_pm_grouped))

colnames(grouped_precision_df) <- 
  c("Prediction", "Truth", "Proportion")

# recall df
grouped_recall_df <- 
  as.data.frame(as.table(discriminant_rm_grouped))

colnames(grouped_recall_df) <- 
  c("Prediction", "Truth", "Proportion")


# Precision heatmap
multiclass_precision <- ggplot(grouped_precision_df,
             aes(x = Truth, y = Prediction, fill = Proportion)) +
  geom_tile(color = "white", alpha = 0.9) +
  geom_text(aes(label = sprintf("%.1f", Proportion)),
            size = 4, colour = "white", fontface = "bold") +
  scale_fill_viridis_c(option = "D", begin = 0.1, end = 0.9) +
  labs(title = "Multiclass Precision Matrix", x = "True Class", y = "Predicted Class") +
  theme_minimal() +
  theme(panel.grid = element_blank())



# ~~~~~~~~~~~~~~~~~~~~~~~~ KNN Binary Classification ~~~~~~~~~~~~~~~~~~~~~~~~~~~

grouped_features <- list(
  AI = rbind(features[[1]], features[[2]], features[[4]]),
  Human = features[[3]]
)

# apply 10-fold cv to grouped featured with KNNCorpus 
knn_cv <- kfold_cv(grouped_features, KNNCorpus, K=10, seed =0)

# create confusion matrix
knn_predictions <- knn_cv[[1]]
knn_truth <- knn_cv[[2]]
knn_cm <- confusionMatrix(as.factor(knn_predictions), as.factor(knn_truth))
knn_cm

# compute standard deviation
knn_sd <- knn_cv[[4]]
knn_sd



# ~~~~~~~~~~~~~~~~~ Discriminant Analysis Binary Classification ~~~~~~~~~~~~~~~~

# apply 10-fold cv to binary discriminant analysis 
discriminant_cv <- kfold_cv(grouped_features, discriminantCorpus, K=10)

# make confusion matrix
discriminant_predictions <- discriminant_cv[[1]]
discriminant_truth <- discriminant_cv[[2]]
discriminant_cm <- confusionMatrix(as.factor(discriminant_predictions), as.factor(discriminant_truth))
discriminant_cm

# compute standard deviation
discriminant_sd <- discriminant_cv[[4]]
discriminant_sd

# make precision and recall matrices
discriminant_pm <- prop.table(discriminant_cm$table, margin = 1)*100
discriminant_rm <- prop.table(discriminant_cm$table, margin = 2)*100

# plot precision matrices
discriminant_precision_df <- 
  as.data.frame(as.table(discriminant_pm))

colnames(discriminant_precision_df) <- 
  c("Prediction", "Truth", "Proportion")

lookup <- c(
  "1" = "AI",
  "2" = "Human"
)

discriminant_precision_df$Prediction <- lookup[as.character(discriminant_precision_df$Prediction)]
discriminant_precision_df$Truth <- lookup[as.character(discriminant_precision_df$Truth)]
discriminant_precision_df


# Precision heatmap
binary_precision <- ggplot(discriminant_precision_df,
                               aes(x = Truth, y = Prediction, fill = Proportion)) +
  geom_tile(color = "white", alpha = 0.9) +
  geom_text(aes(label = sprintf("%.1f", Proportion)),
            size = 4, colour = "white", fontface = "bold") +
  scale_fill_viridis_c(option = "D", begin = 0.1, end = 0.9) +
  labs(title = "Binary Precision Matrix", x = "True Class", y = "Predicted Class") +
  theme_minimal() +
  theme(panel.grid = element_blank())


# Combine side by side
binary_precision + multiclass_precision + plot_layout(guides = "collect")  # collects one common legend
