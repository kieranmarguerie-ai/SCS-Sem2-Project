
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Libraries ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

library(caret)
library(ggplot2)
library(viridis)
library(patchwork)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Load Data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

rosie_wd <- "~/University/Year 4/Statistical Case Studies/SCS-Sem2-Project/Data/FunctionWords/"
ella_wd <- "C:/Users/Ella Park/Desktop/Year 4/Sem 1/Stats Case Study/A3/SCS-Sem2-Project/Data/FunctionWords/"
kieran_wd <- "~/SCS-Sem2-Project/Data/FunctionWords/"

# words <- loadCorpus(rosie_wd) # only run if necessary - it takes forever!!!

# Combined the data into one
X <- rbind(
  words$features[[1]],
  words$features[[2]],
  words$features[[3]],
  words$features[[4]]
)

# Label data based on author 
y <- c(
  rep("Gemini", nrow(words$features[[1]])),
  rep("GPT",   nrow(words$features[[2]])), 
  rep("Human",   nrow(words$features[[3]])), 
  rep("Llama",   nrow(words$features[[4]]))
)

# ~~~~~~~~~~~~~~~~~~~~ Source Useful Functions ~~~~~~~~~~~~~~~~~~~~~~~~~


source("Functions//stylometryfunctions.R")


# ~~~~~~~~~~~ KNN (Documents) Cross Validation Functions ~~~~~~~~~~~~~~~~~

# knn cv func
knn_cv_k <- function(X, y, k_values = 1:10, numfolds = 10, seed = 0) {
  
  set.seed(seed)
  
  N <- nrow(X)
  
  # create stratified folds 
  folds <- sample(rep(1:numfolds, length.out = N))
  
  # initialise to store results
  results <- numeric(length(k_values))
  
  # loops across k for k nearest neighbours
  for (j in seq_along(k_values)) {
    
    k <- k_values[j]
    fold_acc <- numeric(numfolds)
    
    
    # loop across folds
    for (i in 1:numfolds) {
      
      # test and train indexes
      test_idx <- which(folds == i)
      train_idx <- which(folds != i)
      
      # extract train and test data and corresponding labels
      traindata <- X[train_idx, , drop = FALSE]
      testdata  <- X[test_idx, , drop = FALSE]
      trainlabels <- y[train_idx]
      testlabels  <- y[test_idx]
      
      # make predictions using myKNN
      preds <- myKNN(traindata, testdata, trainlabels, k)
      
      # compute accuracies
      fold_acc[i] <- mean(preds == testlabels)
    }
    
    results[j] <- mean(fold_acc)
  }
  
  return(results)
}

knn_cv_k.2 <- function(X, y, k_values = 1:10, numfolds = 10, seed = 1,
                     relative_freq = TRUE) {
  
  set.seed(seed)
  N <- nrow(X)
  
  # convert counts to relative frequencies
  if (relative_freq) {
    row_totals <- rowSums(X)
    
    # avoid divide-by-zero for empty documents
    row_totals[row_totals == 0] <- 1
    
    X <- X / row_totals
  }
  
  # create stratified folds 
  folds <- sample(rep(1:numfolds, length.out = N))
  
  # initialise to store results
  results <- numeric(length(k_values))
  
  # loops across k for k nearest neighbours
  for (j in seq_along(k_values)) {
    
    k <- k_values[j]
    fold_acc <- numeric(numfolds)
    
    
    # loop across folds
    for (i in 1:numfolds) {
      
      # test and train indexes
      test_idx <- which(folds == i)
      train_idx <- which(folds != i)
      
      # extract train and test data and corresponding labels
      traindata <- X[train_idx, , drop = FALSE]
      testdata  <- X[test_idx, , drop = FALSE]
      trainlabels <- y[train_idx]
      testlabels  <- y[test_idx]
      
      # make predictions using myKNN
      preds <- myKNN(traindata, testdata, trainlabels, k)
      
      # compute accuracies
      fold_acc[i] <- mean(preds == testlabels)
    }
    
    results[j] <- mean(fold_acc)
  }
  
  return(results)
}

knn_cv_k.3 <- function(X, y, k_values = 1:10, numfolds = 10, seed = 1,
                       relative_freq = TRUE, group = FALSE) {
  
  set.seed(seed)
  N <- nrow(X)
  
  # convert counts to relative frequencies
  if (relative_freq) {
    row_totals <- rowSums(X)
    row_totals[row_totals == 0] <- 1
    X <- X / row_totals
  }
  
  # create stratified folds
  folds <- sample(rep(1:numfolds, length.out = N))
  
  # initialise to store results
  results <- numeric(length(k_values))
  
  # loops across k for k nearest neighbours
  for (j in seq_along(k_values)) {
    
    k <- k_values[j]
    fold_acc <- numeric(numfolds)
    
    
    # loop across folds
    for (i in 1:numfolds) {
      
      # test and train indexes
      test_idx <- which(folds == i)
      train_idx <- which(folds != i)
      
      # extract train and test data and corresponding labels
      traindata <- X[train_idx, , drop = FALSE]
      testdata  <- X[test_idx, , drop = FALSE]
      trainlabels <- y[train_idx]
      testlabels  <- y[test_idx]
      
      # make predictions using myKNN
      preds <- myKNN(traindata, testdata, trainlabels, k)
      
      # grouping step
      if (group) {
        
        # convert to character 
        preds <- as.character(preds)
        testlabels <- as.character(testlabels)
        
        # group predictions
        preds[preds %in% c("Gemini","GPT","Llama")] <- "AI"
        preds[preds == "Human"] <- "Human"
        
        # group true labels
        testlabels[testlabels %in% c("Gemini","GPT","Llama")] <- "AI"
        testlabels[testlabels == "Human"] <- "Human"
      }
      
      # compute accuracies
      fold_acc[i] <- mean(preds == testlabels)
    }
    
    results[j] <- mean(fold_acc)
  }
  
  return(results)
}

# ~~~~~~~~~~~~~~~~~~~~~~~~~ Binary KNN (Documents) ~~~~~~~~~~~~~~~~~~~~~~~~~ 

# convert to binary labels
y_binary <- ifelse(y == "Human", "Human", "AI")

# k-values to test knn on
k_values <- 1:15

# apply 10-fold cv to knn for each of these k-vals
cv_acc <- knn_cv_k.2(X, y_binary, k_values, numfolds = 10, seed = 0)

# store results in df for plotting 
df1 <- data.frame(
  k = k_values,
  accuracy = cv_acc
)

# plot binary accuracy for each value of k
p1 <- ggplot(df1, aes(x = k, y = accuracy)) +
  geom_point() +
  geom_line() +
  labs(
    x = "k",
    y = "Cross-validated accuracy",
    title = "Binary KNN performance: tuning for k"
  ) +
  theme_minimal()

# find best k
best_k <- k_values[which.max(cv_acc)]
best_k

# ~~~~~~~~~~~~~~~~~~~~~~~~~ Multiclass KNN (Documents) ~~~~~~~~~~~~~~~~~~~~~~~~~ 

# apply 10-fold cv to multiclass knn for each of the same k-vals
cv_acc_multi <- knn_cv_k.2(X, y, k_values, numfolds = 10, seed = 0)

# df for plotting
df2 <- data.frame(
  k = k_values,
  accuracy = cv_acc_multi
)

# plot accuracies against k for multiclass
p2 <- ggplot(df2, aes(x = k, y = accuracy)) +
  geom_point() +
  geom_line() +
  labs(
    x = "k",
    y = "Cross-validated accuracy",
    title = "Multiclass KNN performance: tuning for k"
  ) +
  theme_minimal() +
  scale_y_continuous(limits = c(0.375, 0.44))

# display plot side by side
p1 + p2


# ~~~~~~~~~~~~~~~~~~ Grouped Multiclass KNN (Documents) ~~~~~~~~~~~~~~~~~~~~~~~~~ 

# apply 10-fold cv to multiclass knn for each of the same k-vals then group
cv_acc_grouped <- knn_cv_k.3(X, y, k_values, numfolds = 10, seed = 0, group=TRUE)

# plot accuracies against k
df3 <- data.frame(
  k = k_values,
  accuracy = cv_acc_grouped
)

p3 <- ggplot(df3, aes(x = k, y = accuracy)) +
  geom_point() +
  geom_line() +
  labs(
    x = "k",
    y = "Cross-validated accuracy",
    title = "Multiclass KNN performance: tuning for k"
  ) +
  theme_minimal() 

p3

# ~~~~~~~~~~~~~~~~~~ Final Results KNN (Documents) ~~~~~~~~~~~~~~~~~~~~~~~~~ 

binary_k <- k_values[which.max(cv_acc)]
multi_k <- k_values[which.max(cv_acc_multi)]

best_acc_binary <- max(cv_acc)
best_acc_multi <- max(cv_acc_multi)
best_acc_grouped <- max(cv_acc_grouped)

