

# ============================= Useful Functions ==============================


# run discriminant corpus on 1 document - returns predicted class for that document & the brier score and log score associated 
discriminant_corpus <- function(traindata, testdata, truelabels) {
  
  D <- length(traindata) # number of classes: 2 or 4 
  V <- ncol(testdata) # number of words in vocab: 201
  
  thetas <- matrix(0, nrow = D, ncol = V)
  priors <- numeric(D) 
  
  for (k in 1:D) {
    
    # make author profile: adds up counts of all words in doc for each author 
    words <- colSums(traindata[[k]])
    
    # smoothing: force unseen words to appear once
    words[words == 0] <- 1
    
    thetas[k, ] <- words / sum(words) 
    priors[k] <- nrow(traindata[[k]]) 
  }
  
  priors <- priors / sum(priors)

  # compute likelihoods
  m <- nrow(testdata)
  loglik <- matrix(0, nrow = m, ncol = D)
  
  for (i in 1:m) {
    x <- testdata[i, ]
    for (k in 1:D) {
      loglik[i, k] <- dmultinom(x, prob = thetas[k, ], log = TRUE)
    }
  }
  
  # compute log posterior by adding log priors 
  logpost <- sweep(loglik, 2, log(priors), "+")
  
  # compute predictions
  pred <- apply(logpost, 1, which.max)
  
  # convert to normal posterior for score calculation
  post <- matrix(0, nrow = m, ncol = D)
  
  for (i in 1:m) {
    z <- logpost[i, ]
    z <- z - max(z)
    post[i, ] <- exp(z) / sum(exp(z))
  }
  
  # logscore 
  log_score <- sum(log(post[cbind(1:m, truelabels)])) # indexes 1:m test data and chooses posterior corresponding to true value
  
  # brier score 
  brier_score <- 0
  for (i in 1:m) {
    y <- rep(0, D)
    y[truelabels[i]] <- 1
    brier_score <- brier_score + sum((post[i, ] - y)^2)
  }
  
  return(list(
    pred = pred,
    log_score = log_score,
    brier_score = brier_score,
    post = post
  ))
}

# KNN function - add logscore & brier score as NULL to work with cross validation function
knn_corpus <- function(traindata, testdata, truelabels) {
  train <- NULL
  
  # train is a matrix where: each row = one “author vector” (sum of all texts for that author, normalized)
  for (i in 1:length(traindata)) {
    train <- rbind(train, apply(traindata[[i]],2,sum))
  }
  
  for (i in 1:nrow(train)) {
    train[i,] <- train[i,]/sum(train[i,])
  }
  for (i in 1:nrow(testdata)) {
    testdata[i,] <- testdata[i,]/sum(testdata[i,])
  }

  trainlabels <- 1:nrow(train) # indicates what author profile each row of train corresponds to
  
  pred <- myKNN(train, testdata, trainlabels, k=1)
 
  
  return(list(
    pred = pred
  ))
}


# cross validation must work for each method - returns predictions for each method vs truth
loocv <- function(data, func, scores = FALSE){
  
  predictions <- NULL
  truth <- NULL
  log_scores <- NULL
  brier_scores <- NULL 
  
  C <- length(data) # number of classes
  
  for (i in 1:C){
    
    D <- nrow(data[[i]]) # number of documents in class i
    
    for (j in 1:D){
      
      
      test_data <- matrix(data[[i]][j,],nrow=1)
      train_data <- data 
      train_data[[i]] <- train_data[[i]][-j,,drop=FALSE]
      
      results <- func(train_data, test_data, c(i))
      
      pred <- results[[1]]
      predictions <- c(predictions, pred)
      truth <- c(truth, i)
      
      if (scores == TRUE){
        log_score <- results[[2]]
        brier_score <- results[[3]]
        log_scores <- c(log_scores, log_score)
        brier_scores <- c(brier_scores, brier_score)
      } else {
        # do nothing
      }
    }
  }
  
  if (scores == TRUE){
    return(list(
      predictions,
      truth,
      log_scores,
      brier_scores))
  } else {
    return(list(
      predictions,
      truth
    ))
  }
}

# ================================ AI vs Human ================================


grouped_features <- list(
  AI = rbind(features[[1]], features[[2]], features[[4]]),
  Human = features[[3]]
)

discriminant_cv <- loocv(grouped_features, discriminant_corpus)

discriminant_predictions <- discriminant_cv[[1]]
discriminant_truth <- discriminant_cv[[2]]
discriminant_log <- discriminant_cv[[3]]
discriminant_brier <- discriminant_cv[[4]]

# evaluation
sum(discriminant_predictions==discriminant_truth) / length(discriminant_truth) # proportion of correct predictions
-mean(discriminant_log) # mean of log score accross test cases
mean(discriminant_brier) # mean of brier score across test cases


# ======================= Gemini vs GPT vs Human vs Llama ======================


# Discriminant Analysis
discriminant_cv <- loocv(features, discriminant_corpus)

discriminant_predictions <- discriminant_cv[[1]]
discriminant_truth <- discriminant_cv[[2]]
discriminant_log <- discriminant_cv[[3]]
discriminant_brier <- discriminant_cv[[4]]

# evaluation
sum(discriminant_predictions==discriminant_truth) / length(discriminant_truth) # proportion of correct predictions
-mean(discriminant_log) # mean of log score accross test cases
mean(discriminant_brier) # mean of brier score across test cases

# KNN 
knn_cv <- loocv(features, knn_corpus)
knn_predictions <- knn_cv[[1]]
knn_truth <- knn_cv[[2]]

# evaluation
sum(knn_predictions==knn_truth) / length(knn_truth) # proportion of correct predictions




