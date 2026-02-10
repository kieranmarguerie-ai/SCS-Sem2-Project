getwd()
source("Functions//stylometryfunctions.R")

rosie_wd <- "~/University/Year 4/Statistical Case Studies/SCS-Sem2-Project/Data/FunctionWords/"
ella_wd <- "C:/Users/Ella Park/Desktop/Year 4/Sem 1/Stats Case Study/A3/SCS-Sem2-Project/Data/FunctionWords/"
kieran_wd <- "~/SCS-Sem2-Project/Data/FunctionWords/"

words <- loadCorpus(rosie_wd)

# Set seed
set.seed(0)

# Looking at data
words$authornames
words$features[[1]]

length(words$features[[1]][1, ])
dim(words$features[[2]])
length(words$features[[3]])
length(words$features[[4]])

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

# Create test data for each authors
X_test = NULL
y_test = NULL
  
X_train <- X
X_train_corpus <- list(
  Gemini = X_train[y_train == "Gemini", , drop = FALSE],
  GPT    = X_train[y_train == "GPT", , drop = FALSE],
  Human  = X_train[y_train == "Human", , drop = FALSE],
  Llama  = X_train[y_train == "Llama", , drop = FALSE]
)
y_train <- y


for (author in c("Gemini", "GPT", "Human", "Llama")) {
  
  author_rows <- which(y == author)
  testind <- sample(author_rows, size = floor(0.1 * length(author_rows)))
  
  X_test <- rbind(X_test, X[testind, , drop = FALSE])
  y_test <- c(y_test, y[testind])
  
  X_train <- X_train[-testind, ]
  y_train <- y_train[-testind]

}

# Apply myKNN 
KNN_1 <- myKNN(X_train, X_test, y_train, k=1) 
KNN_1 == y_test


# Apply KNNCorpus
KNN_Corpus <- KNNCorpus(X_train_corpus, X_test) 
KNN_Corpus == y_test

results <- as.character(KNN_Corpus) 
true <- as.character(y_test)

# Define which values to replace
ai_names <- c("1", "2", "4")

# Replace them with "AI"
results[results %in% ai_names] <- "AI"
true[true %in% ai_names] <- "AI"


# Check result
results == true

# Apply discriminant analysis 
discriminant <- discriminantCorpus(X_train_corpus, X_test, labels = c("Gemini", "GPT", "Human", "Llama")) 
discriminant == y_test


# -----------------  AI vs AI vs AI vs Human ----------------------

features <- words$features
authors <- words$authornames  

predictions <- NULL
KNNpredictions <- NULL
truth <- NULL

#discard unknown texts
for (i in 1:length(features)) {
  for (j in 1:nrow(features[[i]])) {
    testdata <- matrix(features[[i]][j,],nrow=1)
    traindata <- features
    traindata[[i]] <- traindata[[i]][-j,,drop=FALSE]
    pred <- discriminantCorpus(traindata, testdata)[1]
    predictions <- c(predictions, pred)
    pred <- KNNCorpus(traindata, testdata)
    KNNpredictions <- c(KNNpredictions, pred)
    truth <- c(truth, i)
  }
}

sum(predictions==truth)/length(truth)
sum(KNNpredictions==truth)/length(truth)

# checking which of human are true
gemini_ind <- which(truth == 1)
gemini_accuracy <- sum(predictions[gemini_ind] == 1) / length(gemini_ind)
gemini_accuracy

# checking which of ai are true 
GPT_ind <- which(truth == 2)
GPT_accuracy <- sum(predictions[GPT_ind] == 2) / length(GPT_ind)
ai_accuracy

# checking which of ai are true 
human_ind <- which(truth == 3)
human_accuracy <- sum(predictions[human_ind] == 3) / length(human_ind)
human_accuracy

# checking which of ai are true 
llama_ind <- which(truth == 4)
llama_accuracy <- sum(predictions[llama_ind] == 4) / length(llama_ind)
llama_accuracy


# ---------------------- AI vs Human ------------------------

restructured_features <- list(
  AI = rbind(features[[1]], features[[2]], features[[4]]),
  Human = features[[3]]
)

restructured_features

predictions <- NULL
KNNpredictions <- NULL
truth <- NULL
log_score <- 0 
brier_score <- 0

#discard unknown texts
for (i in 1:length(restructured_features)) {
  for (j in 1:nrow(restructured_features[[i]])) {
    testdata <- matrix(restructured_features[[i]][j,],nrow=1)
    traindata <- restructured_features
    traindata[[i]] <- traindata[[i]][-j,,drop=FALSE]
    pred <- discriminantCorpus_unbalanced2(traindata, testdata, i)[[1]]
    predictions <- c(predictions, pred)
    pred <- KNNCorpus(traindata, testdata)
    KNNpredictions <- c(KNNpredictions, pred)
    truth <- c(truth, i)
    log_score <- log_score + discriminantCorpus_unbalanced2(traindata, testdata, i)[[3]]
    brier_score <- brier_score + discriminantCorpus_unbalanced2(traindata, testdata, i)[[4]]
  }
}


n <- sum(sapply(restructured_features, nrow))
brier_score <- (1/n) * brier_score 

restructured_features

sum(predictions==truth)/length(truth)
sum(KNNpredictions==truth)/length(truth)

# checking which of human are true
human_ind <- which(truth == 2)
human_accuracy <- sum(predictions[human_ind] == 2) / length(human_ind)
human_accuracy

# checking which of ai are true 
ai_ind <- which(truth == 1)
ai_accuracy <- sum(predictions[ai_ind] == 1) / length(ai_ind)
ai_accuracy
# Run PCA
pca <- prcomp(X, scale. = TRUE)

# Create dataframe with first few PCs
pca_df <- data.frame(
  PC1 = pca$x[,1],
  PC2 = pca$x[,2],
  PC3 = pca$x[,3],
  PC4 = pca$x[,4],
  Author = y
)

# PC1 vs PC2
ggplot(pca_df, aes(PC1, PC2, colour = Author)) +
  geom_point(alpha = 0.7) +
  labs(title = "PCA Visualisation of Authors") +
  theme_minimal()

# PC1 vs PC3
ggplot(pca_df, aes(PC1, PC3, colour = Author)) +
  geom_point(alpha = 0.7) +
  labs(title = "PCA Visualisation of Authors") +
  theme_minimal()

# PC2 vs PC3
ggplot(pca_df, aes(PC2, PC3, colour = Author)) +
  geom_point(alpha = 0.7) +
  labs(title = "PCA Visualisation of Authors") +
  theme_minimal()

