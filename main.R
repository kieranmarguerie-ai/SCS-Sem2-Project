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
y_train <- y

for (author in c("Gemini", "GPT", "Human", "Llama")) {
  
  author_rows <- which(y == author)
  testind <- sample(author_rows, size = floor(0.1 * length(author_rows)))
  
  X_test <- rbind(X_test, X[testind, , drop = FALSE])
  y_test <- c(y_test, y[testind])
  
  X_train <- X_train[-testind, ]
  y_train <- y_train[-testind]

}

# Apply KNN 
KNN_1 <- myKNN(X_train, X_test, y_train, k=1) 
KNN_1 == y_test

KNN_3 <- myKNN(X_train, X_test, y_train, k=3) 
KNN_3 == y_test


# 
# # Randomly splitting up the data into test and training data 
# traindata <- words$features
# testdata <- NULL
# 
# testlabels <- NULL #true authors for the test set
# for (i in 1:length(traindata)) {
#   #select a random book by this author by choosing a row (= book)
#   testind <- sample(1:nrow(traindata[[i]]), 1)
#   #add this book to the test set
#   testdata <- rbind(testdata, traindata[[i]][testind,])
#   testlabels <- c(testlabels, i)
#   #now discard the book from the training set
#   traindata[[i]] <- traindata[[i]][-testind,,drop=FALSE]
# }
# 
# 
# 


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

