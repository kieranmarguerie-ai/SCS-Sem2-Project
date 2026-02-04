getwd()
source("Functions//stylometryfunctions.R")

rosie_wd <- "~/University/Year 4/Statistical Case Studies/SCS-Sem2-Project/Data/FunctionWords/"
ella_wd <- "C:/Users/Ella Park/Desktop/Year 4/Sem 1/Stats Case Study/A3/SCS-Sem2-Project/Data/FunctionWords/"
kieran_wd <- "~/SCS-Sem2-Project/Data/FunctionWords/"

words <- loadCorpus(rosie_wd)

words$authornames
words$features[[1]]

traindata <- words$features
testdata <- NULL

testlabels <- NULL #true authors for the test set
for (i in 1:length(traindata)) {
  #select a random book by this author by choosing a row (= book)
  testind <- sample(1:nrow(traindata[[i]]), 1)
  #add this book to the test set
  testdata <- rbind(testdata, traindata[[i]][testind,])
  testlabels <- c(testlabels, i)
  #now discard the book from the training set
  traindata[[i]] <- traindata[[i]][-testind,,drop=FALSE]
}

