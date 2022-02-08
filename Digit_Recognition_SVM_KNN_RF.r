#Toni Hanrahan
#SVMs, kNN, and Random Forest for handwriting recognition
#____________________________________



#Step 1:  Install packages

#install.packages("RWeka")
library(RWeka)



#Step 2: read in data

trainset <- read.csv("~/R/thanrahan_scripts/IST707/Kaggle-digit-train-sample-small-1400.csv")
#trainset <- read.csv("~/R/thanrahan_scripts/IST707/Kaggle-digit-train.csv")
trainset$label=factor(trainset$label)

summary(trainset)
str(trainset)


#Step 3: Visualize some of the characters
#code sourced from # Digit Recognizer in R
#by [Koba Khitalishvili](http://www.kobakhit.com/) https://www.kaggle.com/nikhil2008v/digit-recognizer-in-r

#install.packages("readr")
library(readr)

#We can quickly plot the pixel color values to obtain a picture of the digit.
# Create a 28*28 matrix with pixel color values
m = matrix(unlist(trainset[10,-1]),nrow = 28,byrow = T)
# Plot that matrix
image(m,col=grey.colors(255))


#This image needs to be rotated to the right. I will rotate the matrix and plot a bunch of images.
rotate <- function(x) t(apply(x, 2, rev)) # reverses (rotates the matrix)

# Plot a bunch of images
par(mfrow=c(2,3))
lapply(1:6, 
       function(x) image(
         rotate(matrix(unlist(trainset[x,-1]),nrow = 28,byrow = T)),
         col=grey.colors(255),
         xlab=trainset[x,1]
       )
)
par(mfrow=c(1,1)) # set plot options back to default



#Step 3: J48 decision tree

WOW("J48")
m=J48(label~., data = trainset)
m=J48(label~., data = trainset, control=Weka_control(U=FALSE, M=2, C=0.5))
e1=evaluate_Weka_classifier(m, seed=1, numFolds=3)
e1


#Step 4: NB

WOW("NaiveBayes")
NB <- make_Weka_classifier("weka/classifiers/bayes/NaiveBayes")
# build default NB model
nb_model=NB(label~., data=trainset)
# turn on discretization
nb_model=NB(label~., data=trainset, control=Weka_control(D=TRUE))
# turn on kernel estimation
nb_model=NB(label~., data=trainset, control=Weka_control(K=TRUE))
e2 <- evaluate_Weka_classifier(nb_model, numFolds = 3, seed = 1, class = TRUE)
e2


#Step 5: K nearest neighbor (knn)

WOW("IBk")
knn <- make_Weka_classifier("weka/classifiers/lazy/IBk")
knn_model=knn(label~., data=trainset)
knn_model=NB(label~., data=trainset)
knn_model=NB(label~., data=trainset, control=Weka_control(K=3))
e3 <- evaluate_Weka_classifier(knn_model, numFolds = 3, seed = 1, class = TRUE)
e3


#Step 6: SVM


WOW("SMO")
svm <- make_Weka_classifier("weka/classifiers/functions/SMO")
svm_model=svm(label~., data=trainset)
e4 <- evaluate_Weka_classifier(svm_model, numFolds = 3, seed = 1, class = TRUE)
e4



#Step 7: Random Forest

WOW("weka/classifiers/trees/RandomForest")
rf <- make_Weka_classifier("weka/classifiers/trees/RandomForest")
# build default model with 100 trees
rf_model=rf(label~., data=trainset)
# build a model with 10 trees instead
rf_model=rf(label~., data=trainset, control=Weka_control(I=10))
e5 <- evaluate_Weka_classifier(rf_model, numFolds = 3, seed = 1, class = TRUE)
e5


#Step 8: create output file

# submit to Kaggle
# first use textwrangler to insert "?," to each row, change "?," in the first row to "label,"
# create test ids
#testid = seq(1, 28000, by=1)
# apply model to all test data
#pred=predict(rf_model, newdata = testset, type = c("class"))
#pred=predict(rf_model, newdata = testset)
#newpred=cbind(testid, pred)
#colnames(newpred)=c("ImageId", "Label")
#write.csv(newpred, file="~/R/thanrahan_scripts/IST707/digit-RF-pred.csv", row.names=FALSE)