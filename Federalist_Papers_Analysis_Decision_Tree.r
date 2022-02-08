#Toni Hanrahan
#Fed Papers Decision Tree
#____________________________________



#Step 1:  Install packages



#install.packages("tm")
#install.packages("stringr")
#install.packages("wordcloud")
#install.packages("Snowball")
#install.packages("slam")
#install.packages("proxy")
#install.packages("factoextra")
#install.packages("mclust") 
library(tm)
library(stringr)
library(wordcloud)
library(Snowball)
library(slam)
library(quanteda)
library(SnowballC)
library(arules)
library(proxy)
library(cluster)
library(stringi)
library(proxy)
library(Matrix)
library(tidytext) # convert DTM to DF
library(plyr) ## for adply
library(ggplot2)
library(factoextra) # for fviz
library(mclust) # for Mclust EM clustering
library(rpart)
library(rpart.plot)
library(rattle)


#Step 2: load corpus and view summary

setwd("~/R/thanrahan_scripts/IST707/fedPapers/")
## Next, load in the documents (the corpus)
FedCorpus <- Corpus(DirSource("txt"))
getTransformations()
ndocs<-length(FedCorpus)


##view summary of all the documents
summary(FedCorpus)
meta(FedCorpus[[1]])
meta(FedCorpus[[1]],5)



#Step 3: cleanup dtm

# ignore extremely rare words i.e. terms that appear in less then 1% of the documents
minTermFreq <- ndocs * 0.0001
# ignore overly common words i.e. terms that appear in more than 50% of the documents
maxTermFreq <- ndocs * 1
MyStopwords <- c("alexand", "jay", "madison", "hamilton", "jame")

STOPS <-stopwords('english')
Fed_dtm <- DocumentTermMatrix(FedCorpus,
                         control = list(
                           stopwords = TRUE, 
                           wordLengths=c(3, 15),
                           removePunctuation = T,
                           removeNumbers = T,
                           tolower=T,
                           stemming = T,
                           remove_separators = T,
                           stopwords = STOPS,
                           removeWords = STOPS,
                           removeWords = MyStopwords,
                           #removeWords(FedCorpus, stopwords('english')),
                           #removeWords(FedCorpus, MyStopwords),
                           bounds = list(global = c(minTermFreq, maxTermFreq))
                         ))



DTM_mat <- as.matrix(Fed_dtm)


## Look at word freuqncies
WordFreq <- colSums(as.matrix(Fed_dtm))
head(WordFreq)
length(WordFreq)
ord <- order(WordFreq)
WordFreq[head(ord)]
WordFreq[tail(ord)]

## Row Sums
Row_Sum_Per_doc <- rowSums((as.matrix(Fed_dtm)))
Row_Sum_Per_doc 




#Step 4: normalize matrix



## Create a normalized version of Fed_dtm
Fed_M <- as.matrix(Fed_dtm)
Fed_M_N1 <- apply(Fed_M, 1, function(i) round(i/sum(i),3))
## transpose
Fed_Matrix_Norm <- t(Fed_M_N1)
## Have a look at the original and the norm to make sure
Fed_M[c(10:20),c(1000:1005)]
Fed_Matrix_Norm[c(10:20),c(1000:1005)]
Row_Sum_Per_doc



#Step 5: convert to a matrix and a df


## Convert to matrix and view
Fed_dtm_matrix = as.matrix(Fed_dtm)
str(Fed_dtm_matrix)
Fed_dtm_matrix[c(1:3),c(2:4)]

## Also convert to DF
Fed_DF <- as.data.frame(as.matrix(Fed_dtm))
str(Fed_DF)
Fed_DF$advantag
nrow(Fed_DF)  ## Each row is a paper

Fed_DF[c(1:11),c(140:150)]



#Step 6: create word cloud to visualize


wordcloud(colnames(Fed_dtm_matrix), Fed_dtm_matrix[13, ], max.words = 70)
head(sort(as.matrix(Fed_dtm)[13,], decreasing = TRUE), n=20)






#Step 7:   Remove columns that did not get removed with 'removewords' function
#run once without this and run again to include this to show the difference

Fed_DF <- subset(Fed_DF, select = -c(alexand, jay, madison, hamilton, jame))

#now do another word cloud

wordcloud(colnames(Fed_dtm_matrix), Fed_dtm_matrix[13, ], max.words = 70)
head(sort(as.matrix(Fed_dtm)[13,], decreasing = TRUE), n=20)


#Step 8: Prep data frame to build test and train datasets

#add a column with the author name 
Fed_DF <- cbind(rownames(Fed_DF), data.frame(Fed_DF, row.names=NULL))

colnames(Fed_DF)[colnames(Fed_DF)=="rownames(Fed_DF)"] <- "Author"
str(Fed_DF)

#convert the column to character and then standardize the values
Fed_DF$Author <- as.character(Fed_DF$Author)

#i'm sure there is a smarter way to do this, but this worked the fastest
Fed_DF$Author[Fed_DF$Author=="dispt_fed_49.txt"] <- "disputed"
Fed_DF$Author[Fed_DF$Author=="dispt_fed_50.txt"]   <- "disputed"
Fed_DF$Author[Fed_DF$Author=="dispt_fed_51.txt"]   <- "disputed"
Fed_DF$Author[Fed_DF$Author=="dispt_fed_52.txt"]   <- "disputed"
Fed_DF$Author[Fed_DF$Author=="dispt_fed_53.txt"]   <- "disputed"
Fed_DF$Author[Fed_DF$Author=="dispt_fed_54.txt"]   <- "disputed"
Fed_DF$Author[Fed_DF$Author=="dispt_fed_55.txt"]   <- "disputed"
Fed_DF$Author[Fed_DF$Author=="dispt_fed_56.txt"]   <- "disputed"
Fed_DF$Author[Fed_DF$Author=="dispt_fed_57.txt"]   <- "disputed"
Fed_DF$Author[Fed_DF$Author=="dispt_fed_62.txt"]   <- "disputed"
Fed_DF$Author[Fed_DF$Author=="dispt_fed_63.txt"]   <- "disputed"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_1.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_11.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_12.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_13.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_15.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_16.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_17.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_21.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_22.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_23.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_24.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_25.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_26.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_27.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_28.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_29.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_30.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_31.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_32.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_33.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_34.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_35.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_36.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_59.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_6.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_60.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_61.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_65.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_66.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_67.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_68.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_69.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_7.txt"]  <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_70.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_71.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_72.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_73.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_74.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_75.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_76.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_77.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_78.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_79.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_8.txt"]  <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_80.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_81.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_82.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_83.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_84.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_85.txt"] <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="Hamilton_fed_9.txt"]  <- "Hamilton"
Fed_DF$Author[Fed_DF$Author=="HM_fed_18.txt"]       <- "HM"
Fed_DF$Author[Fed_DF$Author=="HM_fed_19.txt"]       <- "HM"
Fed_DF$Author[Fed_DF$Author=="HM_fed_20.txt"]       <- "HM"
Fed_DF$Author[Fed_DF$Author=="Jay_fed_2.txt"]       <- "Jay"
Fed_DF$Author[Fed_DF$Author=="Jay_fed_3.txt"]      <- "Jay"
Fed_DF$Author[Fed_DF$Author=="Jay_fed_4.txt"]      <- "Jay"
Fed_DF$Author[Fed_DF$Author=="Jay_fed_5.txt"]      <- "Jay"
Fed_DF$Author[Fed_DF$Author=="Jay_fed_64.txt"]     <- "Jay"
Fed_DF$Author[Fed_DF$Author=="Madison_fed_10.txt"] <- "Madison"
Fed_DF$Author[Fed_DF$Author=="Madison_fed_14.txt"] <- "Madison"
Fed_DF$Author[Fed_DF$Author=="Madison_fed_37.txt"] <- "Madison"
Fed_DF$Author[Fed_DF$Author=="Madison_fed_38.txt"] <- "Madison"
Fed_DF$Author[Fed_DF$Author=="Madison_fed_39.txt"] <- "Madison"
Fed_DF$Author[Fed_DF$Author=="Madison_fed_40.txt"] <- "Madison"
Fed_DF$Author[Fed_DF$Author=="Madison_fed_41.txt"] <- "Madison"
Fed_DF$Author[Fed_DF$Author=="Madison_fed_42.txt"] <- "Madison"
Fed_DF$Author[Fed_DF$Author=="Madison_fed_43.txt"] <- "Madison"
Fed_DF$Author[Fed_DF$Author=="Madison_fed_44.txt"] <- "Madison"
Fed_DF$Author[Fed_DF$Author=="Madison_fed_45.txt"] <- "Madison"
Fed_DF$Author[Fed_DF$Author=="Madison_fed_46.txt"] <- "Madison"
Fed_DF$Author[Fed_DF$Author=="Madison_fed_47.txt"] <- "Madison"
Fed_DF$Author[Fed_DF$Author=="Madison_fed_48.txt"] <- "Madison"
Fed_DF$Author[Fed_DF$Author=="Madison_fed_58.txt"] <- "Madison"



Fed_DF$Author



#Step 9:  Create test and train dataframes

# test 1
Fed_DF_Train1 <- Fed_DF
Fed_DF_Train1 <- Fed_DF_Train1[-c(1:11), ] #remove rows of disputed texts


Fed_DF_Test1 <- Fed_DF[c(1:11), ] #keep only rows of disputed texts
Test_Labels1 <- Fed_DF_Test1$Author
Fed_DF_Test1 <- Fed_DF_Test1[-c(1)] ## remove the Author




# test 2
Fed_DF_Train2 <- Fed_DF
Fed_DF_Train2 <- Fed_DF_Train2[-c(1:11), ] #remove rows of disputed texts
Fed_DF_Train2 <- Fed_DF_Train2[-c(1:36), ] #remove about half the data

Fed_DF_Test2 <- Fed_DF[c(12:47), ] #keep the other half
Test_Labels2 <- Fed_DF_Test2$Author
Fed_DF_Test2 <- Fed_DF_Test2[-c(1) ] ## remove the Author


Fed_DF$Author

# test 3

#Fed_DF_Train3 <- Fed_DF[c(12:47), ] #reverse the test/train that was done in test 2

#Fed_DF_Test3 <- Fed_DF
#Fed_DF_Test3 <- Fed_DF_Test3[-c(1:11), ] #remove rows of disputed texts
#Fed_DF_Test3 <- Fed_DF_Test3[-c(1:36), ] #reverse the test/train that was done in test 2
#Test_Labels3 <- Fed_DF_Test3$Author
#Fed_DF_Test3 <- Fed_DF_Test3[-c(1)] ## remove the Author




## ---------------------------------------------------------------------------
## Decision Tree Classification
## ---------------------------------------------------------------------------
## Next, we will use Decision Trees to see if we can classify data
## by author type : Hamilton, Madison, Jay or HM


## Step 5 --------------------------------------------

## decision tree 1
Tree1 <- rpart(Fed_DF_Train1$Author ~ ., data = Fed_DF_Train1, method="class")
summary(Tree1)

predicted1= predict(Tree1, Fed_DF_Test1, type="class")
Results1 <- data.frame(Predicted=predicted1,Actual=Test_Labels1)


table(Results1)

fancyRpartPlot(Tree1)







## decision tree 2
Tree2 <- rpart(Fed_DF_Train2$Author ~ ., data = Fed_DF_Train2, method="class")
summary(Tree2)

predicted2= predict(Tree2,Fed_DF_Test2, type="class")
(Results2 <- data.frame(Predicted=predicted2,Actual=Test_Labels2))

(table(Results2))

fancyRpartPlot(Tree2)





## decision tree 3

summary(Fed_DF_Train1)
summary(Fed_DF_Train2)
summary(Fed_DF_Train3)

Tree3 <- rpart(Fed_DF_Train3$Author ~ ., data = Fed_DF_Train3, method="class")
summary(Tree3)

predicted3= predict(Tree3,Fed_DF_Test3, type="class")
(Results3 <- data.frame(Predicted=predicted3,Actual=Test_Labels3))

(table(Results3))

fancyRpartPlot(Tree3)



