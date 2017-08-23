################################################################################################################
##                                                                                                            ## 
## Domain     :  Health Care                                                                                  ##
## Project    :  Diabetes Diagnose analysis                                                                   ##
## Data       :  Data has been provided by UpX Academy in csv format.                                         ## 
## Objective  :  Classify positive and negative cases by splitting your dataset into training and testing, also 
##               implement cross validation to avoid overfitting. Evaluate your model on the basis of AUC curve.                                 ##                                                               ##
##                                                                                                            ##
################################################################################################################
##
## Install Required Packages
##
## install.packages("Amelia")
## install.packages("class")
## install.packages("gmodels")
## install.packages("GMD")
## install.packages("sjPlot")
## install.packages("survival")
## install.packages("ROCR")
##
## Declare the installed needed packages so that we can use their functions in our R workbook
library(ggplot2)                          
library(dplyr)
library(Amelia)      #To use MISSMAP function to find out any missing Values
library(tree)        #To use tree modelling algorithms/functions
library(caret)       #To use Classification and regression Mehtods alogorithms/functions
library(class)
library(gmodels)     #To use validation the model performance of knn model
library(GMD)   
library(randomForest)
library(survival)    # To perform survival analysis 
library(e1071)       # To use svm method
library(ROCR)
##
##----------------------------------------------------------------------------------------------------------##
##  Step 1 - Data Collection                                                                                ##
##  In the this step we will be loading our input diabetes data in csv format into our system. As the provided 
##  dataset has no columns in the input datase, we will provide the column names separately. 

columns <- c('Pregnant_Times','glucose','bp','skin_fold.','insulin','bmi','pedigree',
             'age','Diabetes')
str(columns)

medical <- read.csv('diabetes.csv',sep=',',col.names = columns)

##----------------------------------------------------------------------------------------------------------##
## Step 2 - Data Exploration & Data Preparation.                                                                      ##
## In this Step we will  :                                                                                  ##
##        - Perform basic analysis to understand the spread,dimensions & volume of the input data           ## 
##        - Will use various EDA tools like Box plot to identify the patterns &                             ##
##          relations among input variables                                                                 ##
##        - Will try to identify the variables which are impacting our predicting variable Diabetes which   ##
##          signifies whether a patient is diabetic or not.                                                 ##
##        - Will find out the missing values and clean the data so that it can be feeded to ML algorithms   ##
##          in subsequent steps.                                                                            ##
##                                                                                                          ##  

head(medical,20)           
tail(medical,20)             
dim(medical)               # Columns = 9, Observation = 767, Predicted Variable  is - Diabetes
str(medical)                
summary(medical)             

medical.eda <- medical          # Copying our input Dataframe into another df medical.eda so that any changes we do for
                                 # EDA is not impacted on the input dataset.
str(medical.eda$Diabetes)

# Lets plot a Pie Chart to show the proportions percentage of patients who are diabetic & not.

tab <- as.data.frame(table(medical.eda$Diabetes))
slices <- c(tab[1,2], tab[2,2]) 
lbls <- c("Not Diabetic", "Diabetic")
pct <- round(slices/sum(slices)*100,digits = 2)  # calculating % rounded to 2 digits
lbls <- paste(lbls, pct)                         # add percents to labels 
lbls <- paste(lbls,"%",sep="")                   # ad % to labels 
pie(slices,labels = lbls, col=rainbow(length(lbls)),angle = 90,
    main="Proportion of Diabetic Patients")         # Plot shows 34.81 % of people are diabetic from the provided data.

# Lets plot varios plots to understand the various attributes of input variables for Left and not Left Employees 

#plot(hr)

medical.eda$Diabetes <- as.factor(medical.eda$Diabetes)
medical.eda.Diabetes <- filter(medical.eda,medical.eda$Diabetes==1)       # df with observations for Diabetic Patients

# Boxplot to see the pattern of Pregnant Times for Diabetic/Non Diabetic.
boxplot(medical.eda$Pregnant_Times ~ medical.eda$Diabetes, data = medical.eda, col = "red",
        xlab = "Diabetes Condition",ylab = "No of Times Pregnant")


# Boxplot to see the pattern of Plasma Glucose Concentration for Diabetic/Non Diabetic.
boxplot(medical.eda$glucose ~ medical.eda$Diabetes, data = medical.eda, col = "red",
        xlab = "Diabetes Condition",ylab = "Plasma Glucose Concentration")

# Boxplot to see the pattern of Blod Pressure for Diabetic/Non Diabetic.
boxplot(medical.eda$bp ~ medical.eda$Diabetes, data = medical.eda, col = "red",
        xlab = "Diabetes Condition",ylab = "Blod Pressure")

# Boxplot to see the pattern of Skin Fold for Diabetic/Non Diabetic.
boxplot(medical.eda$skin_fold. ~ medical.eda$Diabetes, data = medical.eda, col = "red",
        xlab = "Diabetes Condition",ylab = "Skin Fold Thcikness")


# Boxplot to see the pattern of 2 hr Serum Insulin for Diabetic/Non Diabetic.
boxplot(medical.eda$insulin ~ medical.eda$Diabetes, data = medical.eda, col = "red",
        xlab = "Diabetes Condition",ylab = "2 Hr Serun Insulin")


# Boxplot to see the pattern of BMI for Diabetic/Non Diabetic.
boxplot(medical.eda$bmi ~ medical.eda$Diabetes, data = medical.eda, col = "red",
        xlab = "Diabetes Condition",ylab = "Body Mass Index")


# Boxplot to see the pattern of Diabetes pedigree fucntion for Diabetic/Non Diabetic.
boxplot(medical.eda$pedigree ~ medical.eda$Diabetes, data = medical.eda, col = "red",
        xlab = "Diabetes Condition",ylab = "Diabetes Pedigree")


# Boxplot to see the pattern of Age for Diabetic/Non Diabetic.
boxplot(medical.eda$age ~ medical.eda$Diabetes, data = medical.eda, col = "red",
        xlab = "Diabetes Condition",ylab = "Age")


#Now lets find out if there are any Missing values in our dataset.
missmap(medical.eda)        # No missing values as the map shows no white cells which signifies for missing values


# 
#  Observation from EDA 
#  - Plasma Glucose Concentration for Diabetic Patients is significantly higher as compared to
#    non diabetic ones.High levels of Q1,Q2, Q3 and upper whiskers along with IQR for diabetic patients
#    signifies the same.
#  - No. of Times Pregant & BMI is also high for Diabetic Patients.
#  - Age also plays quite relative role , as majority of the patients who are diabetic has
#    are relatively older. Q1, Q2, Q3 and upper whiskers for diabetic ones are higher than
#    the not diabetic one, which signifies the roles of Age here.
#  - To some extent, skin Fold Thickness & blood pressure is also marginally high for Diabteic Patients.

##----------------------------------------------------------------------------------------------------------##
##  Now lets create training and test Data sets. For this project although our objective is to predict who is diabetic
##  ,we are not provided with test dataset.Hence here we will seggregate the input dataset
##  into 2 datasets i.e. train & test in proportions 70 & 30% respectively. So, we can train our model
##  on train dataset & validate/test on test dataset and finally by improving the final performance.

set.seed(1000)
idx <- sample(seq(1, 2), size = nrow(medical), replace = TRUE, prob = c(.7,.3))
train <- medical[idx == 1,]
test <-  medical[idx == 2,]
dim(train)
dim(test)


###--------------------------------------------------------------------------------------------------------------------###
# Step 3 - selection of Models and Model building.
# Planning to run algorithms using 10-fold cross validation type of resampling                                          #
###---------------------------------------------------------------------------------------------------

train$Diabetes <- as.factor(train$Diabetes)
test$Diabetes <- as.factor(test$Diabetes)
dim(train)
dim(test)
tC <- trainControl(method="cv", number=10)

###--------------------------------------------------------------------------------------------------------------------###
#  Build Models                                                                                                          #
#  - Linear Methods     - LDA, GLM                                                                                       #
#  - Non Linear Methods - kNN, SVM                                                                                       #
#  - Tree Methods       - CART                                                                                           #
#  - Ensemble Methods   - RF, GBM                                                                                        #
###-------------------------------------------------------------------------------------------

set.seed(100)
fit.lda <- train(Diabetes~., data=train, method="lda", preProcess=c("center", "scale"), maximize = TRUE, trControl=tC)
fit.lda

set.seed(100)
fit.glm <- train(Diabetes~., data=train, method="glm", preProcess=c("center", "scale"), maximize = TRUE, trControl=tC)
fit.glm

set.seed(100)
fit.knn <- train(Diabetes~., data=train, method="knn", preProcess=c("center", "scale"), maximize = TRUE, trControl=tC)
fit.knn

set.seed(100)
fit.svm <- train(Diabetes~., data=train, method="svmRadial", preProcess=c("center", "scale"), maximize = TRUE, trControl=tC)
fit.svm

set.seed(100)
fit.cart <- train(Diabetes~., data=train, method="rpart", metric="Accuracy", trControl=tC)
fit.cart
plot(varImp(fit.cart),top=20)   # shows glucose, bmi, age , Pregnant_Times, insulin & skin_fold are most important.

set.seed(100)
fit.rf <- train(Diabetes~., data=train, method="rf", metric="Accuracy", trControl=tC)
fit.rf
plot(varImp(fit.rf),top=20) #shows glucose,bmi, age, pedigree, bp, Pregnant Times & skin fold most important.

set.seed(100)
fit.gbm <- train(Diabetes~., data=train, method="gbm", metric="Accuracy", trControl=tC, verbose=FALSE)
fit.gbm


###--------------------------------------------------------------------------------------------------------------------###
#  Summarize the results of the models                                                                                   #
###--------------------------------------------------------------------------------------------------------------------###

results_fit <- resamples(list(m_lda=fit.lda, m_glm=fit.glm, m_cart=fit.cart, m_knn=fit.knn, m_svm=fit.svm, m_rf=fit.rf, m_gbm=fit.gbm))
results_fit

summary(results_fit)
dotplot(results_fit)
bwplot(results_fit, col="red", main="Accuracy Results of Models")

print(fit.lda)
print(fit.glm)
print(fit.cart)
print(fit.knn)
print(fit.svm)
print(fit.rf)
print(fit.gbm)

###--------------------------------------------------------------------------------------------------------------------###
# Step 4-  Use models build in above Step to predict Dianetic or not.
##--------------------------------------------------------------------------------------------------------------------###

predict_lda <- predict(fit.lda, test)
confusionMatrix(predict_lda,test$Diabetes)    #Accuracy 77.42 Kappa 47.54

#--> Code for ROC - Sensitivity / Specificity 
predict_lda_val <- prediction(as.numeric(predict_lda), test$Diabetes)
perf_lda_VAL_ROC <- performance(predict_lda_val, "tpr", "fpr") 
plot(perf_lda_VAL_ROC, colorize=T, lwd=3, main="ROC Curve for LDA ") 
abline(a=0,b=1)  
#---------------------------------------------------

predict.glm <- predict(fit.glm, test)
confusionMatrix(predict.glm,test$Diabetes)    #Accuracy 76.96 Kappa 46.28

#--> Code for ROC - Sensitivity / Specificity 
predict_glm_val <- prediction(as.numeric(predict.glm), test$Diabetes)
perf_glm_VAL_ROC <- performance(predict_glm_val, "tpr", "fpr") 
plot(perf_glm_VAL_ROC, colorize=T, lwd=3, main="ROC Curve for GLM ") 
abline(a=0,b=1)                         
#-----------------------------------

predict.cart <- predict(fit.cart,test)
confusionMatrix(predict.cart,test$Diabetes)    #Accuracy 75.12 Kappa 41.98

#--> Code for ROC - Sensitivity / Specificity 
predict_cart_val <- prediction(as.numeric(predict.cart), test$Diabetes)
perf_cart_VAL_ROC <- performance(predict_cart_val, "tpr", "fpr") 
plot(perf_cart_VAL_ROC, colorize=T, lwd=3, main="ROC Curve for Cart ") 
abline(a=0,b=1)                         
#-----------------------------------

predict.svm <- predict(fit.svm,test)
confusionMatrix(predict.svm,test$Diabetes)    #Accuracy 74.65 Kappa 40.03

#--> Code for ROC - Sensitivity / Specificity 
predict_svm_val <- prediction(as.numeric(predict.svm), test$Diabetes)
perf_svm_VAL_ROC <- performance(predict_svm_val, "tpr", "fpr") 
plot(perf_svm_VAL_ROC, colorize=T, lwd=3, main="ROC Curve for SVM ") 
abline(a=0,b=1)     
#------------------------------------------------

predict.rf <- predict(fit.rf,test)
confusionMatrix(predict.rf,test$Diabetes)    #Accuracy 72.81 Kappa 39.31

#--> Code for ROC - Sensitivity / Specificity 
predict_rf_val <- prediction(as.numeric(predict.rf), test$Diabetes)
perf_rf_VAL_ROC <- performance(predict_rf_val, "tpr", "fpr") 
plot(perf_rf_VAL_ROC, colorize=T, lwd=3, main="ROC Curve for RF ") 
abline(a=0,b=1)                         

#---------------------------------------------------------

predict.gbm <- predict(fit.gbm,test)
confusionMatrix(predict.gbm,test$Diabetes)    #Accuracy 78.34 Kappa 49.68

#--> Code for ROC - Sensitivity / Specificity 

predict_gbm_val <- prediction(as.numeric(predict.gbm), test$Diabetes)
perf_gbm_VAL_ROC <- performance(predict_gbm_val, "tpr", "fpr") 
plot(perf_gbm_VAL_ROC, colorize=T, lwd=3, main="ROC Curve for GBM ") 
abline(a=0,b=1)                           
                       

## Conclusion:- One can see from the above stats that model performance that either of GBM (Accuracy 78.34 Kappa 49.68) Or
##              LDA (Accuracy 77.42 Kappa 47.54) looks close the best models for our business. To decide
##              further we have ploted the ROC for both these models and could see that the area under the ROC 
##              (AUC) is slightly larger for GBM as compared to LDA. This is quite obvious as the Accuracy, Sensitvity and Specificity
##              for GBM is slightly greater than the LDA. hence we wil consider GBM model for this problem. 
##            




