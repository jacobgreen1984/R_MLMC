# MLMC linux
# jacobgreen1984@2e.co.kr


# -------------------------------------------------------------------------
# pre-processing 
# -------------------------------------------------------------------------
library(ROSE)
library(FSelector)
library(woeBinning)
library(data.table)
library(caret)
library(mice)
options(warn=-1)
#options(warn=0)

# read dataset
# convert to factor 
# sampling for test 
dataXY <- fread("/media/jacob/database/INGlife/dataXY.csv",stringsAsFactors = T)
dataXY[, which(sapply(dataXY,is.character))] <- lapply(dataXY[, which(sapply(dataXY,is.character))], as.factor)
set.seed(1234)
#tmp    <- sample(nrow(dataXY),round(nrow(dataXY)*0.1))
#dataXY <- dataXY[tmp,]
dataXY <- as.data.frame(dataXY)

# missing values 
#mice(dataXY,m=5,maxit=100,meth='pmm',seed=1234)

# near zero variance predictors
tmpVar         <- nearZeroVar(dataXY,names=F,saveMetrics=T,allowParallel=T)
zero_vars      <- which(tmpVar$zeroVar)
near_zero_vars <- which(tmpVar$nzv)
dataXY         <- dataXY[,-zero_vars]

# numeric outlier treatment 
p.cap <- function(x){
  for (i in which(sapply(x, is.numeric))) {
    quantiles <- quantile( x[,i], c(.05, .95 ), na.rm =TRUE)
    x[,i] = ifelse(x[,i] < quantiles[1] , quantiles[1], x[,i])
    x[,i] = ifelse(x[,i] > quantiles[2] , quantiles[2], x[,i])
  }
  return(x)
}
dataXY <- p.cap(dataXY)

# feature selection
# ig_values      <- information.gain(Y~., dataXY)
# top_k_features <- rownames(ig_values)[which(ig_values$attr_importance!=0)]
# cat("number of features:", length(top_k_features))
# dataXY         <- dataXY[,c("Y",top_k_features)]

# split to train, valid, test 
set.seed(1234)
intrain       <- createDataPartition(dataXY$Y, p=0.6, list=FALSE)
train         <- dataXY[intrain,]
total_valid   <- dataXY[-intrain,]
set.seed(1234)
intotal_valid <- createDataPartition(total_valid$Y, p=0.5, list=FALSE)
test          <- total_valid[intotal_valid,]
valid         <- total_valid[-intotal_valid,]

# feature engineering using woe
binning <- woe.binning(train,'Y',train)
train   <- woe.binning.deploy(train, binning, min.iv.total=0.3,add.woe.or.dum.var='woe') # both dum and woe
valid   <- woe.binning.deploy(valid, binning, min.iv.total=0.3,add.woe.or.dum.var='woe') # both dum and woe
test    <- woe.binning.deploy(test, binning, min.iv.total=0.3,add.woe.or.dum.var='woe') # both dum and woe
#train   <- woe.binning.deploy(train, binning, min.iv.total=0.3,add.woe.or.dum.var='dum') # only dum

# create dummy
train   <- data.frame(Y=train$Y,model.matrix(Y~+.-1, data=train))
valid   <- data.frame(Y=valid$Y,model.matrix(Y~+.-1, data=valid))
test    <- data.frame(Y=test$Y,model.matrix(Y~+.-1, data=test))

# scale(need for tensorflow)
pp      <- preProcess(train, method = "range")
saveRDS(pp,"/media/jacob/database/INGlife/pp.Rda")
train   <- predict(pp, train)
valid   <- predict(pp, valid)
test    <- predict(pp, test)

# create artificial dataset
# p_for_ROSE = 0.5
# source("/media/jacob/database/INGlife/code/00_CreateSample_using_ROSE.R")
# table(train$Y)

# check
#summary(train)
#table(train$Y)

# export train/test dataset for TensorFlow 
write.csv(train,"/media/jacob/database/INGlife/train.csv",row.names = F)
write.csv(valid,"/media/jacob/database/INGlife/valid.csv",row.names = F)
write.csv(test,"/media/jacob/database/INGlife/test.csv",row.names = F)
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# set h2o 
# -------------------------------------------------------------------------
# turn on h2o 
library(h2o)
# h2o.shutdown()
h2o.init(nthreads=-1)

# convert to h2oDF
train <- as.h2o(train)
valid <- as.h2o(valid)
test  <- as.h2o(test)

# set x and y
x <- which(colnames(train)!="Y")
y <- which(colnames(train)=="Y")

# h2o_reduce dimentionality 
# k_for_pca = 100
# source("/media/jacob/database/INGlife/code/00_PCA_using_GLRM.R")
# h2o.saveModel(object=compressor, path="/media/jacob/database/INGlife", force=TRUE)
# summary(compressor)
# trainX     <- predict(compressor,train[,x])
# validX     <- predict(compressor,valid[,x])
# testX      <- predict(compressor,test[,x])
# train      <- h2o.cbind(trainX,train[,y])
# valid      <- h2o.cbind(validX,valid[,y])
# test       <- h2o.cbind(testX,test[,y])
# x          <- which(colnames(train)!="Y")
# y          <- which(colnames(train)=="Y")
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# Machine Learning 
# -------------------------------------------------------------------------
# eval function 
Eval_Model1 <- function(model){
  pred = h2o.predict(model,test)
  h2o.varimp_plot(model)
  print(table(as.data.frame(test$Y)$Y,as.data.frame(pred$predict)$predict))
  print(pROC::auc(as.data.frame(test$Y)$Y,as.data.frame(pred$p1)$p1))
}

Eval_Model2 <- function(model){
  pred = h2o.predict(model,test)
  h2o.varimp_plot(model)
  print(table(as.data.frame(test$Y)$Y,as.data.frame(pred$C1)$C1))
  print(pROC::auc(as.data.frame(test$Y)$Y,as.data.frame(pred$C3)$C3))
}

# options for grid search 
max_runtime_secs <- 60*20
max_models       <- 100
savepath         <- "/media/jacob/database/INGlife/model"

source("/media/jacob/database/INGlife/code/04_GBM_w_Train.R")
source("/media/jacob/database/INGlife/code/04_GBM_w_Tune.R")
h2o.saveModel(object=GBM_w_Train, path=savepath, force=TRUE)
h2o.saveModel(object=GBM_w_Tune, path=savepath, force=TRUE)
Eval_Model1(GBM_w_Train)
Eval_Model1(GBM_w_Tune)
# Area under the curve: 0.9215

source("/media/jacob/database/INGlife/code/05_RF_w_Train.R")
source("/media/jacob/database/INGlife/code/05_RF_w_Tune.R")
h2o.saveModel(object=RF_w_Train, path=savepath, force=TRUE)
h2o.saveModel(object=RF_w_Tune, path=savepath, force=TRUE)
Eval_Model1(RF_w_Train)
Eval_Model1(RF_w_Tune)

source("/media/jacob/database/INGlife/code/06_DL_w_Train.R")
source("/media/jacob/database/INGlife/code/06_DL_w_Tune.R")
h2o.saveModel(object=DL_w_Train, path=savepath, force=TRUE)
h2o.saveModel(object=DL_w_Tune, path=savepath, force=TRUE)
Eval_Model1(DL_w_Train)
Eval_Model1(DL_w_Tune)

source("/media/jacob/database/INGlife/code/07_XGB_w_Train.R")
source("/media/jacob/database/INGlife/code/07_XGB_w_Tune.R")
h2o.saveModel(object=XGB_w_Train, path=savepath, force=TRUE)
h2o.saveModel(object=XGB_w_Tune, path=savepath, force=TRUE)
Eval_Model2(XGB_w_Train)
Eval_Model2(XGB_w_Tune)

# Tensorflow using Python
Sys.setenv(PATH = paste("/home/jacob/anaconda3/bin", Sys.getenv("PATH"), sep=":"))
system("python3 /media/jacob/database/INGlife/TF_w_Train_v1.0.py")
class_test_TF <- read.csv("/media/jacob/database/INGlife/class_test_TF.csv")
proba_test_TF <- read.csv("/media/jacob/database/INGlife/proba_test_TF.csv")
table(as.data.frame(test$Y)$Y,class_test_TF$class_test_TF)
pROC::auc(as.data.frame(test$Y)$Y, proba_test_TF$proba_test_TF)
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# 2layer_stacking
# -------------------------------------------------------------------------
# base learners
pred_GBM <- as.data.frame(h2o.predict(GBM_w_Train,test)$p1)
pred_RF  <- as.data.frame(h2o.predict(RF_w_Train,test)$p1)
pred_DL  <- as.data.frame(h2o.predict(DL_w_Train,test)$p1)
pred_XGB <- as.data.frame(h2o.predict(XGB_w_Train,test)$C3)
pred_TF  <- read.csv("/media/jacob/database/INGlife/proba_test_TF.csv")

# rank 
pred_GBM$rank <- as.integer(rank(-pred_GBM$p1))
pred_RF$rank  <- as.integer(rank(-pred_RF$p1))
pred_DL$rank  <- as.integer(rank(-pred_DL$p1))
pred_XGB$rank <- as.integer(rank(-pred_XGB$C3))
pred_TF$rank  <- as.integer(rank(-pred_TF$proba_test_TF))

# stack
pred_stack    <- data.frame(pred_GBM$rank
                            ,pred_RF$rank
                            ,pred_DL$rank
                            ,pred_XGB$rank
                            ,pred_TF$rank)

# rank avearge
pred_stack$rank_avg <- apply(pred_stack,1,mean)
pred_stack$rank_avg <- with(pred_stack,(rank_avg-min(rank_avg))/(max(rank_avg)-min(rank_avg)))
pROC::auc(as.data.frame(test$Y)$Y, 1-pred_stack$rank_avg)
# Area under the curve: 0.92
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# 3layer_stacking
# -------------------------------------------------------------------------
# XGB variable importances
h2o.varimp_plot(XGB_w_Train)
x_varimp <- h2o.varimp(XGB_w_Train)$variable[1:1]

# train dataset for stacking model
pred_GBM    <- as.data.frame(h2o.predict(GBM_w_Train,valid)$p1)
pred_RF     <- as.data.frame(h2o.predict(RF_w_Train,valid)$p1)
pred_DL     <- as.data.frame(h2o.predict(DL_w_Train,valid)$p1)
pred_XGB    <- as.data.frame(h2o.predict(XGB_w_Train,valid)$C3)
pred_TF     <- read.csv("/media/jacob/database/INGlife/proba_valid_TF.csv")$proba_valid_TF
train_base  <- data.frame(pred_GBM$p1,pred_RF$p1,pred_DL$p1,pred_XGB$C3,pred_TF)
train_stack <- cbind(as.data.frame(valid$Y),train_base,as.data.frame(valid[,x_varimp]))

# test dataset for stacking model
pred_GBM    <- as.data.frame(h2o.predict(GBM_w_Train,test)$p1)
pred_RF     <- as.data.frame(h2o.predict(RF_w_Train,test)$p1)
pred_DL     <- as.data.frame(h2o.predict(DL_w_Train,test)$p1)
pred_XGB    <- as.data.frame(h2o.predict(XGB_w_Train,test)$C3)
pred_TF     <- read.csv("/media/jacob/database/INGlife/proba_test_TF.csv")$proba_test_TF
test_base   <- data.frame(pred_GBM$p1,pred_RF$p1,pred_DL$p1,pred_XGB$C3,pred_TF) 
test_stack  <- cbind(as.data.frame(test$Y),test_base,as.data.frame(test[,x_varimp]))

# feature engineering using woe
binning     <- woe.binning(train_stack,'Y',train_stack)
train_stack <- woe.binning.deploy(train_stack, binning, min.iv.total=0.5,add.woe.or.dum.var='woe') # both dum and woe
test_stack  <- woe.binning.deploy(test_stack, binning, min.iv.total=0.5,add.woe.or.dum.var='woe') # both dum and woe

# convert to h2oDF
train_stack <- as.h2o(train_stack)
test_stack  <- as.h2o(test_stack)

# set x and y
x <- which(colnames(train_stack)!="Y")
y <- which(colnames(train_stack)=="Y")

# meta learner
meta_GLM      <- h2o.glm(training_frame=train_stack,family="binomial",x=x,y=y)
meta_RF       <- h2o.randomForest(training_frame=train_stack,x=x,y=y,seed=1234)
meta_GBM      <- h2o.gbm(training_frame=train_stack,x=x,y=y,seed=1234)
meta_XGB      <- h2o.xgboost(training_frame=train_stack,x=x,y=y,seed=1234)
meta_DL       <- h2o.deeplearning(training_frame=train_stack,x=x,y=y,seed=1234)

pred_meta_GLM <- predict(meta_GLM,test_stack)
pred_meta_RF  <- predict(meta_RF,test_stack)
pred_meta_GBM <- predict(meta_GBM,test_stack)
pred_meta_XGB <- predict(meta_XGB,test_stack)
pred_meta_DL  <- predict(meta_DL,test_stack)

pred_meta_GLM <- as.data.frame(pred_meta_GLM$p1)
pred_meta_RF  <- as.data.frame(pred_meta_RF$p1)
pred_meta_GBM <- as.data.frame(pred_meta_GBM$p1)
pred_meta_XGB <- as.data.frame(pred_meta_XGB$C3)
pred_meta_DL  <- as.data.frame(pred_meta_DL$p1)

# rank 
pred_meta_GLM$rank <- as.integer(rank(-pred_meta_GLM$p1))
pred_meta_RF$rank  <- as.integer(rank(-pred_meta_RF$p1))
pred_meta_GBM$rank <- as.integer(rank(-pred_meta_GBM$p1))
pred_meta_XGB$rank <- as.integer(rank(-pred_meta_XGB$C3))
pred_meta_DL$rank  <- as.integer(rank(-pred_meta_DL$p1))

# stack
pred_meta_stack    <- data.frame(pred_meta_GLM$rank
                                 ,pred_meta_RF$rank
                                 ,pred_meta_GBM$rank
                                 ,pred_meta_XGB$rank
                                 ,pred_meta_DL$rank)

# rank avearge
pred_meta_stack$rank_avg <- apply(pred_meta_stack,1,mean)
pred_meta_stack$rank_avg <- with(pred_meta_stack,(rank_avg-min(rank_avg))/(max(rank_avg)-min(rank_avg)))
pROC::auc(as.data.frame(test$Y)$Y, 1-pred_meta_stack$rank_avg)
# Area under the curve: 0.9136
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# summary 
# -------------------------------------------------------------------------
# output
#Yhat   <- 1-pred_stack$rank_avg
Yhat   <- pred_GBM$p1
output <- data.frame(Y=as.data.frame(test$Y)$Y, Yhat=Yhat)
output <- output[order(output$Yhat,decreasing=T),]

# calibrate prob
output$rate[output$Yhat>=0.9]                   <- "A"
output$rate[output$Yhat<0.9 & output$Yhat>=0.8] <- "B"
output$rate[output$Yhat<0.8 & output$Yhat>=0.7] <- "C"
output$rate[output$Yhat<0.7 & output$Yhat>=0.6] <- "D"
output$rate[output$Yhat<0.6 & output$Yhat>=0.5] <- "E"
output$rate[output$Yhat<0.5 & output$Yhat>=0.4] <- "F"
output$rate[output$Yhat<0.4 & output$Yhat>=0.3] <- "G"
output$rate[output$Yhat<0.3 & output$Yhat>=0.2] <- "H"
output$rate[output$Yhat<0.2 & output$Yhat>=0.1] <- "I"
output$rate[output$Yhat<0.1 & output$Yhat>=0.0] <- "J"
table_calib      <- as.data.frame.matrix(t(table(output$Y,output$rate)))
table_calib$sum  <- rowSums(table_calib)
table_calib$prob <- round(table_calib$"1"/table_calib$sum,2)
print(table_calib)
# -------------------------------------------------------------------------




