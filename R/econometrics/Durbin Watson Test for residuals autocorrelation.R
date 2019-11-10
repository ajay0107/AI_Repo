

library("egcm")
library("tseries")
library("urca")
library("AER")
library("dplyr")
library("vars")
library(ggplot2)

# The Durbin Watson (DW) statistic is a test for autocorrelation in the residuals 
# from a statistical regression analysis. The Durbin-Watson statistic will always 
# have a value between 0 and 4. A value of 2.0 means that there is 
# no autocorrelation detected in the sample. Values from 0 to less 
# than 2 indicate positive autocorrelation and values from from 2 to 4 
# indicate negative autocorrelation.

# Durbin–Watson statistic is a test statistic used to detect the presence of autocorrelation at 
# lag 1 in the residuals (prediction errors) from a regression analysis.

# if residuals are auto-correlated then you can infer that your model is 
# misspecified, that is you have missed some explanatory variable that can
# capture that trend (information in dependent variable). In this case, consider 
# adding lagged independed or dependent variable

# The result of Durbin-Watson showes that either you fitted a linear function to 
# your data which have nonlinear relationship or you didn't condconsidered an important 
# variable in your model. Try to transform your data in a way that you can guarantee 
# a linear relationship and think if you omitted an important variable or not. 

# In case of durbin.watson problem use a new variable which is a lagged transformation 
# of the independent time series variables, then apply two stage least regression 
# procedure.it means part of the variation is explained by linear trend of either 
# the dependent or independent variable

# The Hypotheses for the Durbin Watson test are:
#   H0 = no first order autocorrelation.
#   H1 = first order correlation exists.

# The Durbin Watson test reports a test statistic, with a value from 0 to 4, where:
#   
#   2 is no autocorrelation.
# 0 to <2 is positive autocorrelation (common in time series data).
# >2 to 4 is negative autocorrelation (less common in time series data).


library(lmtest)

data=read.csv(file = "data/oilgspc.csv",stringsAsFactors = F, check.names = F)
head(data)
model_lm <-  lm(gspc~oil, data = data)
summary(model_lm)

# Durbin-Watson test
durbin_watson_test <- dwtest(model_lm)
durbin_watson_test
