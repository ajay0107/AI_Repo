
# ===============================================
# Engle-Granger cointegration using egcm()
# ===============================================

# Both series are I(1)
# install.packages("egcm")
library("egcm")
library("tseries")
library("urca")
library("AER")
library("dplyr")
library(ggplot2)
# H0 : There is no cointegration in between black and white pepper
data("PepperPrice")
PepperPriceDf <- as.data.frame(as.matrix(PepperPrice))
PepperPriceDf <- PepperPriceDf %>% mutate(time=1:nrow(PepperPriceDf))
# plot
ggplot(PepperPriceDf, aes(time)) + 
  geom_line(aes(y = black, colour = "black")) + 
  geom_line(aes(y = white, colour = "white"))

engle_granger_test <- egcm(PepperPrice)
summary(engle_granger_test)


# =====================================================
# Engle-Granger cointegration test using dynlm() 
# =====================================================
install.packages("dynlm")
library(dynlm) # dynamic linear model
library(urca)

# step-1: store the values of residuals ( using dynlm)
# step-2: Test for the stationarity of residuals (using urca package)

model_1 <-  dynlm(black~white,data = PepperPrice)
summary(model_1)
res_pepper <- residuals(model_1)
summary(ur.df(res_pepper))












