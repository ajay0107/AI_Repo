
# ===============================================
# Johanson cointegration test - it test for multiple time series for cointegration 
# ===============================================

# Both series are I(1)
# install.packages("egcm")
library("egcm")
library("tseries")
library("urca")
library("AER")
library("dplyr")
library("vars")
library(ggplot2)
# H0 : r =0: Number of cointegrating vectors is zero.
# H0 : r <= 1 : Number of cointegrating vectors is less than or equal to 1.
# H0 : r <= 2 : Number of cointegrating vectors is less than or equal to 2.

data("PepperPrice")
PepperPriceDf <- as.data.frame(as.matrix(PepperPrice))
PepperPriceDf <- PepperPriceDf %>% mutate(time=1:nrow(PepperPriceDf))
random_n <- rnorm(nrow(PepperPriceDf),mean = 400,sd=200)
PepperPriceDf$red <- PepperPriceDf$black+random_n

# plot
ggplot(PepperPriceDf, aes(time)) + 
  geom_line(aes(y = black, colour = "black")) + 
  geom_line(aes(y = white, colour = "white"))+
  geom_line(aes(y = red, colour = "red"))

# varselect gives appropriate lags 
varmodel <- VARselect(PepperPriceDf[-3],lag.max = 10,type = "const")
varmodel
# conduct test 
cointest <- ca.jo(PepperPriceDf[-3],K = 4,type = "eigen",
                  ecdet = "const",spec = "transitory")
summary(cointest)


