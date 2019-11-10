
# ===============================================
# Phillips Ouliaries Cointegratioin Test 
# ===============================================

# Both series are I(1)
# install.packages("egcm")
library("egcm")
library("tseries") # po.test()
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

# test 
po.test(PepperPrice)
