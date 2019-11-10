


library("egcm")
library("tseries")
library("urca")
library("AER")
library("dplyr")
library("vars")
library(ggplot2)

data <- as.data.frame(readxl::read_xlsx(path = "../practical_time_series_coursera/female_births_california.xlsx"))
plot(data$Date,data$`Daily total female births in California, 1959`, type = "l",
     main = "female births california")
numBirths <- data$`Daily total female births in California, 1959`
# test if autocorrelation exist for different lags 
Box.test(numBirths,lag = log(length(numBirths)))
# H0 :  No auto correlation exist for given lag 
# as we see p-value <0.05 so there exist autocorrelation for given lag
# we can see from plot that there exist trend in the time series so we take differencing 

data1 <- rnorm(1000)
Box.test(data1,lag = log(length(data1)))
# as we see p-value > 0.05 so there doesn't exist autocorrelation for given lag
