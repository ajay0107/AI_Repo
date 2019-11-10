

library(vars)
library(urca)
library("egcm")
dataoil=read.csv(file = "data/oilgspc.csv",stringsAsFactors = F, check.names = F)
attach(dataoil)

# doing ADF test
adf.test(gspc) # tells that gspc is non-stationary
plot(gspc,type="l")
adf.test(oil) # tells that oil is non-stationary
plot(oil,type="l")

# check for cointegration
E <- egcm(dataoil[-1])
is.cointegrated(E) # not cointegrated

# fit a VAR model with appropriate lags
# gives appropriate lag criteria  
VARselect(dataoil[-1],lag.max = 12,type = "const")

# conduct cointegration test (johansen test)
cointest <- ca.jo(dataoil[-1],K = 2,type = "eigen",ecdet = "const",
                  spec = "transitory")
summary(cointest)

# H0 : r =0: Number of cointegrating vectors is zero.
# H0 : r <= 1 : Number of cointegrating vectors is less than or equal to 1.
# H0 : r <= 2 : Number of cointegrating vectors is less than or equal to 2.

# Run VCECM
vecm <-  cajorls(cointest)
vecm







