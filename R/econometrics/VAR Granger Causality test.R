
library(lmtest)
library(tseries)
dataoil=read.csv(file = "data/oilgspc.csv",stringsAsFactors = F, check.names = F)
attach(dataoil)

# Granger Causality test
# series which we are considering must be stationary

# doing ADF test
adf.test(gspc) # tells that gspc is non-stationary
plot(gspc,type="l")
adf.test(oil) # tells that oil is non-stationary
plot(oil,type="l")
# taking difference to make it stationary
gspcdiff <- diff(gspc)
plot(gspcdiff,type="l")
oildiff <- diff(oil)
plot(oildiff,type="l")
adf.test(gspcdiff)
adf.test(oildiff)

datatest <-  cbind(oildiff,gspcdiff)
dataVAR <-  VAR(datatest,type = "const",p = 2) # lag =2
summary(dataVAR)
#  Since Granger-Causality test is sensitive to the lag length
# let's try to take information criteria to select lags
dataVAR1 <-  VAR(datatest,type="const",lag.max = 10,ic = "AIC")
summary(dataVAR1)

# implementing Granger test 
# H0: oildiff does not granger cause gspcdiff 
causality(dataVAR1,cause = "oildiff")

# H0: gspcdiff does not granger cause oildiff 
causality(dataVAR1,cause = "gspcdiff")

# experiment with random numbers
r1 <- rnorm(5000)
r2 <- rnorm(5000)
r <- cbind(r1,r2)
datar <- VAR(r,type = "const",lag.max = 15,ic = "AIC")
summary(datar)
causality(datar,cause = "r1")
causality(datar,cause = "r2")




















