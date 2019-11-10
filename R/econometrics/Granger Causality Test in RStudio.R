
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

# Implementing granger-causality test

# H0 : oil does not cause gspc , then
# oil would be independent variable
# gspc would be dependent variable

grangertest(gspc~oil, order=2)

# Interpretation
# if p value <0.05 then, Reject the Null hypothesis and oil causes gspc
# if p value > 0.05 then, accept the Null hypothesis and oil does not causes gspc

# H0 : gspc does not cause oil , then
# gspc would be independent variable
# oil would be dependent variable

grangertest(oil~gspc, order=2)

# Interpretation
# if p value <0.05 then, Reject the Null hypothesis and gspc causes oil
# if p value > 0.05 then, accept the Null hypothesis and gspc does not causes oil

# overall here, we get to know nobody causes anybody
# granger-test is sensitive to selection of lags

















