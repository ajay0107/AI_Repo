# install.packages("faraway")
# install.packages("astsa")
# install.packages("tseries")
# install.packages("isdals")
# install.packages("ppcor")
# install.packages("forecast")

library(bev)
library(faraway)
library(astsa)
library(isdals) # bodyfat dataset
library(ppcor)
library(ggplot2)
library(forecast) # auto.arima()
library(lubridate)
library(dplyr)



data <- data(package = "faraway")
data("coagulation", package="faraway")
coagulation
summary(coagulation)

# producting time plot 
plot(jj,type="o",main="johnson earnings",xlab="time",ylab="earnings")
plot(flu, type="o", main="flu deaths", xlab="months",ylab="deaths per 10k people")
plot(globtemp,type="o",main="global temp deviations", xlab="years",
     ylab="temperature deviations")

# calculation of autocovariance coefficients 
purelyRandomprocess <- ts(rnorm(100))
acf1 <- acf(purelyRandomprocess, type = "covariance")
# calcuaton of autocorrelation coefficients
acf2 <- acf(purelyRandomprocess, main= "auto-correlation coefficients")

# Random Walk 
randomWalk <- vector()
randomWalk[1] <- 0
randomvar <- rnorm(10000)
for(i in 2:10000){
  randomWalk[i] <- randomWalk[i-1]+randomvar[i]
}
 
randomWalk <- ts(randomWalk) # converting vector to time series
# plotting random walk 
plot(randomWalk, main="Random Walk", ylab="value", xlab="days",col="blue", lwd=1)
# acf is defined for stationary process and Random Walk is not a stationary process
# but let's just find the acf 
acf(randomWalk)
# Difference x2-x1, x3-x2 etc of random walk data, creates stationary process (purely random process) data
purelyRandomprocess_1 <- diff(randomWalk)
acf(purelyRandomprocess_1, main="diff of random walk process")
plot(purelyRandomprocess_1,main="diff of random walk process")

# simulating moving average (MA(2)) process
# generate random normal distribution random numbers 
noise = rnorm(10000)
# empty vector
ma_2 <- vector()
# loop for generating MA(2) process
for (i in 3:10000) {
  ma_2[i] <- noise[i]+0.7*noise[i-1]+0.2*noise[i-2]
}
# shift the intial 2 empty elements
moving_average_process <- ma_2[3:10000]
# put time series structure on this vanilla data 
moving_average_process <- ts(moving_average_process)
# plotting multiframe plots 
par(mfrow=c(2,1))
plot(moving_average_process, main="Moving average process (2)",col="blue")
# particular structure of acf graphs tells us whether or not the data is from MA(2) process
# Autocorrelation cuts of at lag q for MA(q) process
acf(moving_average_process, main="autocorrelation graph of MA(2)")

# exercise - simulating MA(5) process
ma_5 <- vector()
for(i in 6:10000){
  ma_5[i] <- noise[i]+0.7*noise[i-1]+0.5*noise[i-2]+0.1*noise[i-3]+0.3*noise[i-4]+0.2*noise[i-5]
}
ma_5 <- ma_5[!is.na(ma_5)]
ma_5 <- ts(ma_5)
par(mfrow=c(2,1))
plot(ma_5, main="moving average order 5", col="red")
acf(ma_5,main="autocorrelation graph of MA(5)")

# simulating the AR(1) process
set.seed(2016)
N <- 1000
phi <- 0.4
z <- rnorm(N,0,1)
x <- vector()
x[1] <- z[1]
for (i in 2:N) {
  x[i] <- z[i]+phi*x[i-1]
}
# converting x as ts object
x <- ts(x)
par(mfrow=c(2,1))
plot(x, main="AR(1) process with white noise,phi=0.4")
x_acf <- acf(x,main="ACF of AR(1) process with white noise,phi=0.4")
# if phi=1
set.seed(2016)
N <- 1000
phi <- 1
z <- rnorm(N,0,1)
x <- vector()
x[1] <- z[1]
for (i in 2:N) {
  x[i] <- z[i]+phi*x[i-1]
}
# converting x as ts object
x <- ts(x)
par(mfrow=c(2,1))
plot(x, main="AR(1) process with white noise,phi=1")
x_acf <- acf(x,main="ACF of AR(1) process with white noise,phi=1")

# simulating AR(2) process xt=zt+phi1*xt-1+phi2*xt-2
set.seed(2017)
phi1 <- 0.6
phi2 <- 0.2
x <- arima.sim(list(ar=c(phi1,phi2)),n=1000)
par(mfrow=c(3,1))
plot(x,main=paste("AR(2) process phi1=",phi1," and phi2=",phi2,sep = ""))
x_acf2 <- acf(x, main=paste("AR(2) process phi1=",phi1," and phi2=",phi2,sep = ""))
x_acf_partial <- acf(x,type = "partial" ,main=paste("PACF of AR(2) process phi1=",phi1," and phi2=",phi2,sep = ""))

# loading beveridge wheat prices index
data(bev)
plot(bev,ylab="price",main="Beveridge Wheat Price Data")
beveridge_MA <- filter(bev,rep(1/31,31),sides = 2)
lines(beveridge_MA,col="red")
par(mfrow=c(3,1))
y=bev/beveridge_MA
plot(y,ylab="scaled price",main="Transformed Beveridge wheat price data")
acf(na.omit(y),main="Autocorrelation function of Transformed Beveridge wheat price data")
acf(na.omit(y),type = "partial",main="PACF of Transformed Beveridge wheat price data")

# remember AR(p) has PACF that cuts after p lags 
# ar() command automatically selects the order of AR process
ar(na.omit(y),order.max = 6)
data(bodyfat)
head(bodyfat)
pairs(bodyfat)
attach(bodyfat)
cor(Fat,Triceps)
fat_hat <- predict(lm(Fat~Thigh))
triceps_hat <- predict(lm(Triceps~Thigh))
cor((Fat-fat_hat),(Triceps-triceps_hat))
# calculating partial correlation using ppcor
pcor(bodyfat)
pcor(bodyfat[,!colnames(bodyfat) %in% c("Midarm")])
# removing effects of Thigh, Midarm
fat_hat1 <- predict(lm(Fat~Thigh+Midarm))
triceps_hat1 <- predict(lm(Triceps~Thigh+Midarm))
cor((Fat-fat_hat1),(Triceps- triceps_hat1))

# Finding coefficients of AR models 
# AR(2) model
set.seed(2017)
n <- 10000
ar_process=arima.sim(n,model = list(ar=c(1/3,1/2)),sd=4)
r <- acf(ar_process)
rPartial <- acf(ar_process, type = "partial")
R_matrix <- matrix(data=1,2,2)
R_matrix[1,2] <- r$acf[2]
R_matrix[2,1] <- r$acf[3]
B <- matrix(data = c(r$acf[2],r$acf[3]),ncol = 1,nrow = 2)
coeff <- solve(R_matrix,B)
c0 <- acf(ar_process,type = "covariance")
sdSquared<- c0$acf[1]*(1-r$acf[2]*coeff[1,1]-r$acf[3]*coeff[2,1])
# AR(3) model
set.seed(2017)
n <- 10000
ar_process=arima.sim(n,model = list(ar=c(1/3,1/2,7/100)),sd=4)
r <- acf(ar_process)
rPartial <- acf(ar_process, type = "partial")
R=matrix(1,3,3) 
R[1,2]=r$acf[1] 
R[1,3]=r$acf[2]
R[2,1]=r$acf[1]
R[2,3]=r$acf[1]
R[3,1]=r$acf[2]
R[3,2]=r$acf[1]
R
# b-column vector on the right
b=matrix(1,3,1)# b- column vector with no entries
b[1,1]=r$acf[1]
b[2,1]=r$acf[2]
b[3,1]=r$acf[3]
b
coeff <- solve(R,b)
c0 <- acf(ar_process,type = "covariance")
sdSquared<- c0$acf[1]*(1-r$acf[2]*coeff[1,1]-r$acf[3]*coeff[2,1]-
                         coeff[3,1]*r$acf[4])

# fit AR(p) process to stationary time series data and find its parameters
# Recruitment dataset
myData <- rec
# substracting mean to get time series with mean zero
arProcess <- myData-mean(myData)
# ACF and PACF of AR process
par(mfrow=c(3,1))
plot(myData, main="Recruitment time series",col="blue",lwd=3)
acfmodel <- acf(arProcess, main="ACF recruitment", col="red", lwd=2)
pacfModel <- acf(arProcess,type = "partial",main="PACF recruitment", col="green", lwd=2)
acfmodel$acf
# order
p=2
# ACFs 
acfVec <- vector()
acfVec[1:p] <- acfmodel$acf[2:(p+1)]
# defining Matrix-R
RMatrix <- matrix(1,p,p)
for (i in 1:p) {
  for (j in 1:p) {
    if (i!=j) {
      RMatrix[i,j] <- acfVec[abs(i-j)]
    }
  }
}
# b vector
b <- vector()
b <- matrix(acfVec,p,1)
coeffNew <- solve(RMatrix,b)
c0 <- acf(arProcess, type = "covariance")
c0 <- c0$acf[1]
varHat <- c0*(1-t(coeffNew) %*% matrix(data = acfVec,nrow = 2, ncol = 1))
# getting constant term
phi0Hat <- mean(myData)*(1-sum(coeffNew))

# fit AR(p) process to Non-stationary time series data and find its parameters
# Johnson & Johnson-model fitting
data("JohnsonJohnson")
plot(JohnsonJohnson, main ="johnson and johnson quarterly earnings",
     col="blue",lwd=2)
# mean is changing (increasing) and variance is changing
# we can't fit a stationary AR model on this data, we have to transform this dataset
# one such famous transformation is log return of time series, which makes it stationary time series
logReturn <- diff(log(JohnsonJohnson))
par(mfrow=c(3,1))
plot(logReturn, main ="johnson and johnson dataset log return",
     col="red",lwd=2)
acfjohnson <- acf(logReturn,main="ACF johnson",col="green")
pacfjohnson <- acf(logReturn, type = "partial",main="PACF johnson",
                   col="magenta")
# we will try to fit AR(4) model here
arProcessJohnson <- logReturn-mean(logReturn)
par(mfrow=c(2,1))
acfjohnsonMeanZero <- acf(arProcessJohnson, main="ACF johnson mean zero",
                          col="red",lwd=3)
pacfjohnsonMeanZero <- acf(arProcessJohnson,type="partial",main="PACF johnson mean zero",
                          col="green",lwd=3)
p <- 4
johnsonAcfs <- acfjohnsonMeanZero$acf[2:(p+1)]
Rjohnson <- matrix(data = 1, nrow = p,ncol = p)
for (i in 1:p) {
  for (j in 1:p) {
    if(i!=j){
      Rjohnson[i,j] <- johnsonAcfs[abs(i-j)]  
    }
  }
}

bjohnson=matrix(johnsonAcfs,p,1)
coeffJohnson <- solve(Rjohnson,bjohnson)
# variance estimation
c0 <- acf(arProcessJohnson,type = "covariance",plot = F)$acf[1]
varjohnson <- c0*(1-sum(coeffJohnson*johnsonAcfs))
# constant term
phi0johnson <- mean(logReturn)*(1-sum(coeffJohnson))
# we have equation with gives log return 


# data simulation
phi1 <- 0.7
phi2 <- -0.2
data <- arima.sim(list(ar=c(phi1,phi2)),n=2000)
par(mfrow=c(2,1))
acfSim <- acf(data, main="ACF of AR(2) process")
pacfSim <- acf(data,type = "partial",main="ACF of AR(2) process")
# using arima function to know coefficients of AR process
arimaModel <- arima(data, order = c(2,0,0),include.mean = F)
df <- as.data.frame(matrix(data = NA, nrow = 5000,ncol = 3))
colnames(df) <- c("ar_order","aic","sse")
for (p in 1:20) {
  print(p)
  arimaModel <- arima(data, order = c(p,0,0),include.mean = F)
  df$ar_order[p] <- p
  df$aic[p] <- arimaModel$aic
  df$sse[p] <- sum(resid(arimaModel)^2)
}

View(df)
par(mfrow=c(2,1))
plot(x=df$ar_order, y=df$aic, col="red")
plot(x=df$ar_order, y=df$sse, col="blue")

# simluating data of ARMA(p,q)
set.seed(500)
phi1 <- 0.7
phi2 <- 0.2
dataArma <- arima.sim(list(order=c(1,0,1),ar=phi1,ma=phi2),n=1000000)
par(mfcol=c(3,1))
plot(data,main="ARMA(1,1)",xlim=c(0,400))
acfArma <- acf(data,main="ACF of ARMA(1,1)")
pacfArma <- acf(data,type = "partial",main="PACF of ARMA(1,1)")

# estimate coefficients of ARMA models
data("discoveries")
plot(discoveries,main="Discoveries")
stripchart(discoveries,method = "stack",offset = 0.5,at=0.15,
           pch=19,main="Number of discoveries dotplot",
           xlab="number of discoveries in year",
           ylab="frequency")
par(mfrow=c(2,1))
acfdiscoveries <- acf(discoveries, main="acf discoveries", col="red", lwd=3)
acfdiscoveries <- acf(discoveries,type = "partial",main="acf discoveries", col="blue", lwd=3)
# difficult to tell the order of ARMA models by acf anf pacf plot 
# we try several competing models and choose the best.. i.e. which has lowest AIC
df <- as.data.frame(matrix(data = NA, nrow = 5000,ncol = 3))
colnames(df) <- c("ar_order(p)","ma_order(q)","aic")
count <- 1
for (p in 0:3) {
  for (q in 0:3) {
    arimamodelDis <- arima(discoveries, order = c(p,0,q))
    df$aic[count] <- arimamodelDis$aic
    df$`ar_order(p)`[count] <- p
    df$`ma_order(q)`[count] <- q
    count <- count+1
  }
}
View(df)
# select the p,q for which we have lowest AIC
# automatic function which gives best model
auto.arima(discoveries,d=0,ic="bic",approximation = F)
auto.arima(discoveries,d=0,ic="bic",approximation = T)
auto.arima(discoveries,d=0,ic="aic",approximation = T)
auto.arima(discoveries,d=0,ic="aic",approximation = F)

# fitting ARIMA model
data <- as.data.frame(readxl::read_xlsx(path = "female_births_california.xlsx"))
par(mfrow=c(2,1))
plot(data$Date,data$`Daily total female births in California, 1959`, type = "l",
     main = "female births california")
numBirths <- data$`Daily total female births in California, 1959`
# test if autocorrelation exist for different lags 
Box.test(numBirths,lag = log(length(numBirths)))
# null hypothesis is that no auto correlation exist for any lag, 
# as we see p-value <0.05 so there exist autocorrelation for any lag
# we can see from plot that there exist trend in the time series so we take differencing 
plot(x=data$Date[1:364],y=diff(numBirths),type = "l", 
     main="female births after differencing")
# again doing box.test 
Box.test(diff(numBirths), lag = log(length(numBirths)))
par(mfrow=c(2,1))
acfBirth <- acf(diff(numBirths), main="acf of diff-births", col="red",lwd=2)
pacfBirth <- acf(diff(numBirths),type = "partial",main="pacf of diff-births", 
                 col="green",lwd=2)
# fitting various ARIMA models 
model1 <- arima(numBirths, order = c(0,1,1))
SSE1 <- sum(model1$residuals^2)
model1Test <- Box.test(model1$residuals, lag = log(length(model1$residuals)))

model2 <- arima(numBirths, order = c(0,1,2))
SSE2 <- sum(model2$residuals^2)
model2Test <- Box.test(model2$residuals, lag = log(length(model2$residuals)))

model3 <- arima(numBirths, order = c(7,1,1))
SSE3 <- sum(model3$residuals^2)
model3Test <- Box.test(model3$residuals, lag = log(length(model3$residuals)))

model4 <- arima(numBirths, order = c(7,1,2))
SSE4 <- sum(model4$residuals^2)
model4Test <- Box.test(model4$residuals, lag = log(length(model4$residuals)))


df<-data.frame(row.names=c('AIC', 'SSE', 'p-value'), c(model1$aic, SSE1, model1Test$p.value), 
               c(model2$aic, SSE2, model2Test$p.value), c(model3$aic, SSE3, model3Test$p.value),
               c(model4$aic, SSE4, model4Test$p.value))
colnames(df)<-c('Arima(0,1,1)','Arima(0,1,2)', 'Arima(7,1,1)', 'Arima(7,1,2)')
format(df,scientific=F)
# here, we select model which has lowest AIC 
# we select model = arima(0,1,2)
sarima(numBirths,0,1,2,0,0,0)

# steps to do SARIMA/ARIMA modelling- SARIMA(p,d,q,P,D,Q)
# (1) time plot
# (2) if variance changes with time, transform the data for example, take log of data
# (3) if there is trend (seasonal/Non seasonal), then do differencing to remove trend- seasonal and non-seasonal differencing
# (3') if there is both trend + variance change, then we can do log return = diff(log(ts))
# (4) Do Ljung-Box Test- to check if there is an auto-correlation between lags
# (5) ACF -> Adjacent spikes -> MA order
# (6) ACF -> Spikes around seasonal lags -> SMA order
# (7) PACF -> Adjacent spikes -> AR order
# (8) PACF -> Spikes around seasonal lags -> SAR order
# (9) Fit few different models
# (10) Compare AIC, choose a model with minimum AIC
# (11) The parsimony principle- p+d+q+P+D+Q <= 6
# (12) Time plot, ACF and PACF of residuals
# (13) Ljung-Box test for residuals - we expect white noise
# (14) choose the model, which follows parsimony principle, along with lowest AIC

plot(JohnsonJohnson,main="quarterly earnings johnson and johnson", col="red", lwd=2)
jjLogreturn <- diff(log(JohnsonJohnson))

# plot after log return
par(mfrow=c(3,1))
plot(jjLogreturn, main="log return quarterly earnings johnson and johnson", col="blue", lwd=1.5)
acfjjLogreturn <- acf(jjLogreturn, main="ACF of log JJ log return", col="green", lwd=3)
pacfjjLogreturn <- acf(jjLogreturn,type = "partial" ,main="PACF of log JJ log return", col="magenta", lwd=3)
# taking seasonal differencing with lag=4
par(mfrow=c(3,1))
jjLogreturnSeasonalDiff <- diff(jjLogreturn,4)
plot(jjLogreturnSeasonalDiff, main="seasonal diff quarterly earnings johnson and johnson", col="blue", lwd=1.5)
acfjjLogreturn <- acf(jjLogreturnSeasonalDiff, main="ACF of log JJ log return", col="green", lwd=3)
pacfjjLogreturn <- acf(jjLogreturnSeasonalDiff,type = "partial" ,main="PACF of log JJ log return", col="magenta", lwd=3)

# LjungBox Test
Box.test(jjLogreturnSeasonalDiff,lag = log(length(jjLogreturnSeasonalDiff)))
# if p-value <0.05, we reject the null hypothesis that there is no auto-correlation between lags
df <- as.data.frame(matrix(data = NA, ncol = 4, nrow = 1000000))
colnames(df) <- c("parameters","aic","sse","pValue")
count <- 1
for (p in 0:3) {
    for (q in 0:3) {
      for (capP in 0:3) {
          for (capQ in 0:3) {
              if(p+1+q+capP+1+capQ <=7){
              print(paste(p,1,q,capP,1,capQ,sep = "-"))
              model <- arima(log(JohnsonJohnson),order = c(p,1,q),seasonal = list(order=c(capP,1,capQ),period=4))
              df$parameters[count] <- paste(p,1,q,capP,1,capQ,sep = "-")
              df$aic[count] <- model$aic
              df$sse[count] <- sum(model$residuals^2)
              boxTest <- Box.test(model$residuals,lag=log(length(model$residuals)))
              df$pValue[count] <- boxTest$p.value
              count <- count+1
              print(count)
              }
            }
            
          }
          
        }
        
      }
      
# using SARIMA for fitting sarima model
df <- as.data.frame(matrix(data = NA, ncol = 4, nrow = 1000000))
colnames(df) <- c("parameters","aic","sse","pValue")
count <- 1
for (p in 0:3) {
  for (q in 0:3) {
    for (capP in 0:3) {
      for (capQ in 0:3) {
        if(p+1+q+capP+1+capQ <=7){
          print(paste(p,1,q,capP,1,capQ,sep = "-"))
          model <- sarima(log(JohnsonJohnson),p,1,q,capP,1,capQ,4)
          df$parameters[count] <- paste(p,1,q,capP,1,capQ,sep = "-")
          df$aic[count] <- model$fit$aic
          df$sse[count] <- sum(model$fit$residuals^2)
          boxTest <- Box.test(model$fit$residuals,lag=log(length(model$fit$residuals)))
          df$pValue[count] <- boxTest$p.value
          count <- count+1
          print(count)
        }
      }
      
    }
    
  }
  
}

# using auto.arima
autoModelarima <- auto.arima(y=log(JohnsonJohnson),max.p = 2,max.q = 2, max.P = 2,max.Q = 2, max.order = 8, start.p = 0,start.q = 0,
           start.P = 0,start.Q = 0,stationary = F,seasonal = TRUE,ic="aic",trace = T,max.d = 3,max.D = 3)
plot(forecast(autoModelarima))
forecast(autoModelarima)
plot(forecast(model))
forecast(model)


# Monthly Milk production per cow dataset
dataMilk <- as.data.frame(readxl::read_xlsx(path = "monthly_milk_production_per_cow.xlsx"))
dataMilk$Month <- as.Date(paste(dataMilk$Month,"01",sep = "-"),"%Y-%m-%d")
par(mfrow=c(3,1))
plot(x=dataMilk$Month,y=dataMilk$Milk, main="monthly milk production per cow",col="red",lwd=1.5, type = "l")
acfMilk <- acf(dataMilk$Milk,main="acf of milk data", col="blue",lwd=2,lag.max=50)
pacfMilk <- acf(dataMilk$Milk,type = "partial",main="pacf of milk data", col="green",lwd=2,lag.max=50)
# from figure, it suggest cyclic behaviour after 12 lags - seasonal differencing 12 lags
# and there is also a trend, given by graph - non seasonal differencing 
nonseasonaldiff <- diff(dataMilk$Milk)
seasonaldiff <- diff(nonseasonaldiff,12)
# no need to transform the data because variance is same 
plot(seasonaldiff, main="with nonseasonal and season diff", col="red", lwd=1.5, type = "l")
# now, our data becomes stationary- only stationary data can be fitted by models 
acfMilk <- acf(seasonaldiff,main="acf of milk data with nonseasonal and season diff", col="blue",lwd=2,lag.max=50)
pacfMilk <- acf(seasonaldiff,type = "partial",main="pacf of milk data with nonseasonal and season diff", col="green",lwd=2,lag.max=50)
# we can see that acf and pacf are significant at lags = 12, 24, 36 etc after nonseasonal and season diff. So, there is seasonal AR and seasonal MA
# from acf and pacf plot, we can get idea of parameters 
# MA= q <- 0, Q=0,1,2,3
# AR = p <- 0, P = 0,1,2,3
df <- as.data.frame(matrix(data = NA, ncol = 4, nrow = 1000000))
colnames(df) <- c("parameters","aic","sse","pValue")
count <- 1
d <- 0
DD <- 0
p <- 0
q <- 0
per <- 12
    for (capP in 0:3) {
      for (capQ in 0:3) {
        if(p+d+q+capP+DD+capQ <=7){
          print(paste(p,d,q,capP,DD,capQ,sep = "-"))
          model <- arima(seasonaldiff,order = c(p,d,q),seasonal = list(order=c(capP,DD,capQ),period=per))
          df$parameters[count] <- paste(p,d,q,capP,DD,capQ,sep = "-")
          df$aic[count] <- model$aic
          df$sse[count] <- sum(model$residuals^2)
          boxTest <- Box.test(model$residuals,lag=log(length(model$residuals)))
          df$pValue[count] <- boxTest$p.value
          count <- count+1
          print(count)
        }
      }
      
    }
    
# if you manual do.....seasonal and non seasonal differencing , then put d=0, D=0 in arima function. It gives same result
 # now, forecasting
model <- arima(dataMilk$Milk,order = c(0,1,0),seasonal = list(order=c(0,1,1),period=12))
plot(forecast(model))  
forecast(model)

# Airplane_Crashes_and_Fatalities_Since_1908 dataset
GTemp <- read.csv("GlobalTemperatures.csv",stringsAsFactors = F, check.names = F)
GTemp["month"] <- month(as.POSIXlt(GTemp$dt, format="%Y-%m-%d"))
GTemp["year"] <- year(as.POSIXlt(GTemp$dt, format="%Y-%m-%d"))
# GTemp <- GTemp[, c("yearCrash","monthCrash","Fatalities")]
# GTemp <- as.data.frame(GTemp %>% group_by(yearCrash,monthCrash) %>% summarise(Fatalities=sum(Fatalities,na.rm = T)))
# xx <- as.data.frame(GTemp %>% group_by(year) %>% summarise(months=sum(month,na.rm = T)))
# taking data from 1960-1991 - only at these time data is uniformly present
GTemp <- GTemp[GTemp$year %in% as.character(1753:2015),]
GTemp["yearMonth"] <- c(1:length(GTemp$dt))
timeVec <-GTemp$yearMonth
landTemp <- GTemp$LandAverageTemperature
plot(x=timeVec,y=landTemp, main="GTemp monthly", col="red", lwd=2, type = "l")
# there is trend + seasonality in data
# making data stationary
logreturntemp <- diff(landTemp)
par(mfrow=c(3,1))
plot(x=timeVec[2:length(landTemp)],y=logreturntemp, main="GTemp monthly", col="red", lwd=2, type = "l")
acfGTemp <- acf(logreturntemp, main= "ACF of GTemp monthly",col="blue", lwd=3 )
pacfGTemp <- acf(logreturntemp,type = "partial",main= "PACF of GTemp monthly",col="green", lwd=3 )
Box.test(fatalities,lag=log(length(landTemp)))










































