# install.packages("faraway")
# install.packages("astsa")
library(faraway)
library(astsa)

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
phi1 <- 0.5
phi2 <- -0.4
x <- arima.sim(list(ar=c(phi1,phi2)),n=1000)
par(mfrow=c(2,1))
plot(x,main=paste("AR(2) process phi1=",phi1," and phi2=",phi2,sep = ""))
x_acf2 <- acf(x, main=paste("AR(2) process phi1=",phi1," and phi2=",phi2,sep = ""))





















