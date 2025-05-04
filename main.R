# Description of Data
data <- read.csv(file = 'Nasdaq100_extreme_interval_data.csv')

ei_list <- c(3,5,8,13,20)
for(ei in ei_list){
  load_df <-paste0('Local_Extrema_',ei,'/QQQ_combination.csv')
  ei_data <- read.csv(load_df)
  print(paste('ei=', ei))
  print(paste('0_prob: mean=',mean(ei_data$X0_prob),'sd=',sd(ei_data$X0_prob)))
  print(paste('1_prob: mean=',mean(ei_data$X1_prob),'sd=',sd(ei_data$X1_prob)))
  print(paste('2_prob: mean=',mean(ei_data$X2_prob),'sd=',sd(ei_data$X2_prob)))
}

# Draw Distribution

density_compare <- function(var_name = 'prob_diff'){
  ei=20
  load_df <-paste0('Local_Extrema_',ei,'/QQQ_combination.csv')
  ei_data <- read.csv(load_df)
  des <- density(as.numeric(unlist(ei_data[var_name])))
  plot(des, xlim=c(-0.5,0.5), ylim=c(0, max(des$y)), col='red')
  col_list <- c('blue','green', 'orange', 'black')
  i <- 1
  for(ei in c(3, 5,8,13)){
    load_df <-paste0('Local_Extrema_',ei,'/QQQ_combination.csv')
    ei_data <- read.csv(load_df)
    lines(density(as.numeric(unlist(ei_data[var_name]))), col=col_list[i])
    i <- i + 1
  }
  legend(-0.4, 4,legend=c('20','3','5','8','13'), col=c('red','blue','green', 'orange', 'black'), lty=1)
}
# Fig 1.4.2~1.4.4
density_compare('prob_diff')
density_compare('X1_prob')
density_compare('X2_prob')


################################################################################
# Model
main_model <- function(ei_data, var_name='X1_prob', ei=3){
  n <- nrow(ei_data)
  df1 <-  structure(list(date=1:n, prob1=as.numeric(unlist(ei_data[var_name]))),.Names = c("date","prob"), class = "data.frame",row.names = c(NA, -n))
  
  ma <- function(x,n=7){stats::filter(x,rep(1/n,n), sides=2)}
  lines(df1$date, ma(df1$prob, n = 4), col='red')
  repl <- 1000
  S <- sum(df1$prob)
  YL <- cumsum(df1$prob)[1:(n-1)]
  YU <- (S - YL)
  
  #priors
  mu_init <- mean(df1$prob)
  sigma_init <- sd(df1$prob)
  a <- (mu_init^2)/(sigma_init^2)
  b <- mu_init/(sigma_init^2)
  a1 <- a2 <- a
  b1 <- b2 <- b
  
  #priors
  # curve(dgamma(x, shape = a, rate = b), from = 0, to = 1)
  # (qq <- qgamma(p= c(0.025, 0.5, 0.975), shape = a, rate = b))
  # abline(v= qq, lty = 2, col = 'red')
  
  #starting values
  m <- 1
  lambda1 <- lambda2 <- mu_init
  
  #Initializing sampler
  trace <- array(NA, dim=c(repl, 3), dimnames=list(iteration=NULL, parameters=c('lambda1', 'lambda2', 'changepoint')))
  trace[1,] <- c(lambda1, lambda2, m) #initial values
  
  
  #Sampling
  for(i in 2:repl){
    lambda1 <- rgamma(1, shape = a1 + YL[m], rate = b1 + m)
    lambda2 <- rgamma(1, shape = a2 + YU[m], rate = b2 + n - m)
    b <- (log(lambda1) + log(lambda2))/2*S + (lambda2-lambda1)*n/2
    a_m <- YL*log(lambda1) + YU*log(lambda2) + (1:(n-1))*(lambda2-lambda1)
    logsum <- b + log(sum(exp(a_m-b)))
    p <- exp(a_m - logsum)
    m <- sample(1:(n-1), size = 1, prob=p)
    trace[i,] <- c(lambda1, lambda2, m)
  }
  
  
  require(coda)
  bi <- 100
  trace.mcmc <- mcmc(trace, start=bi) 
  summary(trace.mcmc)
  
  
  #MCMC Diagnostics.----------
  x11(10,20); par(mfrow = c(3,1))
  plot(trace[,1], main = 'Trace of lambda1', xlab='Iteration', ylab = expression(lambda[1]), type = 'l')
  plot(trace[,2], main = 'Trace of lambda2', xlab='Iteration', ylab = expression(lambda[2]), type= 'l')
  plot(trace[,3] + 1970 -1, main = 'Trace of Year', xlab='Iteration', ylab = 'Year', type = 'l')
  
  x11(10,20); par(mfrow=c(3,1))
  acf(trace.mcmc[,1], main = 'Autocorrelation lambda1')
  acf(trace.mcmc[,2], main = 'Autocorrelation lambda2')
  acf(trace.mcmc[,3], main = 'Autocorrelation Year')
  
  effectiveSize(trace.mcmc)
  
  
  #Posterior Densities-------
  #lambdas
  #plot(c(7, 20), c(0,.6), type='n', xlab = 'Year', ylab = 'Density', main = 'Density of lambdas')
  plot(density(trace[,1]))
  lines(density(trace[,2]), col='red')
  legend('topright', col = c('black', 'red'), lty = c(1,1), legend = c(expression(lambda[1]), expression(lambda[2])))
  quantile(trace[,1],probs=c(0.025, 0.5, 0.975))
  quantile(trace[,2],probs=c(0.025, 0.5, 0.975))
  
  #Dates
  dates <- factor(trace.mcmc[,3], levels=1:n)
  distrib_dates <- table(dates)/length(dates)
  plot(distrib_dates, ylab = 'Probability', xlab = 'Date_index', main = paste0('G',ei,'-',var_name,':PMF of Change Point'))
  
  #Joint Densities
  pairs(x=trace)
  
  #point estimates---------
  cp <- which(distrib_dates == max(distrib_dates)) + 1
  l1 <- median(trace[,1])
  l2 <- median(trace[,2])
  plot(df1,main = paste0('G',ei,'-',var_name,':Point Estimates'))
  lines(c(1, cp), c(l1, l1))
  lines(c(cp, n), c(l2, l2), col = 'red')
  legend('topleft', col = c('black', 'red'), lty = c(1,1), legend = c(expression(lambda[1]), expression(lambda[2])))
}

for(ei in ei_list){
  load_df <-paste0('Local_Extrema_',ei,'/QQQ_combination.csv')
  ei_data <- read.csv(load_df)
  print('#################################')
  print(paste('ei=', ei))
  print('1_prob:')
  main_model(ei_data, var_name='X1_prob', ei=ei)
  print('#################################')
  print('2_prob:')
  main_model(ei_data, var_name='X2_prob', ei=ei)
  }

