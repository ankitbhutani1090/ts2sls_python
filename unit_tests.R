rm(list=ls())
library(reticulate)
ivregress <- import("ivregress")


# Prepare demo data
df1 <- read.csv("auto_data.csv")
df1$weight2 <- with(df1, weight*weight)
df2 <- data.frame(df1)
df1$mpg <- list(NULL)

# ivregress - single instrument
S <- data.frame(df2)

y_var <- 'price'
regs <- list('weight', 'mpg')
ev <- list('mpg')
inst <- list('headroom')
result <- ivregress$ivregress_2sls(S, y_var, regs, ev, inst)
print(result)


#ts2sls - signle instrument
S1 <- df1 # sample 1 - this doesn't have the endogenous variables
S2 <- df2 # sample 2 - this doesn't have the target variable

y_var <- 'price'
regs <- list('weight', 'mpg')
ev <- list('mpg')
inst <- list('headroom')
result <- ivregress$ts2sls(S1, S2, y_var, regs, ev, inst)
print(result)


# ivregress - multiple instruments
S <- df2

y_var <- 'price'
regs <- list('headroom', 'mpg')
ev <- list('mpg')
inst <- list('weight', 'weight2')
result <- ivregress$ivregress_2sls(S, y_var, regs, ev, inst)
print(result)



# ts2sls - Multiple instruments
S1 <- df1 # sample 1 - this doesn't have the endogenous variables
S2 <- df2 # sample 2 - this doesn't have the target variable

y_var <- 'price'
regs <- list('headroom', 'mpg')
ev <- list('mpg')
inst <- list('weight', 'weight2')
result <- ivregress$ts2sls(S1, S2, y_var, regs, ev, inst)
print(result)




