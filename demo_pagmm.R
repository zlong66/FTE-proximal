source("data_generation_pagmm.R")
source("pagmm.R")

### Setup ###

kernel.type <- "gauss"
sim.setting <- "linear_1" # other settings are "nonlinear_1", "nonlinear_2"
n.sample <- 400 # 300 training samples + 100 independent test samples

# generate simulated datasets
seed <- 100
data = generation(seed,n.sample = n.sample,setting = sim.setting)

# PAGMM with cross-fitting
RKHS <- RKHSCV(data,ker1=kernel.type)
mse <- RKHS$mse.ha.in # mse of estimated FTE of 300 training samples
mse.out <- RKHS$mse.ha.out # mse of estimated FTE of 100 test samples
runtime <- RKHS$time # run time of 

# PAGMM without cross-fitting
RKHS_noCF <- RKHSCV(data,ker1=kernel.type, CF=F)
mse_noCF <- RKHS_noCF$mse.ha.in
