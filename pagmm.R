library(stats)
library(expm)
library(caret)
library(bigsnpr)


#################################### Kernels #################################################

K.gauss <- function(x,y,sigma=1){
  return(exp(-sum((x-y)^2) / (2*sigma^2)))
}

################################
##### Sobolev RK function ######
################################
# not optimized

#########################################
#########################################

k1 <- function(t){
  return(t-.5)
}

k2 <- function(t){
  return( (k1(t)^2-1/12)/2 )
}

k4 <- function(t){
  return( (k1(t)^4-k1(t)^2/2+7/240)/24 )
}

#########################################
#########################################

K.sob <- function(s,t){
  ans <- 1 + k1(s)*k1(t) + k2(s)*k2(t) - k4(abs(s-t))
  return(ans)
}

#########################################
#########################################

K.sob.prod <- function(x, y){
  p <- length(x)
  out <- 1
  for (j in (1:p)){
    out <- out*K.sob(x[j], y[j])
  }
  return(out)
}

#########################################
#########################################

standard <- function(x){
  return((x-mean(x))/sd(x))
}

#########################################
#########################################

transform.sob <- function(X){
  Xlim <- apply(X, 2, range)
  Xstd <- matrix(nr=nrow(X), nc=ncol(X))
  for (i in (1:ncol(X))){
    Xstd[,i] <- (X[,i]-Xlim[1,i])/diff(Xlim[,i])
  }
  return(list(Xstd=Xstd, Xlim=Xlim))
}
#########################################
####### Kernel for function input #######
#########################################

K.gauss.l2 <- function(x,y,sigma=1){
  len <- length(x)
  t <- seq(0,1,len=len)
  integral <- trapz(t, (x-y)^2)
  return(exp(-integral/(2*sigma^2)))
}

#########################################
#########################################

K.poly <- function(x,y,c=1,d=1){
  t <- seq(0,1, len=length(x))
  in.prod <- trapz(t, x*y)
  return((c+in.prod)^d)
}

##################################################
#### general function to produce Gram matrix #####
##################################################

getGram <- function(X, A, ker1="gauss", ker2="gauss",
                    sigma1=1, sigma2=1, poly= c(1,1)){
  # note that sobolev kernel is not compatible with standardize=T, and it requires X to be
  # transformed to [0,1]^p
  N <- nrow(X)
  K <- matrix(nr=N, nc=N)
  for (i in (1:N)){
    for (j in (1:N)){
      
      if (ker1=="gauss" & ker2=="gauss"){
        K[i,j] <- K.gauss(X[i,],X[j,], sigma = sigma1)*K.gauss.l2(A[i,],A[j,],sigma=sigma2)
      }else if (ker1=="sob" & ker2=="gauss"){
        K[i,j] <- K.sob.prod(X[i,],X[j,])*K.gauss.l2(A[i,],A[j,],sigma=sigma2)
      }else if (ker1=="gauss" & ker2=="poly"){
        K[i,j] <- K.gauss(X[i,],X[j,], sigma = sigma1)*K.poly(A[i,],A[j,],c=poly[1], d=poly[2])
      }else if (ker1=="sob" & ker2=="poly"){
        K[i,j] <- K.sob.prod(X[i,],X[j,])*K.poly(A[i,],A[j,],c=poly[1], d=poly[2])
      }
    } 
  }
  return(K)
}


get_kernel <- function(X, A.X, Y, A.Y, ker1="gauss", ker2="gauss",
                       sigma1=1, sigma2=1, poly= c(1,1)){
  K <- matrix(NA, nrow=nrow(X),ncol=nrow(Y))
  for (i in 1:nrow(X)){
    for (j in 1:nrow(Y)){
      
      if (ker1=="gauss" & ker2=="gauss"){
        K[i,j] <- K.gauss(X[i,],Y[j,], sigma = sigma1)*K.gauss.l2(A.X[i,],A.Y[j,],sigma=sigma2)
      }else if (ker1=="sob" & ker2=="gauss"){
        K[i,j] <- K.sob.prod(X[i,],Y[j,])*K.gauss.l2(A.X[i,],A.Y[j,],sigma=sigma2)
      }else if (ker1=="gauss" & ker2=="poly"){
        K[i,j] <- K.gauss(X[i,],Y[j,], sigma = sigma1)*K.poly(A.X[i,],A.Y[j,],c=poly[1], d=poly[2])
      }else if (ker1=="sob" & ker2=="poly"){
        K[i,j] <- K.sob.prod(X[i,],Y[j,])*K.poly(A.X[i,],A.Y[j,],c=poly[1], d=poly[2])
      }
    }
  }
  return(K)
}


###########################Calculate the median heuristic ##############################


# function for median heuristic for scaler
get_median_s <- function(X){
  pairwise_dist <- dist(X, diag=F, method= "euclidean")^2
  median <- sqrt(median(pairwise_dist))
  return(median)
}


# functin for median heuristic for functions
get_median_f <- function(A){
  pair_dist_fun = dist_fun(A)
  median <- sqrt(median(pair_dist_fun))
  return(median)
}


####  ####
#t = seq(0,1,len=20)

dist_fun <- function(A){
  t = seq(0,1,len=ncol(A))
  dist <- matrix(NA, nrow=nrow(A),ncol = nrow(A))
  for (i in 1:nrow(A)){
    for (j in 1:i){
      dist[i,j] <- trapz(t, (A[i,]-A[j,])^2)
    }
  }
  # only use the lower diagonal of the matrix
  return(dist[lower.tri(dist, diag = FALSE)])
}


dist_fun_A <- function(A1, A2){
  if(is.null(dim(A1))){A1 <- matrix(A1, nrow=1)}
  if(is.null(dim(A2))){A2 <- matrix(A2, nrow=1)}
  t = seq(0,1,len=ncol(A1))
  dist <- matrix(NA, nrow=nrow(A1),ncol = nrow(A2))
  for (i in 1:nrow(A1)){
    for (j in 1:nrow(A2)){
      dist[i,j] <- trapz(t, (A1[i,]-A2[j,])^2)
    }
  }
  # only use the lower diagonal of the matrix
  return(dist)
}

G.A <- function(A1, A2=NULL, type="gauss", sigma = 1, poly=c(1,1)){
  
  if (is.null(A2)){A2=A1}
  if(type=="gauss"){
    dist_A <- dist_fun_A(A1, A2)
    G.A <- exp(-dist_A/(2*sigma^2))
  }else if(type=="poly"){
    c <- poly[1]; d <- poly[2]
    prod <- in_prod(A1, A2)
    G.A <- (c+prod)^d
  }
  return(G.A)
}

G.cont <- function(cont1, cont2=NULL, type="gauss", sigma=1){
  
  if(is.null(cont2)){
    dist.cont <- as.matrix(dist(cont1, diag=T,upper=T, method= "euclidean")^2)
  }else{
    cont12 <- rbind(cont1, cont2)
    dist.cont.tol <- as.matrix(dist(cont12, diag=T, upper=T, method= "euclidean")^2)
    # dist.cont <- dist_cont(cont1, cont2)
    idx1 <- nrow(cont1)
    idx2 <- idx1 + 1
    dist.cont <- dist.cont.tol[1:idx1,idx2:nrow(cont12)]
  }
  if (type=="gauss"){
    G <- exp(-dist.cont/(2*sigma^2))
  }
  return(G)
}


get_causal_effect <- function(new_a, A.train, Gram.conf, a, ...){
  
  parms <- list(...)
  for (name in names(parms)){
    assign(name, parms[[name]])
  }
  
  # t1 <- Sys.time()
  n.obs <- nrow(Gram.conf)
  # trt_rep <- rbind(new_treat)[rep(1, n.obs), ]
  g.aA <- G.A(A1 = new_a, A2=A.train, type=ker2, sigma = sigma2, poly=poly)
  G.aA <- rbind(g.aA)[rep(1, n.obs), ]
  Kh.a <- Gram.conf * G.aA
  ate_est <- mean(Kh.a %*% a)
  # t2 <- Sys.time()
  # print(t2 - t1)
  # print(paste0("The estimated h(a) is ",ate_est))
  return(ate_est)
}

crossfit <- function(nk=3, n, Gram.f, Gram.h,...){
  
  parms <- list(...)
  for (name in names(parms)){
    assign(name, parms[[name]])
  }
  
  fold_idx <- (1:n) %% nk + 1
  haCross <- matrix(NA, nrow = n, ncol = nk)
  # t1 <- Sys.time()
  for (i in 1:nk){
    
    ind <- (fold_idx != i)
    
    A.train <- treatment[ind,]
    A.val <- treatment[!ind,]
    Y.train <- Y[ind,]
    
    n.train <- nrow(A.train); n.val <- nrow(A.val)
    delta_train = delta_scale/(n.train^delta_exp)
    alpha = alpha_scale* (delta_train**4)
    
    Kf <- Gram.f[ind, ind]
    RootKf <- sqrtm(Kf)
    Kh <- Gram.h[ind, ind]
    M <- RootKf %*% solve(Kf/(2*n.train*delta_train^2) + diag(n.train)/2) %*% RootKf
    a <- ginv(Kh %*% M %*% Kh + alpha * Kh) %*% Kh %*% M %*% Y.train
    # Gram.conf <- Gram.h.cont[!ind, ind] * Gram.h.cate[!ind, ind]
    
    Gram.conf <- G.cont(WX[!ind,], WX[ind,], type=ker1, sigma=sigma1_h)
    A <- as.list(as.data.frame(t(treatment)))
    
    ate_est <- sapply(A, get_causal_effect, A.train=A.train, Gram.conf=Gram.conf, a=a,
                      ker2=ker2, sigma2 = sigma2, poly=poly)
    haCross[,i] <- ate_est
  }
  # t2 <- Sys.time()
  # t2 - t1
  ha_bar <- apply(haCross, 1, mean)
  
  return(ha_bar)
}

no_crossfit <- function(nk=3, n, Gram.f, Gram.h,...){
  
  parms <- list(...)
  for (name in names(parms)){
    assign(name, parms[[name]])
  }
  
  fold_idx <- (1:n) %% nk + 1
  # haCross <- matrix(NA, nrow = n, ncol = nk)
  haNoCross <- matrix(NA, nrow = n, ncol = 1)
  t1 <- Sys.time()
  
  # only calculate the FTE once
  # for (i in 1:nk){
  i <- 1
  ind <- (fold_idx != i)
  
  A.train <- treatment[ind,]
  A.val <- treatment[!ind,]
  Y.train <- Y[ind,]
  
  n.train <- nrow(A.train); n.val <- nrow(A.val)
  delta_train = delta_scale/(n.train^delta_exp)
  alpha = alpha_scale* (delta_train**4)
  
  Kf <- Gram.f[ind, ind]
  RootKf <- sqrtm(Kf)
  Kh <- Gram.h[ind, ind]
  M <- RootKf %*% solve(Kf/(2*n.train*delta_train^2) + diag(n.train)/2) %*% RootKf
  a <- ginv(Kh %*% M %*% Kh + alpha * Kh) %*% Kh %*% M %*% Y.train
  # Gram.conf <- Gram.h.cont[!ind, ind] * Gram.h.cate[!ind, ind]
  
  Gram.conf <- G.cont(WX[!ind,], WX[ind,], type=ker1, sigma=sigma1_h)
  A <- as.list(as.data.frame(t(treatment)))
  
  ate_est <- sapply(A, get_causal_effect, A.train=A.train, Gram.conf=Gram.conf, a=a,
                    ker2=ker2, sigma2 = sigma2, poly=poly)
  haNoCross[,i] <- ate_est
  # }
  t2 <- Sys.time()
  t2 - t1
  # ha_bar <- apply(haCross, 1, mean)
  
  return(haNoCross)
}


################################### RKHS ######################################

RKHSCV <- function(data, ker1="gauss",ker2="gauss", CF=T, n_alphas=30, cv=5, 
                   delta_scale=5, delta_exp=.4, A.standard = T){
  
  start.time <- Sys.time()
  
  # data = generation(seed,n.sample = n.sample,setting = sim.setting)
  sim.data = data$sim.data
  alpha_scales = seq_log(exp(-2), exp(6), n_alphas) 
  
  # n.total = nrow(sim.data)
  # n.train= n.total*(2/3); n.test= n.total*(1/3)
  n.train= nrow(sim.data)*(3/4); n.test= nrow(sim.data)*(1/4)
  
  train.data = sim.data[1:n.train,]
  h0.train = data$h0[1:n.train,]
  treatment.train = data$treatment[1:n.train,]
  
  test.data = sim.data[-(1:n.train),]
  h0.test = data$h0[-(1:n.train),]
  treatment.test = data$treatment[-(1:n.train),]
  
  WX.test <- test.data[,c(2,5,6)]
  
  best_score = c()
  best_alpha_scale = c()
  
  # Set parameters in kernel: if A.standard is T, then set mean.norm of A as sigma2
  if (A.standard) sigma2 <- get_median_f(treatment.train)
  
  fold_idx <- (1:n.train) %% cv + 1
  n = n.train*(cv-1)/cv
  n.out = n.train/cv
  delta_train = delta_scale/(n^delta_exp)
  delta_test = delta_scale/(n.out^delta_exp)
  
  scores = matrix(NA, nrow=n_alphas,ncol=cv)
  
  #standardize the data
  if (ker1=="sob"){
    transform1 = transform.sob(train.data[,c(2,5,6)])
    WX.all = transform1$Xstd
    lim = transform1$Xlim
    transform2 = transform.sob(train.data[,c(4,5,6)])
    ZX.all = transform2$Xstd
    
    # standardize WX.test
    WXstd <- matrix(nr=nrow(WX.test), nc=ncol(WX.test))
    for (l in (1:ncol(WX.test))){WXstd[,l] <- (WX.test[,l]-lim[1,l])/diff(lim[,l])}
    WX.test <- WXstd
    
  }else if(ker1=="gauss"){
    
    preprocessParams = preProcess(train.data[,c(2,5,6)], method=c("center", "scale"))
    WX.all = predict(preprocessParams, train.data[,c(2,5,6)])
    prepParams = preProcess(train.data[,c(4,5,6)], method=c("center", "scale"))
    ZX.all = predict(prepParams, train.data[,c(4,5,6)])
    
    # standardize WX.test
    WX.test <- predict(preprocessParams, WX.test)
    
    sigma1_h = get_median_s(WX.all)
    sigma1_f = get_median_s(ZX.all)
  }
  
  for (i in 1:cv){
    ind <- (fold_idx != i)
    # training set
    train <- train.data[ind,]
    treatment <- treatment.train[ind,]
    h0 <- h0.train[ind]        
    WX <- WX.all[ind,] 
    ZX <- ZX.all[ind,]
    Y <- train[,1]
    # validation set
    test <- train.data[!ind,]
    treatment.out <- treatment.train[!ind,]
    h0.out <- h0.train[!ind]
    WX.out <- WX.all[!ind,]
    ZX.out <- ZX.all[!ind,]
    Y.out <- test[,1]
    
    Kf <- getGram(ZX, A=treatment, ker1 = ker1, ker2=ker2, sigma1 = sigma1_f , sigma2=sigma2, poly=poly)
    RootKf <- sqrtm(Kf)
    Kh <- getGram(WX, A=treatment, ker1 = ker1, ker2=ker2, sigma1 = sigma1_h , sigma2=sigma2, poly=poly)
    M <- RootKf %*% solve(Kf/(2*n*delta_train^2) + diag(n)/2) %*% RootKf
    
    Kf.test <- getGram(ZX.out, A=treatment.out, ker1 = ker1, ker2=ker2, sigma1 = sigma1_f , sigma2=sigma2, poly=poly)
    RootKf.test <- sqrtm(Kf.test)
    M.test <- RootKf.test %*% solve(Kf.test/(2*n.out*delta_test^2) + diag(n.out)/2) %*% RootKf.test
    
    for (j in 1:n_alphas){
      #j = 1
      alpha = alpha_scales[j] * (delta_train**4)
      a <- ginv(Kh %*% M %*% Kh + alpha * Kh) %*% Kh %*% M %*% Y 
      
      # Calculate out-sample MSE
      K.out <- get_kernel(X=WX.out,A.X=treatment.out,Y=WX,A.Y=treatment,ker1=ker1,ker2=ker2,sigma1=sigma1_h,sigma2=sigma2,poly=poly)
      h0.hat.out <- K.out %*% a
      res = Y.out - h0.hat.out
      scores[j,i] <- t(res) %*% M.test %*% res/nrow(res)
    }
    
  }
  
  avr_scores = apply(scores, 1, mean)
  best_ind = which.min(avr_scores)
  best_score = min(avr_scores)
  best_alpha_scale = alpha_scales[best_ind]
      
  
  delta = delta_scale/(n.train^delta_exp)
  alpha = best_alpha_scale* (delta**4)
  
  Kh <- getGram(WX.all, A=treatment.train, ker1 = ker1, ker2=ker2, sigma1 = sigma1_h , sigma2=sigma2, poly=poly)
  Kf <- getGram(ZX.all, A=treatment.train, ker1 = ker1, ker2=ker2, sigma1 = sigma1_f , sigma2=sigma2, poly=poly)
  RootKf <- sqrtm(Kf)
  M <- RootKf %*% solve(Kf/(2*n.train*delta^2) + diag(n.train)/2) %*% RootKf
  
  a <- ginv(Kh %*% M %*% Kh + alpha * Kh) %*% Kh %*% M %*% train.data[,1]
  
  
  K.test <- get_kernel(X=WX.test,A.X=treatment.test,Y=WX.all,A.Y=treatment.train,ker1=ker1,ker2=ker2,sigma1=sigma1_h,sigma2=sigma2,poly=poly)
  h0.hat.test <- K.test %*% a
  mse.test <- sum((h0.test-h0.hat.test)**2/n.test) 
  
  time.h <- Sys.time()
  
  Y.train <- matrix(train.data[,1], ncol=1)
  
  if(CF){
    ha.hat.train <- crossfit(nk=3, n=nrow(train.data), Gram.f=Kf, Gram.h=Kh, 
                             treatment=treatment.train, Y=Y.train, delta_scale=delta_scale, delta_exp=delta_exp,
                             ker1=ker1, ker2=ker2, sigma1_h=sigma1_h, sigma2=sigma2, poly=poly, alpha_scale=best_alpha_scale, WX=WX.all)
  }else{
    ha.hat.train <- no_crossfit(nk=3, n=nrow(train.data), Gram.f=Kf, Gram.h=Kh, 
                                treatment=treatment.train, Y=Y.train, delta_scale=delta_scale, delta_exp=delta_exp,
                                ker1=ker1, ker2=ker2, sigma1_h=sigma1_h, sigma2=sigma2, poly=poly, alpha_scale=best_alpha_scale, WX=WX.all)
  }
   
  Gram.test <- G.cont(WX.test, WX.all, type=ker1, sigma=sigma1_h)
  A.test <- as.list(as.data.frame(t(treatment.test)))
  
  ha.hat.test <- sapply(A.test, get_causal_effect, A.train=treatment.train, Gram.conf=Gram.test, a=a,
                     ker2=ker2, sigma2 = sigma2, poly=poly)
 
  
  mse.ha.in <- mean((data$ha.train - ha.hat.train)^2)
  # remove
  mse.ha.out <- mean((data$ha.test - ha.hat.test)^2)
  
  time.ha <- Sys.time()
  
  time.taken1 <- time.h - start.time
  time.taken2 <- time.ha - start.time
  
  time <- c(time.taken1, time.taken2)
  
  return(list(mse.test=mse.test, mse.ha.in=mse.ha.in, mse.ha.out=mse.ha.out,time = time))
}

