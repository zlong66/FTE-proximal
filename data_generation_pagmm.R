

library(pracma)
library(MASS)




FPCA = function(X, pve = 0.95){
  #demeanX=as.matrix(X-matrix(apply(X,2,mean),nrow=nrow(X), ncol=ncol(X), byrow=TRUE))
  #eigenX=eigen(cov(demeanX))
  eigenX=eigen(cov(X))
  L=which.max(cumsum(eigenX$values)/sum(eigenX$values)>pve) # truncate at FVE=95%
  eigenfunX=eigenX$vectors[,1:L] 
  eigenvalX=eigenX$values[1:L]/ncol(X) 
  eigenfunX.rescale=eigenfunX*sqrt(ncol(X)) # make the norm of the eigenfunction = 1
  #scr = demeanX%*%eigenfunX.rescale/ncol(X)
  scr = X%*%eigenfunX.rescale/ncol(X)
  R = list()
  R$perc = eigenX$values/sum(eigenX$values)
  R$efn = eigenfunX.rescale
  R$eval = eigenvalX
  R$scr = scr
  return(R)
}

#################################### Data Generation #################################################



mu <- c(0, 0)
sigma <- matrix(c(1, 0.6, 0.6, 1), 2, 2)

# parameters for Z,W,X
alpha0 <- mu0 <- k0 <- 0.25
alpha.x <- mu.x <- k.x <- c(0.25,0.25)
cov <- matrix(c(1,.25,.5,.25,1,.5,.5,.5,1),3, 3)

# parameters for Y
b0 = 2; b.x <-c(0.25,0.25); b.w <- 4; omega <- 2; b.wx <- 2


generation <- function(seed, n.sample=400, setting="linear_1"){
  # for each treatment, generate 100 replications
  
  set.seed(seed)
  sim.data <- matrix(NA, nrow=n.sample, ncol = 5)
  treatment <- matrix(NA, nrow = n.sample, ncol=20)
  mu.w.ux <- matrix(NA, nrow=n.sample)
  intgr_a <- matrix(NA, nrow=n.sample)
  #mean.y <- matrix(NA, nrow = n.sample)
  Y <- matrix(NA, nrow = n.sample)
  h0 <- matrix(NA, nrow = n.sample)
  
  n.train <- n.sample*(3/4); n.test = n.sample*(1/4)
  h.train <- matrix(NA, nrow = n.train, ncol = n.train)
  h.test <- matrix(NA, nrow = n.test, ncol = n.test)
  
  # generate x
  X <- mvrnorm(n=n.sample, mu, sigma)
  noi <- rnorm(n=n.sample, 0, 1)
  noi.y <- rnorm(n=n.sample, 0, sd=2)
  
  t <- seq(0,1, len=20)
  
  for (i in 1:n.sample){
    x <- X[i,]
    x1 <- x[1]
    x2 <- x[2]
    
    # generate treatment
    beta1 <- sin(2*pi*t)
    beta2 <- cos(2*pi*t)
    e <- cos(4*pi*t)
    noise <- noi[i]
    A <- x1*beta1 + x2*beta2 + e*noise
    
    # generate z,w,u
    s <- seq(0,1, len = 1000)
    f0 <- exp(s)*(x1*sin(2*pi*s)+x2*cos(2*pi*s)+noise*cos(4*pi*s))
    f1 <- (-1/2*exp(s)+cos(6*pi*s)+s^2/2)*(x1*sin(2*pi*s)+x2*cos(2*pi*s)+noise*cos(4*pi*s))
    f2 <- (-exp(s)+s^2)*(x1*sin(2*pi*s)+x2*cos(2*pi*s)+noise*cos(4*pi*s))
    
    intgr0 <- trapz(s, f0)
    intgr1 <- trapz(s, f1)
    intgr2 <- trapz(s, f2)
    
    mean.z <- alpha0 + intgr0 + t(alpha.x)%*%x
    mean.w <- mu0 + intgr1 + t(mu.x)%*%x
    mean.u <- k0 + intgr2 + t(k.x)%*%x
    
    zwu <- mvrnorm(n=1, c(mean.z, mean.w, mean.u), cov)
    # generate y
    mu.w.ux[i,] <- mu0 + t(mu.x)%*%x + cov[2,3]/cov[3,3]*(zwu[3]-k0-t(k.x)%*%x)
    
    f3 <- sin(2*pi*s)*(x1*sin(2*pi*s) + x2*cos(2*pi*s) + noise*cos(4*pi*s))
    intgr_a[i,] <- trapz(s, f3)
    
    sim.data[i,] <- c(zwu[2],zwu[3],zwu[1],x)
    
    treatment[i,] <- A
  }
  colnames(sim.data) <- c("W","U","Z","X1","X2")

  ## generate h(A,W,X) and Y
  # h is linear w.r.t A,W,X
  if(setting=="linear_1"){
    
    h <- b0+intgr_a + sim.data[,c("X1","X2")]%*%b.x + b.w*sim.data[,"W"]
    mu.y <- b0+intgr_a + sim.data[,c("X1","X2")]%*%b.x + (b.w-omega)*mu.w.ux + omega*sim.data[,"W"]
    
  }else if (setting=="linear_2"){
    
    omega <- b.w
    h <- b0+intgr_a + sim.data[,c("X1","X2")]%*%b.x + b.w*sim.data[,"W"]
    mu.y <- b0+intgr_a + sim.data[,c("X1","X2")]%*%b.x + (b.w-omega)*mu.w.ux + omega*sim.data[,"W"]
    
  }else if (setting=="nonlinear_1"){
    
    h <- b0+intgr_a + sim.data[,c("X1","X2")]%*%b.x + b.w*sim.data[,"W"] + b.wx*sim.data[,"X1"]*sim.data[,"W"]
    mu.y <- b0+intgr_a + sim.data[,c("X1","X2")]%*%b.x + (b.w+b.wx*sim.data[,"X1"]-omega)*mu.w.ux + omega*sim.data[,"W"]
    
  }else if (setting=="nonlinear_2"){
 
    ## find the FPCA of treatments and get the FPC scores of treatments
    Atrain <- treatment[1:n.train,]
    Atest <- treatment[-(1:n.train),]

    fpcaA <- FPCA(Atrain, pve = 0.99)
    a1a2.train <- fpcaA$scr[,c(1,2)]
    a1.test = apply(Atest, 1, function(x){trapz(t, x*fpcaA$efn[,1])})
    a2.test = apply(Atest, 1, function(x){trapz(t, x*fpcaA$efn[,2])})
    
    a1a2 <- rbind(a1a2.train,cbind(a1.test, a2.test))

    ## obtain h and mu.y
    b1 <- 2; b2 <- 2
    h <- b0 + b1*(a1a2[,1])^2 + b2*sin(a1a2[,2]) + sim.data[,c("X1","X2")]%*%b.x + b.w*sim.data[,"W"]
    mu.y <- b0 + b1*(a1a2[,1])^2 + b2*sin(a1a2[,2]) + sim.data[,c("X1","X2")]%*%b.x + (b.w-omega)*mu.w.ux + omega*sim.data[,"W"]
    
  }
  
  Y <- mu.y + noi.y
  h0 <- h
  sim.data <- cbind(Y, sim.data)
  colnames(sim.data) <- c("Y","W","U","Z","X1","X2")
  #mean.y[i,] <- mu.y
  #Y[i,] <- y
  
  
  for (i in 1:n.train) {
    t <- seq(0, 1, len = 20)
    intgr <- trapz(t, sin(2*pi*t)*treatment[i,])
    for (j in 1:n.train) {
      x.j <- sim.data[j,c(5,6)]; w.j <- sim.data[j,2]
      if (setting=="linear_1" | setting=="linear_2"){
        h.train[i,j] <- b0+intgr + t(b.x)%*%x.j + b.w*w.j
        # h.train[i,j] <- b0+intgr_a[i] + t(b.x)%*%x.j + b.w*w.j
      }else if (setting=="nonlinear_1"){
        h.train[i,j] <- b0+intgr + t(b.x)%*%x.j + b.w*w.j + b.wx*x.j[1]*w.j
        # h.train[i,j] <- b0+intgr_a[i] + t(b.x)%*%x.j + b.w*w.j + b.wx*x.j[1]*w.j
      }else if (setting=="nonlinear_2"){
        # mu.w.ux <- mean.w.ux[j]
        # h.train[i,j] <- b0+intgr + intgr^2+ t(b.x)%*%x.j + b.w*w.j
        # h.train[i,j] <- b0+b1*a1[i] + t(b.x)%*%x.j + b.w*w.j
        h.train[i,j] <- b0+b1*(a1a2[i,1])^2 + b2*sin(a1a2[i,2]) + t(b.x)%*%x.j + b.w*w.j
      }
    }
  }
  
  for (i in (n.train+1):n.sample) {
    t <- seq(0, 1, len = 20)
    intgr <- trapz(t, sin(2*pi*t)*treatment[i,])
    for (j in (n.train+1):n.sample) {

      x.j <- sim.data[j,c(5,6)]; w.j <- sim.data[j,2]
      if (setting=="linear_1" | setting=="linear_2"){
        h.test[i-n.train,j-n.train] <- b0+intgr + t(b.x)%*%x.j + b.w*w.j
        # h.test[i-n.train,j-n.train] <- b0+intgr_a[i] + t(b.x)%*%x.j + b.w*w.j
      }else if (setting=="nonlinear_1"){
        h.test[i-n.train,j-n.train] <- b0+intgr + t(b.x)%*%x.j + b.w*w.j + b.wx*x.j[1]*w.j
        # h.test[i-n.train,j-n.train] <- b0+intgr_a[i] + t(b.x)%*%x.j + b.w*w.j + b.wx*x.j[1]*w.j
      }else if (setting=="nonlinear_2"){
        #mu.w.ux <- mean.w.ux[j]
        # h.test[i-n.train,j-n.train] <- b0+intgr + intgr^2 + t(b.x)%*%x.j + b.w*w.j
        # h.test[i-n.train,j-n.train] <- b0+ b1*a1[i] + t(b.x)%*%x.j + b.w*w.j
        h.test[i-n.train,j-n.train] <- b0 + b1*(a1a2[i,1])^2 + b2*sin(a1a2[i,2]) + t(b.x)%*%x.j + b.w*w.j
      }
    }
  }
  
  ha.train <- apply(h.train, 1, mean)
  ha.test <- apply(h.test, 1, mean)
  
  return(list(sim.data=sim.data, treatment=treatment, mean.y=mu.y, h0=h0, ha.train=ha.train, ha.test=ha.test))
}





