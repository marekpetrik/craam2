// linear regression with multiplicative noise model

data {
    int<lower=0> N;   // number of data points
    int<lower=0> K;   // number of features
    matrix[N,K] X;
    vector[N] y;
} 

transformed data{
}

parameters {
  real<lower=0> alpha;
  vector<lower=0, upper=1>[K] w;
}

transformed parameters {
    vector[N] noise;
    vector<lower=0.1, upper=3.0>[N] z; // the lower 0.1 to prevent a division by 0
    
    z = X * w;
    for (i in 1:N){
        noise[i] = y[i] / z[i];    
    }
}

model {
    // uses a shape and rate definition
    // alpha = beta because we are assuming that the mean is 1
    // and essentially we are fitting the standard deviation only
    noise ~ gamma(alpha, alpha); 
  
    // this is necessary to handle the nonlinear pdf transformation that
    // arises from multiplicative noise
    for(i in 1:N){
        target += -2 * log(z[i]);   
    }
} 


