// linear regression with multiplicative noise model

data {
    int<lower=0> N;
    int<lower=0> K;
    matrix[N,K] x;
    vector[N] y;
}

parameters {
  real alpha;             // intercept
  vector[K] beta;         // coefficients
  real<lower=0> sigma;    // noise standard deviation
  
  real<lower=0> sigma2;
}


transformed parameters {
  vector[N] noise;
  vector[N] true_y;
  
  true_y = alpha + x * beta;
  noise = y ./ true_y;
  
  noise = y ./ true_y;
}

model {
  //beta ~ normal(0,1);
  //alpha ~ normal(0,1);
  //sigma ~ normal(0, 1);
  
  noise ~ gamma(1, sigma); 
  
  // this is necessary to handle the nonlinear pdf transformation that
  // arises from multiplicative noise
  target += -2 * sum(log(true_y));
} 


