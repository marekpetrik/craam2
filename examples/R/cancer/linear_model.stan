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
  vector[N] true_y;
  real<lower=0> sigma2;
}


transformed parameters {
  vector[N] noise;

  noise = y ./ true_y;
}

model {
  //beta ~ normal(0,1);
  //alpha ~ normal(0,1);
  //sigma ~ normal(0, 1);
  
  true_y ~ normal(alpha + x * beta, sigma2);

  noise ~ normal(1, sigma) ; 
  
  target += -2 * sum(log(true_y));
} 


