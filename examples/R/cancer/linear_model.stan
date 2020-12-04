// linear regression with multiplicative noise model

data {
    int<lower=0> N;
    int<lower=0> K;
    matrix[N,K] x;
    vector[N] y;
}

parameters {
  real alpha;           // intercept
  vector[K] beta;       // coefficients
  real<lower=0> sigma;    // noise standard deviation
}

transformed parameters {
  vector[N] errors;
  vector[N] predictions;
  
  predictions = (alpha + x * beta);

  errors = y ./ predictions;

}

model {
  //beta ~ normal(0,1);
  //alpha ~ normal(0,1);
  sigma ~ normal(0, 1);
  
  errors ~ normal(1, sigma) ; // this is the additive version for now
  target += -2 * sum(log(predictions));
} 


