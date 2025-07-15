data {
  int<lower=0> N;   // number of individuals
  // int<lower=0> N_pred;    // number of "individuals" (cells) to make predictions for post stratification
  int<lower=0> N_dep; // number of categories of dependent variable
  
  
  int N_reg;
 
  int N_treat;
  
  array[N] int y_pre;
  // array[N_treat, N] int y_post;
  array[N] int y_post;
  
  array[N] int treat;
  
}

parameters {

vector[N_dep - 1] mu_theta_treat; // mean for dirichlets for treatment effects
real mu_beta_treat; // mean for coefficient for treatment effects
real <lower = 0> sigma_beta_treat; // standard deviation for coefficients for treatment effects

//////////////////
// pre treatment //
//////////////////

// real alpha_pre;


ordered[N_dep-1] c_pre;

///////////////////////
// treatment effects //
///////////////////////

array[N_treat] simplex[N_dep - 1] theta_treat;
array[N_treat] real beta_treat;

// array[N_treat] real alpha;
  
array[N_treat] ordered[N_dep-1] c; // cutpoints for ordered logistic // !! should do hierarchical as well

}

model {

// for (n in 1:N){
//	y_pre[n] ~ ordered_logistic(alpha_pre, c_pre);
//	y_post[n] ~ ordered_logistic( beta_treat[treat[n]] * sum( theta_treat[treat[n]][ : y_pre[n] - 1] ) + alpha[treat[n]], c[treat[n]]);
//	}
	
 for (n in 1:N){
	y_pre[n] ~ ordered_logistic(0, c_pre);
	y_post[n] ~ ordered_logistic( beta_treat[treat[n]] * sum( theta_treat[treat[n]][ : y_pre[n] - 1] ), c[treat[n]]);
	}


// alpha_pre ~ normal(0, 2);
  
mu_theta_treat ~ gamma(1,1);
mu_beta_treat ~ normal(0,3);
sigma_beta_treat ~ gamma(1,1);

for (t in 1:N_treat){
  theta_treat[t] ~ dirichlet(mu_theta_treat);
  beta_treat[t] ~ normal(mu_beta_treat, sigma_beta_treat);
}

// for (t in 1:N_treat){
//  alpha[t] ~ normal(0, 2);
//}

}







