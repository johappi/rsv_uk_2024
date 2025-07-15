functions {
	vector ordered_pred(vector alpha, real gamma){
	int n = num_elements(alpha);
	vector[n+1] beta;
	vector[n] z = reverse( cumulative_sum(rep_vector(1,n)) );
	beta[1] = - sum(z .* alpha) / (n+1);
	for (i in 2:n+1){
		beta[i] = beta[i-1] + alpha[i-1];
	}
	return gamma * beta;
	}
}


data {
  int<lower=0> N;   // number of individuals
  int<lower=0> N_pred;    // number of "individuals" (cells) to make predictions for post stratification
  int<lower=0> N_dep; // number of categories of dependent variable
  
  int N_age; // age 6 categories
  int N_rel; // religion
  int N_eth; // ethnicity
  int N_lan; // language
  int N_emp; // employment status
  int N_edu; // eudcation level

  int N_dis; // disability
  int N_hea; // health in general
  
  int N_reg;
  
  array[N] int age;
  array[N] int rel;
  array[N] int eth;
  array[N] int lan;
  array[N] int emp;
  array[N] int edu;
  array[N] int sex;
  array[N] int reg;

  array[N] int dis;
  array[N] int hea;
  array[N] int buk;
  array[N] int mig;
  
  array[N_pred] int age_pred;
  array[N_pred] int rel_pred;
  array[N_pred] int eth_pred;
  array[N_pred] int lan_pred;
  array[N_pred] int emp_pred;
  array[N_pred] int edu_pred;
  array[N_pred] int sex_pred;
  array[N_pred] int reg_pred;

  array[N_pred] int dis_pred;
  array[N_pred] int hea_pred;
  array[N_pred] int buk_pred;
  array[N_pred] int mig_pred;
  
  array[N] int y_ordinal;
}

parameters {
  real<lower=0.01> sigma_age;
  real<lower=0.01> sigma_edu;
  real<lower=0.01> sigma_eth;
  real<lower=0.01> sigma_lan;
  real<lower=0.01> sigma_rel;
  real<lower=0.01> sigma_emp;
  real<lower=0.01> sigma_reg;

  real<lower=0.01> sigma_dis;
  real<lower=0.01> sigma_hea;
  
vector[N_age - 1] beta_age_raw;
vector[N_edu - 1] beta_edu_raw;
vector[N_eth - 1] beta_eth_raw;
vector[N_lan - 1] beta_lan_raw;
vector[N_rel - 1] beta_rel_raw;
vector[N_emp - 1] beta_emp_raw;
vector[N_reg - 1] beta_reg_raw;

vector[N_dis - 1] beta_dis_raw;
vector[N_hea - 1] beta_hea_raw;

real beta_sex;

real beta_buk;
real beta_mig;

simplex[N_dep - 2] c_tilde;
real<lower=0> c_scale;
real c_loc;


}

transformed parameters {
ordered[N_dep-1] c = c_loc + ordered_pred( c_tilde, c_scale );

vector[N_age] beta_age_raw2 = append_row(beta_age_raw, -sum(beta_age_raw));
vector[N_edu] beta_edu_raw2 = append_row(beta_edu_raw, -sum(beta_edu_raw));
vector[N_eth] beta_eth_raw2 = append_row(beta_eth_raw, -sum(beta_eth_raw));
vector[N_lan] beta_lan_raw2 = append_row(beta_lan_raw, -sum(beta_lan_raw));
vector[N_rel] beta_rel_raw2 = append_row(beta_rel_raw, -sum(beta_rel_raw));
vector[N_emp] beta_emp_raw2 = append_row(beta_emp_raw, -sum(beta_emp_raw));
vector[N_reg] beta_reg_raw2 = append_row(beta_reg_raw, -sum(beta_reg_raw));

vector[N_dis] beta_dis_raw2 = append_row(beta_dis_raw, -sum(beta_dis_raw));
vector[N_hea] beta_hea_raw2 = append_row(beta_hea_raw, -sum(beta_hea_raw));

vector[N_age] beta_age = sigma_age * beta_age_raw2;
vector[N_edu] beta_edu = sigma_edu * beta_edu_raw2;
vector[N_eth] beta_eth = sigma_eth * beta_eth_raw2;
vector[N_lan] beta_lan = sigma_lan * beta_lan_raw2;
vector[N_rel] beta_rel = sigma_rel * beta_rel_raw2;
vector[N_emp] beta_emp = sigma_emp * beta_emp_raw2;
vector[N_reg] beta_reg = sigma_reg * beta_reg_raw2;

vector[N_dis] beta_dis = sigma_dis * beta_dis_raw2;
vector[N_hea] beta_hea = sigma_hea * beta_hea_raw2;

}

model {

  vector[N] lin_pred = beta_age[age] + [beta_sex, -beta_sex][sex]' + beta_edu[edu] + beta_eth[eth] + beta_lan[lan] + beta_rel[rel] + beta_emp[emp] + beta_reg[reg]
                       + beta_dis[dis] + beta_hea[hea]  // + beta_enp[enp]
                       + [beta_buk,-beta_buk][buk]' + [beta_mig,-beta_mig][mig]';

  for (n in 1:N){
  	y_ordinal[n] ~ ordered_logistic(lin_pred[n], c);//
  }

c_tilde ~ dirichlet( rep_vector( 1, N_dep - 2 ) );
c_scale ~ normal(0,5);
c_loc ~ normal(0,7);


beta_age_raw2 ~ normal(0, 1);
beta_edu_raw2 ~ normal(0, 1);
beta_eth_raw2 ~ normal(0, 1);
beta_lan_raw2 ~ normal(0, 1);
beta_rel_raw2 ~ normal(0, 1);
beta_emp_raw2 ~ normal(0, 1);
beta_reg_raw2 ~ normal(0, 1);

beta_dis_raw2 ~ normal(0, 1);
beta_hea_raw2 ~ normal(0, 1);
  
  beta_sex ~ normal(0,2);
  beta_buk ~ normal(0,2);
  beta_mig ~ normal(0,2);
	
  { sigma_age, sigma_edu, sigma_eth, sigma_lan, sigma_rel, sigma_emp, sigma_reg, sigma_dis, sigma_hea } ~ normal(0, 1); // sigma_enp,
}

generated quantities {
array[N_pred, N_dep] real y_log_prob;
array[N_pred] int y_pred;

{
                       
vector[N_pred] lin_pred = beta_age[age_pred] + [beta_sex, -beta_sex][sex_pred]' + beta_edu[edu_pred] + beta_eth[eth_pred] + beta_lan[lan_pred] + beta_rel[rel_pred] + beta_emp[emp_pred] + beta_reg[reg_pred]
                       + beta_dis[dis_pred] + beta_hea[hea_pred] // + beta_enp[enp_pred]
                       + [beta_buk,-beta_buk][buk_pred]' + [beta_mig,-beta_mig][mig_pred]';

for (n in 1:N_pred){
	y_pred[n] = ordered_logistic_rng(lin_pred[n], c);
	for (k in 1:N_dep) {
		y_log_prob[n,k] = ordered_logistic_lpmf(k | lin_pred[n], c);
	}
}
}
}
