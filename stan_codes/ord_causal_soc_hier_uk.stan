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
    
  int N_treat;
  
  array[N] int y_pre;
  array[N] int y_post;
  
  array[N] int treat;
  
}

parameters {
//////////////////
// pre treatment //
//////////////////

real<lower=0.01> sigma_age_pre;
real<lower=0.01> sigma_edu_pre;
real<lower=0.01> sigma_eth_pre;
real<lower=0.01> sigma_lan_pre;
real<lower=0.01> sigma_rel_pre;
real<lower=0.01> sigma_emp_pre;
real<lower=0.01> sigma_reg_pre;

real<lower=0.01> sigma_dis_pre;
real<lower=0.01> sigma_hea_pre;
  
vector[N_age - 1] beta_age_raw_pre;
vector[N_edu - 1] beta_edu_raw_pre;
vector[N_eth - 1] beta_eth_raw_pre;
vector[N_lan - 1] beta_lan_raw_pre;
vector[N_rel - 1] beta_rel_raw_pre;
vector[N_emp - 1] beta_emp_raw_pre;
vector[N_reg - 1] beta_reg_raw_pre;

vector[N_dis - 1] beta_dis_raw_pre;
vector[N_hea - 1] beta_hea_raw_pre;

real beta_sex_pre;

real beta_buk_pre;
real beta_mig_pre;

simplex[N_dep - 2] c_tilde_pre;
real<lower=0> c_scale_pre;
real c_loc_pre;

///////////////////////
// treatment effects //
///////////////////////

array[N_treat] real<lower=0.01> sigma_age;
array[N_treat] real<lower=0.01> sigma_edu;
array[N_treat] real<lower=0.01> sigma_eth;
array[N_treat] real<lower=0.01> sigma_lan;
array[N_treat] real<lower=0.01> sigma_rel;
array[N_treat] real<lower=0.01> sigma_emp;

array[N_treat] real<lower=0.01> sigma_dis;
array[N_treat] real<lower=0.01> sigma_hea;
  
array[N_treat] vector[N_age - 1] beta_age_raw;
array[N_treat] vector[N_edu - 1] beta_edu_raw;
array[N_treat] vector[N_eth - 1] beta_eth_raw;
array[N_treat] vector[N_lan - 1] beta_lan_raw;
array[N_treat] vector[N_rel - 1] beta_rel_raw;
array[N_treat] vector[N_emp - 1] beta_emp_raw;

array[N_treat] vector[N_dis - 1] beta_dis_raw;
array[N_treat] vector[N_hea - 1] beta_hea_raw;

array[N_treat] real beta_sex;
array[N_treat] real beta_buk;
array[N_treat] real beta_mig;

array[N_treat, N_dep] simplex[N_dep - 2] c_tilde;
array[N_treat, N_dep] real<lower=0> c_scale_raw;
array[N_treat, N_dep] real c_loc_raw;


array[N_dep] simplex[N_dep-2] c_tilde_prior;
array[N_dep] real<lower=0> c_tilde_prior_scale;
array[N_dep] real<lower=0> c_scale_mu;
array[N_dep] real<lower=0> c_scale_sigma;

array[N_dep] real c_loc_mu;
array[N_dep] real<lower=0> c_loc_sigma;
}

transformed parameters {
array[N_treat, N_dep] real c_scale;
array[N_treat, N_dep] real c_loc;

for (t in 1:N_treat){
	for (n in 1:N_dep){
		c_scale[t,n] = c_scale_mu[n] + c_scale_raw[t,n] * c_scale_sigma[n];
		c_loc[t,n] = c_loc_mu[n] + c_loc_raw[t,n] * c_loc_sigma[n];
	}
}

// pre treatment
ordered[N_dep-1] c_pre = c_loc_pre + ordered_pred( c_tilde_pre, c_scale_pre );

vector[N_age] beta_age_raw2_pre = append_row(beta_age_raw_pre, -sum(beta_age_raw_pre));
vector[N_edu] beta_edu_raw2_pre = append_row(beta_edu_raw_pre, -sum(beta_edu_raw_pre));
vector[N_eth] beta_eth_raw2_pre = append_row(beta_eth_raw_pre, -sum(beta_eth_raw_pre));
vector[N_lan] beta_lan_raw2_pre = append_row(beta_lan_raw_pre, -sum(beta_lan_raw_pre));
vector[N_rel] beta_rel_raw2_pre = append_row(beta_rel_raw_pre, -sum(beta_rel_raw_pre));
vector[N_emp] beta_emp_raw2_pre = append_row(beta_emp_raw_pre, -sum(beta_emp_raw_pre));
vector[N_reg] beta_reg_raw2_pre = append_row(beta_reg_raw_pre, -sum(beta_reg_raw_pre));

vector[N_dis] beta_dis_raw2_pre = append_row(beta_dis_raw_pre, -sum(beta_dis_raw_pre));
vector[N_hea] beta_hea_raw2_pre = append_row(beta_hea_raw_pre, -sum(beta_hea_raw_pre));


vector[N_age] beta_age_pre = sigma_age_pre * beta_age_raw2_pre;
vector[N_edu] beta_edu_pre = sigma_edu_pre * beta_edu_raw2_pre;
vector[N_eth] beta_eth_pre = sigma_eth_pre * beta_eth_raw2_pre;
vector[N_lan] beta_lan_pre = sigma_lan_pre * beta_lan_raw2_pre;
vector[N_rel] beta_rel_pre = sigma_rel_pre * beta_rel_raw2_pre;
vector[N_emp] beta_emp_pre = sigma_emp_pre * beta_emp_raw2_pre;
vector[N_reg] beta_reg_pre = sigma_reg_pre * beta_reg_raw2_pre;

vector[N_dis] beta_dis_pre = sigma_dis_pre * beta_dis_raw2_pre;
vector[N_hea] beta_hea_pre = sigma_hea_pre * beta_hea_raw2_pre;

// treatment effects

array[N_treat, N_dep] ordered[N_dep-1] c;

array[N_treat] vector[N_age] beta_age_raw2;
array[N_treat] vector[N_edu] beta_edu_raw2;
array[N_treat] vector[N_eth] beta_eth_raw2;
array[N_treat] vector[N_lan] beta_lan_raw2;
array[N_treat] vector[N_rel] beta_rel_raw2;
array[N_treat] vector[N_emp] beta_emp_raw2;
array[N_treat] vector[N_dis] beta_dis_raw2;
array[N_treat] vector[N_hea] beta_hea_raw2;

array[N_treat] vector[N_age] beta_age;
array[N_treat] vector[N_edu] beta_edu;
array[N_treat] vector[N_eth] beta_eth;
array[N_treat] vector[N_lan] beta_lan;
array[N_treat] vector[N_rel] beta_rel;
array[N_treat] vector[N_emp] beta_emp;
array[N_treat] vector[N_dis] beta_dis;
array[N_treat] vector[N_hea] beta_hea;

for (t in 1:N_treat){
	for (n in 1:N_dep){
		c[t,n] = c_loc[t,n] + ordered_pred( c_tilde[t,n], c_scale[t,n] );
	}
	beta_age_raw2[t] = append_row(beta_age_raw[t], -sum(beta_age_raw[t]));
	beta_edu_raw2[t] = append_row(beta_edu_raw[t], -sum(beta_edu_raw[t]));
	beta_eth_raw2[t] = append_row(beta_eth_raw[t], -sum(beta_eth_raw[t]));
	beta_lan_raw2[t] = append_row(beta_lan_raw[t], -sum(beta_lan_raw[t]));
	beta_rel_raw2[t] = append_row(beta_rel_raw[t], -sum(beta_rel_raw[t]));
	beta_emp_raw2[t] = append_row(beta_emp_raw[t], -sum(beta_emp_raw[t]));
	beta_dis_raw2[t] = append_row(beta_dis_raw[t], -sum(beta_dis_raw[t]));
	beta_hea_raw2[t] = append_row(beta_hea_raw[t], -sum(beta_hea_raw[t]));


	beta_age[t] = sigma_age[t] * beta_age_raw2[t];
	beta_edu[t] = sigma_edu[t] * beta_edu_raw2[t];
	beta_eth[t] = sigma_eth[t] * beta_eth_raw2[t];
	beta_lan[t] = sigma_lan[t] * beta_lan_raw2[t];
	beta_rel[t] = sigma_rel[t] * beta_rel_raw2[t];
	beta_emp[t] = sigma_emp[t] * beta_emp_raw2[t];
	beta_dis[t] = sigma_dis[t] * beta_dis_raw2[t];
	beta_hea[t] = sigma_hea[t] * beta_hea_raw2[t];

}


}

model {
vector[N] lin_pred_pre = beta_reg_pre[reg] + beta_age_pre[age] + [beta_sex_pre, -beta_sex_pre][sex]' + beta_edu_pre[edu] + beta_eth_pre[eth] + beta_lan_pre[lan] + beta_rel_pre[rel] 
  			+ beta_emp_pre[emp] + beta_dis_pre[dis] + beta_hea_pre[hea]
                        + [beta_buk_pre,-beta_buk_pre][buk]' + [beta_mig_pre,-beta_mig_pre][mig]';
  array[N_treat] vector[N] lin_pred_treat;

// no treatment effect per region
for (t in 1:N_treat){
  	lin_pred_treat[t] = lin_pred_pre + beta_age[t][age] + [beta_sex[t], -beta_sex[t]][sex]' + beta_edu[t][edu] + beta_eth[t][eth] 
  			+ beta_lan[t][lan] + beta_rel[t][rel] 
  			+ beta_emp[t][emp] + beta_dis[t][dis] + beta_hea[t][hea]
                        + [beta_buk[t],-beta_buk[t]][buk]' + [beta_mig[t],-beta_mig[t]][mig]';
}


for (n in 1:N){
	y_pre[n] ~ ordered_logistic(lin_pred_pre[n], c_pre);
	y_post[n] ~ ordered_logistic( lin_pred_treat[treat[n]][n], c[treat[n], y_pre[n]]);
	}

c_tilde_pre ~ dirichlet( rep_vector( 1, N_dep - 2 ) );
c_scale_pre ~ normal(0,5);
c_loc_pre ~ normal(0,7);
  
beta_age_raw2_pre ~ normal(0, 1);
beta_edu_raw2_pre ~ normal(0, 1);
beta_eth_raw2_pre ~ normal(0, 1);
beta_lan_raw2_pre ~ normal(0, 1);
beta_rel_raw2_pre ~ normal(0, 1);
beta_emp_raw2_pre ~ normal(0, 1);
beta_reg_raw2_pre ~ normal(0, 1);
beta_dis_raw2_pre ~ normal(0, 1);
beta_hea_raw2_pre ~ normal(0, 1);
  
  
  beta_sex_pre ~ normal(0,2);
  beta_buk_pre ~ normal(0,2);
  beta_mig_pre ~ normal(0,2);
	
  { sigma_age_pre, sigma_edu_pre, sigma_eth_pre, sigma_lan_pre, sigma_rel_pre, sigma_emp_pre, sigma_reg_pre, sigma_dis_pre, sigma_hea_pre } ~ normal(0, 1);

/////////////
for (n in 1:N_dep){
	c_tilde_prior_scale[n] ~ gamma(1,1);
	c_tilde_prior[n] ~ dirichlet( rep_vector(1, N_dep-2) );
	c_scale_mu[n] ~ normal(0,5);
	c_loc_mu[n] ~ normal(0,7);
	c_scale_sigma[n] ~ normal(0,5);
	c_loc_sigma[n] ~ normal(0,5);
}

for (t in 1:N_treat){
for (n in 1:N_dep){
 c_tilde[t,n] ~ dirichlet( c_tilde_prior_scale[n] * c_tilde_prior[n] );
 c_scale_raw[t,n] ~ normal(0,1);
 c_loc_raw[t,n] ~ normal(0,1);
 }
beta_age_raw2[t] ~ normal(0, 1);
beta_edu_raw2[t] ~ normal(0, 1);
beta_eth_raw2[t] ~ normal(0, 1);
beta_lan_raw2[t] ~ normal(0, 1);
beta_rel_raw2[t] ~ normal(0, 1);
beta_emp_raw2[t] ~ normal(0, 1);
beta_dis_raw2[t] ~ normal(0, 1);
beta_hea_raw2[t] ~ normal(0, 1);
  
  
  beta_sex[t] ~ normal(0,2);
  beta_buk[t] ~ normal(0,2);
  beta_mig[t] ~ normal(0,2);

  { sigma_age[t], sigma_edu[t], sigma_eth[t], sigma_lan[t], sigma_rel[t], sigma_emp[t], sigma_dis[t], sigma_hea[t] } ~ normal(0, 1);
}
}


generated quantities {
    vector[N] log_lik;
    { 
    vector[N] lin_pred_pre = beta_reg_pre[reg] + beta_age_pre[age] + [beta_sex_pre, -beta_sex_pre][sex]' + beta_edu_pre[edu] + beta_eth_pre[eth] + beta_lan_pre[lan] + beta_rel_pre[rel] 
  			+ beta_emp_pre[emp] + beta_dis_pre[dis] + beta_hea_pre[hea]
                        + [beta_buk_pre,-beta_buk_pre][buk]' + [beta_mig_pre,-beta_mig_pre][mig]';
  array[N_treat] vector[N] lin_pred_treat;

for (t in 1:N_treat){
  	lin_pred_treat[t] = lin_pred_pre + beta_age[t][age] + [beta_sex[t], -beta_sex[t]][sex]' + beta_edu[t][edu] + beta_eth[t][eth] 
  			+ beta_lan[t][lan] + beta_rel[t][rel] 
  			+ beta_emp[t][emp] + beta_dis[t][dis] + beta_hea[t][hea]
                        + [beta_buk[t],-beta_buk[t]][buk]' + [beta_mig[t],-beta_mig[t]][mig]';
}
    
    for (n in 1:N) {
        log_lik[n] = ordered_logistic_lpmf( y_post[n] |  lin_pred_treat[treat[n]][n], c[treat[n], y_pre[n]]);
    }
    
    }
}







