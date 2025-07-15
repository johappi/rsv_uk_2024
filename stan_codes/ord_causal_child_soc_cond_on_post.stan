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
  // int N_yuk; // year of arrival in UK
  
  // int N_sch; // = 2 // Are you a student in full-time education?
  // int N_buk; // = 2 // Were you born in the United Kingdom?
  // int N_mig; // = 2 // Did you live outside the UK more than a year ago?
  // int N_hhlan; // = 2 // Does anyone in your household have English or Welsh as their main language?
  //int N_sex; // =2
  
  // int N_reg;
  
  array[N] int age;
  array[N] int rel;
  array[N] int eth;
  array[N] int lan;
  array[N] int emp;
  array[N] int edu;
  array[N] int sex;
  // array[N] int reg;

  array[N] int dis;
  array[N] int hea;
  // array[N] int yuk;
  // array[N] int sch;
  array[N] int buk;
  array[N] int mig;
  // array[N] int hhlan;

  int N_treat;
  
  array[N] int y_child;
  // array[N_treat, N] int y_post;
  array[N] int y_post;
  
  array[N] int treat;
  
  // optional
  array[N] int rel_ch_i;
  array[N] int sex_ch_i;
  array[N] int per_ch_i;
  
}

parameters {

// vector[N_dep - 1] mu_theta_treat; // mean for dirichlets for treatment effects
// real mu_beta_treat; // mean for coefficient for treatment effects
// real <lower = 0> sigma_beta_treat; // standard deviation for coefficients for treatment effects

//////////////////
// shared parameters //
//////////////////

real<lower=0.01> sigma_age;
real<lower=0.01> sigma_edu;
real<lower=0.01> sigma_eth;
real<lower=0.01> sigma_lan;
real<lower=0.01> sigma_rel;
real<lower=0.01> sigma_emp;
// real<lower=0.01> sigma_reg;

real<lower=0.01> sigma_dis;
real<lower=0.01> sigma_hea;
// real<lower=0.01> sigma_yuk;
  
vector[N_age - 1] beta_age_raw;
vector[N_edu - 1] beta_edu_raw;
vector[N_eth - 1] beta_eth_raw;
vector[N_lan - 1] beta_lan_raw;
vector[N_rel - 1] beta_rel_raw;
vector[N_emp - 1] beta_emp_raw;
// vector[N_reg - 1] beta_reg_raw;

vector[N_dis - 1] beta_dis_raw;
vector[N_hea - 1] beta_hea_raw;
// vector[N_yuk - 1] beta_yuk_raw;

real beta_sex;

// real beta_sch;
real beta_buk;
real beta_mig;
// real beta_hhlan;

real beta_rel_ch_i;
real beta_sex_ch_i;
real beta_per_ch_i;


///////////////////////
//// child effects ////
///////////////////////
// simplex[N_dep - 1] theta_child;
// real beta_child;


array[N_dep] simplex[N_dep - 2] c_tilde_child;
array[N_dep] real<lower=0> c_scale_child;
array[N_dep] real c_loc_child;

// alternatively
// array[N_treat] simplex[N_dep - 2] c_tilde_child;
// array[N_treat] real<lower=0> c_scale_child;
// array[N_treat] real c_loc_child;

real<lower=0.01> sigma_age_child;
real<lower=0.01> sigma_edu_child;
real<lower=0.01> sigma_eth_child;
real<lower=0.01> sigma_lan_child;
real<lower=0.01> sigma_rel_child;
real<lower=0.01> sigma_emp_child;
// real<lower=0.01> sigma_reg_child;

real<lower=0.01> sigma_dis_child;
real<lower=0.01> sigma_hea_child;
// real<lower=0.01> sigma_yuk_child;
  
vector[N_age - 1] beta_age_raw_child;
vector[N_edu - 1] beta_edu_raw_child;
vector[N_eth - 1] beta_eth_raw_child;
vector[N_lan - 1] beta_lan_raw_child;
vector[N_rel - 1] beta_rel_raw_child;
vector[N_emp - 1] beta_emp_raw_child;
// vector[N_reg - 1] beta_reg_raw_child;

vector[N_dis - 1] beta_dis_raw_child;
vector[N_hea - 1] beta_hea_raw_child;
// vector[N_yuk - 1] beta_yuk_raw_child;

real beta_sex_child;

// real beta_sch_child;
real beta_buk_child;
real beta_mig_child;
// real beta_hhlan_child;

real beta_rel_ch_i_child;
real beta_sex_ch_i_child;
real beta_per_ch_i_child;
///////////////////////
// treatment effects //
///////////////////////

// array[N_treat] simplex[N_dep - 1] theta_treat;
// array[N_treat] real beta_treat;

// ordered[N_dep-1] c; // cutpoints for ordered logistic
// alternatively
array[N_treat] simplex[N_dep - 2] c_tilde;
array[N_treat] real<lower=0> c_scale;
array[N_treat] real c_loc;


}

transformed parameters {
// alternatively for c
array[N_treat] ordered[N_dep-1] c;
array[N_dep] ordered[N_dep-1] c_child;

// shared parameters

vector[N_age] beta_age_raw2 = append_row(beta_age_raw, -sum(beta_age_raw));
vector[N_edu] beta_edu_raw2 = append_row(beta_edu_raw, -sum(beta_edu_raw));
vector[N_eth] beta_eth_raw2 = append_row(beta_eth_raw, -sum(beta_eth_raw));
vector[N_lan] beta_lan_raw2 = append_row(beta_lan_raw, -sum(beta_lan_raw));
vector[N_rel] beta_rel_raw2 = append_row(beta_rel_raw, -sum(beta_rel_raw));
vector[N_emp] beta_emp_raw2 = append_row(beta_emp_raw, -sum(beta_emp_raw));
// vector[N_reg] beta_reg_raw2 = append_row(beta_reg_raw, -sum(beta_reg_raw));

vector[N_dis] beta_dis_raw2 = append_row(beta_dis_raw, -sum(beta_dis_raw));
vector[N_hea] beta_hea_raw2 = append_row(beta_hea_raw, -sum(beta_hea_raw));
// vector[N_yuk] beta_yuk_raw2 = append_row(beta_yuk_raw, -sum(beta_yuk_raw));


vector[N_age] beta_age = sigma_age * beta_age_raw2;
vector[N_edu] beta_edu = sigma_edu * beta_edu_raw2;
vector[N_eth] beta_eth = sigma_eth * beta_eth_raw2;
vector[N_lan] beta_lan = sigma_lan * beta_lan_raw2;
vector[N_rel] beta_rel = sigma_rel * beta_rel_raw2;
vector[N_emp] beta_emp = sigma_emp * beta_emp_raw2;
// vector[N_reg] beta_reg = sigma_reg * beta_reg_raw2;

vector[N_dis] beta_dis = sigma_dis * beta_dis_raw2;
vector[N_hea] beta_hea = sigma_hea * beta_hea_raw2;
// vector[N_yuk] beta_yuk = sigma_yuk * beta_yuk_raw2;


// child
// alternatively for c
for (d in 1:N_dep){
	c_child[d] = c_loc_child[d] + ordered_pred( c_tilde_child[d], c_scale_child[d] );
}

vector[N_age] beta_age_raw2_child = append_row(beta_age_raw_child, -sum(beta_age_raw_child));
vector[N_edu] beta_edu_raw2_child = append_row(beta_edu_raw_child, -sum(beta_edu_raw_child));
vector[N_eth] beta_eth_raw2_child = append_row(beta_eth_raw_child, -sum(beta_eth_raw_child));
vector[N_lan] beta_lan_raw2_child = append_row(beta_lan_raw_child, -sum(beta_lan_raw_child));
vector[N_rel] beta_rel_raw2_child = append_row(beta_rel_raw_child, -sum(beta_rel_raw_child));
vector[N_emp] beta_emp_raw2_child = append_row(beta_emp_raw_child, -sum(beta_emp_raw_child));
// vector[N_reg] beta_reg_raw2_child = append_row(beta_reg_raw_child, -sum(beta_reg_raw_child));

vector[N_dis] beta_dis_raw2_child = append_row(beta_dis_raw_child, -sum(beta_dis_raw_child));
vector[N_hea] beta_hea_raw2_child = append_row(beta_hea_raw_child, -sum(beta_hea_raw_child));
// vector[N_yuk] beta_yuk_raw2_child = append_row(beta_yuk_raw_child, -sum(beta_yuk_raw_child));


vector[N_age] beta_age_child = sigma_age_child * beta_age_raw2_child;
vector[N_edu] beta_edu_child = sigma_edu_child * beta_edu_raw2_child;
vector[N_eth] beta_eth_child = sigma_eth_child * beta_eth_raw2_child;
vector[N_lan] beta_lan_child = sigma_lan_child * beta_lan_raw2_child;
vector[N_rel] beta_rel_child = sigma_rel_child * beta_rel_raw2_child;
vector[N_emp] beta_emp_child = sigma_emp_child * beta_emp_raw2_child;
// vector[N_reg] beta_reg_child = sigma_reg_child * beta_reg_raw2_child;

vector[N_dis] beta_dis_child = sigma_dis_child * beta_dis_raw2_child;
vector[N_hea] beta_hea_child = sigma_hea_child * beta_hea_raw2_child;
// vector[N_yuk] beta_yuk_child = sigma_yuk_child * beta_yuk_raw2_child;

// treatment effects
for (t in 1:N_treat){
	c[t] = c_loc[t] + ordered_pred( c_tilde[t], c_scale[t] );
}
}


model {
vector[N] lin_pred = beta_age[age] + [beta_sex, -beta_sex][sex]' + beta_edu[edu] + beta_eth[eth] + beta_lan[lan] + beta_rel[rel] 
  			+ beta_emp[emp] + beta_dis[dis] + beta_hea[hea]
                        + [beta_buk,-beta_buk][buk]' + [beta_mig,-beta_mig][mig]'
                        + [beta_sex_ch_i, -beta_sex_ch_i][sex_ch_i]' + [beta_per_ch_i, -beta_per_ch_i][per_ch_i]' + [beta_rel_ch_i, -beta_rel_ch_i][rel_ch_i]';
vector[N] lin_pred_child = lin_pred + beta_age_child[age] + [beta_sex_child, -beta_sex_child][sex]' + beta_edu_child[edu] + beta_eth_child[eth] + beta_lan_child[lan] + beta_rel_child[rel] 
  			+ beta_emp_child[emp] + beta_dis_child[dis] + beta_hea_child[hea]
                        + [beta_buk_child,-beta_buk_child][buk]' + [beta_mig_child,-beta_mig_child][mig]'
                        + [beta_sex_ch_i_child, -beta_sex_ch_i_child][sex_ch_i]' + [beta_per_ch_i_child, -beta_per_ch_i_child][per_ch_i]' + [beta_rel_ch_i_child, -beta_rel_ch_i_child][rel_ch_i]';

for (n in 1:N){
	// y_child[n] ~ ordered_logistic( beta_child * sum( theta_child[ : y_post[n] - 1] ) + lin_pred_child[n], c_child  ); //+ c[treat[n]]
	// y_child[n] ~ ordered_logistic( beta_child * sum( theta_child[ : y_post[n] - 1] ) + lin_pred_child[n], c_child[treat[n]] );
	y_child[n] ~ ordered_logistic( lin_pred_child[n], c_child[y_post[n]]   ); 
	y_post[n] ~ ordered_logistic( lin_pred[n], c[treat[n]] );
	}
 
/*
c_tilde_child ~ dirichlet( rep_vector( 1, N_dep - 2 ) );
// c_scale_child ~ normal(0,5);
// c_loc_child ~ normal(0,7);
c_scale_child ~ normal(0,5);
c_loc_child ~ normal(0,7);
*/


for (d in 1:N_dep){
c_tilde_child[d] ~ dirichlet( rep_vector( 1, N_dep - 2 ) );
c_scale_child[d] ~ normal(0,5);
c_loc_child[d] ~ normal(0,7);
}

 
for (t in 1:N_treat){
	c_tilde ~ dirichlet( rep_vector( 1, N_dep - 2 ) );
	c_scale ~ normal(0,5);
	c_loc ~ normal(0,7); 	
}
 
beta_age_raw2_child ~ normal(0, 1);
beta_edu_raw2_child ~ normal(0, 1);
beta_eth_raw2_child ~ normal(0, 1);
beta_lan_raw2_child ~ normal(0, 1);
beta_rel_raw2_child ~ normal(0, 1);
beta_emp_raw2_child ~ normal(0, 1);
// beta_reg_raw2_child ~ normal(0, 1);

beta_dis_raw2_child ~ normal(0, 1);
beta_hea_raw2_child ~ normal(0, 1);
// beta_yuk_raw2_child ~ normal(0, 1);
  
  
  beta_sex_child ~ normal(0,2);
  // beta_sch_child ~ normal(0,2);
  beta_buk_child ~ normal(0,2);
  beta_mig_child ~ normal(0,2);
  // beta_hhlan_child ~ normal(0,2);
	
  { sigma_age_child, sigma_edu_child, sigma_eth_child, sigma_lan_child, sigma_rel_child, sigma_emp_child, sigma_dis_child, sigma_hea_child } ~ normal(0, 1);

beta_age_raw2 ~ normal(0, 1);
beta_edu_raw2 ~ normal(0, 1);
beta_eth_raw2 ~ normal(0, 1);
beta_lan_raw2 ~ normal(0, 1);
beta_rel_raw2 ~ normal(0, 1);
beta_emp_raw2 ~ normal(0, 1);
// beta_reg_raw2 ~ normal(0, 1);

beta_dis_raw2 ~ normal(0, 1);
beta_hea_raw2 ~ normal(0, 1);
// beta_yuk_raw2 ~ normal(0, 1);
  
  
  beta_sex ~ normal(0,2);
  // beta_sch ~ normal(0,2);
  beta_buk ~ normal(0,2);
  beta_mig ~ normal(0,2);
  // beta_hhlan ~ normal(0,2);
	
  { sigma_age, sigma_edu, sigma_eth, sigma_lan, sigma_rel, sigma_emp, sigma_dis, sigma_hea } ~ normal(0, 1);


// theta_child ~ dirichlet( rep_vector(1, N_dep - 1 ) );
// beta_child ~ normal(0,7);


// mu_theta_child ~ gamma(1,1);
// mu_beta_child ~ normal(0,3);
// sigma_beta_child ~ gamma(1,1);

// for (t in 1:N_treat){
//  theta_treat[t] ~ dirichlet(mu_theta_treat);
//  beta_treat[t] ~ normal(mu_beta_treat, sigma_beta_treat);
//}

beta_sex_ch_i ~ normal(0,2);
beta_per_ch_i ~ normal(0,2);
beta_rel_ch_i ~ normal(0,2);

beta_sex_ch_i_child ~ normal(0,2);
beta_per_ch_i_child ~ normal(0,2);
beta_rel_ch_i_child ~ normal(0,2);

}


generated quantities {

array[N, N_dep, N_dep] real y_log_prob_child;
array[N, N_dep] real y_log_prob_post;
// array[N] int y_pred;
{
vector[N] lin_pred = beta_age[age] + [beta_sex, -beta_sex][sex]' + beta_edu[edu] + beta_eth[eth] + beta_lan[lan] + beta_rel[rel] 
  			+ beta_emp[emp] + beta_dis[dis] + beta_hea[hea]
                        + [beta_buk,-beta_buk][buk]' + [beta_mig,-beta_mig][mig]'
                        + [beta_sex_ch_i, -beta_sex_ch_i][sex_ch_i]' + [beta_per_ch_i, -beta_per_ch_i][per_ch_i]' + [beta_rel_ch_i, -beta_rel_ch_i][rel_ch_i]';
vector[N] lin_pred_child = lin_pred + beta_age_child[age] + [beta_sex_child, -beta_sex_child][sex]' + beta_edu_child[edu] + beta_eth_child[eth] + beta_lan_child[lan] + beta_rel_child[rel] 
  			+ beta_emp_child[emp] + beta_dis_child[dis] + beta_hea_child[hea]
                        + [beta_buk_child,-beta_buk_child][buk]' + [beta_mig_child,-beta_mig_child][mig]'
                        + [beta_sex_ch_i_child, -beta_sex_ch_i_child][sex_ch_i]' + [beta_per_ch_i_child, -beta_per_ch_i_child][per_ch_i]' + [beta_rel_ch_i_child, -beta_rel_ch_i_child][rel_ch_i]';

for (n in 1:N){
	// y_pred[n] = ordered_logistic_rng(lin_pred[n], c);
	for (k_post in 1:N_dep) {
		// y_log_prob_child[n,k] = ordered_logistic_lpmf(k | beta_child * sum( theta_child[ : y_post[n] - 1] ) + lin_pred_child[n], c_child);
		y_log_prob_post[n,k_post] = ordered_logistic_lpmf(k_post | lin_pred[n], c[treat[n]] );
		for (k_child in 1:N_dep){
			y_log_prob_child[n, k_post, k_child] = ordered_logistic_lpmf( k_child | lin_pred_child[n], c_child[k_post] ); //  + c[treat[n]]
		}
	}
}

}
}


