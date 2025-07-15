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
  
  array[N] int age;
  array[N] int rel;
  array[N] int eth;
  array[N] int lan;
  array[N] int emp;
  array[N] int edu;
  array[N] int sex;
  array[N] int dis;
  array[N] int hea;
  array[N] int buk;
  array[N] int mig;

  int N_treat;
  
  array[N] int y_child;
  array[N] int y_pre;
  array[N] int y_post;
  
  array[N] int treat;
  
  // optional
  array[N] int rel_ch_i;
  array[N] int sex_ch_i;
  array[N] int per_ch_i;
  
}

parameters {

//////////////////
// shared parameters //
//////////////////

real<lower=0.01> sigma_age;
real<lower=0.01> sigma_edu;
real<lower=0.01> sigma_eth;
real<lower=0.01> sigma_lan;
real<lower=0.01> sigma_rel;
real<lower=0.01> sigma_emp;

real<lower=0.01> sigma_dis;
real<lower=0.01> sigma_hea;
  
vector[N_age - 1] beta_age_raw;
vector[N_edu - 1] beta_edu_raw;
vector[N_eth - 1] beta_eth_raw;
vector[N_lan - 1] beta_lan_raw;
vector[N_rel - 1] beta_rel_raw;
vector[N_emp - 1] beta_emp_raw;

vector[N_dis - 1] beta_dis_raw;
vector[N_hea - 1] beta_hea_raw;

real beta_sex;
real beta_buk;
real beta_mig;

real beta_rel_ch_i;
real beta_sex_ch_i;
real beta_per_ch_i;

//////////////////
// post parameters //
//////////////////

real<lower=0.01> sigma_age_post;
real<lower=0.01> sigma_edu_post;
real<lower=0.01> sigma_eth_post;
real<lower=0.01> sigma_lan_post;
real<lower=0.01> sigma_rel_post;
real<lower=0.01> sigma_emp_post;

real<lower=0.01> sigma_dis_post;
real<lower=0.01> sigma_hea_post;
  
vector[N_age - 1] beta_age_raw_post;
vector[N_edu - 1] beta_edu_raw_post;
vector[N_eth - 1] beta_eth_raw_post;
vector[N_lan - 1] beta_lan_raw_post;
vector[N_rel - 1] beta_rel_raw_post;
vector[N_emp - 1] beta_emp_raw_post;

vector[N_dis - 1] beta_dis_raw_post;
vector[N_hea - 1] beta_hea_raw_post;

real beta_sex_post;

real beta_buk_post;
real beta_mig_post;

real beta_rel_ch_i_post;
real beta_sex_ch_i_post;
real beta_per_ch_i_post;


///////////////////////
//// child effects ////
///////////////////////

array[N_dep] simplex[N_dep - 2] c_tilde_child_pre;
array[N_dep] real<lower=0> c_scale_child_pre;
array[N_dep] real c_loc_child_pre;

array[N_dep] simplex[N_dep - 2] c_tilde_child_post;
array[N_dep] real<lower=0> c_scale_child_post;
array[N_dep] real c_loc_child_post;

real<lower=0.01> sigma_age_child;
real<lower=0.01> sigma_edu_child;
real<lower=0.01> sigma_eth_child;
real<lower=0.01> sigma_lan_child;
real<lower=0.01> sigma_rel_child;
real<lower=0.01> sigma_emp_child;

real<lower=0.01> sigma_dis_child;
real<lower=0.01> sigma_hea_child;
  
vector[N_age - 1] beta_age_raw_child;
vector[N_edu - 1] beta_edu_raw_child;
vector[N_eth - 1] beta_eth_raw_child;
vector[N_lan - 1] beta_lan_raw_child;
vector[N_rel - 1] beta_rel_raw_child;
vector[N_emp - 1] beta_emp_raw_child;

vector[N_dis - 1] beta_dis_raw_child;
vector[N_hea - 1] beta_hea_raw_child;

real beta_sex_child;

real beta_buk_child;
real beta_mig_child;

real beta_rel_ch_i_child;
real beta_sex_ch_i_child;
real beta_per_ch_i_child;

vector[N_treat-1] beta_treat_raw_child;
real<lower=0> sigma_treat_child;

///////////////////////
///// pre intent //////
///////////////////////

simplex[N_dep - 2] c_tilde_pre;
real<lower=0> c_scale_pre;
real c_loc_pre;

///////////////////////
// treatment effects //
///////////////////////
vector[N_dep - 1] mu_theta_treat; // mean for dirichlets for treatment effects
real mu_beta_treat; // mean for coefficient for treatment effects
real <lower = 0> sigma_beta_treat; // standard deviation for coefficients for treatment effects
array[N_treat] simplex[N_dep - 1] theta_treat;
array[N_treat] real beta_treat_tilde; 

array[N_treat] simplex[N_dep - 2] c_tilde_post;
array[N_treat] real<lower=0> c_scale_post;
array[N_treat] real c_loc_post;


}

transformed parameters {
array[N_treat] real beta_treat;

ordered[N_dep-1] c_pre;
array[N_treat] ordered[N_dep-1] c_post;
array[N_dep] ordered[N_dep-1] c_child_pre;
array[N_dep] ordered[N_dep-1] c_child_post;

// shared parameters

vector[N_age] beta_age_raw2 = append_row(beta_age_raw, -sum(beta_age_raw));
vector[N_edu] beta_edu_raw2 = append_row(beta_edu_raw, -sum(beta_edu_raw));
vector[N_eth] beta_eth_raw2 = append_row(beta_eth_raw, -sum(beta_eth_raw));
vector[N_lan] beta_lan_raw2 = append_row(beta_lan_raw, -sum(beta_lan_raw));
vector[N_rel] beta_rel_raw2 = append_row(beta_rel_raw, -sum(beta_rel_raw));
vector[N_emp] beta_emp_raw2 = append_row(beta_emp_raw, -sum(beta_emp_raw));

vector[N_dis] beta_dis_raw2 = append_row(beta_dis_raw, -sum(beta_dis_raw));
vector[N_hea] beta_hea_raw2 = append_row(beta_hea_raw, -sum(beta_hea_raw));

vector[N_age] beta_age = sigma_age * beta_age_raw2;
vector[N_edu] beta_edu = sigma_edu * beta_edu_raw2;
vector[N_eth] beta_eth = sigma_eth * beta_eth_raw2;
vector[N_lan] beta_lan = sigma_lan * beta_lan_raw2;
vector[N_rel] beta_rel = sigma_rel * beta_rel_raw2;
vector[N_emp] beta_emp = sigma_emp * beta_emp_raw2;

vector[N_dis] beta_dis = sigma_dis * beta_dis_raw2;
vector[N_hea] beta_hea = sigma_hea * beta_hea_raw2;

///////////
// child //
///////////

for (d in 1:N_dep){
	c_child_pre[d] = c_loc_child_pre[d] + ordered_pred( c_tilde_child_pre[d], c_scale_child_pre[d] );
	c_child_post[d] = c_loc_child_post[d] + ordered_pred( c_tilde_child_post[d], c_scale_child_post[d] );
}

vector[N_age] beta_age_raw2_child = append_row(beta_age_raw_child, -sum(beta_age_raw_child));
vector[N_edu] beta_edu_raw2_child = append_row(beta_edu_raw_child, -sum(beta_edu_raw_child));
vector[N_eth] beta_eth_raw2_child = append_row(beta_eth_raw_child, -sum(beta_eth_raw_child));
vector[N_lan] beta_lan_raw2_child = append_row(beta_lan_raw_child, -sum(beta_lan_raw_child));
vector[N_rel] beta_rel_raw2_child = append_row(beta_rel_raw_child, -sum(beta_rel_raw_child));
vector[N_emp] beta_emp_raw2_child = append_row(beta_emp_raw_child, -sum(beta_emp_raw_child));

vector[N_dis] beta_dis_raw2_child = append_row(beta_dis_raw_child, -sum(beta_dis_raw_child));
vector[N_hea] beta_hea_raw2_child = append_row(beta_hea_raw_child, -sum(beta_hea_raw_child));


vector[N_age] beta_age_child = sigma_age_child * beta_age_raw2_child;
vector[N_edu] beta_edu_child = sigma_edu_child * beta_edu_raw2_child;
vector[N_eth] beta_eth_child = sigma_eth_child * beta_eth_raw2_child;
vector[N_lan] beta_lan_child = sigma_lan_child * beta_lan_raw2_child;
vector[N_rel] beta_rel_child = sigma_rel_child * beta_rel_raw2_child;
vector[N_emp] beta_emp_child = sigma_emp_child * beta_emp_raw2_child;

vector[N_dis] beta_dis_child = sigma_dis_child * beta_dis_raw2_child;
vector[N_hea] beta_hea_child = sigma_hea_child * beta_hea_raw2_child;

vector[N_treat] beta_treat_raw2_child = append_row( beta_treat_raw_child, -sum( beta_treat_raw_child) );
vector[N_treat] beta_treat_child = sigma_treat_child * beta_treat_raw2_child;

////////////////////////
///// pre effects //////
////////////////////////
c_pre = c_loc_pre + ordered_pred( c_tilde_pre, c_scale_pre );

///////////////////////
// treatment effects //
///////////////////////
for (t in 1:N_treat){
	c_post[t] = c_loc_post[t] + ordered_pred( c_tilde_post[t], c_scale_post[t] );
	beta_treat[t] = mu_beta_treat + sigma_beta_treat * beta_treat_tilde[t];
}

//////////////////////////////////
vector[N_age] beta_age_raw2_post = append_row(beta_age_raw_post, -sum(beta_age_raw_post));
vector[N_edu] beta_edu_raw2_post = append_row(beta_edu_raw_post, -sum(beta_edu_raw_post));
vector[N_eth] beta_eth_raw2_post = append_row(beta_eth_raw_post, -sum(beta_eth_raw_post));
vector[N_lan] beta_lan_raw2_post = append_row(beta_lan_raw_post, -sum(beta_lan_raw_post));
vector[N_rel] beta_rel_raw2_post = append_row(beta_rel_raw_post, -sum(beta_rel_raw_post));
vector[N_emp] beta_emp_raw2_post = append_row(beta_emp_raw_post, -sum(beta_emp_raw_post));

vector[N_dis] beta_dis_raw2_post = append_row(beta_dis_raw_post, -sum(beta_dis_raw_post));
vector[N_hea] beta_hea_raw2_post = append_row(beta_hea_raw_post, -sum(beta_hea_raw_post));

vector[N_age] beta_age_post = sigma_age_post * beta_age_raw2_post;
vector[N_edu] beta_edu_post = sigma_edu_post * beta_edu_raw2_post;
vector[N_eth] beta_eth_post = sigma_eth_post * beta_eth_raw2_post;
vector[N_lan] beta_lan_post = sigma_lan_post * beta_lan_raw2_post;
vector[N_rel] beta_rel_post = sigma_rel_post * beta_rel_raw2_post;
vector[N_emp] beta_emp_post = sigma_emp_post * beta_emp_raw2_post;

vector[N_dis] beta_dis_post = sigma_dis_post * beta_dis_raw2_post;
vector[N_hea] beta_hea_post = sigma_hea_post * beta_hea_raw2_post;
}


model {
vector[N] lin_pred = beta_age[age] + [beta_sex, -beta_sex][sex]' + beta_edu[edu] + beta_eth[eth] + beta_lan[lan] + beta_rel[rel] 
  			+ beta_emp[emp] + beta_dis[dis] + beta_hea[hea]
                        + [beta_buk,-beta_buk][buk]' + [beta_mig,-beta_mig][mig]'
                        + [beta_sex_ch_i, -beta_sex_ch_i][sex_ch_i]' + [beta_per_ch_i, -beta_per_ch_i][per_ch_i]' + [beta_rel_ch_i, -beta_rel_ch_i][rel_ch_i]';
vector[N] lin_pred_child =  lin_pred + beta_age_child[age] + [beta_sex_child, -beta_sex_child][sex]' + beta_edu_child[edu] + beta_eth_child[eth] + beta_lan_child[lan] + beta_rel_child[rel] 
  			+ beta_emp_child[emp] + beta_dis_child[dis] + beta_hea_child[hea]
                        + [beta_buk_child,-beta_buk_child][buk]' + [beta_mig_child,-beta_mig_child][mig]'
                        + [beta_sex_ch_i_child, -beta_sex_ch_i_child][sex_ch_i]' + [beta_per_ch_i_child, -beta_per_ch_i_child][per_ch_i]'
                        + [beta_rel_ch_i_child, -beta_rel_ch_i_child][rel_ch_i]'
                        + beta_treat_child[treat];
vector[N] lin_pred_post = lin_pred + beta_age_post[age] + [beta_sex_post, -beta_sex_post][sex]' + beta_edu_post[edu] + beta_eth_post[eth] + beta_lan_post[lan] + beta_rel_post[rel] 
                     + beta_emp_post[emp] + beta_dis_post[dis] + beta_hea_post[hea]
                     + [beta_buk_post, -beta_buk_post][buk]' + [beta_mig_post, -beta_mig_post][mig]'
                     + [beta_sex_ch_i_post, -beta_sex_ch_i_post][sex_ch_i]' + [beta_per_ch_i_post, -beta_per_ch_i_post][per_ch_i]' + [beta_rel_ch_i_post, -beta_rel_ch_i_post][rel_ch_i]';


for (n in 1:N){
	y_child[n] ~ ordered_logistic( lin_pred_child[n], c_child_pre[y_pre[n]] + c_child_post[y_post[n]] ); 
	y_pre[n] ~ ordered_logistic( lin_pred[n], c_pre );
	y_post[n] ~ ordered_logistic( beta_treat[treat[n]] * sum( theta_treat[treat[n]][ : y_pre[n] - 1] ) + lin_pred_post[n], c_post[treat[n]] );
	}
 
for (d in 1:N_dep){
	c_tilde_child_pre[d] ~ dirichlet( rep_vector( 1, N_dep - 2 ) );
	c_scale_child_pre[d] ~ normal(0,5);
	c_loc_child_pre[d] ~ normal(0,7);

	c_tilde_child_post[d] ~ dirichlet( rep_vector( 1, N_dep - 2 ) );
	c_scale_child_post[d] ~ normal(0,5);
	c_loc_child_post[d] ~ normal(0,7);
}

c_tilde_pre ~ dirichlet( rep_vector( 1, N_dep - 2 ) );
c_scale_pre ~ normal(0,5);
c_loc_pre ~ normal(0,7); 	
 
for (t in 1:N_treat){
	c_tilde_post ~ dirichlet( rep_vector( 1, N_dep - 2 ) );
	c_scale_post ~ normal(0,5);
	c_loc_post ~ normal(0,7); 	
}
 
beta_age_raw2_child ~ normal(0, 1);
beta_edu_raw2_child ~ normal(0, 1);
beta_eth_raw2_child ~ normal(0, 1);
beta_lan_raw2_child ~ normal(0, 1);
beta_rel_raw2_child ~ normal(0, 1);
beta_emp_raw2_child ~ normal(0, 1);

beta_dis_raw2_child ~ normal(0, 1);
beta_hea_raw2_child ~ normal(0, 1);
  
  
  beta_sex_child ~ normal(0,2);
  beta_buk_child ~ normal(0,2);
  beta_mig_child ~ normal(0,2);
	
 beta_treat_raw2_child ~ normal(0,1);
 sigma_treat_child ~ normal(0,1);
	
  { sigma_age_child, sigma_edu_child, sigma_eth_child, sigma_lan_child, sigma_rel_child, sigma_emp_child, sigma_dis_child, sigma_hea_child } ~ normal(0, 1);

beta_age_raw2 ~ normal(0, 1);
beta_edu_raw2 ~ normal(0, 1);
beta_eth_raw2 ~ normal(0, 1);
beta_lan_raw2 ~ normal(0, 1);
beta_rel_raw2 ~ normal(0, 1);
beta_emp_raw2 ~ normal(0, 1);

beta_dis_raw2 ~ normal(0, 1);
beta_hea_raw2 ~ normal(0, 1);
  
  
  beta_sex ~ normal(0,2);
  beta_buk ~ normal(0,2);
  beta_mig ~ normal(0,2);
	
  { sigma_age, sigma_edu, sigma_eth, sigma_lan, sigma_rel, sigma_emp, sigma_dis, sigma_hea } ~ normal(0, 1);

beta_age_raw2_post ~ normal(0, 1);
beta_edu_raw2_post ~ normal(0, 1);
beta_eth_raw2_post ~ normal(0, 1);
beta_lan_raw2_post ~ normal(0, 1);
beta_rel_raw2_post ~ normal(0, 1);
beta_emp_raw2_post ~ normal(0, 1);

beta_dis_raw2_post ~ normal(0, 1);
beta_hea_raw2_post ~ normal(0, 1);
  
beta_sex_post ~ normal(0, 2);
beta_buk_post ~ normal(0, 2);
beta_mig_post ~ normal(0, 2);

{ sigma_age_post, sigma_edu_post, sigma_eth_post, sigma_lan_post, sigma_rel_post, sigma_emp_post, sigma_dis_post, sigma_hea_post } ~ normal(0, 1);


mu_theta_treat ~ gamma(1,1);
mu_beta_treat ~ normal(0,3);
sigma_beta_treat ~ gamma(1,1);

for (t in 1:N_treat){
  theta_treat[t] ~ dirichlet(mu_theta_treat);
  beta_treat_tilde ~ normal(0,1);
}

beta_sex_ch_i ~ normal(0,2);
beta_per_ch_i ~ normal(0,2);
beta_rel_ch_i ~ normal(0,2);

beta_sex_ch_i_post ~ normal(0,2);
beta_per_ch_i_post ~ normal(0,2);
beta_rel_ch_i_post ~ normal(0,2);

beta_sex_ch_i_child ~ normal(0,2);
beta_per_ch_i_child ~ normal(0,2);
beta_rel_ch_i_child ~ normal(0,2);

}


generated quantities {

array[N, N_dep, N_dep, N_dep] real y_log_prob_child;
array[N, N_dep] real y_log_prob_pre;
array[N, N_dep, N_dep] real y_log_prob_post;
array[N, N_dep, N_dep,N_treat] real y_log_prob_post_treat;
array[N, N_dep, N_dep, N_dep,N_treat] real y_log_prob_child_treat;
{
vector[N] lin_pred = beta_age[age] + [beta_sex, -beta_sex][sex]' + beta_edu[edu] + beta_eth[eth] + beta_lan[lan] + beta_rel[rel] 
  			+ beta_emp[emp] + beta_dis[dis] + beta_hea[hea]
                        + [beta_buk,-beta_buk][buk]' + [beta_mig,-beta_mig][mig]'
                        + [beta_sex_ch_i, -beta_sex_ch_i][sex_ch_i]' + [beta_per_ch_i, -beta_per_ch_i][per_ch_i]' + [beta_rel_ch_i, -beta_rel_ch_i][rel_ch_i]';

vector[N] lin_pred_post = lin_pred + beta_age_post[age] + [beta_sex_post, -beta_sex_post][sex]' + beta_edu_post[edu] + beta_eth_post[eth] + beta_lan_post[lan] + beta_rel_post[rel] 
                     + beta_emp_post[emp] + beta_dis_post[dis] + beta_hea_post[hea]
                     + [beta_buk_post, -beta_buk_post][buk]' + [beta_mig_post, -beta_mig_post][mig]'
                     + [beta_sex_ch_i_post, -beta_sex_ch_i_post][sex_ch_i]' + [beta_per_ch_i_post, -beta_per_ch_i_post][per_ch_i]' + [beta_rel_ch_i_post, -beta_rel_ch_i_post][rel_ch_i]';



vector[N] lin_pred_child = lin_pred + beta_age_child[age] + [beta_sex_child, -beta_sex_child][sex]' + beta_edu_child[edu] + beta_eth_child[eth] + beta_lan_child[lan] + beta_rel_child[rel] 
  			+ beta_emp_child[emp] + beta_dis_child[dis] + beta_hea_child[hea]
                        + [beta_buk_child,-beta_buk_child][buk]' + [beta_mig_child,-beta_mig_child][mig]'
                        + [beta_sex_ch_i_child, -beta_sex_ch_i_child][sex_ch_i]' + [beta_per_ch_i_child, -beta_per_ch_i_child][per_ch_i]'
                        + [beta_rel_ch_i_child, -beta_rel_ch_i_child][rel_ch_i]';

vector[N] beta_treat_child_pred = beta_treat_child[treat];


for (n in 1:N){
	for (k_pre in 1:N_dep) {
		y_log_prob_pre[n,k_pre] = ordered_logistic_lpmf(k_pre | lin_pred[n], c_pre );
	}
	for (k_pre in 1:N_dep) {
		for (k_post in 1:N_dep) {
			y_log_prob_post[n, k_pre, k_post] = ordered_logistic_lpmf(k_post | beta_treat[treat[n]] * sum( theta_treat[treat[n]][ : k_pre - 1] ) + lin_pred_post[n], c_post[treat[n]] );
			for (k_child in 1:N_dep){
				y_log_prob_child[n, k_pre, k_post, k_child] = ordered_logistic_lpmf( k_child | lin_pred_child[n] + beta_treat_child_pred[n], c_child_pre[k_pre] + c_child_post[k_post] );
			}
		}
		for (t in 1:N_treat){
			for (k_post in 1:N_dep){
				y_log_prob_post_treat[n, k_pre, k_post, t] = ordered_logistic_lpmf(k_post | beta_treat[t] * sum( theta_treat[t][ : k_pre - 1] ) + lin_pred_post[n], c_post[t] );
				for (k_child in 1:N_dep){
					y_log_prob_child_treat[n, k_pre, k_post, k_child,t] = ordered_logistic_lpmf( k_child | lin_pred_child[n] + beta_treat_child[t], c_child_pre[k_pre] + c_child_post[k_post] );
				}
			}
		}
	}
}
}
}


