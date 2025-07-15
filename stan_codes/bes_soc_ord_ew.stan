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
  int N_yuk; // year of arrival in UK
    
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
  array[N] int yuk;
  array[N] int mig;
  array[N] int hhlan;
    
  array[N] int y_ordinal;
  
  int N_besd_ord; // number of ordinal besd variables (=6)
  int num_besd_ord_cat; // number of categories per ordinal variable (=2 or 4)
  int N_besd_ber; // number of binary besd variables (=8)
  int N_besd6; // numcategories for categorical variable
  int N_besd7; // numcategories for categorical variable
  
  array[N] int besd6;
  array[N] int besd7;
  
  array[N_besd_ord, N] int besd_ord;
  array[N_besd_ber, N] int besd_ber;
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
  real<lower=0.01> sigma_yuk;

  real<lower=0.01> sigma_besd6;
  real<lower=0.01> sigma_besd7;
  
vector[N_age - 1] beta_age_raw;
vector[N_edu - 1] beta_edu_raw;
vector[N_eth - 1] beta_eth_raw;
vector[N_lan - 1] beta_lan_raw;
vector[N_rel - 1] beta_rel_raw;
vector[N_emp - 1] beta_emp_raw;
vector[N_reg - 1] beta_reg_raw;

vector[N_dis - 1] beta_dis_raw;
vector[N_hea - 1] beta_hea_raw;
vector[N_yuk - 1] beta_yuk_raw;

real beta_sex;

real beta_mig;
real beta_hhlan;

simplex[N_dep - 2] c_tilde;
real<lower=0> c_scale;
real c_loc;



vector[N_besd6 - 1]  beta_besd6_raw;
vector[N_besd7 - 1]  beta_besd7_raw;

array[N_besd_ord] real gamma_besd_ord; // scale

array[N_besd_ord] simplex[num_besd_ord_cat-1] alpha_besd_ord; 
array[N_besd_ber] real beta_besd_ber;
}

transformed parameters {
ordered[N_dep-1] c = c_loc + ordered_pred( c_tilde, c_scale );

array[N_besd_ord] vector[num_besd_ord_cat] beta_besd_ord;

vector[N_age] beta_age_raw2 = append_row(beta_age_raw, -sum(beta_age_raw));
vector[N_edu] beta_edu_raw2 = append_row(beta_edu_raw, -sum(beta_edu_raw));
vector[N_eth] beta_eth_raw2 = append_row(beta_eth_raw, -sum(beta_eth_raw));
vector[N_lan] beta_lan_raw2 = append_row(beta_lan_raw, -sum(beta_lan_raw));
vector[N_rel] beta_rel_raw2 = append_row(beta_rel_raw, -sum(beta_rel_raw));
vector[N_emp] beta_emp_raw2 = append_row(beta_emp_raw, -sum(beta_emp_raw));
vector[N_reg] beta_reg_raw2 = append_row(beta_reg_raw, -sum(beta_reg_raw));

vector[N_dis] beta_dis_raw2 = append_row(beta_dis_raw, -sum(beta_dis_raw));
vector[N_hea] beta_hea_raw2 = append_row(beta_hea_raw, -sum(beta_hea_raw));
vector[N_yuk] beta_yuk_raw2 = append_row(beta_yuk_raw, -sum(beta_yuk_raw));

vector[N_besd6]  beta_besd6_raw2 = append_row(beta_besd6_raw, -sum(beta_besd6_raw));
vector[N_besd7]  beta_besd7_raw2 = append_row(beta_besd7_raw, -sum(beta_besd7_raw));

vector[N_age] beta_age = sigma_age * beta_age_raw2;
vector[N_edu] beta_edu = sigma_edu * beta_edu_raw2;
vector[N_eth] beta_eth = sigma_eth * beta_eth_raw2;
vector[N_lan] beta_lan = sigma_lan * beta_lan_raw2;
vector[N_rel] beta_rel = sigma_rel * beta_rel_raw2;
vector[N_emp] beta_emp = sigma_emp * beta_emp_raw2;
vector[N_reg] beta_reg = sigma_reg * beta_reg_raw2;

vector[N_dis] beta_dis = sigma_dis * beta_dis_raw2;
vector[N_hea] beta_hea = sigma_hea * beta_hea_raw2;
vector[N_yuk] beta_yuk = sigma_yuk * beta_yuk_raw2;

vector[N_besd6]  beta_besd6 = sigma_besd6 * beta_besd6_raw2;
vector[N_besd7]  beta_besd7 = sigma_besd7 * beta_besd7_raw2;

for (b in 1:N_besd_ord){
	beta_besd_ord[b] = ordered_pred( alpha_besd_ord[b], gamma_besd_ord[b] );
}
}

model {         
  vector[N] lin_pred = beta_age[age] + [beta_sex, -beta_sex][sex]' + beta_edu[edu] + beta_eth[eth] + beta_lan[lan] + beta_rel[rel] + beta_emp[emp] + beta_reg[reg]
                       + beta_dis[dis] + beta_hea[hea] + beta_yuk[yuk]
                       + [beta_mig,-beta_mig][mig]' + [beta_hhlan,-beta_hhlan][hhlan]';
                       
   for (b in 1:N_besd_ord){
   	lin_pred = lin_pred + beta_besd_ord[b][besd_ord[b]];
   }
   
   for (b in 1:N_besd_ber){
   	real beta_b = beta_besd_ber[b];
   	lin_pred = lin_pred + [beta_b, -beta_b][besd_ber[b]]';
   }
  
  lin_pred = lin_pred + beta_besd6[besd6] + beta_besd7[besd7];  //+ beta_besd16[besd16] + beta_besd18[besd18];
  
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
beta_yuk_raw2 ~ normal(0, 1);
  
  
  beta_sex ~ normal(0,2);
  beta_mig ~ normal(0,2);
  beta_hhlan ~ normal(0,2);
  
  for (b in 1:N_besd_ord){
  	gamma_besd_ord ~ normal(0, 2);
  	alpha_besd_ord ~ dirichlet( rep_vector(1, num_besd_ord_cat - 1) ); //normal(0, 1);
  }
  
  for (b in 1:N_besd_ber){
  	beta_besd_ber ~ normal(0, 2);
  }
  
  beta_besd6_raw2 ~ normal(0, 1);
  beta_besd7_raw2 ~ normal(0, 1);

  { sigma_age, sigma_edu, sigma_eth, sigma_lan, sigma_rel, sigma_emp, sigma_reg, sigma_dis, sigma_hea, sigma_yuk } ~ normal(0, 1);
{ sigma_besd6, sigma_besd7 } ~ normal(0, 1);

}

