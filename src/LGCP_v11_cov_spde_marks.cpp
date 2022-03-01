#include <TMB.hpp>

template <class Type>
Type objective_function<Type>::operator()()
{
  
  using namespace R_inla;
  using namespace density;
  using namespace Eigen;
  
  // ------------------------------------------------------------------------ //
  // Spatial field data
  // ------------------------------------------------------------------------ //
  
  // The A matrices are for projecting the mesh to a point for the pixel and point data respectively.
  DATA_SPARSE_MATRIX(Apixel);
  DATA_STRUCT(spde, spde_t);
  
  // Shape data. Cases and region id.
  DATA_VECTOR(y); // point count in each pixel
  DATA_VECTOR(A); //area of pixel
  DATA_VECTOR(marks); // mark of points
  DATA_VECTOR(weight); // weight of mark liklelihood
  DATA_VECTOR(gal1); //covariate 1 stack
  DATA_VECTOR(gal2); //covaraite 2 stack
  
  // Point process parameters
  PARAMETER(intercept);
  PARAMETER(beta1);
  PARAMETER(beta2);
  PARAMETER(log_R1);
  PARAMETER(log_a1);
  PARAMETER(log_R2);
  PARAMETER(log_a2);
  Type a1 = exp(log_a1);
  Type a2 = exp(log_a2);
  
  DATA_SCALAR(priormean_intercept);
  DATA_SCALAR(priorsd_intercept);
  DATA_SCALAR(priormean_beta1);
  DATA_SCALAR(priorsd_beta1);
  DATA_SCALAR(priormean_beta2);
  DATA_SCALAR(priorsd_beta2);
  DATA_SCALAR(priormean_R1);
  DATA_SCALAR(priorsd_R1);
  DATA_SCALAR(priormean_R2);
  DATA_SCALAR(priorsd_R2);
  
  // mark process parameters
  PARAMETER(intercept_m);
  PARAMETER(beta1_m);
  PARAMETER(beta2_m);
  PARAMETER(log_R1_m);
  PARAMETER(log_a1_m);
  PARAMETER(log_R2_m);
  PARAMETER(log_a2_m);
  PARAMETER(alpha);
  Type a1_m = exp(log_a1_m);
  Type a2_m = exp(log_a2_m);
  
  DATA_SCALAR(priormean_intercept_m);
  DATA_SCALAR(priorsd_intercept_m);
  DATA_SCALAR(priormean_beta1_m);
  DATA_SCALAR(priorsd_beta1_m);
  DATA_SCALAR(priormean_beta2_m);
  DATA_SCALAR(priorsd_beta2_m);
  DATA_SCALAR(priormean_R1_m);
  DATA_SCALAR(priorsd_R1_m);
  DATA_SCALAR(priormean_R2_m);
  DATA_SCALAR(priorsd_R2_m);
  DATA_SCALAR(priormean_alpha);
  DATA_SCALAR(priorsd_alpha);
  
  // spde hyperparameters
  PARAMETER(log_sigma);
  PARAMETER(log_rho);
  Type sigma = exp(log_sigma);
  Type rho = exp(log_rho);
  
  // Priors on spde hyperparameters
  DATA_SCALAR(prior_rho_min);
  DATA_SCALAR(prior_rho_prob);
  DATA_SCALAR(prior_sigma_max);
  DATA_SCALAR(prior_sigma_prob);
  
  // Convert hyperparameters to natural scale
  DATA_SCALAR(nu);
  Type kappa = sqrt(8.0) / rho;
  
  // Random effect parameters
  PARAMETER_VECTOR(nodemean);
  
  int n_pixels = y.size();
  int n_points = marks.size();
  
  Type nll = 0.0;
  
  // ------------------------------------------------------------------------ //
  // Likelihood from priors
  // ------------------------------------------------------------------------ //
  nll -= dnorm(alpha, priormean_alpha, priorsd_alpha, true);
  
  nll -= dnorm(intercept, priormean_intercept, priorsd_intercept, true);
  nll -= dnorm(beta1, priormean_beta1, priorsd_beta1, true);
  nll -= dnorm(beta2, priormean_beta2, priorsd_beta2, true);
  nll -= dnorm(log_R1, priormean_R1, priorsd_R1, true);
  nll -= dnorm(log_R2, priormean_R2, priorsd_R2, true);
  nll -= CppAD::CondExpGe(a1,Type(0), log(1.0)-1.0*a1, Type(-INFINITY)) + log_a1;
  nll -= CppAD::CondExpGe(a2,Type(0), log(1.0)-1.0*a2, Type(-INFINITY)) + log_a2;
  
  nll -= dnorm(intercept_m, priormean_intercept_m, priorsd_intercept_m, true);
  nll -= dnorm(beta1_m, priormean_beta1_m, priorsd_beta1_m, true);
  nll -= dnorm(beta2_m, priormean_beta2_m, priorsd_beta2_m, true);
  nll -= dnorm(log_R1_m, priormean_R1_m, priorsd_R1_m, true);
  nll -= dnorm(log_R2_m, priormean_R2_m, priorsd_R2_m, true);
  nll -= CppAD::CondExpGe(a1_m,Type(0), log(1.0)-1.0*a1_m, Type(-INFINITY)) + log_a1_m;
  nll -= CppAD::CondExpGe(a2_m,Type(0), log(1.0)-1.0*a2_m, Type(-INFINITY)) + log_a2_m;
  // if(field) {
  // Likelihood of hyperparameters for field. 
  // From https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1415907 (Theorem 2.6)
  Type lambdatilde1 = -log(prior_rho_prob) * prior_rho_min;
  Type lambdatilde2 = -log(prior_sigma_prob) / prior_sigma_max;
  Type log_pcdensity = log(lambdatilde1) + log(lambdatilde2) - 2*log_rho - lambdatilde1 * pow(rho, -1) - lambdatilde2 * sigma;
  // log_rho and log_sigma from the Jacobian
  nll -= log_pcdensity + log_rho + log_sigma;
  
  // Build spde matrix
  SparseMatrix<Type> Q = Q_spde(spde, kappa);
  
  // From Lindgren (2011) https://doi.org/10.1111/j.1467-9868.2011.00777.x, see equation for the marginal variance
  Type scaling_factor = sqrt(exp(lgamma(nu)) / (exp(lgamma(nu + 1)) * 4 * M_PI * pow(kappa, 2*nu)));
  
  // Likelihood of the random field.
  nll += SCALE(GMRF(Q), sigma / scaling_factor)(nodemean);
  //}
  
  Type nll_priors = nll;
  
  // ------------------------------------------------------------------------ //
  // Likelihood from data
  // ------------------------------------------------------------------------ //
  
  vector<Type> pixel_linear_pred(n_pixels);
  for(int i = 0; i < n_pixels; i++) {
    pixel_linear_pred[i] = intercept + beta1 * exp(-pow(gal1[i]/pow(exp(log_R1), 2.0), a1)) +  beta2 * exp(-pow(gal2[i]/pow(exp(log_R2), 2.0), a2));
  }
  
  vector<Type> linear_pred_field(n_pixels);
  linear_pred_field = Apixel * nodemean;
  pixel_linear_pred += linear_pred_field;
  
  vector<Type> pixel_pred(n_pixels);
  pixel_pred = exp(pixel_linear_pred);
  
  for(int i = 0; i < n_pixels; i++){
    nll -= y[i]*pixel_linear_pred[i] - A[i]*pixel_pred[i];
  }
  
  vector<Type> linear_pred_mark(n_points);
  int j;
  for(int i = 0; i < n_points; i++) {
    j = n_pixels - n_points + i;
    linear_pred_mark[i] = intercept_m + beta1_m * exp(-pow(gal1[j]/pow(exp(log_R1_m), 2.0), a1_m)) +  beta2_m * exp(-pow(gal2[j]/pow(exp(log_R2_m), 2.0), a2_m)) + 1 / alpha * linear_pred_field[j];
  }
  for(int i = 0; i < n_points; i++){
    nll -= weight[i] * (linear_pred_mark[i] - exp(linear_pred_mark[i])*marks[i]);
  }
  
  REPORT(rho);
  REPORT(sigma);
  REPORT(intercept);
  REPORT(beta1);
  REPORT(beta2);
  REPORT(log_R1);
  REPORT(log_R2);
  REPORT(a1);
  REPORT(a2);
  REPORT(intercept_m);
  REPORT(beta1_m);
  REPORT(beta2_m);
  REPORT(log_R1_m);
  REPORT(log_R2_m);
  REPORT(a1_m);
  REPORT(a2_m);
  REPORT(alpha);
  REPORT(nodemean);
  REPORT(nll_priors);
  REPORT(nll);
  // if(family == 0) {
  //   REPORT(reportpolygonsd);
  // }
  // 
  return nll;
}