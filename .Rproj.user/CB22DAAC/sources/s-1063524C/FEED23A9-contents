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
  DATA_SPARSE_MATRIX(Apixel_m);
  DATA_STRUCT(spde, spde_t);
  
  // Shape data. Cases and region id.
  DATA_VECTOR(y); // point count in each pixel
  DATA_VECTOR(A); //area of pixel
  DATA_VECTOR(marks); // mark of points
  DATA_MATRIX(gal); //covariate matrix for normal galaxies
  
  int Ng = gal.cols(); //number of normal galaxies
  
  // Point process parameters
  PARAMETER(intercept);
  PARAMETER_VECTOR(beta);
  PARAMETER_VECTOR(log_R);
  PARAMETER_VECTOR(log_a);
  vector<Type> a(Ng);
  for(int i = 0; i < Ng; i++){
    a(i) = exp(log_a(i));
  }
  
  DATA_SCALAR(priormean_intercept);
  DATA_SCALAR(priorsd_intercept);
  DATA_VECTOR(priormean_beta);
  DATA_VECTOR(priorsd_beta);
  DATA_VECTOR(priormean_R);
  DATA_VECTOR(priorsd_R);
  
  PARAMETER(intercept_m); // intercept of marks
  
  DATA_SCALAR(priormean_intercept_m);
  DATA_SCALAR(priorsd_intercept_m);
  
  // mark process parameters
  PARAMETER(intercept_v);
  PARAMETER_VECTOR(beta_v);
  PARAMETER_VECTOR(log_R_v);
  PARAMETER_VECTOR(log_a_v);
  PARAMETER(alpha);
  vector<Type> a_v(Ng);
  for(int i = 0; i < Ng; i++){
    a_v(i) = exp(log_a_v(i));
  }
  
  DATA_SCALAR(priormean_intercept_v);
  DATA_SCALAR(priorsd_intercept_v);
  DATA_VECTOR(priormean_beta_v);
  DATA_VECTOR(priorsd_beta_v);
  DATA_VECTOR(priormean_R_v);
  DATA_VECTOR(priorsd_R_v);
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
  
  // spde hyperparameters for color
  PARAMETER(log_sigma_m);
  PARAMETER(log_rho_m);
  Type sigma_m = exp(log_sigma_m);
  Type rho_m = exp(log_rho_m);
  
  // Priors on spde hyperparameters
  DATA_SCALAR(prior_rho_min_m);
  DATA_SCALAR(prior_rho_prob_m);
  DATA_SCALAR(prior_sigma_max_m);
  DATA_SCALAR(prior_sigma_prob_m);
  
  // Convert hyperparameters to natural scale
  DATA_SCALAR(nu);
  Type kappa = sqrt(8.0) / rho;
  Type kappa_m = sqrt(8.0) / rho_m;
  
  // Random effect parameters
  PARAMETER_VECTOR(nodemean);
  // Random effect parameters for color
  PARAMETER_VECTOR(nodemean_m);
  
  int n_pixels = y.size();
  int n_points = marks.size();
  
  Type nll = 0.0;
  
  // ------------------------------------------------------------------------ //
  // Likelihood from priors
  // ------------------------------------------------------------------------ //
  nll -= dnorm(alpha, priormean_alpha, priorsd_alpha, true);
  nll -= dnorm(intercept, priormean_intercept, priorsd_intercept, true);
  nll -= dnorm(intercept_m, priormean_intercept_m, priorsd_intercept_m, true);
  nll -= dnorm(intercept_v, priormean_intercept_v, priorsd_intercept_v, true);
  for(int i = 0; i < Ng; i++){
    nll -= dnorm(beta(i), priormean_beta(i), priorsd_beta(i), true);
    nll -= dnorm(log_R(i), priormean_R(i), priorsd_R(i), true);
    nll -= CppAD::CondExpGe(a(i),Type(0), log(1.0)-1.0*a(i), Type(-INFINITY)) + log_a(i);
    nll -= dnorm(beta_v(i), priormean_beta_v(i), priorsd_beta_v(i), true);
    nll -= dnorm(log_R_v(i), priormean_R_v(i), priorsd_R_v(i), true);
    nll -= CppAD::CondExpGe(a_v(i),Type(0), log(1.0)-1.0*a_v(i), Type(-INFINITY)) + log_a_v(i);
  }
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
  
  Type lambdatilde1_m = -log(prior_rho_prob_m) * prior_rho_min_m;
  Type lambdatilde2_m = -log(prior_sigma_prob_m) / prior_sigma_max_m;
  Type log_pcdensity_m = log(lambdatilde1_m) + log(lambdatilde2_m) - 2*log_rho_m - lambdatilde1_m * pow(rho_m, -1) - lambdatilde2_m * sigma_m;
  // log_rho and log_sigma from the Jacobian
  nll -= log_pcdensity_m + log_rho_m + log_sigma_m;
  
  // Build spde matrix
  SparseMatrix<Type> Q_m = Q_spde(spde, kappa_m);
  
  // From Lindgren (2011) https://doi.org/10.1111/j.1467-9868.2011.00777.x, see equation for the marginal variance
  Type scaling_factor_m = sqrt(exp(lgamma(nu)) / (exp(lgamma(nu + 1)) * 4 * M_PI * pow(kappa_m, 2*nu)));
  
  // Likelihood of the random field.
  nll += SCALE(GMRF(Q_m), sigma_m / scaling_factor_m)(nodemean_m);
  //}
  
  Type nll_priors = nll;
  
  // ------------------------------------------------------------------------ //
  // Likelihood from data
  // ------------------------------------------------------------------------ //
  
  vector<Type> pixel_linear_pred(n_pixels);
  for(int i = 0; i < n_pixels; i++) {
    pixel_linear_pred[i] = intercept;
    for(int j = 0; j < Ng; j++) {
      pixel_linear_pred[i] += beta(j) * exp(-pow(gal(i,j)/pow(exp(log_R(j)), 2.0), a(j)));
    }
  }
  
  vector<Type> linear_pred_field(n_pixels);
  linear_pred_field = Apixel * nodemean;
  pixel_linear_pred += linear_pred_field;
  
  vector<Type> linear_pred_marks(n_points);
  linear_pred_marks = Apixel_m * nodemean_m;
  
  vector<Type> pixel_pred(n_pixels);
  pixel_pred = exp(pixel_linear_pred);
  
  for(int i = 0; i < n_pixels; i++){
    nll -= y[i]*pixel_linear_pred[i] - A[i]*pixel_pred[i];
  }
  
  vector<Type> sd_pred_marks(n_points);
  int j;
  for(int i = 0; i < n_points; i++) {
    j = n_pixels - n_points + i;
    sd_pred_marks[i] = intercept_v + 1 / alpha * linear_pred_field[j];
    for(int k = 0; k < Ng; k++) {
      sd_pred_marks[i] += beta_v(k) * exp(-pow(gal(j,k)/pow(exp(log_R_v(k)), 2.0), a_v(k)));
    }
  }
  for(int i = 0; i < n_points; i++){
    nll -= dnorm(marks[i], intercept_m + linear_pred_marks[i], exp(-sd_pred_marks[i]), true);
  }
  
  REPORT(rho);
  REPORT(sigma);
  REPORT(rho_m);
  REPORT(sigma_m);
  REPORT(intercept);
  REPORT(beta);
  REPORT(log_R);
  REPORT(a);
  REPORT(intercept_v);
  REPORT(beta_v);
  REPORT(log_R_v);
  REPORT(a_v);
  REPORT(intercept_m);
  REPORT(alpha);
  REPORT(nodemean);
  REPORT(nodemean_m);
  REPORT(nll_priors);
  REPORT(nll);
  // if(family == 0) {
  //   REPORT(reportpolygonsd);
  // }
  // 
  return nll;
}