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
  DATA_VECTOR(p); // probability that a GC is an outlier
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
  
  // mark process parameters
  PARAMETER(intercept_m);
  PARAMETER_VECTOR(beta_m);
  PARAMETER_VECTOR(log_R_m);
  PARAMETER_VECTOR(log_a_m);
  PARAMETER(alpha);
  vector<Type> a_m(Ng);
  for(int i = 0; i < Ng; i++){
    a_m(i) = exp(log_a_m(i));
  }
  
  DATA_SCALAR(priormean_intercept_m);
  DATA_SCALAR(priorsd_intercept_m);
  DATA_VECTOR(priormean_beta_m);
  DATA_VECTOR(priorsd_beta_m);
  DATA_VECTOR(priormean_R_m);
  DATA_VECTOR(priorsd_R_m);
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
  nll -= dnorm(intercept_m, priormean_intercept_m, priorsd_intercept_m, true);
  for(int i = 0; i < Ng; i++){
    nll -= dnorm(beta(i), priormean_beta(i), priorsd_beta(i), true);
    nll -= dnorm(log_R(i), priormean_R(i), priorsd_R(i), true);
    nll -= CppAD::CondExpGe(a(i),Type(0), log(1.0)-1.0*a(i), Type(-INFINITY)) + log_a(i);
    nll -= dnorm(beta_m(i), priormean_beta_m(i), priorsd_beta_m(i), true);
    nll -= dnorm(log_R_m(i), priormean_R_m(i), priorsd_R_m(i), true);
    nll -= CppAD::CondExpGe(a_m(i),Type(0), log(1.0)-1.0*a_m(i), Type(-INFINITY)) + log_a_m(i);
  }
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
    pixel_linear_pred[i] = intercept;
    for(int j = 0; j < Ng; j++) {
      pixel_linear_pred[i] += beta(j) * exp(-pow(gal(i,j)/pow(exp(log_R(j)), 2.0), a(j)));
    }
  }
  
  vector<Type> linear_pred_field(n_pixels);
  linear_pred_field = Apixel * nodemean;
  pixel_linear_pred += linear_pred_field;
  
  vector<Type> pixel_pred(n_pixels);
  pixel_pred = exp(pixel_linear_pred);
  
  for(int i = 0; i < n_pixels; i++){
    nll -= y[i]*pixel_linear_pred[i] - A[i]*pixel_pred[i];
  }
  
  vector<Type> linear_pred_mark_no(n_points);
  vector<Type> linear_pred_mark_o(n_points);
  int j;
  for(int i = 0; i < n_points; i++) {
    j = n_pixels - n_points + i;
    linear_pred_mark_no[i] = intercept_m + 1 / alpha * linear_pred_field[j];
    linear_pred_mark_o[i] = intercept_m;
    for(int k = 0; k < Ng; k++) {
      linear_pred_mark_no[i] += beta_m(k) * exp(-pow(gal(j,k)/pow(exp(log_R_m(k)), 2.0), a_m(k)));
      linear_pred_mark_o[i] += beta_m(k) * exp(-pow(gal(j,k)/pow(exp(log_R_m(k)), 2.0), a_m(k)));
    }
  }
  for(int i = 0; i < n_points; i++){
    nll -= log((1-p[i])*exp(linear_pred_mark_no[i])*exp(-pow(marks[i]*linear_pred_mark_no[i], 2.0)*0.5) + p[i]*exp(linear_pred_mark_o[i])*exp(-pow(marks[i]*linear_pred_mark_o[i], 2.0)*0.5)); //(linear_pred_mark[i] - exp(linear_pred_mark[i])*marks[i]);
  }
  
  REPORT(rho);
  REPORT(sigma);
  REPORT(intercept);
  REPORT(beta);
  REPORT(log_R);
  REPORT(a);
  REPORT(intercept_m);
  REPORT(beta_m);
  REPORT(log_R_m);
  REPORT(a_m);
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