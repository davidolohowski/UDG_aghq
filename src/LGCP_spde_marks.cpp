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
  DATA_VECTOR(y); // point count
  DATA_VECTOR(marks); //marks of points
  DATA_VECTOR(A); //area of pixel
  DATA_VECTOR(weight); //weight of mark likelihood
  
  PARAMETER(intercept);
  PARAMETER(intercept_m);
  PARAMETER(alpha);
  
  DATA_SCALAR(priormean_intercept);
  DATA_SCALAR(priorsd_intercept);
  DATA_SCALAR(priormean_intercept_m);
  DATA_SCALAR(priorsd_intercept_m);
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
  
  nll -= dnorm(intercept, priormean_intercept, priorsd_intercept, true);
  nll -= dnorm(intercept_m, priormean_intercept_m, priorsd_intercept_m, true);
  nll -= dnorm(alpha, priormean_alpha, priorsd_alpha, true);
  
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
  
  // point porcess likelihood
  vector<Type> pixel_linear_pred(n_pixels);
  for(int i = 0; i < n_pixels; i++) pixel_linear_pred[i] = intercept;
  
  vector<Type> linear_pred_field(n_pixels);
  linear_pred_field = Apixel * nodemean;
  pixel_linear_pred += linear_pred_field;
  
  vector<Type> pixel_pred(n_pixels);
  pixel_pred = exp(pixel_linear_pred);
  
  for(int i = 0; i < n_pixels; i++){
    nll -= y[i]*pixel_linear_pred[i] - A[i]*pixel_pred[i];
  }
  
  //mark likelihood
  vector<Type> linear_pred_mark(n_points);
  for(int i = 0; i < n_points; i++) linear_pred_mark[i] = intercept_m + 1/alpha * linear_pred_field[n_pixels - n_points + i];
    
  for(int i = 0; i < n_points; i++){
      nll -=  weight[i] * (linear_pred_mark[i] - exp(linear_pred_mark[i])*marks[i]);
  }
  
  REPORT(rho);
  REPORT(sigma);
  REPORT(intercept);
  REPORT(nodemean);
  REPORT(nll_priors);
  REPORT(nll);
  REPORT(intercept_m);
  REPORT(alpha);

  return nll;
}