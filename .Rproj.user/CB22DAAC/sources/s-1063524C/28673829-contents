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
  DATA_MATRIX(gal); // covariate matrix for normal galaxies
  
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
  
  Type nll = 0.0;
  
  // ------------------------------------------------------------------------ //
  // Likelihood from priors
  // ------------------------------------------------------------------------ //
  nll -= dnorm(intercept, priormean_intercept, priorsd_intercept, true);
  for(int i = 0; i < Ng; i++){
    nll -= dnorm(beta(i), priormean_beta(i), priorsd_beta(i), true);
    nll -= dnorm(log_R(i), priormean_R(i), priorsd_R(i), true);
    nll -= CppAD::CondExpGe(a(i),Type(0), log(1.0)-1.0*a(i), Type(-INFINITY)) + log_a(i);
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
  
  REPORT(rho);
  REPORT(sigma);
  REPORT(intercept);
  REPORT(beta);
  REPORT(log_R);
  REPORT(a);
  REPORT(nodemean);
  REPORT(nll_priors);
  REPORT(nll);
  // if(family == 0) {
  //   REPORT(reportpolygonsd);
  // }
  // 
  return nll;
}