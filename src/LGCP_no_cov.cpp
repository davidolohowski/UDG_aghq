#include <TMB.hpp>                                // Links in the TMB libraries
//#include <fenv.h>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(y); // Point counts in each pixel
  DATA_SCALAR(A); // Grid area
  //DATA_SPARSE_MATRIX(design); // eta = design * W
  DATA_SCALAR(nu); // Matern shape
  DATA_SCALAR(rho_u);
  DATA_SCALAR(rho_alpha);
  DATA_SCALAR(sigma_u);
  DATA_SCALAR(sigma_alpha);
  DATA_SCALAR(betaprec);
  
  DATA_MATRIX(D); // Distance matrix for calculating matern
  
  // Params
  PARAMETER_VECTOR(W); // W = (U,V,beta); eta = design * W
  PARAMETER(logkappa); // Transformed matern params
  PARAMETER(logtau);
  
  // Constants
  int n = y.size();
  double pi = 3.141592653589793115998;
  
  // rho and sigma
  Type kappa = exp(logkappa);
  Type tau = exp(logtau);
  Type rho = sqrt(8.0*nu) / kappa;
  Type sigma = tau / ( pow(kappa,nu) * sqrt( exp(lgamma(nu + 1.0)) * (4.0*pi) / exp(lgamma(nu)) ) );
  
  // Log posterior
  Type lp = 0;
  // Prior for sigma,rho. with dimension 2 fixed, formula simplifies
  // the logkappa + logtau at the end is the jacobian
  
  Type lambda1 = -1.0 * (rho_u / sqrt(8.0*nu)) * log(rho_alpha);
  Type lambda2 = ( -1.0 * pow(kappa,-1.0 * nu) * sqrt( exp(lgamma(nu))  / ( exp(lgamma(nu + 1.0)) * (4.0*pi) ) ) ) * log(sigma_alpha) / sigma_u;
  
  Type lpt = log(lambda1) + log(lambda2) - lambda1 * kappa - lambda2 * tau + logkappa + logtau;
  lp += lpt;
  
  // Prior for W
  // N(0,Matern())
  // From documentation: https://kaskr.github.io/adcomp/matern_8cpp-example.html
  // Incorporate the sqrt(8nu) difference...
  matrix<Type> C(D);
  for(int i=0; i<C.rows(); i++)
    for(int j=0; j<C.cols(); j++)
      C(i,j) = pow(sigma,2.0) * matern(D(i,j), rho / sqrt(8.0 * nu), nu);
  // Now split up the parameter vector
  // W[1:n] = zeroprob part
  // W[n+1] = beta for zeroprob part
  // W[(n+2):(2*n+1)] = infectprob part
  // W[2*n+2] = beta for infectprob part
  // UPDATE: this is true for the full model but only because ncol(design) = 2*n + 2.
  // Need to use ncol(design)
  int d = W.size() - 1;
  int Wd = d;
  vector<Type> U(Wd);
  for (int i=0;i<Wd;i++) U(i) = W(i);
  Type beta = W(Wd);
  
  // Now add the priors
  Type nll = density::MVNORM_t<Type>(C)(U);
  lp -= nll; // Negative prior (?)
  
  // Part for beta
  Type betapart = -0.5 * log(2*pi) + 0.5 * log(betaprec) - betaprec * 0.5 * (pow(beta,2.0));
  lp += betapart;
  
  // Transformations
  vector<Type> eta(n); // dim(eta) = nrow(design) = 2*n;
  for(int i=0; i < n; i++) eta(i) = beta + U(i);
  
  // Log likelihood
  for (int i = 0;i < n;i++) {
    lp += y(i)*eta(i) - A*exp(eta(i));
  }
  
  REPORT(rho);
  REPORT(sigma);
  REPORT(kappa);
  REPORT(tau);
  REPORT(eta);
  REPORT(C);
  REPORT(nll);
  REPORT(betapart);
  REPORT(U);
  REPORT(beta);
  REPORT(lpt);
  
  // Return NEGATED log posterior
  return -1.0 * lp;
}
