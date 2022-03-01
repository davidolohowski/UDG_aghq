#include <TMB.hpp>                                // Links in the TMB libraries
//#include <fenv.h>

template<class Type>
Type objective_function<Type>::operator() ()
{
  using namespace density;
  // Data
  DATA_VECTOR(y); // Point counts in each pixel
  DATA_VECTOR(m); // Marks of each point
  DATA_IVECTOR(ref_loc); // Pixel index of where each point is
  DATA_SCALAR(weight); // Weight of mark likleihood
  DATA_SCALAR(A); // Grid area
  // DATA_SPARSE_MATRIX(design1); // eta1 = design1 * W for point process
  // DATA_SPARSE_MATRIX(design2); // eta2 = design2 * W for marks
  //DATA_SCALAR(nu); // Matern shape
  //DATA_SCALAR(rho_0);
  //DATA_SCALAR(rho_alpha);
  //DATA_SCALAR(sigma_0);
  //DATA_SCALAR(sigma_alpha);
  DATA_SCALAR(fixed_prec);
  
  DATA_MATRIX(D); // Distance matrix for calculating matern
  
  // Params
  PARAMETER_VECTOR(W); // W = (U, beta, beta', alpha);
  PARAMETER(logprec); // Nugget effect of mark process
  PARAMETER(logrho); // Transformed matern params
  PARAMETER(logsigma);
  
  // Constants
  int n = y.size();
  int np = ref_loc.size();
  double pi = 3.141592653589793115998;
  
  // rho, sigma and prec
  Type prec = exp(logprec);
  Type sigma = exp(logsigma);
  Type rho = exp(logrho);
  // Type rho = sqrt(8.0*nu) / kappa;
  // Type sigma = tau / ( pow(kappa,nu) * sqrt( exp(lgamma(nu + 1.0)) * (4.0*pi) / exp(lgamma(nu)) ) );
  
  // Log posterior
  Type lp = 0.0;
  // Prior for sigma,rho. with dimension 2 fixed, formula simplifies
  // the logkappa + logtau at the end is the jacobian
  
  Type lpt = -sigma - 0.005 * rho + log(0.005) + logsigma + logrho ;
  lp += lpt;
  
  // Prior for W
  // N(0,Matern())
  // From documentation: https://kaskr.github.io/adcomp/matern_8cpp-example.html
  // Incorporate the sqrt(8nu) difference...
  matrix<Type> C(D);
  for(int i=0; i<C.rows(); i++)
    for(int j=0; j<C.cols(); j++)
      C(i,j) = sigma * exp(-D(i,j)/rho);
  // Now split up the parameter vector
  // W[1:n] = zeroprob part
  // W[n+1] = beta for zeroprob part
  // W[(n+2):(2*n+1)] = infectprob part
  // W[2*n+2] = beta for infectprob part
  // UPDATE: this is true for the full model but only because ncol(design) = 2*n + 2.
  // Need to use ncol(design)
  int Wd = W.size() - 3;;
  vector<Type> U(Wd);
  for (int i=0;i<Wd;i++) U(i) = W(i);
  Type beta = W(Wd);
  Type beta_p = W(Wd + 1);
  Type alpha = W(Wd + 2);
  
  // Now add the priors
  Type nll = density::MVNORM_t<Type>(C)(U);
  lp -= nll; // Negative prior (?)
  
  // Part for beta, beta', and alpha
  Type fixed_part = -1.5 * log(2.0 * pi) + 1.5 * log(fixed_prec) - fixed_prec * 0.5 * (pow(beta,2.0) + pow(beta_p, 2.0) + pow(alpha, 2.0));
  lp += fixed_part;
  
  Type prec_part = dgamma(prec, 1, 5e-05, true);
  lp += prec_part;
  
  // eta for point process
  vector<Type> eta1(n);
  for (int i=0; i < n; i++) eta1[i] = beta + U[i];
  
  // eta for marks
  vector<Type> eta2(np);
  for (int i=0; i < np; i++) eta2[i] = beta_p + alpha * U[ref_loc[i]];
  
  // Log likelihood
  for (int i = 0;i < n;i++) {
    lp += y[i]*eta1[i] - A*exp(eta1[i]);
  }
  
  for (int i = 0; i < np; i++) {
    lp += - weight * 0.5 * prec * pow(m[i] - eta2[i], 2.0) + 0.5 * weight * log(prec) - 0.5 * weight * log(2.0 * pi);
  }
  
  REPORT(rho);
  REPORT(sigma);
  REPORT(prec);
  REPORT(eta1);
  REPORT(eta2);
  REPORT(C);
  REPORT(nll);
  REPORT(fixed_part);
  REPORT(prec_part);
  REPORT(U);
  REPORT(beta);
  REPORT(beta_p);
  REPORT(alpha);
  REPORT(lpt);
  
  // Return NEGATED log posterior
  return -1.0 * lp;
}
