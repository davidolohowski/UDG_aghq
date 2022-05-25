library(TMB)
precompile()
library(geostatsp)
library(aghq)
library(tmbstan)
library(tidyverse)
library(spatstat)
library(spatial)
library(sp)
library(sf)
library(maptools)
library(raster)
library(fields)
library(viridis)
library(rstan)
library(INLA)
library(adehabitatMA)
library(rgeos)
library(plotly)
library(inlabru)
library(excursions)

# Compile TMB template-- only need to do once
compile('src/LGCP_cov_spde_marks.cpp')
source('src/inla_mesh_dual.R')
dyn.load(dynlib(file.path("src/LGCP_cov_spde")))
dyn.load(dynlib(file.path("src/LGCP_cov_spde_marks")))

# ================================================================ Data pre-processing and Preparation ======================================================================= #
v_acs <- read_csv('data/v11acs_GC.csv')
GC_color <- readRDS('data/v11_acs_col.RDS')
X <- c(1, 4095, 4214, 219)
Y <- c(1, 100.5, 4300, 4044)
region <- Polygon(cbind(X,Y))
region <- SpatialPolygons(list(Polygons(list(region),'region')))
region <- SpatialPolygonsDataFrame(region, data.frame(id = region@polygons[[1]]@ID, row.names = region@polygons[[1]]@ID))
plot(region)
points(v_acs[,3:4])

spv <- SpatialPoints(v_acs[,3:4])

# construct spde mesh
v_mesh <- inla.mesh.2d(loc = spv, boundary = region,
                       max.edge = c(80, 800), cutoff = 40, max.n.strict = 1300L)

# dual mesh for the spde mesh nodes to get the exposure weights
dmesh <- inla.mesh.dual(v_mesh)

nv <- v_mesh$n

domain.polys <- Polygons(list(Polygon(cbind(X,Y))), '0')
domainSP <- SpatialPolygons(list(domain.polys))

# exposure weights
w <- sapply(1:length(dmesh), function(i) {
  if (gIntersects(dmesh[i, ], domainSP))
    return(gArea(gIntersection(dmesh[i, ], domainSP)))
  else return(0)
})

# response vector
y.pp <- rep(0:1, c(nv, nrow(spv@coords)))
# exposure vector
e.pp <- c(w, rep(0, nrow(spv@coords))) 

# mesh nodes locations
loc_mesh <- v_mesh$loc[,1:2]

colnames(loc_mesh) <- c('x','y')

imat <- inla.spde.make.A(v_mesh, loc = v_mesh$loc[,1:2])

lmat <- inla.spde.make.A(v_mesh, loc = spv)

# projection matrix for the mesh nodes and the point locations
A.pp <- rbind(imat, lmat)

loc <- rbind(loc_mesh, v_acs[,3:4])

# covariate values at the locations
gal1 <- (loc$x*cos(pi/18) - loc$y*sin(pi/18) - 800)^2 + (loc$x*sin(pi/18) + loc$y*cos(pi/18)-1950)^2/1.733522
gal2 <- ((loc$x-2300)*cos(pi/6)- (loc$y-5800)*sin(pi/6))^2 + ((loc$x-2300)*sin(pi/6) + (loc$y-5800)*cos(pi/6))^2/2.25

gal <- cbind(gal1, gal2)
# spde stuff
nu <- 1

# spde model
spde <- (inla.spde2.pcmatern(mesh = v_mesh, alpha = 2,
                             prior.range = c(400, 0.5),
                             prior.sigma = c(1.5, 0.5))$param.inla)[c("M0", "M1", "M2")]

# random weights for the basis functions
n_s <- nrow(spde$M0)
nodemean <- rep(0, n_s)

# ============================================================== Non-marked point process model =============================================================== #
# # input data for the non-marked point process
input_data <- list(Apixel = A.pp,
                   spde = spde,
                   y = y.pp,
                   A = e.pp,
                   nu = nu,
                   gal = gal,
                   priormean_intercept = 0,
                   priorsd_intercept = sqrt(1000),
                   priormean_beta = rep(0, ncol(gal)),
                   priorsd_beta = rep(sqrt(1000), ncol(gal)),
                   priormean_R = c(6.56, 6.8),
                   priorsd_R = c(0.25, 0.2),
                   prior_rho_min = 400,
                   prior_rho_prob = 0.5,
                   prior_sigma_max = 1.5,
                   prior_sigma_prob = 0.5)

parameters <- list(intercept = -11,
                   beta = c(2, 6),
                   log_R = c(6.56, 6.8),
                   log_a = c(0,0),
                   log_sigma = log(1.5),
                   log_rho = log(400),
                   nodemean = nodemean)

obj <- MakeADFun(
  data = input_data,
  parameters = parameters,
  random = c('nodemean', 'intercept', 'beta', 'log_R', 'log_a'),
  silent = TRUE,
  DLL = "LGCP_cov_spde")

# non-marked point process
tm <- Sys.time()
cat("Doing AGHQ, time = ",format(tm),"\n")
LGCP_fit_no_marks <- aghq::marginal_laplace_tmb(
  obj,
  3,
  startingvalue = c(parameters$log_sigma, parameters$log_rho)
)
aghqtime <- difftime(Sys.time(),tm,units = 'secs')
saveRDS(LGCP_fit_no_marks,file = "LGCP_v11_cov_spde_marks/LGCP_v11_non_marked.RDS")
cat("AGHQ took: ",format(aghqtime),"\n")

# ================================================================= Fully marked model ======================================================================= #
# full marked point process
input_data <- list(Apixel = A.pp,
                   spde = spde,
                   y = y.pp,
                   A = e.pp,
                   marks = GC_color,
                   nu = nu,
                   gal = gal,
                   priormean_intercept = 0,
                   priorsd_intercept = sqrt(1000),
                   priormean_beta = rep(0, ncol(gal)),
                   priorsd_beta = rep(sqrt(1000), ncol(gal)),
                   priormean_R = c(6.56, 6.8),
                   priorsd_R = c(0.25, 0.2),
                   priormean_intercept_m = 0,
                   priorsd_intercept_m = sqrt(1000),
                   priormean_alpha = 1,
                   priorsd_alpha = sqrt(10),
                   prior_rho_min = 400,
                   prior_rho_prob = 0.5,
                   prior_sigma_max = 1.5,
                   prior_sigma_prob = 0.5)

parameters <- list(intercept = -11,
                   beta = c(2,6),
                   log_R = c(6.56,6.8),
                   log_a = c(0,0),
                   intercept_m = -1.7,
                   alpha = -1,
                   log_sigma = log(1.5),
                   log_rho = log(400),
                   nodemean = nodemean)

obj <- MakeADFun(
  data = input_data,
  parameters = parameters,
  random = c('nodemean', 'intercept', 'beta', 'log_R', 'log_a',
             'intercept_m'),
  silent = TRUE,
  DLL = "LGCP_cov_spde_marks")

tm <- Sys.time()
cat("Doing AGHQ, time = ",format(tm),"\n")
LGCP_fit_marked <- aghq::marginal_laplace_tmb(
  obj,
  3,
  startingvalue = c(parameters$alpha, parameters$log_sigma, parameters$log_rho)
)
aghqtime <- difftime(Sys.time(),tm,units = 'secs')
saveRDS(LGCP_fit_marked,file = "LGCP_v11_cov_spde_marks/LGCP_v11_marked.RDS")
cat("AGHQ took: ",format(aghqtime),"\n")

# ============================================================ Post model fitting data analysis ============================================================ #
# construct grids for the entire study region
grid <- makegrid(region, n = 50000)
grid <- SpatialPoints(grid, proj4string = CRS(proj4string(region)))
grid <- crop(grid, region)
grid <- as.data.frame(grid)
names(grid) <- c('x', 'y')
coordinates(grid) <- ~x+y
gridded(grid) <- T
df <- as.data.frame(grid)

U1 <- c(2628, 1532)
U2 <- c(2517, 3289)

U <- data.frame(x = c(U1[1],U2[1]), y = c(U1[2], U2[2]))

# projection matrix for the grids
Apix <- inla.mesh.project(v_mesh, loc = grid)$A

samps1.0 <- sample_marginal(LGCP_fit_marked, 1000)
post_node1.0 <- samps1.0$samps[9:1460,]

post_U1.0 <- exp(Apix %*% post_node1.0)


exc_prob <- data.frame()

set.seed(376823)
for (i in 1:50) {
  # samples from the non-marked model
  samps0.0 <- sample_marginal(LGCP_fit_no_marks, 1000)
  post_node0.0 <- samps0.0$samps[8:1459,]
  
  post_U0.0 <- exp(Apix %*% post_node0.0)
  
  sigma0.0 <- exp(samps0.0$thetasamples[[1]])
  
  # excursion functions for the non-marked model
  df$exc_q0.5_0.0 <- excursions.mc(post_U0.0, 0.5, 1, '>')$`F`
  df$exc_q0.75_0.0 <- excursions.mc(post_U0.0, 0.5, exp(quantile(rnorm(1000, 0, sigma0.0), 0.75)), '>')$`F`
  df$exc_q0.9_0.0 <- excursions.mc(post_U0.0, 0.5, exp(quantile(rnorm(1000, 0, sigma0.0), 0.9)), '>')$`F`
  df$exc_q0.95_0.0 <- excursions.mc(post_U0.0, 0.5, exp(quantile(rnorm(1000, 0, sigma0.0), 0.95)), '>')$`F`
  
  # samples from the fully marked model
  samps1.0 <- sample_marginal(LGCP_fit_marked, 1000)
  post_node1.0 <- samps1.0$samps[9:1460,]
  
  post_U1.0 <- exp(Apix %*% post_node1.0)
  
  sigma1.0 <- exp(samps1.0$thetasamples[[2]])
  
  # excursion functions for the fully marked model
  df$exc_q0.5_1.0 <- excursions.mc(post_U1.0, 0.5, 1, '>')$`F`
  df$exc_q0.75_1.0 <- excursions.mc(post_U1.0, 0.5, exp(quantile(rnorm(1000, 0, sigma1.0), 0.75)), '>')$`F`
  df$exc_q0.9_1.0 <- excursions.mc(post_U1.0, 0.5, exp(quantile(rnorm(1000, 0, sigma1.0), 0.9)), '>')$`F`
  df$exc_q0.95_1.0 <- excursions.mc(post_U1.0, 0.5, exp(quantile(rnorm(1000, 0, sigma1.0), 0.95)), '>')$`F`
  
  # ================================================================ Detection Comparison ========================================================================= #
  
  Fs <- df %>%
    mutate(dist1 = sqrt((x - U1[1])^2 + (y - U1[2])^2), dist2 = sqrt((x - U2[1])^2 + (y - U2[2])^2)) %>%
    dplyr::filter(dist1 < 220 | dist2 < 220) %>%
    mutate(U1 = dist1 < 220, U2 = dist2 < 220)
  
  # Detection probability for non-marked model
  Fs_0.5_U1_0.0 <- Fs %>%
    filter(U1 == T) %>%
    summarise(p = max(exc_q0.5_0.0))
  
  Fs_0.5_U2_0.0 <- Fs %>%
    filter(U2 == T) %>%
    summarise(p = max(exc_q0.5_0.0))
  
  Fs_0.75_U1_0.0 <- Fs %>%
    filter(U1 == T) %>%
    summarise(p = max(exc_q0.75_0.0))
  
  Fs_0.75_U2_0.0 <- Fs %>%
    filter(U2 == T) %>%
    summarise(p = max(exc_q0.75_0.0))
  
  Fs_0.9_U1_0.0 <- Fs %>%
    filter(U1 == T) %>%
    summarise(p = max(exc_q0.9_0.0))
  
  Fs_0.9_U2_0.0 <- Fs %>%
    filter(U2 == T) %>%
    summarise(p = max(exc_q0.9_0.0))
  
  Fs_0.95_U1_0.0 <- Fs %>%
    filter(U1 == T) %>%
    summarise(p = max(exc_q0.95_0.0))
  
  Fs_0.95_U2_0.0 <- Fs %>%
    filter(U2 == T) %>%
    summarise(p = max(exc_q0.95_0.0))
  
  # Detection probability for fully marked model
  Fs_0.5_U1_1.0 <- Fs %>%
    filter(U1 == T) %>%
    summarise(p = max(exc_q0.5_1.0))
  
  Fs_0.5_U2_1.0 <- Fs %>%
    filter(U2 == T) %>%
    summarise(p = max(exc_q0.5_1.0))
  
  Fs_0.75_U1_1.0 <- Fs %>%
    filter(U1 == T) %>%
    summarise(p = max(exc_q0.75_1.0))
  
  Fs_0.75_U2_1.0 <- Fs %>%
    filter(U2 == T) %>%
    summarise(p = max(exc_q0.75_1.0))
  
  Fs_0.9_U1_1.0 <- Fs %>%
    filter(U1 == T) %>%
    summarise(p = max(exc_q0.9_1.0))
  
  Fs_0.9_U2_1.0 <- Fs %>%
    filter(U2 == T) %>%
    summarise(p = max(exc_q0.9_1.0))
  
  Fs_0.95_U1_1.0 <- Fs %>%
    filter(U1 == T) %>%
    summarise(p = max(exc_q0.95_1.0))
  
  Fs_0.95_U2_1.0 <- Fs %>%
    filter(U2 == T) %>%
    summarise(p = max(exc_q0.95_1.0))
  
  
  # Compare detection probability between models
  exc_prob <- bind_rows(exc_prob, data.frame(prob = as.numeric(c(Fs_0.5_U1_0.0, Fs_0.75_U1_0.0, Fs_0.9_U1_0.0, Fs_0.95_U1_0.0,
                                                                 Fs_0.5_U1_1.0, Fs_0.75_U1_1.0, Fs_0.9_U1_1.0, Fs_0.95_U1_1.0,
                                                                 Fs_0.5_U2_0.0, Fs_0.75_U2_0.0, Fs_0.9_U2_0.0, Fs_0.95_U2_0.0,
                                                                 Fs_0.5_U2_1.0, Fs_0.75_U2_1.0, Fs_0.9_U2_1.0, Fs_0.95_U2_1.0)),
                                             model = rep(c('PPP', 'MPP', 'PPP', 'MPP'), each = 4),
                                             q = rep(c('0.5', '0.75', '0.9', '0.95'), 4),
                                             ID = rep(c('U1', 'U2'), each = 8)))
}

saveRDS(exc_prob, "LGCP_v11_cov_spde_marks/v11_exc_prob.RDS")

df_exc <- exc_prob %>%
  group_by(model, q, ID) %>%
  summarise(mean = median(prob),
            upper = quantile(prob, 0.75),
            lower = quantile(prob, 0.25))

ggplot(df_exc, aes(model, mean)) + geom_point(aes(color = q), position = position_dodge(0.5)) + 
  geom_errorbar(aes(ymin = lower, ymax = upper, color = q), position = position_dodge(0.5), 
                width = 0.1) + facet_wrap(~ID)

theBreaks <- c(0.9, 1, 1.025, 1.05, 1.1, 1.15, 1.2, 1.225, 1.25, 1.3, 1.35)
theCol = rev(RColorBrewer::brewer.pal(length(theBreaks)-1, 'Spectral'))

df$mean <- rowMeans(post_U1.0)

ggplot(df, aes(x, y)) +
  geom_contour_filled(aes(z = mean), breaks = theBreaks) +
  scale_fill_manual(values = theCol, name = '$\\exp(\\mathcal{U}(s))$', guide = guide_legend(reverse = T)) +
  coord_fixed() +
  geom_point(data = v_acs, aes(x,y), size = 0.1, stroke = 0.25, shape = 20) +
  xlab('X (pixels)') + ylab('Y (pixels)') +
  theme_minimal() +
  theme(legend.text = element_text(size = 7),
        legend.title = element_text(size = 8),
        strip.background = element_rect(color = NULL, fill = 'white', linetype = 'blank'))




# sigma1 <- exp(LGCP1_postsamples$thetasamples[[2]])
# 
# pal <- rev(RColorBrewer::brewer.pal(11, 'Spectral'))
# 
# ggplot(df, aes(x, y)) +
#   geom_contour_filled(aes(z = exc0.5), breaks = c(0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)) +
#   scale_fill_manual(values = pal, name = '$F(s)$', guide = guide_legend(reverse = T)) +
#   coord_fixed() +
#   geom_point(data = v_acs, aes(x,y), size = 0.1, stroke = 0.25, shape = 20) +
#   xlab('X (pixels)') + ylab('Y (pixels)') +
#   theme_minimal() +
#   theme(legend.text = element_text(size = 7),
#         legend.title = element_text(size = 8),
#         strip.background = element_rect(color = NULL, fill = 'white', linetype = 'blank'))










