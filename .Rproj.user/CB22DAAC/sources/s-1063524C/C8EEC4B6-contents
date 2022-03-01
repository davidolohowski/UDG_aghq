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
library(rgeos)
library(inlabru)

source('src/inla_mesh_dual.R')
savestamp <- "20220213-v1"
# plotpath <- paste0(globalpath,"figures/loaloazip/")
globalpath <- getwd()
plotpath <- file.path(globalpath,"LGCP_spde_marks")
if (!dir.exists(plotpath)) dir.create(plotpath)
savepath <- plotpath

file.copy(system.file('LGCP_spde_marks.cpp',package='aghq'),globalpath)

# Compile TMB template-- only need to do once
compile('src/LGCP_spde_marks.cpp')
dyn.load(dynlib(file.path(globalpath,"src/LGCP_spde_marks")))

v_acs <- read_csv('data/v7acs_GC.csv')

X <- c(1, 4095, 4214, 223)
Y <- c(1, 100.5, 4300, 4044)
region <- Polygon(cbind(X,Y))
region <- SpatialPolygons(list(Polygons(list(region),'region')))
region <- SpatialPolygonsDataFrame(region, data.frame(id = region@polygons[[1]]@ID, row.names = region@polygons[[1]]@ID))
plot(region)
points(v_acs[,3:4])

#no normal galaxies in this field or in its surroudning, hence no covariates.

spv <- SpatialPoints(v_acs[,3:4])

v_mesh <- inla.mesh.2d(loc = spv, boundary = region,
                       max.edge = c(80, 800), cutoff = 40, max.n.strict = 1300L)

plot(v_mesh, asp = 1)
plot(region, add = T)

dmesh <- inla.mesh.dual(v_mesh)

plot(dmesh)

nv <- v_mesh$n

domain.polys <- Polygons(list(Polygon(cbind(X,Y))), '0')
domainSP <- SpatialPolygons(list(domain.polys))

plot(domainSP)

w <- sapply(1:length(dmesh), function(i) {
  if (gIntersects(dmesh[i, ], domainSP))
    return(gArea(gIntersection(dmesh[i, ], domainSP)))
  else return(0)
})

y.pp <- rep(0:1, c(nv, nrow(spv@coords)))
e.pp <- c(w, rep(0, nrow(spv@coords))) 

loc.w <- matrix(0, nv, 2)

for (i in 1:length(dmesh)) {
  loc.w[i,] <- dmesh@polygons[[i]]@labpt
}

loc_mesh <- v_mesh$loc[,1:2]

colnames(loc_mesh) <- c('x','y')

imat <- inla.spde.make.A(v_mesh, loc = v_mesh$loc[,1:2])

lmat <- inla.spde.make.A(v_mesh, loc = spv)

A.pp <- rbind(imat, lmat)


# color
GC_color <- -readRDS('data/v7_acs_col.RDS')

# spde stuff
nu <- 1
# Sort out mesh bits
spde <- (inla.spde2.pcmatern(v_mesh, alpha = nu + 1,
                             prior.range = c(400, 0.5),
                             prior.sigma = c(1.5, 0.5))$param.inla)[c("M0", "M1", "M2")]

n_s <- nrow(spde$M0)
nodemean <- rep(0, n_s)

input_data <- list(Apixel = A.pp,
                   spde = spde,
                   y = y.pp,
                   A = e.pp,
                   nu = nu,
                   marks = GC_color,
                   weight = rep(1, length(GC_color)),
                   priormean_intercept = 0.0,
                   priorsd_intercept = sqrt(1000),
                   priormean_intercept_m = 0.0,
                   priorsd_intercept_m = sqrt(1000),
                   priormean_alpha = 1,
                   priorsd_alpha = sqrt(10),
                   prior_rho_min = 1000,
                   prior_rho_prob = 0.5,
                   prior_sigma_max = 1.5,
                   prior_sigma_prob = 0.5)

parameters <- list(intercept = 0,
                   intercept_m = 0,
                   alpha = 1,
                   log_sigma = log(1.5),
                   log_rho = log(400),
                   nodemean = nodemean)


obj <- MakeADFun(
  data = input_data, 
  parameters = parameters,
  random = c('nodemean', 'intercept', 'intercept_m'),
  silent = TRUE,
  DLL = "LGCP_spde_marks")

LGCP_fit <- aghq::marginal_laplace_tmb(
  obj,
  3,
  startingvalue = c(parameters$alpha, parameters$log_sigma, parameters$log_rho)
)

grid <- makegrid(region, n = 20000)
grid <- SpatialPoints(grid, proj4string = CRS(proj4string(region)))
grid <- crop(grid, region)
grid <- as.data.frame(grid)
names(grid) <- c('x', 'y')
coordinates(grid) <- ~x+y
gridded(grid) <- T

Apix <- inla.mesh.project(v_mesh, loc = grid)$A

LGCP_postsamples <- sample_marginal(LGCP_fit,500)

post_node <- LGCP_postsamples$samps[3:1461,]

mean(LGCP_postsamples$samps['intercept_m',])


post_U <- rowMeans(exp(Apix %*% post_node))

post_alpha <- mean(LGCP_postsamples$thetasamples[[1]])

post_sigma <- mean(exp(LGCP_postsamples$thetasamples[[2]]))

post_rho <- mean(exp(LGCP_postsamples$thetasamples[[3]]))

df <- as.data.frame(grid)

df$mean <- post_U

ggplot(df, aes(x,y)) + geom_raster(aes(fill = mean)) + scale_fill_viridis() + coord_fixed()


n <- 61
#INLA
imat <- Diagonal(nv, rep(1, nv))
A.pp <- rbind(imat, lmat)

spde <- inla.spde2.pcmatern(mesh = v_mesh,
                            # PC-prior on range: P(practic.range < 0.05) = 0.01
                            prior.range = c(400, 0.5),
                            # PC-prior on sigma: P(sigma > 1) = 0.01
                            prior.sigma = c(1.5, 0.5)) 

stk.u <- inla.stack(
  data = list(y = GC_color),
  A = list(lmat, 1), 
  effects = list(i = 1:nv, b0 = rep(1, length(GC_color))))

u.res <- inla(y ~ 0 + b0 + f(i, model = spde),
              data = inla.stack.data(stk.u), 
              control.predictor = list(A = inla.stack.A(stk.u)))

stk2.y <- inla.stack(
  data = list(y = cbind(GC_color, NA), e = rep(0, n)), 
  A = list(lmat, 1),
  effects = list(i = 1:nv, b0.y = rep(1, n)),
  tag = 'resp2')

stk2.pp <- inla.stack(data = list(y = cbind(NA, y.pp), e = e.pp), 
                      A = list(A.pp, 1),
                      effects = list(j = 1:nv, b0.pp = rep(1, nv + n)),
                      tag = 'pp2')

j.stk <- inla.stack(stk2.y, stk2.pp)


gaus.prior <- list(prior = 'gaussian', param = c(0, 0.001))
# Model formula
jform <- y ~ 0 + b0.pp + b0.y + f(i, model = spde) +
  f(j, copy = 'i', fixed = FALSE,
    hyper = list(beta = gaus.prior))
# Fit model
j.res <- inla(jform, family = c('gaussian', 'poisson'), 
              data = inla.stack.data(j.stk),
              E = inla.stack.data(j.stk)$e,
              control.predictor = list(A = inla.stack.A(j.stk)))

summary(j.res)

post_nodes <- j.res$summary.random$i$mean
post_U <- as.numeric(exp(Apix %*% post_nodes))

df <- as.data.frame(grid)

df$mean <- post_U

ggplot(df, aes(x,y)) + geom_raster(aes(fill = mean)) + scale_fill_viridis() + coord_fixed()


