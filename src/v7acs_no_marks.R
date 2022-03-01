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

compile('src/LGCP_spde.cpp')
source('src/inla_mesh_dual.R')
dyn.load(dynlib("src/LGCP_spde"))

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

imat <- inla.spde.make.A(v_mesh, loc = v_mesh$loc[,1:2])

lmat <- inla.spde.make.A(v_mesh, loc = spv)

A.pp <- rbind(imat, lmat)

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
                   priormean_intercept = 0.0,
                   priorsd_intercept = 2.0,
                   prior_rho_min = 400,
                   prior_rho_prob = 0.5,
                   prior_sigma_max = 1.5,
                   prior_sigma_prob = 0.5)

parameters <- list(intercept = 0,
                   log_sigma = log(1.5),
                   log_rho = log(400),
                   nodemean = nodemean)


obj <- MakeADFun(
  data = input_data, 
  parameters = parameters,
  random = c('nodemean', 'intercept'),
  silent = TRUE,
  DLL = "LGCP_spde")

LGCP_fit <- aghq::marginal_laplace_tmb(
  obj,
  3,
  startingvalue = c(parameters$log_sigma, parameters$log_rho)
)

grid <- makegrid(region, n = 90000)
grid <- SpatialPoints(grid, proj4string = CRS(proj4string(region)))
grid <- crop(grid, region)
grid <- as.data.frame(grid)
names(grid) <- c('x', 'y')
coordinates(grid) <- ~x+y
gridded(grid) <- T

Apix <- inla.mesh.project(v_mesh, loc = grid)$A

LGCP_postsamples <- sample_marginal(LGCP_fit,500)

mean(LGCP_postsamples$samps[1,])

post_node <- LGCP_postsamples$samps[2:1460,]

post_U <- rowMeans(exp(Apix %*% post_node))

post_sigma <- mean(exp(LGCP_postsamples$thetasamples[[1]]))

post_rho <- mean(exp(LGCP_postsamples$thetasamples[[2]]))

df <- as.data.frame(grid)

df$mean <- post_U

ggplot(df, aes(x,y)) + geom_raster(aes(fill = mean)) + scale_fill_viridis()


