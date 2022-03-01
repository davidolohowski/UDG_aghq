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
#compile TMB file
savestamp <- "20220214-v1"
# plotpath <- paste0(globalpath,"figures/loaloazip/")
globalpath <- getwd()
plotpath <- file.path(globalpath,"LGCP_v11_cov_spde")
if (!dir.exists(plotpath)) dir.create(plotpath)
savepath <- plotpath
file.copy(system.file('LGCP_v11_cov_spde.cpp',package='aghq'),globalpath)
# Compile TMB template-- only need to do once
compile('src/LGCP_v11_cov_spde.cpp')
source('src/inla_mesh_dual.R')
dyn.load(dynlib(file.path(globalpath,"src/LGCP_v11_cov_spde")))

v_acs <- read_csv('data/v11acs_GC.csv')
X <- c(1, 4095, 4214, 219)
Y <- c(1, 100.5, 4300, 4044)
region <- Polygon(cbind(X,Y))
region <- SpatialPolygons(list(Polygons(list(region),'region')))
region <- SpatialPolygonsDataFrame(region, data.frame(id = region@polygons[[1]]@ID, row.names = region@polygons[[1]]@ID))
plot(region)
points(v_acs[,3:4])

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

loc <- rbind(loc_mesh, v_acs[,3:4])

loc$gal1 <- (loc$x*cos(pi/18) - loc$y*sin(pi/18) - 800)^2 + (loc$x*sin(pi/18) + loc$y*cos(pi/18)-1950)^2/1.733522
loc$gal2 <- ((loc$x-2300)*cos(pi/6)- (loc$y-5800)*sin(pi/6))^2 + ((loc$x-2300)*sin(pi/6) + (loc$y-5800)*cos(pi/6))^2/2.25

ggplot(loc, aes(x,y)) + geom_point(aes(color = exp(-gal2/900^2)))
# spde stuff
nu <- 1
# Sort out mesh bits
spde <- (inla.spde2.pcmatern(mesh = v_mesh, alpha = 2,
                             prior.range = c(400, 0.5),
                             prior.sigma = c(1.5, 0.5))$param.inla)[c("M0", "M1", "M2")]
n_s <- nrow(spde$M0)
nodemean <- rep(0, n_s)

input_data <- list(Apixel = A.pp,
                   spde = spde,
                   y = y.pp,
                   A = e.pp,
                   nu = nu,
                   gal1 = loc$gal1,
                   gal2 = loc$gal2,
                   priormean_intercept = 0,
                   priorsd_intercept = 1000,
                   priormean_beta1 = 0,
                   priorsd_beta1 = 1000,
                   priormean_beta2 = 0,
                   priorsd_beta2 = 1000,
                   priormean_R1 = 6.56,
                   priorsd_R1 = 0.25,
                   priormean_R2 = 6.8,
                   priorsd_R2 = 0.2,
                   prior_rho_min = 400,
                   prior_rho_prob = 0.5,
                   prior_sigma_max = 1.5,
                   prior_sigma_prob = 0.5)

parameters <- list(intercept = -11,
                   beta1 = 2,
                   beta2 = 6,
                   log_R1 = 6.56,
                   log_R2 = 6.8,
                   log_a1 = 0,
                   log_a2 = 0,
                   log_sigma = log(1.5),
                   log_rho = log(400),
                   nodemean = nodemean)

obj <- MakeADFun(
  data = input_data,
  parameters = parameters,
  random = c('nodemean', 'intercept', 'beta1', 'beta2', 'log_R1', 'log_R2', 'log_a1', 'log_a2'),
  silent = TRUE,
  DLL = "LGCP_v11_cov_spde")

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

Apix <- inla.mesh.project(v_mesh, loc = spv)$A

LGCP_postsamples <- sample_marginal(LGCP_fit,500)

post_node <- LGCP_postsamples$samps[8:1459,]

mean(LGCP_postsamples$samps[7,])
post_U <- rowMeans(exp(Apix %*% post_node))
post_sigma <- mean(exp(LGCP_postsamples$thetasamples[[1]]))
post_rho <- mean(exp(LGCP_postsamples$thetasamples[[2]]))
df <- as.data.frame(spv)

df$mean <- post_U

df$exc <- exc

ggplot(df, aes(x,y)) + geom_point(aes(color = comb_weight)) + scale_fill_viridis()




