LGCP0.0 <- readRDS('LGCP_v11_cov_spde_marks/LGCP_fit_0.0.RDS')
LGCP0.25 <- readRDS('LGCP_v11_cov_spde_marks/LGCP_fit_0.25.RDS')
LGCP0.5 <- readRDS('LGCP_v11_cov_spde_marks/LGCP_fit_0.5.RDS')
LGCP0.75 <- readRDS('LGCP_v11_cov_spde_marks/LGCP_fit_0.75.RDS')
LGCP1.0 <- readRDS('LGCP_v11_cov_spde_marks/LGCP_fit_1.0.RDS')

samps0.0 <- sample_marginal(LGCP0.0, 5000)
samps0.25 <- sample_marginal(LGCP0.25, 5000)
samps0.5 <- sample_marginal(LGCP0.5, 5000)
samps0.75 <- sample_marginal(LGCP0.75, 5000)
samps1.0 <- sample_marginal(LGCP1.0, 5000)

v_acs <- read_csv('data/v11acs_GC.csv')
X <- c(1, 4095, 4214, 219)
Y <- c(1, 100.5, 4300, 4044)
region <- Polygon(cbind(X,Y))
region <- SpatialPolygons(list(Polygons(list(region),'region')))
region <- SpatialPolygonsDataFrame(region, data.frame(id = region@polygons[[1]]@ID, row.names = region@polygons[[1]]@ID))

spv <- SpatialPoints(v_acs[,3:4])

v_mesh <- inla.mesh.2d(loc = spv, boundary = region,
                       max.edge = c(80, 800), cutoff = 40, max.n.strict = 1300L)

grid <- makegrid(region, n = 90000)
grid <- SpatialPoints(grid, proj4string = CRS(proj4string(region)))
grid <- crop(grid, region)
grid <- as.data.frame(grid)
names(grid) <- c('x', 'y')
coordinates(grid) <- ~x+y
gridded(grid) <- T

Apix <- inla.mesh.project(v_mesh, loc = grid)$A

post_node0.0 <- samps0.0$samps[15:1466,]
post_node0.25 <- samps0.25$samps[15:1466,]
post_node0.5 <- samps0.5$samps[15:1466,]
post_node0.75 <- samps0.75$samps[15:1466,]
post_node1.0 <- samps1.0$samps[15:1466,]

post_U0.0 <- exp(Apix %*% post_node0.0)
post_U0.25 <- exp(Apix %*% post_node0.25)
post_U0.5 <- exp(Apix %*% post_node0.5)
post_U0.75 <- exp(Apix %*% post_node0.75)
post_U1.0 <- exp(Apix %*% post_node1.0)

sigma0.0 <- exp(samps0.0$thetasamples[[2]])
sigma0.25 <- exp(samps0.25$thetasamples[[2]])
sigma0.5 <- exp(samps0.5$thetasamples[[2]])
sigma0.75 <- exp(samps0.75$thetasamples[[2]])
sigma1.0 <- exp(samps1.0$thetasamples[[2]])

exc_q0.5_0.0 <- rowMeans(post_U0.0 > 1)
exc_q0.5_0.25 <- rowMeans(post_U0.25 > 1)
exc_q0.5_0.5 <- rowMeans(post_U0.5 > 1)
exc_q0.5_0.75 <- rowMeans(post_U0.75 > 1)
exc_q0.5_1.0 <- rowMeans(post_U1.0 > 1)

exc_q0.75_0.0 <- rowMeans(post_U0.0 > exp(quantile(rnorm(5000, 0, sigma0.0), 0.75)))
exc_q0.75_0.25 <- rowMeans(post_U0.25 > exp(quantile(rnorm(5000, 0, sigma0.25), 0.75)))
exc_q0.75_0.5 <- rowMeans(post_U0.5 > exp(quantile(rnorm(5000, 0, sigma0.5), 0.75)))
exc_q0.75_0.75 <- rowMeans(post_U0.75 > exp(quantile(rnorm(5000, 0, sigma0.75), 0.75)))
exc_q0.75_1.0 <- rowMeans(post_U1.0 > exp(quantile(rnorm(5000, 0, sigma1.0), 0.75)))

exc_q0.9_0.0 <- rowMeans(post_U0.0 > exp(quantile(rnorm(5000, 0, sigma0.0), 0.9)))
exc_q0.9_0.25 <- rowMeans(post_U0.25 > exp(quantile(rnorm(5000, 0, sigma0.25), 0.9)))
exc_q0.9_0.5 <- rowMeans(post_U0.5 > exp(quantile(rnorm(5000, 0, sigma0.5), 0.9)))
exc_q0.9_0.75 <- rowMeans(post_U0.75 > exp(quantile(rnorm(5000, 0, sigma0.75), 0.9)))
exc_q0.9_1.0 <- rowMeans(post_U1.0 > exp(quantile(rnorm(5000, 0, sigma1.0), 0.9)))

exc_q0.95_0.0 <- rowMeans(post_U0.0 > exp(quantile(rnorm(5000, 0, sigma0.0), 0.95)))
exc_q0.95_0.25 <- rowMeans(post_U0.25 > exp(quantile(rnorm(5000, 0, sigma0.25), 0.95)))
exc_q0.95_0.5 <- rowMeans(post_U0.5 > exp(quantile(rnorm(5000, 0, sigma0.5), 0.95)))
exc_q0.95_0.75 <- rowMeans(post_U0.75 > exp(quantile(rnorm(5000, 0, sigma0.75), 0.95)))
exc_q0.95_1.0 <- rowMeans(post_U1.0 > exp(quantile(rnorm(5000, 0, sigma1.0), 0.95)))


U1 <- c(2628, 1532)
U2 <- c(2517, 3289)

U <- data.frame(x = c(U1[1],U2[1]), y = c(U1[2], U2[2]))


df <- as.data.frame(grid)

df$exc <- exc_q0.75_1.0

Fs_0.5 <- df %>%
  mutate(dist1 = sqrt((x - U1[1])^2 + (y - U1[2])^2), dist2 = sqrt((x - U2[1])^2 + (y - U2[2])^2)) %>%
  dplyr::filter(dist1 < 220 | dist2 < 220) %>%
  mutate(U1 = dist1 < 220, U2 = dist2 < 220)

Fs_0.75_U1_1.0 <- Fs_0.5 %>%
  filter(U1 == T) %>%
  summarise(p = max(exc)) #0.99

Fs_0.75_U2_1.0 <- Fs_0.5 %>%
  filter(U2 == T) %>%
  summarise(p = max(exc)) #0.99

pal <- rev(RColorBrewer::brewer.pal(11, 'Spectral'))

ggplot(df, aes(x, y)) +
  geom_contour_filled(aes(z = exc), breaks = c(0, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3)) +
  scale_fill_manual(values = pal, name = '$F(s)$', guide = guide_legend(reverse = T)) +
  coord_fixed() +
  geom_point(data = v_acs, aes(x,y), size = 0.1, stroke = 0.25, shape = 20) +
  xlab('X (pixels)') + ylab('Y (pixels)') +
  theme_minimal() +
  theme(legend.text = element_text(size = 7),
        legend.title = element_text(size = 8),
        strip.background = element_rect(color = NULL, fill = 'white', linetype = 'blank'))

df_U1 <- readRDS('LGCP_v11_cov_spde_marks/')


ggplot(df_U2, aes(rho, prob)) + geom_line(aes(color = q))












