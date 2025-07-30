# =====
library(dplyr) # mutate

##### General settings
wd <- normalizePath(file.path("C:", "Users", Sys.getenv("USERNAME"), "Documents", "GitHub", "EnvHeteroImpactGW"))
source(file.path(wd, "code", "utils.R"))

# Initial year
init_year <- 2011

# Load paths
paths <- ProjPaths$new(wd)

setwd(paths$wd)
paths$add_file("SD6_grid_info.csv", "data")

# ====
# Load data
data <- read.csv(paths$get$SD6_grid_info, header = TRUE, stringsAsFactors = FALSE)
# Adding the new column 'corn'
crop_column <- sprintf("Crop%d", init_year)
data <- data %>%
  mutate(corn = if_else(!!sym(crop_column) == "corn", 1, 0))
data <- data[data$other_freq <= 3, ]
row.names(data) <- NULL
# df <- data[c("lon", "lat", "x_meters", "y_meters", crop_column)]
df <- data[c("lon", "lat", "x_km", "y_km", crop_column)]
names(df)

# SD6 LEMA note:
#   Area: 255  km2
#   W:    24   km (max~29)
#   H:    19.5 km

df <- df %>%
  mutate(corn = if_else(!!sym(crop_column) == "corn", 1, 0))
# plot_crop_dist_ggplot(df, var="corn")

write.csv(df, file.path(paths$get$data, sprintf("4scen_corn_%d.csv", init_year)), row.names = FALSE)

# Fit without specify initial values
# fit <- spglm(
#   corn ~ 1,
#   family = "binomial",
#   data = df,
#   spcov_type = "spherical",
#   xcoord = "x_km",
#   ycoord = "y_km",
#   estmethod = "reml"
# )
#
# summary(fit)
# logLik(fit)

# Fit with initial values (better)
lhs_sample <- gen_lhs_samples(20, bounds = list(sigmasq = c(0.001, 1), range = c(2, 15)), seed = 123)

fitbs <- batch.spglm(
  data = df, y = "corn",
  coords = c("x_km", "y_km"),
  lhs_sample = lhs_sample
)

fitb <- fitbs$fitb
summary(fitb)
logLik(fitb)

save(fitbs, file = file.path(paths$get$models, sprintf("sd6_corn_glm_%d.RData", init_year)))

# =====

# Load data
data <- read.csv(paths$get$SD6_grid_info, header = TRUE, stringsAsFactors = FALSE)
data <- data[data$other_freq <= 3, ]

eff_well <- 0.5
r <- 0.2032  # [m] = 8 inches
pumping_days <- 90
data$well_st <- data[[paste0("wl_ele_m_", init_year)]] - data$well_depth_ele
data$well_tr <- data$well_st * data$well_k
data$B_init <- 1 / (4 * pi * data$well_tr * eff_well) * (
  -0.5772 - log(
    r^2 * data$well_sy / (4 * data$well_tr * pumping_days)
  )
)

# scale B_init to avoid numerical issue when fitting model
data$B_init_scaled <- data$B_init * 1000

# Form the final input df
df <- data[c("lon", "lat", "x_km", "y_km", "B_init_scaled")]
names(df)
write.csv(df, file.path(paths$get$data, sprintf("4scen_B_%d.csv", init_year)), row.names = FALSE)

# # Fit without specify initial values
# fit_g <- spglm(
#   B_init_scaled ~ 1,
#   family = "Gamma",
#   data = df,
#   spcov_type = "spherical",
#   xcoord = "x_km",
#   ycoord = "y_km",
#   estmethod = "reml"
# )
#
# summary(fit_g)
# logLik(fit_g) # -360.9626
#
# fit_ig <- spglm(
#   B_init_scaled ~ 1,
#   family = "inverse.gaussian",
#   data = df,
#   spcov_type = "spherical",
#   xcoord = "x_km",
#   ycoord = "y_km",
#   estmethod = "reml"
# )
#
# summary(fit_ig)
# logLik(fit_ig) # -360.8533

# Fit with initial values (better)
lhs_sample <- gen_lhs_samples(20, bounds = list(sigmasq = c(0.001, 1), range = c(2, 15)), seed = 123)

fitbs <- batch.spglm(
  data = df, y = "B_init_scaled",
  coords = c("x_km", "y_km"),
  lhs_sample = lhs_sample,
  family = "inverse.gaussian"
)

fitb <- fitbs$fitb
summary(fitb)
logLik(fitb)

save(fitbs, file = file.path(paths$get$models, sprintf("sd6_B_glm_ig_%d.RData", init_year)))



# =====
# Archive
# df2 <- data[c("lon", "lat", "x_meters", "y_meters", crop_column)]
# df2 <- df2 %>%
#   mutate(corn = if_else(!!sym(crop_column) == "corn", 1, 0))
# fit2 <- spglm(
#   corn ~ 1,
#   family = "binomial",
#   data = df2,
#   spcov_type = "spherical",
#   xcoord = "x_meters",
#   ycoord = "y_meters",
#   estmethod = "reml",
# )
# summary(fit2)

# (No need. Got the same result.)
# Fit model (This will take some time.)
# lhs_sample <- gen_lhs_samples(100, bounds = list(sigmasq = c(0.001, 5), range = c(2000, 30000)), seed = 123)
#
# fitbs <- batch.spglm(
#   data = df, y = "corn",
#   coords = c("x_meters", "y_meters"),
#   lhs_sample = lhs_sample
# )
#
# fitb <- fitbs$fitb
# summary(fitb)
#
# save(fitbs, file = file.path(paths$get$models, sprintf("sd6_corn_glm_%d.RData", init_year)))

# Results with initial values
# spglm(formula = y ~ 1, family = "binomial", data = df, xcoord = "x_meters",
#     ycoord = "y_meters", spcov_initial = structure(list(initial = c(de = 4.82578843967666,
#     range = 15765.659692185), is_known = c(de = FALSE, range = FALSE
#     )), class = "spherical"), estmethod = "reml")
#
# Deviance Residuals:
#     Min      1Q  Median      3Q     Max
# -1.7848 -1.2889  0.7748  0.8823  1.1411
#
# Coefficients (fixed):
#             Estimate Std. Error z value Pr(>|z|)
# (Intercept)   0.6072     0.2812    2.16   0.0308 *
# ---
# Signif. codes:  0 <U+2018>***<U+2019> 0.001 <U+2018>**<U+2019> 0.01 <U+2018>*<U+2019> 0.05 <U+2018>.<U+2019> 0.1 <U+2018> <U+2019> 1
#
# Coefficients (spherical spatial covariance):
#        de        ie     range
# 4.209e-01 2.235e-03 1.131e+04
#
# Coefficients (Dispersion for binomial family):
# dispersion
#          1
