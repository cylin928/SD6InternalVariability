# =====
library(MASS)
library(spmodel)
library(ggplot2)

##### General settings
wd <- normalizePath(file.path("C:", "Users", Sys.getenv("USERNAME"), "Documents", "GitHub", "EnvHeteroImpactGW"))
source(file.path(wd, "code", "utils.R"))

# Initial year
init_year <- 2011

paths <- ProjPaths$new(wd)
paths$add_subfolder("scenarios")
setwd(paths$wd)

# Load the 'fitbs' object from the file
load(file.path(paths$get$models, sprintf("sd6_corn_glm_%d.RData", init_year)))
fit <- fitbs$fitb
coefs <- fit$coefficient
#summary(fit)

df <- read.csv(file.path(paths$get$data, sprintf("4scen_corn_%d.csv", init_year)))[c("x_km", "y_km")]

# =====
##### Generate scenarios with different ranges
# seed list: 3256
seed = 3256
scen_ranges = c(3, 7, 11, 15, 19, 24, 28)
for(range in scen_ranges){
  print(range)
  sim <- gen_glm_samples_binom(n = 1000000, coefs = coefs, df = df, range = range, seed = seed)
  result <- categorized_range_samples(sim, interval = 0.01, pick = 2, pick_threshold = 2)
  result_sets <- result$selected_data_list

  for(set_ in seq_along(result_sets)){
    result_set <- result_sets[[set_]]
    write.csv(
      result_set,
      file.path(paths$get$scenarios, sprintf("R%02d_S%02d_Rseed%d.csv", range, set_, seed)),
      row.names = FALSE)
  }

  # Keep the records
  save(result, file = file.path(paths$get$scenarios, sprintf("R%02d_Rseed%d.RData", range, seed)))

}

# =====
##### Generate baseline scenarios
# seed list: 3256
# corn ratio in 2011 is 0.672619
seed = 3256
sim <- gen_glm_samples_binom(n = 10000, coefs = coefs, df = df, range = coefs$spcov[["range"]], seed = seed)
result <- categorized_baseline_samples(sim, interval = 0.01, pick = 30, ratio = 0.672619)
write.csv(
  result,
  file.path(paths$get$scenarios, sprintf("R00_C00_Rseed%d.csv", seed)),
  row.names = FALSE)
save(result, file = file.path(paths$get$scenarios, sprintf("R00_C00_Rseed%d.RData", seed)))

# For the OAT experiment
ratios <- c(0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5)
target_averages <- ratios * 0.672619
col_means <- colMeans(sim)
closest_columns <- sapply(target_averages, function(target) {
  # Get the index of the column with the minimum difference to the target
  which.min(abs(col_means - target))
})
closest_columns_data <- sim[, closest_columns]
colnames(closest_columns_data) <- ratios
write.csv(closest_columns_data,
          file.path(paths$get$scenarios, sprintf("corn_dist_Rseed%d.csv", seed)),
          row.names = FALSE)

# For the exp4 Anova experiment
seed = 3256
sim <- gen_glm_samples_binom(n = 1000000, coefs = coefs, df = df, range = coefs$spcov[["range"]], seed = seed)

corn_ratios = c(0.72  , 0.6952, 0.6704, 0.6457, 0.6209, 0.5961, 0.5713, 0.5466, 0.5218, 0.497)
for(corn_ratio in corn_ratios){
  result <- categorized_baseline_samples(sim, interval = 0.003, pick = 30, ratio = corn_ratio)
  write.csv(
    result,
    file.path(paths$get$scenarios, sprintf("anova_C%f_Rseed%d.csv", corn_ratio, seed)),
    row.names = FALSE)
}
# =====
##### Generate task 1 scenarios for just varying ratio
# seed list: 3256
# corn ratio in 2011 is 0.672619
seed = 3256
sim <- gen_glm_samples_binom(n = 100000, coefs = coefs, df = df, range = coefs$spcov[["range"]], seed = seed)
result <- categorized_range_samples(sim, interval = 0.01, pick = 1, pick_threshold = 1)
select_result <- result$selected_data_list[[1]][ , paste0("C", seq(40, 85, by = 5))]
write.csv(
  select_result,
  file.path(paths$get$scenarios, sprintf("R00_Rseed%d.csv", seed)),
  row.names = FALSE)
save(result, file = file.path(paths$get$scenarios, sprintf("R00_Rseed%d.RData", seed)))

#=====
##### Additionally range scenarios for task 4
seed <- 3256
for (rc_seed in c(7834, 3924, 1833)){
  df_rc <- read.csv(file.path(paths$get$scenarios, sprintf("rc_samples_%d.csv", rc_seed)))

  df_rc$sample_corn_ratio <- NA
  df_rc_ini_corn <- list()

  for (i in 1:nrow(df_rc)) {
    print(paste("Processing row:", i, "of", nrow(df_rc)))
    range <- df_rc[i, "range"]
    ratio <- df_rc[i, "corn_ratio"]/100
    rc_idx <- df_rc[i, "rc_idx"]
    sim <- gen_glm_samples_binom(n = 10000, coefs = coefs, df = df, range = range, seed = seed)
    column_means <- colMeans(sim)

    closest_index <- which.min(abs(column_means - ratio))
    if (abs(column_means[closest_index] - ratio) <= 0.001) {
      df_rc[i, "sample_corn_ratio"] <- column_means[closest_index]
      df_rc_ini_corn[[rc_idx]] <- sim[, closest_index]
    }
  }

  df_rc_ini_corn <- data.frame(df_rc_ini_corn)
  write.csv(
    df_rc_ini_corn,
    file.path(paths$get$scenarios,
              sprintf("rc_samples_%d_Rseed%d.csv", rc_seed, seed)),
    row.names = FALSE
  )

  write.csv(
    df_rc,
    file.path(paths$get$scenarios,
              sprintf("rc_samples_%d.csv", rc_seed)),
    row.names = FALSE
  )
}

# =====
# Archive
# =====
# coefs <- fit$coefficient
# spcov_params_val <- spcov_params("spherical", de = coefs$spcov[["de"]], ie = coefs$spcov[["ie"]], range = coefs$spcov[["range"]])
# sim <- sprbinom(samples = 1000000, mean = coefs$fixed, spcov_params = spcov_params_val, data = df, xcoord = "x_km", ycoord = "y_km")
#
#
#
# # Assume 'sim' is your matrix [336, 1000] from the 'sprbinom' function
# column_means <- colMeans(sim)
#
# # Create intervals and categorize column means
# breaks <- seq(0, 1, by = 0.01)
# categories <- cut(column_means, breaks, include.lowest = TRUE)
#
# # Count the number of column means falling into each category
# category_counts <- table(categories)
# print(category_counts)
#
# # Select one column from each category
# selected_columns <- tapply(seq_along(column_means), categories, function(indices) {
#   if (length(indices) > 0) sample(indices, 1) else NA
# })
# print(selected_columns)
#
# # Accessing selected columns from sim, if needed for further processing
# selected_data <- sim[, unlist(selected_columns, use.names = FALSE)]

#=====
# scen_ranges = c(3, 7, 11, 15, 19, 24, 28)
# scen_ranges_category_counts = list()
# for (range in scen_ranges){
#   print(range)
#   coefs <- fit$coefficient
#   spcov_params_val <- spcov_params("spherical", de = coefs$spcov[["de"]], ie = coefs$spcov[["ie"]], range = range)
#   sim <- sprbinom(samples = 1000000, mean = coefs$fixed, spcov_params = spcov_params_val, data = df, xcoord = "x_km", ycoord = "y_km")
#
#   # Assume 'sim' is your matrix [336, 1000] from the 'sprbinom' function
#   column_means <- colMeans(sim)
#
#   # Create intervals and categorize column means
#   breaks <- seq(0, 1, by = 0.01)
#   categories <- cut(column_means, breaks, include.lowest = TRUE)
#
#   # Count the number of column means falling into each category
#   category_counts <- table(categories)
#
#   scen_ranges_category_counts[[as.character(range)]] <- category_counts
# }


# =====
# Customized function for simulation that will fix the corn ratio.

# spherical <- function(locs, sigmasq, range) {
#   n <- dim(locs)[[1]]
#
#   spherical_cor <- function(distance, range) {
#     eta <- distance / range
#     cor <- ifelse(eta <= 1, 1 - 1.5 * eta + 0.5 * eta^3, 0)
#     return(cor)
#   }
#
#   # Compute the distance matrix
#   dist_matrix <- as.matrix(dist(locs))
#
#   # Compute the correlation matrix
#   cor_matrix <- matrix(nrow = nrow(dist_matrix), ncol = ncol(dist_matrix))
#   for (i in 1:nrow(dist_matrix)) {
#     for (j in 1:ncol(dist_matrix)) {
#       cor_matrix[i, j] <- spherical_cor(dist_matrix[i, j], range)
#     }
#   }
#
#   cov_matrix <- cor_matrix * sigmasq
#   random_effects <- mvrnorm(mu = rep(0, n), Sigma = cov_matrix)
#
#   return(random_effects)
# }
#
#
# mvrnorm.residuals <- function(ie, n, mu = 0) {
#   residuals <- mvrnorm(mu = rep(mu, n), Sigma = diag(ie, n, n))
#   return(residuals)
# }
#
# link.bino <- function(y, n = 1) {
#   # Check if the probability parameter is valid
#   if (y < 0 || y > 1) {
#     stop("Probability 'y' must be between 0 and 1.")
#   }
#
#   # Sample from the binomial distribution
#   rbinom(n, size = 1, prob = y)
# }
#
# link.logit <- function(y) {
#   # Inverse logit transformation
#   logit <- 1 / (1 + exp(-y)) # [0, 1]
#
#   return(logit)
# }
#
# simulate.sprbinom <- function(locs, sigmasq, range, intercept, corn_ratio, beta = NA, features = NA, ie = 0,
#                               random.model = spherical,
#                               residual.model = mvrnorm.residuals,
#                               link = link.logit,
#                               seed = NA) {
#   # Set random seed
#   if (!is.na(seed)) {
#     set.seed(seed)
#   }
#
#   n <- dim(locs)[[1]]
#
#   # Fixed effects e.g., intercept + beta * x
#   if (is.na(beta) == 1) {
#     fixed_effects <- intercept
#   } else {
#     fixed_effects <- beta * features + intercept
#   }
#
#   # Random effects. Default using Matern.
#   # print(random.model)
#   random_effects <- random.model(locs = locs, sigmasq = sigmasq, range = range)
#
#   # Residuals (incorporating the nugget effect of Matern) assuming zero mean.
#   if (is.function(residual.model)) {
#     # print(residual.model)
#     # Length of predictors
#     residuals <- residual.model(ie = ie, n = n, mu = 0)
#   } else {
#     residuals <- 0
#   }
#
#   # Linear regression
#   y <- fixed_effects + random_effects + residuals
#
#   # Transform if the link is provided.
#   if (is.function(link)) {
#     print(link)
#     y_transformed <- link(y)
#     if (y_transformed < 0.5){
#         binary_y <- 0
#     } else {
#         binary_y <- 1
#     }
#
#   } else {
#     if (corn_ratio == 0) {
#       binary_y <- numeric(nrow(df))
#     } else if (corn_ratio == 1) {
#       binary_y <- rep(1, nrow(df))
#     } else {
#       # Convert to binary while maintain the given proportion
#       sorted_vec <- sort(y, decreasing = TRUE)
#       threshold <- sorted_vec[n * corn_ratio]
#
#       # Convert the original vector to 0s and 1s based on the threshold
#       binary_y <- ifelse(y >= threshold, 1, 0)
#     }
#   }
#
#   return(binary_y)
# }
#
#
#
# # =====
# # Simulation with given corn ratio
# # sample(1:9999, 20, replace = FALSE)
# seeds <- c(
#   7234, 3285, 6106, 3966, 2676, 3998, 1915, 6033, 4267, 6723, 4434, 4570, 9349, 3267, 6363,
#   459, 1932, 1803, 2055, 2872
# )
# corn_ratios <- seq(0, 1, by = 0.2)
# ranges <- seq(5000, 35000, by = 5000) # m
#
# coe <- fitb$coefficients
# # $fixed
# # (Intercept)
# # 0.2864151
# #
# # $spcov
# # de           ie        range       rotate
# # 9.651200e-01 4.841292e-03 2.548186e+04 0.000000e+00
# # scale
# # 1.000000e+00
# # attr(,"class")
# # [1] "spherical"
# #
# # $dispersion
# # dispersion
# # 1
# # attr(,"class")
# # [1] "binomial"
# #
# # $randcov
# # NULL
#
# locs <- cbind(df$x_meters, df$y_meters)
# corn_ratio_2012 <- sum(df$corn) / nrow(df)
# df_gen <- data[c("gid")]
#
# total <- length(seeds) * (length(corn_ratios) - 2) * length(ranges)
# pb <- txtProgressBar(min = 0, max = total, style = 3)
# i <- 0
# for (seed in seeds) {
#   sim <- simulate.sprbinom(
#     locs = locs, sigmasq = coe$spcov[["de"]], range = coe$spcov[["range"]],
#     intercept = coe$fixed, corn_ratio = corn_ratio_2012, ie = coe$spcov[["ie"]], seed = seed
#   )
#   df_gen[[sprintf("fitted_s%.0f", seed)]] <- sim
#   for (ratio in corn_ratios) {
#     if (ratio == 0 || ratio == 1) {
#       sim <- simulate.sprbinom(
#         locs = locs, sigmasq = coe$spcov[["de"]], range = range,
#         intercept = coe$fixed, corn_ratio = ratio, ie = coe$spcov[["ie"]], seed = seed
#       )
#       df_gen[[sprintf("fitted_c%.1f_rNA_s%.0f", ratio, seed)]] <- sim
#     } else {
#       for (range in ranges) {
#         sim <- simulate.sprbinom(
#           locs = locs, sigmasq = coe$spcov[["de"]], range = range,
#           intercept = coe$fixed, corn_ratio = ratio, ie = coe$spcov[["ie"]], seed = seed
#         )
#         df_gen[[sprintf("fitted_c%.1f_r%.0f_s%.0f", ratio, range, seed)]] <- sim
#         # Update the progress bar
#         i <- i + 1
#         setTxtProgressBar(pb, i)
#       }
#     }
#   }
# }
#
# # Write the data frame to a CSV file without row names
# write.csv(df_gen, "C:/Users/CL/OneDrive/VT/Proj_DIESE-heterogeneity/Code/Inputs/gen_init_corn2.csv", row.names = FALSE)
# ======
# Brute force solution for maintain proportion in binary glm.
# Not working very well. (never stop)

# coe <- fitb$coefficient
# spcov_params_val <- spcov_params("spherical", de = coe$spcov[["de"]], ie = coe$spcov[["ie"]], range = coe$spcov[["range"]])
# sim <- sprbinom(samples = 1, mean=coe$fixed, spcov_params_val, data = df, xcoord = "x_meters", ycoord = "y_meters")
# N <- 10
# n <- 0
# count <- 0
# while(n<N){
#   count <- count + 1
#   if(sum(sim) == sum(df$corn)){
#     df[[paste("crop", n, sep = "")]] <- sim
#     n <- n + 1
#     print(n)
#   }
#   #print(count)
# }

# =====
# Archive
# coe <- fitb$coefficients
# #sim <- sprbinom(samples = 1, mean=coe$fixed, spcov_params_val, data = df, xcoord = "x_meters", ycoord = "y_meters")
# sim <- simulate.sprbinom(locs, sigmasq, range, intercept, corn_ratio, beta=NA, features=NA, ie=0,
# df$sim <- sim
# me$plot_crop_dist_ggplot(df, var="sim")

# lhs_sample <- spcov_initial(
#   spcov_type = "spherical",
#   de = 1.468,
#   ie = 0,
#   range = 1.946e+04
# )
#
# fitb  <- spglm(corn ~ 1,
#                family = "binomial",
#                data = df,
#                xcoord = "x_meters",
#                ycoord = "y_meters",
#                #spcov_type = "spherical",
#                estmethod = "reml",
#                spcov_initial = list(lhs_sample, lhs_sample)
# )
#
# summary(fitb)
#
# logLik(fitb)
# plot(fitb)
#
# df$sim <- sim
# sum(sim)
# me$plot_crop_dist_ggplot(df, var="sim")
