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
load(file.path(paths$get$models, sprintf("sd6_B_glm_ig_%d.RData", init_year)))
fit <- fitbs$fitb
coefs <- fit$coefficient
#summary(fit)

df <- read.csv(file.path(paths$get$data, sprintf("4scen_B_%d.csv", init_year)))[c("x_km", "y_km")]

# =====
##### Generate baseline scenarios (Archive)
# seed list: 3256
# corn ratio in 2011 is 0.672619
seed = 3256
sim <- gen_glm_samples_invgauss(n = 10000, coefs = coefs, df = df, range = coefs$spcov[["range"]], seed = seed)

column_means <- colMeans(sim/1000)
hist(column_means)

# result <- categorized_baseline_samples(sim, interval = 0.001, pick = 30, ratio = 0.672619)
# write.csv(
#   result,
#   file.path(paths$get$scenarios, sprintf("B_R00_C00_Rseed%d.csv", seed)),
#   row.names = FALSE)
# save(result, file = file.path(paths$get$scenarios, sprintf("B_R00_C00_Rseed%d.RData", seed)))
