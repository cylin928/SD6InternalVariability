library(lhs) # Latin Hypercube Sample
library(geoR)
library(spmodel)
library(withr) # Local random seed
library(ggplot2)
library(R6)

ProjPaths <- R6Class("ProjPaths",
  public = list(
    wd = NULL,
    subfolders = list(),
    files = list(),
    otherpaths = list(),
    get = list(), # a collaborative getter for all paths

    initialize = function(wd) {
      self$wd <- wd
      self$add_subfolder("code")
      self$add_subfolder("data")
      self$add_subfolder("figures")
      self$add_subfolder("inputs")
      self$add_subfolder("outputs")
      self$add_subfolder("models")
    },
    check_name_eligibility = function(name) {
      if (!grepl("^[A-Za-z_][A-Za-z0-9_]*$", name) || name %in% names(self)) {
        stop(paste(name, "has been used/is not a valid attribute name."))
      }
    },
    add_subfolder = function(subfolder) {
      self$check_name_eligibility(subfolder)
      full_path <- file.path(self$wd, subfolder)
      if (!dir.exists(full_path)) {
        dir.create(full_path, recursive = TRUE)
      }
      self$subfolders[[subfolder]] <- full_path
      self$get[[subfolder]] <- full_path
    },
    remove_subfolder = function(subfolder) {
      if (subfolder %in% names(self$subfolders)) {
        self$subfolders[[subfolder]] <- NULL
        self$get[[subfolder]] <- NULL
      } else {
        warning(paste(subfolder, "is not a tracked subfolder."))
      }
    },
    add_nested_folder = function(parent_subfolder, nested_folder) {
      nested_attr_name <- paste(parent_subfolder, nested_folder, sep = "_")
      self$check_name_eligibility(nested_attr_name)
      if (parent_subfolder %in% names(self$subfolders)) {
        full_path <- file.path(self$subfolders[[parent_subfolder]], nested_folder)
        if (!dir.exists(full_path)) {
          dir.create(full_path, recursive = TRUE)
        }
        self$get[[nested_attr_name]] <- full_path
      } else {
        stop(paste(parent_subfolder, "is not a tracked subfolder."))
      }
    },
    remove_nested_folder = function(parent_subfolder, nested_folder) {
      nested_attr_name <- paste(parent_subfolder, nested_folder, sep = "_")
      if (nested_attr_name %in% names(self$get)) {
        unlink(self[[nested_attr_name]], recursive = TRUE) # This only works if the directory is empty
        self$get[[nested_attr_name]] <- NULL
      } else {
        warning(paste(nested_attr_name, "is not a tracked nested folder."))
      }
    },
    add_file = function(file_name, subfolder = NULL, name = NULL) {
      if (!is.null(subfolder) && subfolder %in% names(self$subfolders)) {
        full_path <- file.path(self$subfolders[[subfolder]], file_name)
      } else {
        full_path <- file.path(self$wd, file_name)
      }
      if (file.exists(full_path)) {
        if (!is.null(name)) {
          self$check_name_eligibility(name)
          self$files[[name]] <- full_path
          self$get[[name]] <- full_path
        } else {
          file_name_without_extension <- tools::file_path_sans_ext(file_name)
          self$check_name_eligibility(file_name_without_extension)
          self$files[[file_name_without_extension]] <- full_path
          self$get[[file_name_without_extension]] <- full_path
        }
      } else {
        stop(paste(full_path, "does not exist."))
      }
    },
    remove_file = function(file_name) {
      if (file_name %in% names(self$files)) {
        self$files[[file_name]] <- NULL
        self$get[[file_name]] <- NULL
      } else {
        warning(paste(file_name, "is not a tracked file."))
      }
    }
  )
)

gen_glm_samples_binom <- function(n, coefs, df, range, seed) {
  with_seed(seed, {
    spcov_params_val <- spcov_params("spherical", de = coefs$spcov[["de"]], ie = coefs$spcov[["ie"]], range = range)
    sim <- sprbinom(samples = n, mean = coefs$fixed, spcov_params = spcov_params_val, data = df, xcoord = "x_km", ycoord = "y_km")
  })

  return(sim)
}

gen_glm_samples_invgauss <- function(n, coefs, df, range, seed) {
  with_seed(seed, {
    spcov_params_val <- spcov_params("spherical", de = coefs$spcov[["de"]], ie = coefs$spcov[["ie"]], range = range)
    sim <- sprinvgauss(samples = n, mean = coefs$fixed, spcov_params = spcov_params_val, data = df, xcoord = "x_km", ycoord = "y_km")
  })

  return(sim)
}

categorized_range_samples <- function(sim, interval = 0.01, pick = 1, pick_threshold = 2) {
  column_means <- colMeans(sim)
  breaks <- seq(0 + interval/2, 1 - interval/2, by = interval)
  category_names <- seq(interval, 1 - interval, by = interval) * 100
  categories <- cut(column_means, breaks, include.lowest = TRUE)
  category_counts <- table(categories)

  # Create a list to store dataframes
  selected_data_list <- vector("list", pick)

  # Initialize empty dataframes for each pick
  for (i in 1:pick) {
    selected_data_list[[i]] <- data.frame(matrix(nrow = nrow(sim), ncol = 0))
  }

  # Select columns for each interval
  for (ind in seq_along(names(category_counts))) {
    category <- names(category_counts)[ind]
    category_name <- sprintf("C%02d", as.integer(category_names[ind]))

    if (category_counts[category] > pick_threshold) {
      indices <- which(categories == category)
      selected_indices <- sample(indices, min(length(indices), pick), replace = FALSE)

      # Add the selected columns to the corresponding dataframes
      for (i in 1:length(selected_indices)) {
        selected_data_list[[i]] <- cbind(selected_data_list[[i]], sim[, selected_indices[i]])
        colnames(selected_data_list[[i]])[ncol(selected_data_list[[i]])] <- as.character(category_name)
      }
    } else {
      # If no samples are in this category, add NA columns
      for (i in 1:pick) {
        # ignore the column
        # selected_data_list[[i]] <- cbind(selected_data_list[[i]], NA)
        # colnames(selected_data_list[[i]])[ncol(selected_data_list[[i]])] <- as.character(category)
      }
    }
  }

  # Return the list of dataframes and category counts
  return(list(selected_data_list = selected_data_list, category_counts = category_counts))
}

categorized_baseline_samples <- function(sim, interval = 0.001, pick = 10, ratio = 0.672619) {
  column_means <- colMeans(sim)
  breaks <- seq(0 + interval/2, 1 - interval/2, by = interval)
  categories <- cut(column_means, breaks, include.lowest = TRUE)

  # Find the category where the ratio is located
  ratio_category <- cut(ratio, breaks, include.lowest = TRUE)

  # Get indices of columns in the chosen category
  selected_indices <- which(categories == ratio_category)

  # If there are not enough samples, print a warning
  if (length(selected_indices) < pick) {
    warning("Not enough samples in the chosen category. Returning fewer samples.")
    pick <- length(selected_indices)
  }
  print(sprintf("%d samples are available in the category for ratio %s.", length(selected_indices), ratio_category))

  # Randomly sample columns from the chosen category
  sampled_indices <- sample(selected_indices, pick, replace = FALSE)

  # Create a dataframe with the sampled columns
  selected_data <- sim[, sampled_indices]

  # Set column names as "S%02d"
  colnames(selected_data) <- sprintf("S%02d", seq_len(pick))

  return(selected_data)
}

# Generate Latin Hypercube samples
gen_lhs_samples <- function(num_samples, bounds = list(sigmasq = c(0.001, 1), range = c(2000, 10000)), seed = NA) {
  if (is.na(seed)) {
    lhs_sample <- randomLHS(n = num_samples, k = 2)
  } else {
    with_seed(seed, {
      lhs_sample <- randomLHS(n = num_samples, k = 2)
    })
  }

  # Scale
  lhs_sample[, 1] <- (lhs_sample[, 1] * (bounds$sigmasq[2] - bounds$sigmasq[1])) + bounds$sigmasq[1]
  lhs_sample[, 2] <- (lhs_sample[, 2] * (bounds$range[2] - bounds$range[1])) + bounds$range[1]
  return(lhs_sample)
}

plot_crop_dist_ggplot <- function(df, var, x = "lon", y = "lat") {
  # Plotting the scatter plot with updated aesthetics
  dff <- df
  dff$Crops <- factor(df[[var]], levels = c(0, 1), labels = c("others", "corn"))
  ggplot(dff, aes(x = dff[[x]], y = dff[[y]], color = Crops)) +
    geom_point(size = 3) + # Increased dot size
    scale_color_manual(values = c("others" = "chocolate4", "corn" = "gold2")) +
    labs(color = "") + # Remove legend title
    xlab("Longitude") +
    ylab("Latitude") +
    theme(
      legend.position = "top", # Position legend at top
      legend.justification = "right", # Right-align the legend
      legend.direction = "horizontal"
    ) # Legend direction horizontal
}

# Batch fitting for spmodel
batch.spglm <- function(data, y, coords = c("x_km", "y_km"), lhs_sample, family = "binomial") {
  # Rename dataframe
  df <- data[, c(y, coords)]
  names(df) <- c("value", "x", "y")

  num_samples <- nrow(lhs_sample)
  init_list <- list()
  for (i in 1:(num_samples)) {
    init <- spcov_initial(
      spcov_type = "spherical",
      de = lhs_sample[i, 1],
      range = lhs_sample[i, 2]
    )
    init_list[[i]] <- init
  }

  if (family == "inverse.gaussian"){
    print(family)
    fitb_list <- spglm(
    value ~ 1,
    family = "inverse.gaussian",
    data = df,
    xcoord = "x",
    ycoord = "y",
    estmethod = "reml",
    spcov_initial = init_list
    )
  } else if (family == "binomial"){
    print(family)
    fitb_list <- spglm(
      value ~ 1,
      family = "binomial",
      data = df,
      xcoord = "x",
      ycoord = "y",
      estmethod = "reml",
      spcov_initial = init_list
    )
  } else if (family == "Gamma"){
    print(family)
    fitb_list <- spglm(
      value ~ 1,
      family = "Gamma",
      data = df,
      xcoord = "x",
      ycoord = "y",
      estmethod = "reml",
      spcov_initial = init_list
    )
  }
  logLik_list <- list()
  #pb <- txtProgressBar(min = 0, max = num_samples, style = 3)
  for (i in 1:(num_samples)) {
    logLik_list[[i]] <- logLik(fitb_list[[i]])
    print(sprintf("Fit with initial sigmasq: %.5f, phi: %.5f [%d/%d].", lhs_sample[i, 1], lhs_sample[i, 2], i, num_samples))
    print(sprintf("logLik: %.4f", logLik_list[[i]]))

    # Update the progress bar
    #setTxtProgressBar(pb, i)
    #flush.console() # Ensure the output is updated immediately
  }
  #close(pb)

  index_of_max <- which.max(unlist(logLik_list))
  fitb <- fitb_list[[index_of_max]]
  logLik <- logLik_list[[index_of_max]]
  # print("============================")
  # print(summary(fit))
  print("============================")
  print(sprintf("Index: %d (logLik: %.4f)", index_of_max, logLik))
  print("============================")

  results <- list(
    fitb = fitb,
    logLik = logLik,
    index_of_max = index_of_max,
    fitb_list = fitb_list,
    logLik_list = logLik_list,
    lhs_sample = lhs_sample
  )
  return(results)
}

# Batch fitting for spmodel
single.spglm <- function(data, y, coords = c("x_km", "y_km"), family = "binomial") {
  # Rename dataframe
  df <- data[, c(y, coords)]
  names(df) <- c("value", "x", "y")

  if (family == "inverse.gaussian"){
    print(family)
    fitb_list <- spglm(
      value ~ 1,
      family = "inverse.gaussian",
      data = df,
      xcoord = "x",
      ycoord = "y",
      estmethod = "reml",
      spcov_type = "spherical",
    )
  } else if (family == "binomial"){
    print(family)
    fitb_list <- spglm(
      value ~ 1,
      family = "binomial",
      data = df,
      xcoord = "x",
      ycoord = "y",
      estmethod = "reml",
      spcov_type = "spherical",
    )
  } else if (family == "Gamma"){
    print(family)
    fitb_list <- spglm(
      value ~ 1,
      family = "Gamma",
      data = df,
      xcoord = "x",
      ycoord = "y",
      estmethod = "reml",
      spcov_type = "spherical",
    )
  }

  results <- list(
    fitb = fitb_list,
    logLik = logLik(fitb_list)
  )
  return(results)
}
