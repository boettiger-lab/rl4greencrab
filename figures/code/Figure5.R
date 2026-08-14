library(tidyverse)
library(viridis)
library(patchwork)

# read in cluter plot data
data <- read.csv("data/cluster/centroid_plot_data.csv")

params <- list(
  max_action = 3000,
  smin = 5,
  smax = 110
)

#########################################
# functions for translating data scales #
#########################################

# convert actions
convert_action <- function(data, params, action_col) {
  
  data$act_real <- pmax(params$max_action * (1 + data[[action_col]]) / 2, 0)
  
  return(data)
}

# calculate biomass as a function of size
calc_biomass <- function(y) {
  
  biomass <- max(0, -0.071 * y + 0.003 * y ^ 2 + 0.00002 * y ^ 3)
  
  return(biomass)
}

# convert biomass
convert_biomass <- function(data, params, biomass_col) {
  
  bmin <- calc_biomass(params$smin)
  bmax <- calc_biomass(params$smax)
  
  data$biomass_real <- (
    (data[[biomass_col]] + 1) * (bmax - bmin) / 2 + bmin
  )
  
  return(data)
}

# convert CPUE
convert_cpue <- function(data, cpue_col) {
  
  data$cpue_real <- (data[[cpue_col]] + 1) / 2 * 100
  
  return(data)
}

# convert all
convert_all <- function(data, params, action_col, biomass_col, cpue_col) {
  
  data <- convert_action(data, params, action_col)
  
  data <- convert_biomass(data, params, biomass_col)
  
  data <- convert_cpue(data, cpue_col)
  
  return(data)
}


##########################
# make scale conversions #
##########################

data <- convert_all(data, params, "centroid_act", "biomass_x", "CPUE_x")


###############
# plot months #
###############

month_names <- c("4" = "Apr", "5" = "May",
                 "6" = "June", "7" = "July", "8" = "Aug", 
                 "9" = "Sep", "10" = "Oct")

# update order of action and month
data$action <- factor(data$action, levels = rev(levels(factor(data$action))))
data$month <- factor(data$month, levels = c("Apr", "May", "June", "July",
                                            "Aug", "Sep", "Oct"))

figure4 <- ggplot(data) + 
  geom_density_2d_filled(aes(x = biomass_real, y = cpue_real, 
                 fill = factor(act_real))) +
  scale_fill_viridis_d(option = "magma") +
  labs(x = expression("mean biomass (g), " * italic("t - 1")),
       y = expression("CPUE (crabs per trap), " * italic("t - 1")),
       color = expression("action\n(number\nof traps), " * italic(t))) +
  scale_x_continuous(breaks = c(5, 10, 15),
                     labels = c(5, 10, 15)) +
  facet_grid(action ~ month) +
  theme(legend.title = element_text(hjust = 0.5))


ggsave("figures/figure5.png",
       figure4, height = 3, width = 8)

ggplot(data, aes(x = biomass_real, y = cpue_real, color = factor(act_real))) + 
  # 1. Plot the actual data points colored by act_real
  #geom_point(alpha = 0.5) +
  
  # 2. Draw an oval around each group
  stat_ellipse(type = "t", level = 0.95, linewidth = 1) + 
  
  scale_color_viridis_d(option = "magma") +
  labs(x = expression("mean biomass (g), " * italic("t - 1")),
       y = expression("CPUE (crabs per trap), " * italic("t - 1")),
       color = expression("action\n(number\nof traps), " * italic(t))) +
  scale_x_continuous(breaks = c(5, 10, 15), labels = c(5, 10, 15)) +
  facet_grid(action ~ month) +
  theme_minimal() +
  theme(legend.title = element_text(hjust = 0.5))
