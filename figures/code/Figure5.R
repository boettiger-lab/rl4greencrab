library(tidyverse)
library(viridis)
library(patchwork)

source("figures/code/clean_utils.R")
source("figures/code/convert_utils.R")

# read in cluster centroid data
centroids <- read.csv("data/cluster/centroids.csv")

# read in cluster sim data
cl_data <- read.csv("data/cluster/clustered_sim.csv")

# separate CPUE and biomass
clean <- cbind(cl_data, t(sapply(cl_data$crabs, clean_crab_pop)))
rownames(clean) <- NULL

colnames(clean)[which(colnames(clean) == "1")] <- "CPUE"
colnames(clean)[which(colnames(clean) == "2")] <- "biomass"

params <- list(
  max_action = 3000,
  smin = 5,
  smax = 110
)

##########################
# make scale conversions #
##########################

data <- convert_all(clean, params)

# convert to long
data_long <- data %>% 
  select(t, months, rew, rep, act0_real, 
         act1_real, biomass_real, cpue_real) %>% 
  pivot_longer(cols = -c(t, months, rew, rep, biomass_real, cpue_real),
               names_to = "action",
               values_to = "a")

# remove 0 biomass
data_long <- data_long[data_long$biomass_real > 0, ]


###############
# plot months #
###############

month_names <- c("4" = "Apr", "5" = "May",
                 "6" = "June", "7" = "July", "8" = "Aug", 
                 "9" = "Sep", "10" = "Oct")

# update order of action and month
#data$action <- factor(data$action, levels = rev(levels(factor(data$action))))
data_long$months <- factor(data_long$months, levels = c("Apr", "May", "June", 
                                                        "July", "Aug", "Sep", 
                                                        "Oct"))

ggplot(data_long) +
  geom_point(aes(x = biomass_real, y = cpue_real, color = a)) +
  scale_color_viridis(option = "magma") +
  facet_grid(action ~ months, labeller = labeller(months = month_names))

ggplot(data_long) +
  geom_density_2d_filled(aes(x = biomass_real, y = cpue_real, fill = factor(a))) +
  scale_fill_viridis_d(option = "magma") +
  facet_grid(action ~ months, labeller = labeller(months = month_names))

figure5 <- ggplot(data) + 
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

library(ggpubr)

ggplot(data_long, aes(x = biomass_real, y = cpue_real, color = factor(a))) +
  #geom_point() +
  scale_fill_viridis_d(option = "magma") +
  scale_color_viridis_d(option = "magma") +
  stat_chull(aes(fill = factor(a)), alpha = 0.2, geom = "polygon") +
  facet_grid(action ~ months, labeller = labeller(months = month_names))

ggsave("figures/figure5.png",
       figure5, height = 3, width = 8)

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
