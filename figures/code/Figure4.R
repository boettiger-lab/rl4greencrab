library(tidyverse)
library(viridis)
library(patchwork)

source("figures/code/convert_utils.R")

# read in data of best TQC/count-biomass-time replicate
data <- read.csv("data/rl_policies/tqc_clean.csv")

# remove anomalous biomass data
data <- data[!c(data$biomass == -1 | data$biomass > -0.46), ]

params <- list(
  max_action = 3000,
  smin = 5,
  smax = 110
)


##########################
# make scale conversions #
##########################

data <- convert_all(data, params)

# change action to t+1
data <- data[-c(1:5), ]

# convert from wide to long
data_long <- data %>% 
  pivot_longer(cols = c(act0_real, act1_real),
               names_to = "action_type",
               values_to = "action")


###############
# plot months #
###############

month_names <- c("4" = "Apr", "5" = "May",
                 "6" = "June", "7" = "July", "8" = "Aug", 
                 "9" = "Sep", "10" = "Oct")
action_names <- c("act1_real" = "Fukui traps", 
                  "act0_real" = "Minnow traps")

figure4 <- ggplot(data_long) + 
  geom_point(aes(x = biomass_real, y = cpue_real, 
                 color = action)) +
  scale_color_viridis() +
  labs(x = expression("mean biomass (g), " * italic("t - 1")),
       y = expression("CPUE (crabs per trap), " * italic("t - 1")),
       color = expression("action\n(number\nof traps), " * italic(t))) +
  scale_x_continuous(breaks = c(5, 10, 15),
                     labels = c(5, 10, 15)) +
  facet_grid(action_type ~ months, 
             labeller = labeller(months = month_names, 
                                 action_type = action_names)) +
  theme_minimal() +
  theme(legend.title = element_text(hjust = 0.5))


ggsave("figures/figure4.png",
       figure4, height = 3, width = 8)


