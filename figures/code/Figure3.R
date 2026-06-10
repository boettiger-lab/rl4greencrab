library(tidyverse)
library(viridis)
library(patchwork)

# read in data
data <- read.csv("model_evaluations/tqc_clean.csv")

# remove anomalous biomass data
data <- data[!c(data$biomass == -1 | data$biomass > -0.2), ]

params <- list(
  max_action = 3000,
  smin = 5,
  smax = 110
)

#########################################
# functions for translating data scales #
#########################################

# convert actions
convert_action <- function(data, params) {
  
  data$act0_real <- pmax(params$max_action * (1 + data$act0) / 2, 0)
  data$act1_real <- pmax(params$max_action * (1 + data$act1) / 2, 0)
  
  return(data)
}

# calculate biomass as a function of size
calc_biomass <- function(y) {
  
  biomass <- max(0, -0.071 * y + 0.003 * y ^ 2 + 0.00002 * y ^ 3)
  
  return(biomass)
}

# convert biomass
convert_biomass <- function(data, params) {
  
  bmin <- calc_biomass(params$smin)
  bmax <- calc_biomass(params$smax)
  
  data$biomass_real <- (
    (data$biomass + 1) * (bmax - bmin) / 2 + bmin
  )
  
  return(data)
}

# convert CPUE
convert_cpue <- function(data) {
  
  data$cpue_real <- (data$CPUE + 1) / 2 * 100
  
  return(data)
}

# convert all
convert_all <- function(data, params) {
  
  data <- convert_action(data, params)
  
  data <- convert_biomass(data, params)
  
  data <- convert_cpue(data)
  
  return(data)
}


##########################
# make scale conversions #
##########################

data <- convert_all(data, params)

# convert from wide to long
data_long <- data %>% 
  pivot_longer(cols = c(act0_real, act1_real),
               names_to = "action_type",
               values_to = "action")


###############
# plot months #
###############

month_names <- c("3" = "Mar", "4" = "Apr", "5" = "May",
                 "6" = "June", "7" = "July", "8" = "Aug", 
                 "9" = "Sep", "10" = "Oct", "11" = "Nov")
action_names <- c("act1_real" = "Fukui traps", 
                  "act0_real" = "Minnow traps")

figure3 <- ggplot(data_long) + 
  geom_point(aes(x = biomass_real, y = action, 
                 color = cpue_real)) +
  scale_color_viridis() +
  labs(x = "mean biomass", 
       y = "action (number of traps)",
       color = "CPUE\n(crabs\nper trap)") +
  facet_grid(action_type ~ months, 
             labeller = labeller(months = month_names, 
                                 action_type = action_names)) +
  theme(legend.title = element_text(hjust = 0.5))


ggsave("figures/figure3.png",
       figure3, height = 3, width = 8)


