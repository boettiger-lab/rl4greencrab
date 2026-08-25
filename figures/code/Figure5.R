library(tidyverse)
library(viridis)
library(patchwork)
library(ggpubr)

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

action_names <- c("act0_real" = "Minnow", "act1_real" = "Fukui")



scale <- sort(unique(data_long$a))

hull_plot <- ggplot(data_long, aes(x = biomass_real, y = cpue_real, 
                                   color = factor(a))) +
  stat_chull(aes(fill = factor(a)), alpha = 0.4, geom = "polygon") +
  scale_fill_viridis_d(option = "magma",
                       breaks = c(scale[1], scale[9], scale[18], scale[27]),
                       labels = c(round(scale[1]), round(scale[9]), 
                                  round(scale[18]), round(scale[27]))) +
  scale_color_viridis_d(option = "magma",
                        breaks = c(scale[1], scale[9], scale[18], scale[27]),
                        labels = c(round(scale[1]), round(scale[9]), 
                                   round(scale[18]), round(scale[27]))) +
  labs(x = expression("mean biomass (g), " * italic("t - 1")),
       y = expression("CPUE (crabs per trap), " * italic("t - 1")),
       color = expression("action\n(number\nof traps), " * italic(t)),
       fill = expression("action\n(number\nof traps), " * italic(t))) +
  facet_grid(action ~ months, labeller = labeller(months = month_names,
                                                  action = action_names)) +
  theme_minimal() +
  theme(legend.title = element_text(hjust = 0.5))

ggsave("figures/supp_figure_cluster.png",
       hull_plot, height = 3, width = 8)

scale_sub <- sort(unique(data_long[data_long$months %in% c(4, 6, 8, 10), ]$a))

hull_plot_sub <- ggplot(data_long[data_long$months %in% c(4, 6, 8, 10), ], 
                        aes(x = biomass_real, y = cpue_real, 
                                   color = factor(a))) +
  stat_chull(aes(fill = factor(a)), alpha = 0.4, geom = "polygon") +
  scale_fill_viridis_d(option = "magma",
                       breaks = c(scale_sub[1], scale_sub[5], 
                                  scale_sub[10], scale_sub[16]),
                       labels = c(round(scale_sub[1]), round(scale_sub[5]), 
                                  round(scale_sub[10]), round(scale_sub[16]))) +
  scale_color_viridis_d(option = "magma",
                        breaks = c(scale_sub[1], scale_sub[5], 
                                   scale_sub[10], scale_sub[16]),
                        labels = c(round(scale_sub[1]), round(scale_sub[5]), 
                                   round(scale_sub[10]), round(scale_sub[16]))) +
  labs(x = expression("mean biomass (g), " * italic("t - 1")),
       y = expression("CPUE (crabs per trap), " * italic("t - 1")),
       color = expression("action\n(number\nof traps), " * italic(t)),
       fill = expression("action\n(number\nof traps), " * italic(t))) +
  facet_grid(action ~ months, labeller = labeller(months = month_names,
                                                  action = action_names)) +
  ggtitle("A. Discrete, cluster-based policy") +
  theme_minimal() +
  theme(legend.title = element_text(hjust = 0.5),
        plot.title = element_text(size = 12))


###############
# reward plot #
###############

# read in constant action data
const_data <- read.csv("data/constant_action/const_agent_simulations.csv") %>% 
  filter(t == 99) %>% 
  mutate(type = "constant")

# read in top tqc algo
top_data <- read.csv("data/rl_policies/count-biomass-time/tqc_count-biomass-time_sim_3.csv") %>% 
  filter(t == 99) %>% 
  mutate(type = "RL")

# subset to final timestep
data_sub <- data[data$t == 99, ] %>% 
  mutate(type = "discrete")

# combine all
all_histo <- rbind(const_data[, c("rew", "type")],
                   top_data[, c("rew", "type")],
                   data_sub[, c("rew", "type")])

# change order of items in legend
all_histo$type <- factor(all_histo$type, 
                         levels = c("constant", "RL", "discrete"))

histo_plot <- ggplot(data = all_histo) +
  geom_density(aes(x = rew, fill = type), alpha = 0.4, 
               adjust = 1.5) +
  ggtitle("B. Cumulative reward") +
  labs(x = "cumulative reward", y = "density", fill = "policy type") +
  scale_fill_manual(values = c("black", "green4", "cornflowerblue")) +
  theme_minimal() +
  theme(plot.title = element_text(size = 12))


###########
# combine #
###########

figure5 <- hull_plot_sub + histo_plot + plot_layout(ncol = 1)

ggsave("figures/figure5.png",
       figure5, height = 6, width = 6)


