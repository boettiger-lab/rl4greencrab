library(tidyverse)
library(viridis)
library(patchwork)
library(purrr)

# read in rl simulation data
csv_files <- c(
  Sys.glob("data/rl_policies/size-time/*.csv"),
  Sys.glob("data/rl_policies/count/*.csv"),
  Sys.glob("data/rl_policies/count-biomass-time/*.csv"),
  Sys.glob("data/rl_policies/count-time/*.csv")
)

combined_df <- map(csv_files, function(filepath) {
  filename <- basename(filepath)
  
  match <- regmatches(filename, regexec("^([^_]+)_(.+)_sim_(\\d+)\\.csv$", filename))[[1]]
  
  algorithm <- match[2]
  obs_type  <- match[3]
  replicate <- as.integer(match[4])
  
  df <- read.csv(filepath)
  df <- df[df$t == 99, ]
  
  df$algorithm <- algorithm
  df$obs_type  <- obs_type
  df$replicate <- replicate
  df
}) %>%
  compact() %>%
  bind_rows()

# read in constant action data
const_data <- read.csv("data/constant_action/const_agent_simulations.csv")

# combine all data
data_all <- rbind(
  combined_df[, c("rew", "algorithm", "obs_type")],
  data.frame(
    rew = const_data$rew,
    algorithm = "constant",
    obs_type  = "constant"
  )
)

# get means
data_means <- data_all %>% 
  group_by(obs_type, algorithm) %>% 
  summarise(mean_reward = mean(rew))


##########
# reward #
##########
fill_colors <- c(
  setNames(viridis(4), c("count", "count-time", 
                         "count-biomass-time", "size-time")),
  "constant" = "black"
)

figure2 <- ggplot() +
  geom_density(data = data_all[data_all$algorithm == "tqc" |
                                 data_all$algorithm == "constant", ],
               aes(x = rew, fill = obs_type),
               alpha = 0.4, adjust = 2) +
  geom_vline(data = data_means[data_means$algorithm == "tqc" |
                                 data_means$algorithm == "constant", ],
             aes(xintercept = mean_reward, color = obs_type),
             linewidth = 1, linetype = "dashed",
             show.legend = FALSE) +
  scale_fill_manual(
    values = fill_colors,
    labels = c("count" = expression(O[1]),
               "count-time" = expression(O[1]^T),
               "count-biomass-time" = expression(O[2]^T),
               "size-time" = expression(O[22]^T),
               "constant" = "constant\naction")
  ) +
  scale_color_manual(values = fill_colors) +
  labs(x = "reward", y = "density", fill = "observation\ntype") +
  theme_minimal() +
  theme(legend.title = element_text(hjust = 0.5),
        legend.key.spacing.y = unit(0.2, "cm"))


ggsave("figures/figure2.svg",
       figure2, height = 3, width = 4)

algo_names <- c(
  "tqc" = "Truncated Quantile Critic (TQC)",
  "td3" = "Twin-delayed Deep Deterministic (TD3)",
  "ppo" = "Proximal Policy Optimization (PPO)",
  "constant" = "Constant Action"
)

supplemental <- ggplot() +
  geom_density(data = data_all, 
               aes(x = rew, fill = obs_type), 
               alpha = 0.4, 
               adjust = 2) +
  geom_vline(data = data_means, 
             aes(xintercept = mean_reward, color = obs_type),
             linewidth = 1, 
             linetype = "dashed",
             show.legend = FALSE) +
  scale_fill_manual(
    values = fill_colors,
    labels = c("count" = expression(O[1]),
               "count-time" = expression(O[1]^T),
               "count-biomass-time" = expression(O[2]^T),
               "size-time" = expression(O[22]^T),
               "constant" = "constant\naction")
  ) +
  scale_color_manual(values = fill_colors) +
  labs(x = "reward", y = "density", fill = "observation\ntype") +
  facet_wrap(~algorithm, ncol = 1, 
             labeller = as_labeller(algo_names)) + 
  theme_minimal() +
  theme(legend.title = element_text(hjust = 0.5),
        legend.key.spacing.y = unit(0.2, "cm"))

ggsave("figures/supp_figure_reward.png",
       supplemental, height = 8, width = 4)

