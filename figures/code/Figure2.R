library(tidyverse)
library(viridis)
library(patchwork)

# function to read in data
read_data <- function(filepath, algo, obs_type, month = TRUE) {
  
  if (month) {
    data <- read.csv(filepath) %>% 
      filter(months >= 3, t == 99) %>% 
      mutate(algo = algo,
             obs_type = obs_type) %>% 
      select(rew, algo, obs_type)
  } else {
    data <- read.csv(filepath) %>% 
      filter(t == 99) %>% 
      mutate(algo = algo,
             obs_type = obs_type) %>% 
      select(rew, algo, obs_type)
  }
  
  return(data)
  
}

# read in data
data <- rbind(
  read_data("model_evaluations/state21/RecurrentPPO_state21.csv",
            "rppo", "21"),
  read_data("model_evaluations/state21/TD3_state21.csv",
            "td3", "21"),
  read_data("model_evaluations/state21/PPO_state21.csv",
            "ppo", "21"),
  read_data("model_evaluations/state21/TQC_state21.csv",
            "tqc", "21"),
  read_data("model_evaluations/state2/RecurrentPPO_state2.csv",
            "rppo", "2"),
  read_data("model_evaluations/state2/TD3_state2.csv",
            "td3", "2"),
  read_data("model_evaluations/state2/PPO_state2.csv",
            "ppo", "2"),
  read_data("model_evaluations/state2/TQC_state2.csv",
            "tqc", "2"),
  read_data("model_evaluations/state1_time/rppo_state1time.csv",
            "rppo", "1time"),
  read_data("model_evaluations/state1_time/td3_state1time.csv",
            "td3", "1time"),
  read_data("model_evaluations/state1_time/ppo_state1time.csv",
            "ppo", "1time"),
  read_data("model_evaluations/state1_time/tqc_state1time.csv",
            "tqc", "1time"),
  read_data("model_evaluations/state1_notime/rppo_state1.csv",
            "rppo", "1", month = FALSE),
  read_data("model_evaluations/state1_notime/td3_state1.csv",
            "td3", "1", month = FALSE),
  read_data("model_evaluations/state1_notime/ppo_state1.csv",
            "ppo", "1", month = FALSE),
  read_data("model_evaluations/state1_notime/tqc_state1.csv",
            "tqc", "1", month = FALSE)
  
)

# get means
data_means <- data %>% 
  group_by(obs_type, algo) %>% 
  summarise(mean_reward = mean(rew))


##########
# reward #
##########

figure2 <- ggplot() +
  geom_density(data = data[data$algo == "tqc", ],
               aes(x = rew, fill = obs_type), 
               alpha = 0.4, adjust = 2) +
  geom_vline(data = data_means[data_means$algo == "tqc", ],
             aes(xintercept = mean_reward, color = obs_type),
             linewidth = 1, linetype = "dashed",
             show.legend = FALSE) +
  scale_fill_viridis_d(
    labels = c("1" = expression(O[1]),
               "1time" = expression(O[1]^T),
               "2" = expression(O[2]^T),
               "21" = expression(O[21]^T))
  ) +
  scale_color_viridis_d() + 
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
  "rppo" = "Recurrent Proximal\nPolicy Optimization (RPPO)"
)

supplemental <- ggplot() +
  geom_density(data = data, 
               aes(x = rew, fill = obs_type), 
               alpha = 0.4, 
               adjust = 2) +
  geom_vline(data = data_means, 
             aes(xintercept = mean_reward, color = obs_type),
             linewidth = 1, 
             linetype = "dashed",
             show.legend = FALSE) +
  scale_fill_viridis_d(
    labels = c("1" = expression(O[1]),
               "1time" = expression(O[1]^T),
               "2" = expression(O[2]^T),
               "21" = expression(O[21]^T))
  ) +
  scale_color_viridis_d() + 
  labs(x = "reward", y = "density", fill = "observation\ntype") +
  facet_wrap(~algo, ncol = 1, 
             labeller = as_labeller(algo_names)) + 
  theme_minimal() +
  theme(legend.title = element_text(hjust = 0.5),
        legend.key.spacing.y = unit(0.2, "cm"))

ggsave("figures/supp_figure_reward.png",
       supplemental, height = 8, width = 4)

