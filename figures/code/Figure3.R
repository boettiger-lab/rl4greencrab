library(httr)
library(jsonlite)
library(purrr)
library(readr)
library(dplyr)
library(viridis)
library(patchwork)
library(tidyverse)

# set up connection to hugging face to read in data
repo <- "boettiger-lab/rl4eco"
base_path <- "rl4greencrab/data/rl_policies"
subfolders <- c("size-time", "count", "count-biomass-time", "count-time")

api_base <- paste0("https://huggingface.co/api/models/", repo, "/tree/main/")
resolve_base <- paste0("https://huggingface.co/", repo, "/resolve/main/")

# get list of csv files across all subfolders via the API
csv_paths <- map(subfolders, function(sub) {
  listing <- fromJSON(paste0(api_base, base_path, "/", sub))
  listing$path[listing$type == "file" & grepl("\\.csv$", listing$path)]
}) |> unlist()

# read in rl simulation data and add metadata columns
combined_df <- map_dfr(csv_paths, function(path) {
  df <- read_csv(paste0(resolve_base, path), show_col_types = FALSE)
  filename <- basename(path)
  df %>%  mutate(
    source_file = filename,
    config = basename(dirname(path)), 
    algorithm = sub("_.*", "", filename), 
    sim = as.integer(sub(".*_sim_(\\d+)\\.csv", "\\1", filename))
  )
})

# add columns for obs_type
combined_df <- combined_df %>% 
  mutate(
    obs_type = sub("^[^_]+_(.*)_sim_.*", "\\1", source_file)
  ) %>% 
  filter(t == 99)
  

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

figure3_1 <- ggplot() +
  geom_density(data = data_all[c(data_all$algorithm == "tqc" &
                                   data_all$obs_type == "count") |
                                 c(data_all$algorithm == "constant"), ],
               aes(x = rew, fill = algorithm),
               alpha = 0.4, adjust = 2) +
  geom_vline(data = data_means[c(data_means$algorithm == "tqc" &
                                   data_means$obs_type == "count") |
                                 c(data_means$algorithm == "constant"), ],
             aes(xintercept = mean_reward, color = algorithm),
             linewidth = 0.75, linetype = "solid",
             show.legend = FALSE) +
  scale_fill_manual(
    values = unname(fill_colors[c("constant", "count")])
  ) +
  scale_y_continuous(breaks = c(0, 0.1, 0.2)) +
  scale_color_manual(values = unname(fill_colors[c("constant", "count")])) +
  labs(x = "", y = "density") +
  theme_minimal() +
  theme(legend.position = "None",
        axis.text.x = element_blank())

figure3_1t <- ggplot() +
  geom_density(data = data_all[c(data_all$algorithm == "tqc" &
                                   data_all$obs_type == "count-time") |
                                 c(data_all$algorithm == "constant"), ],
               aes(x = rew, fill = algorithm),
               alpha = 0.4, adjust = 2) +
  geom_vline(data = data_means[c(data_means$algorithm == "tqc" &
                                   data_means$obs_type == "count-time") |
                                 c(data_means$algorithm == "constant"), ],
             aes(xintercept = mean_reward, color = algorithm),
             linewidth = 0.75, linetype = "solid",
             show.legend = FALSE) +
  scale_fill_manual(
    values = unname(fill_colors[c("constant", "count-time")])
  ) +
  scale_y_continuous(breaks = c(0, 0.1, 0.2)) +
  scale_color_manual(values = unname(fill_colors[c("constant", 
                                                   "count-time")])) +
  labs(x = "", y = "density") +
  theme_minimal() +
  theme(legend.position = "None",
        axis.text.x = element_blank())

figure3_2t <- ggplot() +
  geom_density(data = data_all[c(data_all$algorithm == "tqc" &
                                   data_all$obs_type == "count-biomass-time") |
                                 c(data_all$algorithm == "constant"), ],
               aes(x = rew, fill = algorithm),
               alpha = 0.4, adjust = 2) +
  geom_vline(data = data_means[c(data_means$algorithm == "tqc" &
                                   data_means$obs_type == "count-biomass-time") |
                                 c(data_means$algorithm == "constant"), ],
             aes(xintercept = mean_reward, color = algorithm),
             linewidth = 0.75, linetype = "solid",
             show.legend = FALSE) +
  scale_fill_manual(
    values = unname(fill_colors[c("constant", "count-biomass-time")])
  ) +
  scale_y_continuous(breaks = c(0, 0.1, 0.2)) +
  scale_color_manual(values = unname(fill_colors[c("constant", 
                                                   "count-biomass-time")])) +
  labs(x = "", y = "density") +
  theme_minimal() +
  theme(legend.position = "None",
        axis.text.x = element_blank())

figure3_22 <- ggplot() +
  geom_density(data = data_all[c(data_all$algorithm == "tqc" &
                                   data_all$obs_type == "size-time") |
                                 c(data_all$algorithm == "constant"), ],
               aes(x = rew, fill = algorithm),
               alpha = 0.4, adjust = 2) +
  geom_vline(data = data_means[c(data_means$algorithm == "tqc" &
                                   data_means$obs_type == "size-time") |
                                 c(data_means$algorithm == "constant"), ],
             aes(xintercept = mean_reward, color = algorithm),
             linewidth = 0.75, linetype = "solid",
             show.legend = FALSE) +
  scale_fill_manual(
    values = unname(fill_colors[c("constant", "size-time")])
  ) +
  scale_y_continuous(breaks = c(0, 0.1, 0.2)) +
  scale_color_manual(values = unname(fill_colors[c("constant", "size-time")])) +
  labs(x = "reward", y = "density") +
  theme_minimal() +
  theme(legend.position = "None")

common_x <- scale_x_continuous(limits = c(-15, -3))

figure3 <- (figure3_1 + common_x) /
  (figure3_1t + common_x) /
  (figure3_2t + common_x) /
  (figure3_22 + common_x) +
  plot_layout(ncol = 1)


ggsave("figures/figure3.svg",
       figure3, height = 6.3, width = 2.625)

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
    limits = c("constant", "count", "count-time", "count-biomass-time", "size-time"),
    values = fill_colors,
    labels = c("count" = expression(O[1]),
               "count-time" = expression(O[1]^T),
               "count-biomass-time" = expression(O[2]^T),
               "size-time" = expression(O[22]^T),
               "constant" = "constant\naction")
  ) +
  scale_x_continuous(limits = c(-20, -3)) +
  scale_color_manual(values = fill_colors) +
  labs(x = "reward", y = "density", fill = "observation\ntype") +
  facet_wrap(~algorithm, ncol = 1, 
             labeller = as_labeller(algo_names)) + 
  theme_minimal() +
  theme(legend.title = element_text(hjust = 0.5),
        legend.key.spacing.y = unit(0.2, "cm"))

ggsave("figures/supp_figure_reward.png",
       supplemental, height = 8, width = 4)

