library(tidyverse)
library(patchwork)

#################
# s transitions #
#################

y <- seq(2.5, 105, 5)

N1 <- 80 * dnorm(y, 15, 5) + 60 * dnorm(y, 60, 10)
N2 <- 80 * dnorm(y, 20, 7) + 40 * dnorm(y, 65, 12)
 
plot_s <- ggplot() +
  geom_col(aes(x = y, y = N1), fill = "#968723") +
  scale_y_continuous(limits = c(0, max(N1))) +
  scale_x_continuous(limits = c(0, 100)) +
  labs(x = "crab size", y = "N") +
  theme_minimal() +
  theme(axis.text = element_blank(),
        axis.ticks = element_blank())
ggsave("figures/figure1_plot_s.svg", plot_s, height = 2, width = 3)

plot_splus <- ggplot() +
  geom_col(aes(x = y, y = N2), fill = "#968723") +
  scale_y_continuous(limits = c(0, max(N1))) +
  scale_x_continuous(limits = c(0, 100)) +
  labs(x = "crab size", y = "N") +
  theme_minimal() +
  theme(axis.text = element_blank(),
        axis.ticks = element_blank())
ggsave("figures/figure1_plot_splus.svg", plot_splus, 
       height = 2, width = 3)


##########
# reward #
##########

params <- list (
  area = 30000,
  a = 0.265,
  b = 2.80, 
  c = 2.99
)

bmass <- seq(0, 2e5, 1000) 
eco_change <- (
  -params$a / (1 + exp(-params$b * (bmass / params$area - params$c)))
)

plot_eco <- ggplot() +
  geom_line(aes(x = bmass, y = eco_change),
            linewidth = 1.5) +
  labs(x = "crab biomass", 
       y = expression(reward ~ (r[E]))) +
  theme_minimal() +
  theme(axis.text = element_blank(),
        axis.ticks = element_blank(),
        axis.title = element_text(size = 16))

plot_cost <- ggplot() +
  geom_line(aes(x = -bmass, y = bmass),
            linewidth = 1.5) +
  labs(x = "number of traps", 
       y = expression(reward ~ (r[C]))) +
  theme_minimal() +
  theme(axis.text = element_blank(),
        axis.ticks = element_blank()) +
  theme(axis.text = element_blank(),
        axis.ticks = element_blank(),
        axis.title = element_text(size = 16))

reward_plot <- plot_eco + plot_cost + plot_layout(nrow = 1)
ggsave("figures/figure1_plot_reward.svg", reward_plot, 
       height = 2, width = 4)
