library(tidyverse)
library(viridis)
library(patchwork)
library(stringr)
library(legendry)

#########################
# no action time series #
#########################

# read in data with no action
data <- read.csv("data/constant_action/no_agent_simulations.csv")

# remove times before colonization
data <- data[c(data$t > 7 & data$t < 99), ]

sizes <- seq(5, 110, 5)

# reformat crab pop
parse_array <- function(x) {
  x %>% 
    str_remove_all("[\\[\\]]") %>% 
    str_replace_all("\\\\n", " ") %>% 
    str_trim() %>% 
    str_split("\\s+")  
}

# parse every row, convert to numeric, bind into a matrix
parsed <- lapply(data$crab_pop, function(s) as.numeric(parse_array(s)[[1]]))
df <- as.data.frame(cbind(do.call(rbind, parsed), data$t, data$rep, 
                          data$months))

# add year
year_seq <- rep(1:13, each = 7)
df$year <- rep(year_seq, 100)


df <- df %>% 
  mutate(small_crabs = rowSums(across(1:13)),
         large_crabs = rowSums(across(14:22))) %>% 
  select(V23, V24, V25, year, small_crabs, large_crabs) %>% 
  pivot_longer(cols = -c(V23, V24, V25, year),
               names_to = "size",
               values_to = "N")

colnames(df)[1:3] <- c("t", "rep", "month")

# set axis labels
lev <- levels(interaction(df$month, df$year))
month_part <- as.integer(sub("\\..*", "", lev))
year_part  <- sub(".*\\.", "", lev)

disp <- ifelse(month_part %% 2 == 1,
               paste0(month_part, ".", year_part),
               paste0(" ", ".", year_part))
names(disp) <- lev

# plot entire time series
timeseries_plot <- ggplot(data = df) +
  geom_violin(aes(x = interaction(month, year), y = N, fill = factor(size), 
                  color = factor(size))) +
  scale_x_discrete(guide = "axis_nested", labels = disp) +
  scale_color_discrete(labels = c("> 65", "<= 65")) +
  scale_fill_discrete(labels = c("> 65", "<= 65")) +
  labs(x = "time (year/month)", y = "crab abundance", color = "crab size\n(mm)",
       fill = "crab size\n(mm)") +
  theme_minimal()

# plot year 6
plot_year6 <- ggplot(data = df[df$year == 6, ]) +
  geom_violin(aes(x = factor(month), y = N, fill = factor(size), 
                  color = factor(size))) +
  scale_color_discrete(labels = c("> 65", "<= 65")) +
  scale_fill_discrete(labels = c("> 65", "<= 65")) +
  labs(x = "month", y = "crab abundance", color = "crab size\n(mm)",
       fill = "crab size\n(mm)") +
  scale_y_continuous(breaks = c(0, 20000, 40000)) +
  ggtitle("Year 6") +
  theme_minimal() + 
  theme(plot.background = element_rect(color = "black", fill = "white", 
                                       linewidth = 1),
        legend.position = "None",
        axis.text = element_text(size = 6),
        axis.title = element_text(size = 8),
        title = element_text(size = 10))

plot_A <- timeseries_plot + 
  inset_element(plot_year6, left = 0.6, bottom = 0.6, right = 0.98, top = 0.98)

########################
# 50 traps time series #
########################

# read in data with 50 traps
data_t50 <- read.csv("data/constant_action/t50_agent_simulations.csv")


ggsave("figures/figure1_timeseries.png", width = )



