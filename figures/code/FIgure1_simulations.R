library(tidyverse)
library(viridis)
library(patchwork)
library(stringr)
library(legendry)

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

ggplot(data = df) +
  geom_violin(aes(x = interaction(month, year), y = N, fill = factor(size), 
                  color = factor(size))) +
  scale_x_discrete(guide = "axis_nested") +
  scale_color_discrete(labels = c("> 65", "<= 65")) +
  scale_fill_discrete(labels = c("> 65", "<= 65")) +
  labs(x = "time (year/month)", y = "count", color = "crab size\n(mm)",
       fill = "crab size\n(mm)") +
  theme_minimal()




