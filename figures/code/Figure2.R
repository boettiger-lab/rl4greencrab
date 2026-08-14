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
  scale_color_manual(labels = c("> 65", "<= 65"), values = c("firebrick",
                                                             "goldenrod")) +
  scale_fill_manual(labels = c("> 65", "<= 65"), values = c("firebrick",
                                                            "goldenrod")) +
  labs(x = "time (year/month)", y = "crab abundance", color = "crab size\n(mm)",
       fill = "crab size\n(mm)") +
  theme_minimal()

# plot year 6
plot_year6 <- ggplot(data = df[df$year == 6, ]) +
  geom_violin(aes(x = factor(month), y = N, fill = factor(size), 
                  color = factor(size))) +
  scale_color_manual(labels = c("> 65", "<= 65"), values = c("firebrick",
                                                             "goldenrod")) +
  scale_fill_manual(labels = c("> 65", "<= 65"), values = c("firebrick",
                                                            "goldenrod")) +
  labs(x = "month", y = "crab\nabundance", color = "crab size\n(mm)",
       fill = "crab size\n(mm)") +
  scale_y_continuous(breaks = c(0, 20000, 40000)) +
  ggtitle("Year 6") +
  theme_minimal() + 
  theme(plot.background = element_rect(color = "black", fill = "white", 
                                       linewidth = 1),
        legend.position = "None",
        axis.text = element_text(size = 6),
        axis.title = element_text(size = 8),
        title = element_text(size = 8))

plot_A <- timeseries_plot + 
  # inset_element(plot_year6, left = 0.6, bottom = 0.6, 
  #               right = 0.98, top = 0.98) +
  plot_annotation(title = "A.",
                  theme = theme(plot.title = element_text(face = "bold", 
                                                        size = 14)))

########################
# 50 traps time series #
########################

# read in data with 50 traps and subset to year 3
data_t50 <- read.csv("data/constant_action/t50_agent_simulations.csv")[65:71, ]

# parse caught crabs and true N
caught <- as.data.frame(
  cbind(do.call(rbind, lapply(data_t50$crabs, 
                              function(s) as.numeric(parse_array(s)[[1]]))))
) %>% 
  mutate(month = 4:10,
         type = "caught") %>% 
  pivot_longer(cols = -c(month, type),
               names_to = "size", 
               values_to = "N")
caught$size <- as.numeric(gsub("\\D", "", caught$size)) 

true <- as.data.frame(
  cbind(do.call(rbind, lapply(data_t50$crab_pop, 
                              function(s) as.numeric(parse_array(s)[[1]]))))
) %>% 
  mutate(month = 4:10,
         type = "true") %>% 
  pivot_longer(cols = -c(month, type),
               names_to = "size", 
               values_to = "N")
true$size <- as.numeric(gsub("\\D", "", true$size)) 
  
plot_caught <- ggplot(data = caught[caught$month %in% c(5, 7, 9), ]) +
  geom_col(aes(x = size, y = N, fill = as.factor(month))) +
  facet_wrap(~ month) +
  ggtitle("Observed crab count (50 traps)") +
  scale_x_continuous(breaks = c(0, 5, 10, 15, 20),
                     labels = c(0, sizes[5], sizes[10], sizes[15], sizes[20])) +
  labs(x = "size (mm)", y = "count",
       fill = "month") +
  theme_minimal() +
  theme(strip.text = element_blank(),
        strip.background = element_blank())

plot_N <- ggplot(data = true[true$month %in% c(5, 7, 9), ]) +
  geom_col(aes(x = size, y = N, fill = as.factor(month))) +
  facet_wrap(~ month) +
  ggtitle("True crab abundance") +
  scale_x_continuous(breaks = c(0, 5, 10, 15, 20),
                     labels = c(0, sizes[5], sizes[10], sizes[15], sizes[20])) +
  labs(x = "", y = "abundance",
       fill = "month") +
  theme_minimal() +
  theme(strip.text = element_blank(),
        strip.background = element_blank())

plot_B <- plot_N + plot_caught + plot_layout(ncol = 1, guides = "collect") +
  plot_annotation(title = "B.",
                  theme = theme(plot.title = element_text(face = "bold", 
                                                          size = 14)))

# combine plots
(wrap_elements(plot_A) / wrap_elements(plot_B))

ggsave("figures/figure1_timeseries.png", width = )



