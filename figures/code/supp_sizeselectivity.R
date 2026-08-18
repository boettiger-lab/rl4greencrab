library(tidyverse)
library(bayestestR)
library(patchwork)
library(cowplot)

# read in samples
out <- read.csv("data/posterior/params.csv")

# make plot of size selectivity
min_size <- 0
max_size <- 110
x <- seq(min_size, max_size, 1)
selective_summaries <- data.frame(
  type = c("minnow", "fukui",
           "minnow", "fukui",
           "minnow", "fukui"),
  stat = c(rep("median", 2), rep("lower_ci", 2), rep("upper_ci", 2))
)
selective_summaries <- cbind(
  selective_summaries,
  as.data.frame(matrix(NA, nrow = dim(selective_summaries), ncol = length(x)))
)
colnames(selective_summaries)[3:113] <- x
size_sel_norm <- function(pmax, xmax, sigma) {
  vector <- pmax * exp(-(x - xmax) ^ 2 / (2 * sigma ^ 2))
  return(1 - exp(-vector))
}
size_sel_log <- function(pmax, k, midpoint) {
  vector <- pmax / (1 + exp(-k * (x - midpoint)))
  return(1 - exp(-vector))
}
get_hdi_low <- function(input, ci) {
  out <- as.numeric(hdi(input, ci)[2])
  return(out)
}
get_hdi_high <- function(input, ci) {
  out <- as.numeric(hdi(input, ci)[3])
  return(out)
}
minnow_select <- mapply(size_sel_norm, out[, "trapm_pmax"],
                        out[, "trapm_xmax"], out[, "trapm_sigma"])
fukui_select <- mapply(size_sel_log, out[, "trapf_pmax"],
                       out[, "trapf_k"], out[, "trapf_midpoint"])
selective_summaries[1, 3:113] <- apply(minnow_select, 1, median)
selective_summaries[2, 3:113] <- apply(fukui_select, 1, median)
selective_summaries[3, 3:113] <- apply(minnow_select, 1,
                                       function(row) get_hdi_low(row, 0.95))
selective_summaries[4, 3:113] <- apply(fukui_select, 1,
                                       function(row) get_hdi_low(row, 0.95))
selective_summaries[5, 3:113] <- apply(minnow_select, 1,
                                       function(row) get_hdi_high(row, 0.95))
selective_summaries[6, 3:113] <- apply(fukui_select, 1,
                                       function(row) get_hdi_high(row, 0.95))

# convert from wide to long
selective_summaries_long <- selective_summaries %>%
  pivot_longer(cols = ! c("type", "stat"),
               values_to = "p",
               names_to = "size") %>%
  pivot_wider(values_from = "p",
              names_from = "stat")

# size selectivity plot
sizesel_plot <- ggplot(data = selective_summaries_long) +
  geom_ribbon(aes(x = as.numeric(size),
                  ymin = lower_ci, ymax = upper_ci,
                  fill = as.factor(type)), alpha = 0.4) +
  geom_line(aes(x = as.numeric(size), y = median,
                color = as.factor(type)),
            linewidth = 1) +
  scale_fill_manual(values = c("violet", "goldenrod"),
                    guide = "none") +
  scale_color_manual(values = c("violet", "goldenrod"),
                     labels = c("Fukui", "Minnow")) +
  labs(x = "size (mm)", y = "probability of capture", color = "trap type") +
  theme_minimal() +
  theme(text = element_text(size = 14, family = "Arial"))

ggsave("figures/supp_sizesel.png", sizesel_plot,
       dpi = 400, width = 5, height = 4,
       bg = "white")
