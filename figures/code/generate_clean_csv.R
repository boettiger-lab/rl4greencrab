library(tidyverse)

source("figures/code/clean_utils.R")



# read in data of best TQC/count-biomass-time replicate
data <- clean_all(read.csv(
  "data/rl_policies/count-biomass-time/tqc_count-biomass-time_sim_3.csv")
)


write.csv(data, "data/rl_policies/tqc_clean.csv")
