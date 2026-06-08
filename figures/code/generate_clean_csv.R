library(tidyverse)

#################################
# functions for processing data #
#################################

# clean data
clean_all <- function(data) {
  
  clean <- cbind(data, t(sapply(data$crabs, clean_crab_pop)))
  rownames(clean) <- NULL
  clean$nonlocal_value <- unname(sapply(clean$nonlocal_crab, get_nonlocal))
  
  clean$nonlocal_sd <- sapply(clean$nonlocal_crab, get_nonlocal_sd)
  
  colnames(clean)[which(colnames(clean) == "1")] <- "CPUE"
  colnames(clean)[which(colnames(clean) == "2")] <- "biomass"
  
  return(clean)
}

clean_crab_pop <- function(string) {
  
  out <- as.numeric(strsplit(gsub("\\[|\\]", "", string), "\\s+")[[1]])
  
  return(as.vector(na.omit(out)))
}

get_nonlocal <- function(string) {
  
  clean_string <- gsub("\\[|\\]", "", string)
  
  last_element_str <- tail(strsplit(clean_string, " ")[[1]], 1)
  
  num <- as.numeric(last_element_str)
  
  if (length(num) == 0) {
    num <- 0
  }
  
  return(num)
}

get_nonlocal_sd <- function(string) {
  
  clean_string <- str_remove_all(string, "\\[|\\]|\\n")
  
  numeric_vector_list <- str_split(clean_string, "\\s+")
  
  numeric_vector <- numeric_vector_list[[1]][numeric_vector_list[[1]] != ""]
  
  return(sd(as.numeric(numeric_vector)))
}

# read in data
data_rppo_var <- clean_all(read.csv(
  "model_evaluations/state2/RecurrentPPO_state2_var.csv"
) %>% filter(months >= 3))
data_td3_var <- clean_all(read.csv(
  "model_evaluations/state2/PPO_state2_var.csv"
) %>% filter(months >= 3))
data_ppo_var <- clean_all(read.csv(
  "model_evaluations/state2/TD3_state2_var.csv"
) %>% filter(months >= 3))
data_tqc_var <- clean_all(read.csv(
  "model_evaluations/state2/TQC_state2_var.csv"
) %>% filter(months >= 3))
data_rppo <- clean_all(read.csv(
  "model_evaluations/state2/RecurrentPPO_state2.csv"
) %>% filter(months >= 3))
data_td3 <- clean_all(read.csv(
  "model_evaluations/state2/PPO_state2.csv"
) %>% filter(months >= 3))
data_ppo <- clean_all(read.csv(
  "model_evaluations/state2/TD3_state2.csv"
) %>% filter(months >= 3))
data_tqc <- clean_all(read.csv(
  "model_evaluations/state2/TQC_state2.csv"
) %>% filter(months >= 3))


write.csv(data_tqc, "model_evaluations/tqc_clean.csv")
