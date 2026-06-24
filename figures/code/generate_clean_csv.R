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

# read in data of best TQC/count-biomass-time replicate
data <- clean_all(read.csv(
  "data/rl_policies/count-biomass-time/tqc_count-biomass-time_sim_3.csv")
)


write.csv(data, "data/rl_policies/tqc_clean.csv")
