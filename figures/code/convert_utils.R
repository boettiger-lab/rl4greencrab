#########################################
# functions for translating data scales #
#########################################

# convert actions
convert_action <- function(data, params) {
  
  data$act0_real <- pmax(params$max_action * (1 + data$act0) / 2, 0)
  data$act1_real <- pmax(params$max_action * (1 + data$act1) / 2, 0)
  
  return(data)
}

# calculate biomass as a function of size
calc_biomass <- function(y) {
  
  biomass <- max(0, -0.071 * y + 0.003 * y ^ 2 + 0.00002 * y ^ 3)
  
  return(biomass)
}

# convert biomass
convert_biomass <- function(data, params) {
  
  bmin <- calc_biomass(params$smin)
  bmax <- calc_biomass(params$smax)
  
  data$biomass_real <- (
    (data$biomass + 1) * (bmax - bmin) / 2 + bmin
  )
  
  return(data)
}

# convert CPUE
convert_cpue <- function(data) {
  
  data$cpue_real <- (data$CPUE + 1) / 2 * 100
  
  return(data)
}

# convert all
convert_all <- function(data, params) {
  
  data <- convert_action(data, params)
  
  data <- convert_biomass(data, params)
  
  data <- convert_cpue(data)
  
  return(data)
}