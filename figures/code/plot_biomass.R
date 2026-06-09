library(tidyverse)

y <- seq(2.5, 110, 5)

biomass <- pmax(0, -0.071 * y + 0.003 * y^2 + 0.00002 * y^3)

biomass_plot <- ggplot() +
  geom_line(aes(x = y, y = biomass)) +
  labs(x = "size (carapace width, mm)", y = "biomass (g)") +
  theme_minimal()
ggsave("figures/supp_figure_biomass.png", biomass_plot, 
       dpi = 300, width = 3, height = 3)
