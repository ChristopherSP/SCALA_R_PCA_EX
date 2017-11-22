####################################################################################
#                                       PCA
####################################################################################

# Plot Iris dataset
pairs(iris[, 1:4], 
      col = iris$Species, 
      lower.panel = NULL)

# Plot normalized Iris dataset
pairs(as.data.frame(scale(iris[, 1:4])),
      col = iris$Species,
      lower.panel = NULL)

# Creates PCA object with normalizing data
pc = prcomp(iris[, 1:4],
            center = T,
            scale. = T)

# Plot principal components
pairs(as.data.frame(pc$x),
      col = iris$Species,
      upper.panel = NULL)

# Another way of plotting
plot(as.data.frame(pc$x), 
     col = iris$Species)

# Information criteria to select number of dimensions
summary(pc)

####################################################################################
#                                     Biplot
####################################################################################

# Import plot Libraries
library(ggplot2)
library(ggfortify)

# Makes Biplot with index vector and explaned variability
autoplot(
  pc,
  data = iris,
  colour = 'Species',
  loadings = TRUE,
  loadings.colour = 'darkblue',
  loadings.label = TRUE,
  loadings.label.size = 3.5,
  loadings.label.colour = 'black'
) + xlab(paste0(
  'CP1 (',
  round(summary(pc)$importance['Proportion of Variance', 1], 4) * 100,
  '% variância explicada)'
)) + ylab(paste0(
  'CP2 (',
  round(summary(pc)$importance['Proportion of Variance', 2], 4) * 100,
  '% variância explicada)'
)) + theme_bw() + scale_color_brewer(palette = "Set1")


####################################################################################################
#                                             Hands On
####################################################################################################

# Read Wine data
wine.fl <- "http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
wine <- read.csv(wine.fl,header = F)

# Names of the variables
wine.names=c("Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
             "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins",
             "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline")

# Change variables names
colnames(wine)[2:14]=wine.names
# Define label colname
colnames(wine)[1]="Class"
