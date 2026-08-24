# ============================================================================
# SECTION 1: Loading Libraries and Data Loading
# ============================================================================
# Loading the required libraries
library(tidyverse)      
library(lubridate)      
library(scales)         
library(gridExtra)      
library(corrplot)       
library(cluster)        
library(factoextra)     
library(caret)          
library(DataExplorer)   
library(skimr)          
library(MASS)           
library(tidymodels)     
library(dplyr)          
library(randomForest)   
library(e1071)          

# Setting working directory to where the dataset is located
setwd('C:/Users/ASUS TUF/Desktop/Iqmal Aidid/USM/Y3 Semester 1/CPC351/Assignment Project/Codes/weather_analysis_Clust_NBC')

# Loading weather dataset
weather_data <- read.csv("06_Weather.csv")

# ============================================================================
# SECTION 2: Data preparation and Pre-processing
# ============================================================================
# Initial data examination
str(weather_data)
summary(weather_data)

# Data quality assessment
cat("\nDataset Dimensions:\n")
cat("Number of rows:", nrow(weather_data), "\n")
cat("Number of columns:", ncol(weather_data), "\n")

# Missing values analysis
cat("\nMissing Values Summary:\n")
missing_summary <- colSums(is.na(weather_data))
print(missing_summary)

# Duplicate row handling
initial_rows <- nrow(weather_data)
weather_data <- weather_data %>% distinct()
removed_duplicates <- initial_rows - nrow(weather_data)
cat("Removed", removed_duplicates, "duplicate rows\n")
cat("Final number of rows:", nrow(weather_data), "\n")

# Data type verification
print(sapply(weather_data, class))

# Data validation checks
cat("\nData Validation Results:\n")
cat("Negative temperatures:", sum(weather_data$AirTemperature < 0), "\n")
cat("Invalid humidity values:", 
    sum(weather_data$RelativeHumidity < 0 | weather_data$RelativeHumidity > 100), "\n")
cat("Unrealistic pressure values:",
    sum(weather_data$BarometricPressure < 870 | weather_data$BarometricPressure > 1090), "\n")

# Timestamp conversion and feature extraction
weather_data <- weather_data %>%
  mutate(
    ObservationTimestamp = as.POSIXct(ObservationTimestamp, format="%m/%d/%Y %H:%M"),
    LocalTimestamp = as.POSIXct(LocalTimestamp, format="%m/%d/%Y %H:%M"),
    Hour = hour(LocalTimestamp),
    Month = month(LocalTimestamp),
    DayOfWeek = wday(LocalTimestamp)
  )

# Timestamp validation
cat("\nTimestamp Validation:\n")
cat("Invalid ObservationTimestamp:", sum(is.na(weather_data$ObservationTimestamp)), "\n")
cat("Invalid LocalTimestamp:", sum(is.na(weather_data$LocalTimestamp)), "\n")

# Distribution analysis visualisations
p1 <- ggplot(weather_data, aes(x=AirTemperature)) +
  geom_histogram(bins=30, fill="skyblue", color="black") +
  theme_minimal() +
  labs(title="Distribution of Air Temperature",
       x="Temperature (°C)",
       y="Count")

p2 <- ggplot(weather_data, aes(x=BarometricPressure)) +
  geom_histogram(bins=30, fill="lightgreen", color="black") +
  theme_minimal() +
  labs(title="Distribution of Barometric Pressure",
       x="Pressure (hPa)",
       y="Count")

p3 <- ggplot(weather_data, aes(x=RelativeHumidity)) +
  geom_histogram(bins=30, fill="pink", color="black") +
  theme_minimal() +
  labs(title="Distribution of Relative Humidity",
       x="Humidity (%)",
       y="Count")

grid.arrange(p1, p2, p3, ncol=2)

# Correlation analysis
numeric_cols <- weather_data %>%
  dplyr::select(AirTemperature, BarometricPressure, RelativeHumidity)

correlation_matrix <- cor(numeric_cols)
corrplot(correlation_matrix, method="color", type="upper",
         addCoef.col="black", number.cex=0.7,
         title="Correlation Matrix of Weather Variables")

# Temporal pattern analysis
p4 <- ggplot(weather_data, aes(x=factor(Hour), y=AirTemperature)) +
  geom_boxplot(fill="skyblue") +
  theme_minimal() +
  labs(title="Temperature Distribution by Hour",
       x="Hour of Day",
       y="Temperature (°C)")

p5 <- ggplot(weather_data, aes(x=factor(Hour), y=RelativeHumidity)) +
  geom_boxplot(fill="pink") +
  theme_minimal() +
  labs(title="Humidity Distribution by Hour",
       x="Hour of Day",
       y="Humidity (%)")

grid.arrange(p4, p5, ncol=2)

# Outlier detection function using IQR method
detect_outliers <- function(x) {
  Q1 <- quantile(x, 0.25)
  Q3 <- quantile(x, 0.75)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  return(x < lower_bound | x > upper_bound)
}

# Outlier analysis
weather_clean <- weather_data %>%
  mutate(across(c(AirTemperature, BarometricPressure, RelativeHumidity),
                list(outlier = detect_outliers)))

cat("\nOutlier Analysis Results:\n")
cat("Temperature outliers:", sum(weather_clean$AirTemperature_outlier), "\n")
cat("Pressure outliers:", sum(weather_clean$BarometricPressure_outlier), "\n")
cat("Humidity outliers:", sum(weather_clean$RelativeHumidity_outlier), "\n")

# Outlier visualisation
p6 <- ggplot(weather_clean, aes(y=AirTemperature)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title="Temperature Outliers")

p7 <- ggplot(weather_clean, aes(y=BarometricPressure)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title="Pressure Outliers")

p8 <- ggplot(weather_clean, aes(y=RelativeHumidity)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title="Humidity Outliers")

grid.arrange(p6, p7, p8, ncol=3)

weather_clean <- weather_data

# Time-based feature engineering
weather_clean <- weather_clean %>%
  mutate(
    TimeOfDay = case_when(
      Hour >= 5 & Hour < 12 ~ "Morning",
      Hour >= 12 & Hour < 17 ~ "Afternoon",
      Hour >= 17 & Hour < 21 ~ "Evening",
      TRUE ~ "Night"
    ),
    Season = case_when(
      Month %in% c(12, 1, 2) ~ "Summer",
      Month %in% c(3, 4, 5) ~ "Autumn",
      Month %in% c(6, 7, 8) ~ "Winter",
      Month %in% c(9, 10, 11) ~ "Spring"
    )
  )

# ============================================================================
# SECTION 3: K-means clustering
# ============================================================================
# Data preparation for clustering
clustering_data <- weather_clean %>%
  dplyr::select(AirTemperature, BarometricPressure, RelativeHumidity) %>%
  scale()

# Optimal cluster determination using elbow method
set.seed(123)
wss <- map_dbl(1:10, function(k) {
  kmeans(clustering_data, centers = k, nstart = 25)$tot.withinss
})

elbow_plot <- tibble(k = 1:10, wss = wss) %>%
  ggplot(aes(k, wss)) +
  geom_line() +
  geom_point() +
  labs(title="Elbow Method for Optimal k",
       x="Number of Clusters (k)",
       y="Total Within-Cluster Sum of Squares")
print(elbow_plot)

# K-means clustering implementation
k <- 4  
kmeans_result <- kmeans(clustering_data, centers = k, nstart = 25)

# Cluster characterisation
cluster_characteristics <- data.frame(
  cluster = 1:k,
  avg_temp = tapply(weather_clean$AirTemperature, kmeans_result$cluster, mean),
  avg_humidity = tapply(weather_clean$RelativeHumidity, kmeans_result$cluster, mean),
  avg_pressure = tapply(weather_clean$BarometricPressure, kmeans_result$cluster, mean)
)

# Cluster naming based on characteristics
cluster_names <- c(
  "1" = "Hot_Dry",        
  "2" = "Warm_Humid",     
  "3" = "Moderate",       
  "4" = "Cool_Humid"      
)

# Cluster assignment
weather_clean$Cluster <- factor(kmeans_result$cluster)
weather_clean$ClusterName <- factor(cluster_names[as.character(weather_clean$Cluster)])

# Cluster visualisation
p9 <- fviz_cluster(kmeans_result, data = clustering_data,
                   geom = "point",
                   ellipse.type = "convex",
                   palette = "Set2",
                   ggtheme = theme_minimal())
print(p9)

# ============================================================================
# SECTION 4: Naive Bayes Classifier
# ============================================================================
# Data partitioning for classification
set.seed(123)
train_index <- createDataPartition(weather_clean$ClusterName, p = 0.6, list = FALSE)
temp_data <- weather_clean[-train_index,]
val_index <- createDataPartition(temp_data$ClusterName, p = 0.5, list = FALSE)

train_data <- weather_clean[train_index,]
val_data <- temp_data[val_index,]
test_data <- temp_data[-val_index,]

# Feature selection using RFE
control <- rfeControl(functions = rfFuncs,
                      method = "cv",
                      number = 5)

features <- train_data %>%
  dplyr::select(AirTemperature, BarometricPressure, RelativeHumidity,
                Hour, Month, DayOfWeek)

rfe_result <- rfe(x = features,
                  y = train_data$ClusterName,
                  sizes = c(1:6),
                  rfeControl = control)

print("Feature Selection Results:")
print(rfe_result)

selected_features <- predictors(rfe_result)

# Feature standardisation for Naive Bayes
preprocess_recipe <- recipe(~ ., data = train_data %>% 
                              dplyr::select(all_of(selected_features))) %>%
  step_normalize(all_numeric_predictors())

prep_recipe <- prep(preprocess_recipe)
train_data_scaled <- bake(prep_recipe, new_data = train_data %>% 
                            dplyr::select(all_of(selected_features)))
val_data_scaled <- bake(prep_recipe, new_data = val_data %>% 
                          dplyr::select(all_of(selected_features)))
test_data_scaled <- bake(prep_recipe, new_data = test_data %>% 
                           dplyr::select(all_of(selected_features)))

# Adding target variable to scaled datasets
train_data_scaled$ClusterName <- train_data$ClusterName
val_data_scaled$ClusterName <- val_data$ClusterName
test_data_scaled$ClusterName <- test_data$ClusterName

# Model formula creation
formula_str <- paste("ClusterName ~", paste(selected_features, collapse = " + "))
formula_obj <- as.formula(formula_str)

# Naive Bayes model training
nb_model <- naiveBayes(formula_obj, data = train_data_scaled)

# Model evaluation
val_pred <- predict(nb_model, val_data_scaled)
val_cm <- confusionMatrix(val_pred, val_data_scaled$ClusterName)
print("Validation Set Performance:")
print(val_cm)

test_pred <- predict(nb_model, test_data_scaled)
test_cm <- confusionMatrix(test_pred, test_data_scaled$ClusterName)
print("Test Set Performance:")
print(test_cm)

# Confusion matrix visualisation
conf_matrix <- as.data.frame(test_cm$table)
conf_matrix_plot <- ggplot(conf_matrix, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = sprintf("%d", Freq)), vjust = 1) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  theme_minimal() +
  labs(title = "Confusion Matrix - Naive Bayes Classification",
       x = "Actual Class",
       y = "Predicted Class",
       fill = "Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
print(conf_matrix_plot)

# Model interpretation
print("Naive Bayes Model Details:")
print(nb_model)

# Feature distribution visualisation
for(feature in selected_features) {
  if(is.numeric(train_data[[feature]])) {
    p <- ggplot(train_data, aes(x = .data[[feature]], fill = ClusterName)) +
      geom_density(alpha = 0.5) +
      theme_minimal() +
      labs(title = paste("Distribution of", feature, "by Weather Pattern"))
    print(p)
  }
}

# ============================================================================
# SECTION 5: Results and Visualisation
# ============================================================================
# Cluster analysis results
cluster_summary <- weather_clean %>%
  group_by(ClusterName) %>%
  summarise(
    avg_temp = mean(AirTemperature),
    avg_pressure = mean(BarometricPressure),
    avg_humidity = mean(RelativeHumidity),
    count = n()
  )

print("Cluster Characteristics:")
print(cluster_summary)

# Temporal pattern visualisation
p10 <- ggplot(weather_clean, aes(x = Hour, fill = ClusterName)) +
  geom_bar(position = "fill") +
  theme_minimal() +
  labs(title = "Distribution of Weather Patterns by Hour",
       x = "Hour of Day",
       y = "Proportion")

p11 <- ggplot(weather_clean, aes(x = Season, fill = ClusterName)) +
  geom_bar(position = "fill") +
  theme_minimal() +
  labs(title = "Distribution of Weather Patterns by Season",
       x = "Season",
       y = "Proportion")

grid.arrange(p10, p11, ncol=2)

# Results storage
saveRDS(weather_clean, "weather_clean.rds")
saveRDS(kmeans_result, "kmeans_model.rds")
saveRDS(nb_model, "naive_bayes_model.rds")

# Analysis summary generation (saving the summary of project in folder as TXT file)
sink("analysis_summary.txt")
cat("Weather Data Analysis Summary\n\n")
cat("1. Data Overview:\n")
print(skim(weather_clean))
cat("\n2. Clustering Results:\n")
print(cluster_summary)
cat("\n3. Classification Performance:\n")
print(test_cm)
sink()

