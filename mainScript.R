library(tidyverse)
library(dplyr)
library(ggplot2)
library(ggthemes)
library(data.table)
library(stringr)
library(caret)
library(lme4)
library(xgboost)
library(randomForest)
library(ltm)

# Set working directory
setwd("~/Library/CloudStorage/GoogleDrive-theoauyeung@gmail.com/My Drive/Rice-University/2024-25/Sport Analytics/final proj data")

# Read data
data <- fread("stats - more.csv")

# Process names (similar to original script)
data <- data %>%
  mutate(Name = str_c(str_split(`last_name, first_name`, ", ", simplify = TRUE)[,2],
                      str_split(`last_name, first_name`, ", ", simplify = TRUE)[,1], sep = " ")) %>%
  dplyr::select(-`last_name, first_name`)


# data %>%
#   # Remove percentage signs and convert to numeric
#   mutate(
#     BBpercent = as.numeric(str_replace(BBpercent, "%", "")),
#     Kpercent = as.numeric(str_replace(Kpercent, "%", "")),
#     # Ensure all relevant columns are numeric
#     across(c(AVG, k_percent, bb_percent, on_base_percent,
#              on_base_plus_slg, babip, exit_velocity_avg,
#              launch_angle_avg, hard_hit_percent, z_swing_percent,
#              whiff_percent, groundballs_percent, flyballs_percent),
#            as.numeric)
#   ) %>%
#   # Remove rows with any NA values in key columns
#   drop_na(BBpercent, Kpercent, wRCplus, wRCplus_next)


# Read year-specific data
data24 <- fread("baseball data - 2024.csv") %>% 
  mutate(year = as.integer(2024)) %>% 
  rename(
    wRCplus = `wRC+`,
    BBpercent = `BB%`,
    Kpercent = `K%`
  )
data23 <- fread("baseball data - 2023.csv") %>% 
  mutate(year = as.integer(2023)) %>% 
  rename(
    wRCplus = `wRC+`,
    BBpercent = `BB%`,
    Kpercent = `K%`
  )
data22 <- fread("baseball data - 2022.csv") %>% 
  mutate(year = as.integer(2022)) %>% 
  rename(
    wRCplus = `wRC+`,
    BBpercent = `BB%`,
    Kpercent = `K%`
  )
data21 <- fread("baseball data - 2021.csv") %>% 
  mutate(year = as.integer(2021)) %>% 
  rename(
    wRCplus = `wRC+`,
    BBpercent = `BB%`,
    Kpercent = `K%`
  )
data19 <- fread("baseball data - 2019.csv")%>% 
  mutate(year = as.integer(2019)) %>% 
  rename(
    wRCplus = `wRC+`,
    BBpercent = `BB%`,
    Kpercent = `K%`
  )
data18<- fread("baseball data - 2018.csv")%>% 
  mutate(year = as.integer(2018)) %>% 
  rename(
    wRCplus = `wRC+`,
    BBpercent = `BB%`,
    Kpercent = `K%`
  )

  


# Protected names handling (as in original script)
protected_names <- c(
  "Bobby Witt Jr.", "Vladimir Guerrero Jr.", "Jazz Chisholm Jr.",
  "Luis García Jr.", "Lourdes Gurriel Jr.", "Ronald Acuña Jr.",
  "Luis Robert Jr.", "Fernando Tatis Jr.", "LaMonte Wade Jr.", "Nelson Cruz", "Jackie Bradley Jr."
)

data$Name <- ifelse(
  data$Name %in% protected_names,
  data$Name,
  ifelse(
    grepl("Jr\\.$", data$Name),
    trimws(sub("Jr\\.$", "", data$Name)),
    data$Name
  )
)

data$Name = ifelse(data$Name == "George Springer III", "George Springer", data$Name)
data$Name = ifelse(data$Name == "Cedric Mullins II", "Cedric Mullins", data$Name)
data$Name = ifelse(data$Name == "Luke Voit III", "Luke Voit", data$Name)
data$Name = ifelse(data$Name == "Cedric Mullins II", "Cedric Mullins", data$Name)

# Merge data across years
merged_data <- rbindlist(list(data24, data23, data22, data21, data19, data18), use.names = TRUE, fill = TRUE)



# Final merged data preparation
final_merged_data <- merge(merged_data, data, by = c("Name", "year"), all = TRUE) %>%
  dplyr::select(Name, year, wRCplus, k_percent, bb_percent, 
          exit_velocity_avg, AVG, ISO,
         launch_angle_avg, hard_hit_percent, opposite_percent, pull_percent, swing_percent, whiff_percent,
         groundballs_percent, flyballs_percent)

# Prepare data for prediction of next year's wRC+
data_processed <- final_merged_data %>%
  arrange(Name, year) %>%
  group_by(Name) %>%
  mutate(
    year_next = year + 1,
    wRCplus_next = case_when(year < 2024 ~ lead(wRCplus), 
                             year == 2024 ~ NA)
  ) %>%
  # Remove rows where we don't have next season's data
  filter(!is.na(wRCplus_next))

data_processed_for_prediction <- final_merged_data %>%
  arrange(Name, year) %>%
  group_by(Name) %>%
  mutate(
    year_next = year + 1,
    wRCplus_next = case_when(year < 2024 ~ lead(wRCplus), 
                             year == 2024 ~ NA)
  ) %>%
  # Remove rows where we don't have next season's data
  filter(ifelse(year == 2024, TRUE, !is.na(wRCplus_next)))

# 
# data_processed <- data_processed %>%
#   # Remove percentage signs and convert to numeric
#   mutate(
#     BBpercent = as.numeric(str_replace(BBpercent, "%", "")),
#     Kpercent = as.numeric(str_replace(Kpercent, "%", "")),
#     # Ensure all relevant columns are numeric
#     across(c(AVG, k_percent, bb_percent, on_base_percent, 
#              on_base_plus_slg, babip, exit_velocity_avg, 
#              launch_angle_avg, hard_hit_percent, z_swing_percent, 
#              whiff_percent, groundballs_percent, flyballs_percent), 
#            as.numeric)
#   ) %>%
#   # Remove rows with any NA values in key columns
#   drop_na(BBpercent, Kpercent, wRCplus, wRCplus_next)

data_processed <- na.omit(data_processed)

# Define predictors (expanded from original)
predictors <- c(
  "k_percent", "bb_percent", 
  "exit_velocity_avg", "AVG", "ISO",
  "launch_angle_avg", "hard_hit_percent", "opposite_percent", "pull_percent", "swing_percent", "whiff_percent",
  "groundballs_percent", "flyballs_percent"
)




# Prepare data for modeling
model_data <- data_processed %>%
  dplyr::select(Name, year, wRCplus_next, all_of(predictors)) %>%
  drop_na()

X <- model_data[, !(names(model_data) %in% c("Name", "year", "wRCplus_next"))]



X<- as.matrix(X)

y <- model_data$wRCplus_next

# Set seed for reproducibility
set.seed(123)

# default RF model
m1 <- randomForest(
  formula = wRCplus_next ~ .,
  data    = model_data
)

which.min(m1$mse)
sqrt(m1$mse[which.min(m1$mse)])

# Feature Importance Random Forest
rf_model <- randomForest(x = X, y = y, 
                         importance = TRUE, 
                         ntree = 402, 
                         mtry = floor(sqrt(ncol(X))))


feature_importance <- data.frame(
  Predictor = predictors,
  Importance = importance(rf_model)[,1]
) %>% 
  arrange(desc(Importance))

ggplot(feature_importance, aes(x = reorder(Predictor, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "#002244", alpha = 1) +
  coord_flip() +
  theme_fivethirtyeight() +
  labs(title = "Feature Importance for Next Season's wRC+",
       x = "Predictors", 
       y = "Importance Score") 

# Predict next season's wRC+ for all players
# Prepare the prediction dataset
prediction_data <- data_processed_for_prediction%>%
  group_by(Name) %>%
  filter(year == 2024) %>%
  dplyr::select(Name, year, all_of(predictors)) %>%
  drop_na()

X_pred <- prediction_data[, !(names(prediction_data) %in% c("Name", "year"))]
# Prepare X for prediction
X_pred <- 
  as.matrix(X_pred)



#Try a simplified cross-validation
control_rf <- trainControl(method = "cv",
                        number = 4,
                        verboseIter = TRUE)

# Modify the train call
rf_cv <- train(
  x = X,
  y = y,
  method = "rf",
  trControl = control_rf,
  importance = TRUE,
  ntree = 402,  # Reduce number of trees if computationally intensive
  tuneLength = 3,
  # Limit the number of tuning parameters
)


# Multiple model ensemble
library(caretEnsemble)

# Define multiple models
control <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  savePredictions = "final",
  classProbs = TRUE
)

# Train multiple models
model_list <- caretList(
  x = X,  # Use X directly
  y = y,  # Use y directly
  trControl = control,
  methodList = c("rf", "xgbTree", "glmnet")
)


# Create ensemble model
ensemble_predictions <- caretEnsemble(
  model_list,
  metric = "RMSE",
  trControl = trainControl(
    method = "repeatedcv",
    number = 5,
    repeats = 3
  )
)

# Print results
# print(rf_cv)

# Predict wRC+ for next season
# prediction_data$predicted_wRCplus_next <- predict(rf_cv, X_pred)

predictions_rf <- predict(model_list$rf, X_pred)
predictions_xgb <- predict(model_list_wo$xgbTree, X_pred)
predictions_glmnet <- predict(model_list$glmnet, X_pred)

# Take the mean or median of these predictions
prediction_data$predicted_wRCplus_next <- (predictions_rf + predictions_xgb + predictions_glmnet) / 3


# Sort predictions
top_predictions_metrics <- prediction_data %>%
  arrange(desc(predicted_wRCplus_next)) 

top_predictions <- prediction_data %>% 
  left_join(data24, by = "Name") %>% 
    dplyr::select(Name, predicted_wRCplus_next, wRCplus)

top_5 <- top_predictions_metrics %>% 
  arrange(desc(predicted_wRCplus_next)) %>% 
  head(5)

top_20 <- top_predictions %>% 
  arrange(desc(predicted_wRCplus_next)) %>% 
  head(20)

top_differences <- top_predictions %>% 
  mutate(difference = predicted_wRCplus_next - wRCplus)


# Assuming top_5 and top_predictions are already defined
top_predictions %>%
  ggplot(aes(x = wRCplus, y = predicted_wRCplus_next)) +
  geom_point(aes(color = ifelse(Name %in% top_5$Name, "Top 5", "Others"))) +
  geom_text_repel(
    data = top_predictions %>% filter(Name %in% top_5$Name),
    aes(label = Name),
    nudge_x = 0.1,
    nudge_y = 0.1,
    box.padding = 0.5,
    max.overlaps = Inf
  ) +
  scale_color_manual(values = c("Top 5" = "red", "Others" = "black")) +
  theme_fivethirtyeight() +
  theme(axis.title = element_text(),
        legend.position = "none") +
  labs(
    x = "wRC+ 2024",
    y = "Predicted 2025 wRC+",
    title = "2024 vs Predicted 2025 wRC+",
    subtitle = "Random Forest Model"
  ) 



##### Prediction for 2024 without 2024 data ####
# Modify merged data to exclude 2024
merged_data_without_2024 <- rbindlist(list(data23, data22, data21, data19, data18), use.names = TRUE, fill = TRUE)

# Final merged data preparation without 2024
final_merged_data_without_2024 <- merge(merged_data_without_2024, data, by = c("Name", "year"), all = TRUE) %>%
  dplyr::select(Name, year, wRCplus, k_percent, bb_percent,
                exit_velocity_avg, AVG, ISO,
                launch_angle_avg, hard_hit_percent, opposite_percent, pull_percent, swing_percent, whiff_percent,
                groundballs_percent, flyballs_percent)


# Prepare data for prediction of next year's wRC+
data_processed_without_2024 <- final_merged_data_without_2024 %>%
  arrange(Name, year) %>%
  group_by(Name) %>%
  mutate(
    year_next = year + 1,
    wRCplus_next = case_when(year < 2023 ~ lead(wRCplus), 
                             year == 2023 ~ NA)
  ) %>%
  # Remove rows where we don't have next season's data
  filter(!is.na(wRCplus_next))

data_processed_for_prediction_without_2024 <- final_merged_data_without_2024 %>%
  arrange(Name, year) %>%
  group_by(Name) %>%
  mutate(
    year_next = year + 1,
    wRCplus_next = case_when(year < 2023 ~ lead(wRCplus), 
                             year == 2023 ~ NA)
  ) %>%
  # Remove rows where we don't have next season's data
  filter(ifelse(year == 2023, TRUE, !is.na(wRCplus_next)))

data_processed_without_2024 <- na.omit(data_processed_without_2024)

# Prepare data for modeling
model_data_without_2024 <- data_processed_without_2024 %>%
  dplyr::select(Name, year, wRCplus_next, all_of(predictors)) %>%
  drop_na()

X_without_2024 <- model_data_without_2024[, !(names(model_data_without_2024) %in% c("Name", "year", "wRCplus_next"))]
X_without_2024 <- as.matrix(X_without_2024)
y_without_2024 <- model_data_without_2024$wRCplus_next

# Set seed for reproducibility
set.seed(123)

# default RF model
m1_without2024 <- randomForest(
  formula = wRCplus_next ~ .,
  data    = model_data_without_2024
)

which.min(m1_without2024$mse)
sqrt(m1$mse[which.min(m1_without2024$mse)])



# #Cross-validation
# control <- trainControl(method = "cv",
#                         number = 6,
#                         verboseIter = TRUE)
# 
# rf_cv_without_2024 <- train(
#   x = X_without_2024,
#   y = y_without_2024,
#   method = "rf",
#   trControl = control,
#   importance = TRUE,
#   ntree = 483,
#   tuneGrid = expand.grid(mtry = c(2, 3, 4, 5, 6, 7))
# )


control_wo <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  savePredictions = "final",
  classProbs = TRUE
)

# Train multiple models
model_list_wo <- caretList(
  x = X,  # Use X directly 
  y = y,  # Use y directly
  trControl = control,
  methodList = c("rf", "xgbTree", "glmnet")
)


#Create ensemble model
ensemble_predictions_wo <- caretEnsemble(
  model_list,
  metric = "RMSE",
  trControl = trainControl(
    method = "repeatedcv",
    number = 5,
    repeats = 3
  )
)


X_pred_without_2024 <- prediction_data_without_2024[, !(names(prediction_data_without_2024) %in% c("Name", "year"))]
X_pred_without_2024 <- as.matrix(X_pred_without_2024)

# Predict 2024 wRC+

# Predict using each model in the ensemble
predictions_rf_24 <- predict(model_list_wo$rf, X_pred_without_2024)
predictions_xgb_24 <- predict(model_list_wo$xgbTree, X_pred_without_2024)
predictions_glmnet_24 <- predict(model_list_wo$glmnet, X_pred_without_2024)

# Take the mean or median of these predictions
prediction_data_without_2024$predicted_wRCplus_2024 <- (predictions_rf_24 + predictions_xgb_24 + predictions_glmnet_24) / 3

# Sort and analyze predictions
top_predictions_without_2024 <- prediction_data_without_2024 %>%
  left_join(data24, by = "Name") %>%
  dplyr::select(Name, predicted_wRCplus_2024, wRCplus)

top_10_without_2024 <- top_predictions_without_2024 %>%
  arrange(desc(predicted_wRCplus_2024)) %>%
  head(10)




top_predictions_without_2024 <- prediction_data_without_2024 %>% 
  left_join(data24, by = "Name") %>% 
  dplyr::select(Name, predicted_wRCplus_2024, wRCplus)

top_predictions_without_2024$predicted_wRCplus_2024 <- as.numeric(top_predictions_without_2024$predicted_wRCplus_2024)

for_graph_wo_2024 <- na.omit(top_predictions_without_2024)

#random forest graph 
ggplot(top_predictions_without_2024, aes(x = predicted_wRCplus_2024, y = wRCplus)) +
  geom_point(alpha = 0.6) +
  stat_smooth(geom='line', alpha=0.5, se=FALSE, method='lm', color = "red")+
  theme_fivethirtyeight() +
  theme(axis.title = element_text()) + 
  labs(
    title = "Predicted vs Actual 2024 wRC+",
    x = "Predicted wRC+",
    y = "Actual wRC+",
    subtitle = sprintf("Correlation: %.3f\nRMSE: %.2f", 
                                 cor(for_graph_wo_2024$predicted_wRCplus_2024, for_graph_wo_2024$wRCplus),
                                 sqrt(mean((for_graph_wo_2024$predicted_wRCplus_2024 - for_graph_wo_2024$wRCplus)^2)))
  )


ggplot(top_predictions_without_2024, aes(x = predicted_wRCplus_2024, y = wRCplus)) +
  geom_point(alpha = 0.6) +
  stat_smooth(geom='line', alpha=0.5, se=FALSE, method='lm', color = "red")+
  theme_fivethirtyeight() +
  theme(axis.title = element_text()) + 
  labs(
    title = "Predicted vs Actual 2024 wRC+ (Ensemble Method)",
    x = "Predicted wRC+",
    y = "Actual wRC+",
    subtitle = sprintf("Correlation: %.3f\nRMSE: %.2f", 
                       cor(for_graph_wo_2024$predicted_wRCplus_2024, for_graph_wo_2024$wRCplus),
                       sqrt(mean((for_graph_wo_2024$predicted_wRCplus_2024 - for_graph_wo_2024$wRCplus)^2)))
  )

