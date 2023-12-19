###################################
# Allstate Project

library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)

train <- vroom("train.csv")
test <- vroom("test.csv")

DataExplorer::plot_intro(train)

train[100:132]


my_recipe <- recipe(loss ~ ., data = train) %>%
  step_rm(id) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(loss)) %>%
  step_normalize(all_numeric_predictors())%>%
  step_zv(all_predictors())#remove any predictors with no variance
prepped_recipe <- prep(my_recipe)      
bake(prepped_recipe, new_data = train)


##################################################################
# BART
##################################################################
my_BART_mod <- bart(mode ="regression",
                    engine = "dbarts", trees =20)

BART_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_BART_mod) %>%
  fit(data=train)
Allstate_preds <- predict(BART_wf, new_data=test)

submission <- Allstate_preds %>%
  mutate(id = test$id) %>%
  mutate(loss = .pred) %>% #transform back to original scale
  select(2, 3)

vroom_write(submission, "submissionBART.csv", delim = ",")  

##################################################################
# BART with updated Recipe
##################################################################

my_recipe2 <- recipe(loss ~ ., data = train) %>%
  step_rm(id) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_corr(all_numeric_predictors(), threshold = 0.6) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(loss)) %>%
  step_normalize(all_numeric_predictors())%>%
  step_zv(all_predictors())#remove any predictors with no variance


my_BART_mod <- bart(mode ="regression",
                    engine = "dbarts", trees =20)

BART_wf <- workflow() %>%
  add_recipe(my_recipe2) %>%
  add_model(my_BART_mod) %>%
  fit(data=train)
Allstate_preds <- predict(BART_wf, new_data=test)

submission <- Allstate_preds %>%
  mutate(id = test$id) %>%
  mutate(loss = .pred) %>% #transform back to original scale
  select(2, 3)

vroom_write(submission, "submissionBART3.csv", delim = ",")  

##########################################################
## Boosted Tree

boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
  set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("regression")

Boost_wf <- workflow() %>%
  add_recipe(my_recipe2) %>%
  add_model(boost_model)

## CV tune, finalize and predict here and save results
## Grid of values to tune over
tuning_grid <- grid_regular(tree_depth(),
                            trees(),
                            learn_rate(),
                            levels = 3) ## L^2 total tuning possibilities

## Split data for CV15
## Split data for CV15
folds <- vfold_cv(train, v = 5, repeats=1)

## Run the CV
CV_results <- Boost_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(mae)) #Or leave metrics NULL

#Find the best tuning parameters
bestTune <- CV_results %>%
  select_best('mae')

final_wf <- Boost_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

Allstate_preds <- predict(final_wf, new_data=test)

#format submission
submission <- Allstate_preds %>%
  mutate(id = test$id) %>%
  mutate(loss = .pred) %>% #transform back to original scale
  select(2, 3)

vroom_write(submission, "Boostedsubmission.csv", delim = ",")

#######################################
# Linear regression for kicks and giggles

my_mod <- linear_reg() %>% #Type of model
  set_engine("lm") # Engine = What R function to use
Lin_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod) %>%
  fit(data = train) #Fit workflow

Allstate_preds <- predict(Lin_wf, new_data=test)

#format submission
submission <- Allstate_preds %>%
  mutate(id = test$id) %>%
  mutate(loss = .pred) %>% #transform back to original scale
  select(2, 3)

vroom_write(submission, "LRsubmission.csv", delim = ",")

