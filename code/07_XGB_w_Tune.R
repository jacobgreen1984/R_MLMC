# MLMC linux 
# jacobgreen1984@2e.co.kr


# -----------------------------------------------------------------------------
# cartesian grid search 
# -----------------------------------------------------------------------------
## Depth 10 is usually plenty of depth for most datasets, but you never know
hyper_params = list(max_depth = c(4,6,8,12,16,20)) 

grid <- h2o.grid(
  ## hyper parameters
  hyper_params = hyper_params,
  
  ## full Cartesian hyper-parameter search
  search_criteria = list(strategy = "Cartesian"),
  
  ## which algorithm to run
  algorithm="xgboost",
  
  ## identifier for the grid, to later retrieve it
  grid_id="XGB_depth_grid",
  
  ## standard model parameters
  x = x, 
  y = y, 
  training_frame = train,
  validation_frame = valid,
  
  ## more trees is better if the learning rate is small enough 
  ## here, use "more than enough" trees - we have early stopping
  ntrees = 10000,                                                            
  
  ## smaller learning rate is better
  learn_rate = 0.01,                                                         
 
  ## early stopping once the validation AUC doesn't improve by at least 0.01% for 5 consecutive scoring events
  stopping_rounds = 5,
  stopping_tolerance = 1e-4,
  stopping_metric = "AUC", 
  
  ## score every 5 trees to make early stopping reproducible (it depends on the scoring interval)
  score_tree_interval = 5,                                                  
  
  ## base random number generator seed for each model (automatically gets incremented internally for each model)
  seed = 1234                                               
)

## by default, display the grid search results sorted by increasing logloss (since this is a classification task)
grid                                                                       

## sort the grid models by decreasing AUC
sortedGrid <- h2o.getGrid("XGB_depth_grid", sort_by="auc", decreasing = TRUE)    
sortedGrid

## find the range of max_depth for the top 3 models
topDepths = sortedGrid@summary_table$max_depth[1:3]                       
minDepth = min(as.numeric(topDepths))
maxDepth = max(as.numeric(topDepths))
print(minDepth)
print(maxDepth)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# random grid search 
# -----------------------------------------------------------------------------
hyper_params = list( 
  ## restrict the search to the range of max_depth established above
  max_depth = seq(minDepth,maxDepth,1),                                      
  
  ## search a large space of row sampling rates per tree
  sample_rate = seq(0.2,1,0.01),                                             
  
  ## search a large space of column sampling rates per split
  col_sample_rate = seq(0.2,1,0.01),                                         
  
  ## search a large space of column sampling rates per tree
  col_sample_rate_per_tree = seq(0.2,1,0.01),                                
  
  ## search a large space of the number of min rows in a terminal node
  min_rows    = 2^seq(0,log2(nrow(train))-1,1),     
  
  ## regularization 
  reg_lambda  = seq(0,1e-4,1e-6), 
  reg_alpha   = seq(0,1e-4,1e-6),                          
  
  ## search a few minimum required relative error improvement thresholds for a split to happen
  min_split_improvement = c(0,1e-8,1e-6,1e-4)                               
)

search_criteria = list(
  ## Random grid search
  strategy = "RandomDiscrete",      
  
  ## limit the runtime to 60 minutes
  max_runtime_secs = max_runtime_secs,         
  
  ## build no more than 100 models
  max_models = max_models,                  
  
  ## random number generator seed to make sampling of parameter combinations reproducible
  seed = 1234,                        
  
  ## early stopping once the leaderboard of the top 5 models is converged to 0.1% relative difference
  stopping_rounds = 5,                
  stopping_metric = "AUC",
  stopping_tolerance = 1e-3
)

grid <- h2o.grid(
  ## hyper parameters
  hyper_params = hyper_params,
  
  ## hyper-parameter search configuration (see above)
  search_criteria = search_criteria,
  
  ## which algorithm to run
  algorithm = "xgboost",
  
  ## identifier for the grid, to later retrieve it
  grid_id = "XGB_grid", 
  
  ## standard model parameters
  x = x, 
  y = y, 
  training_frame = train,
  validation_frame = valid,
  
  ## more trees is better if the learning rate is small enough
  ## use "more than enough" trees - we have early stopping
  ntrees = 10000,                                                            
  
  ## smaller learning rate is better
  learn_rate = 0.01, 

  ## early stopping once the validation AUC doesn't improve by at least 0.01% for 5 consecutive scoring events
  stopping_rounds = 5, stopping_tolerance = 1e-4, stopping_metric = "AUC", 
  
  ## score every 5 trees to make early stopping reproducible (it depends on the scoring interval)
  score_tree_interval = 5,                                                
  
  ## base random number generator seed for each model (automatically gets incremented internally for each model)
  seed = 1234                                                             
)

## Sort the grid models by AUC
sortedGrid <- h2o.getGrid("XGB_grid", sort_by = "auc", decreasing = TRUE)    
XGB_w_Tune <- h2o.getModel(sortedGrid@model_ids[[1]])
# -----------------------------------------------------------------------------