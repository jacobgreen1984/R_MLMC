# MLMC linux 
# jacobgreen1984@2e.co.kr


# -----------------------------------------------------------------------------
# cartesian grid search 
# -----------------------------------------------------------------------------
## Depth 10 is usually plenty of depth for most datasets, but you never know
hyper_params = list(hidden=list(c(100),c(200),c(300),c(100,100),c(200,200),c(300,300))) 

grid <- h2o.grid(
  ## hyper parameters
  hyper_params = hyper_params,
  
  ## full Cartesian hyper-parameter search
  search_criteria = list(strategy = "Cartesian"),
  
  ## which algorithm to run
  algorithm="deeplearning",
  
  ## identifier for the grid, to later retrieve it
  grid_id="DL_depth_grid",
  
  ## standard model parameters
  x = x, 
  y = y, 
  training_frame = train, 
  validation_frame = valid,
  
  # already done!
  standardize = F, 

  ## fix a random number generator seed for reproducibility
  seed = 1234,                                                             
  
  ## early stopping once the validation AUC doesn't improve by at least 0.01% for 5 consecutive scoring events
  stopping_rounds = 5,
  stopping_tolerance = 1e-4,
  stopping_metric = "AUC"                                          
)

## by default, display the grid search results sorted by increasing logloss (since this is a classification task)
grid                                                                       

## sort the grid models by decreasing AUC
sortedGrid = h2o.getGrid("DL_depth_grid", sort_by="auc", decreasing = TRUE)    
topHidden  = list(h2o.getModel(sortedGrid@model_ids[[1]])@allparameters$hidden
                  ,h2o.getModel(sortedGrid@model_ids[[2]])@allparameters$hidden
                  ,h2o.getModel(sortedGrid@model_ids[[3]])@allparameters$hidden)
print(topHidden)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# random grid search 
# -----------------------------------------------------------------------------
hyper_params = list( 
  hidden = topHidden
  ,activation=c("Rectifier","Tanh","Maxout","RectifierWithDropout","TanhWithDropout","MaxoutWithDropout")
  ,input_dropout_ratio=c(0,0.05,0.1,0.15,0.2,0.25,0.3)
  ,l1=seq(0,1e-4,1e-6)
  ,l2=seq(0,1e-4,1e-6)
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
  algorithm = "deeplearning",
  
  ## identifier for the grid, to later retrieve it
  grid_id = "DL_grid", 
  
  ## standard model parameters
  x = x, 
  y = y, 
  training_frame = train, 
  validation_frame = valid,
  
  # already done!
  standardize = F, 
  
  ## the number of epochs
  epochs = 1000000,
  
  ## early stopping once the validation AUC doesn't improve by at least 0.01% for 5 consecutive scoring events
  stopping_rounds = 5, stopping_tolerance = 1e-4, stopping_metric = "AUC", 
  
  ## base random number generator seed for each model (automatically gets incremented internally for each model)
  seed = 1234                                                             
)

## Sort the grid models by AUC
sortedGrid <- h2o.getGrid("DL_grid", sort_by = "auc", decreasing = TRUE)    
DL_w_Tune  <- h2o.getModel(sortedGrid@model_ids[[1]])
# -----------------------------------------------------------------------------
