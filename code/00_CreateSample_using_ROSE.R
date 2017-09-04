# MLMC linux 
# jacobgreen1984@2e.co.kr


# -----------------------------------------------------------------------------
# create aritificial dataset 
# -----------------------------------------------------------------------------
train <- ROSE(Y ~ .
              ,data = train
              ,p=p_for_ROSE      # The probability of the minority class
              ,seed = 1234)$data
# -----------------------------------------------------------------------------


