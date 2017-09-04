# MLMC linux 
# jacobgreen1984@2e.co.kr


# -----------------------------------------------------------------------------
# PCA using GLRM
# -----------------------------------------------------------------------------
compressor <- h2o.prcomp(training_frame = train[,x]
                         ,k = k_for_pca
                         ,transform = "NONE"         # already done!
                         ,pca_method="GLRM"
                         ,use_all_factor_levels=TRUE
                         ,impute_missing=TRUE
                         ,seed=1234)
# -----------------------------------------------------------------------------


