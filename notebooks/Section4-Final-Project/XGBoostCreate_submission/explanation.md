We have tried various techniques to improve the XGBoost model performance provided.

Hyperparameter Tuning: We have done a hyperparameter search for the XGBoost parameters like depth of tree - 3,4,5,6, shrinkage parameter - 0.2,0.3,0.1, using different feature selectors - random, cyclic etc. we are also trying to change the feature selector parameter

KDTreeEncoding: We have tried out various methods like, adding the urban feature information as well to the encoded dataset and tried applying PCA in the train encoder, instead of taking all the points we have compressed the number of features.

Custom Thresholds: We have tried out various custom thresholds in the predict.py to avoid negative scores.

next, we also want to try separate encoding for the urban and rural datasets as done by the code from TA.

Dataset: In this final project we study classification in the context of the Poverty dataset, which is part of the Wilds Project. Problem statement is to classify poor vs. wealthy regions in Africa based on satellite imagery. There are ~20,000 images covering 23 countries in Africa. The satellite images are of the shape 224X224X8. Each pixel corresponds to a 30mX30m area and each image corresponds to a 6.7kmX6.7km square. This dataset comprises images from both urban and rural areas. In general, urban areas are significantly more wealthy than rural areas. To make the problem into a classification task, a threshold is defined on the wealth that separates the poor from wealthy.
