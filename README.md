# APS Failure in Scania Trucks
Xinyu Tian | xt5@illinois.edu

## Summary
The project aimed at predicting the APS failures data collected from heavy Scania trucks and minimizing costs. To deal with a large amount of missing data, the project used Bayesian Principal Component Analysis (BPCA) to impute the missing values. Considering the fact of imbalanced classification, the project adopted random forest with SMOTE resampling and cost sensitive CART as well as Naïve Bayes as the baseline. Overall, the SMOTE-Random Forest performed best in terms of the total cost.

## Preparation
Having read both the training and test datasets, the class column was mapped by the rule: positive to 1 and negative to 0. Here, the positive class consists of component failures for a specific component of the APS system. The negative class consists of trucks with failures for components not related to the APS. All other attributes were converted into numerical ones. 

The histogram below reveals a significant unbalanced pattern between the positive and negative classes. It suggested the necessary to do resampling and use Kappa to evaluate.

 

For convenience in data preprocessing, the predictors were separated from the datasets and stored as tra_fea and tes_fea.

## Heuristic Feature Selection and Missing Data Imputation
Since names of all the attributes are have been anonymized and no more relations can be inferred from the context, the features were initially selected based on their statistics. Here, the columns with too much missing data (missing data percentage > 60%), too much zero (zero value percentage > 90%) and if all observations are identical (standard derivation = 0).

remove columns where over 60% of instances are missing
remove columns where over 90% of instance values are zero
remove columns where the standard derivation is zero

After the filtering, there were 142 predictors. Due to the high dimensions, the project attempted reducing dimensionality by PCA. Given the missing values, Bayesian Principal Component Analysis (BPCA) was selected as it generally performs well in the presence of missing values.  The library pcaMethods with more built-in PCA models were employed here.

However, the result of PCA was not satisfying as there were only 2 PCs extracted and the first one can explain almost all of the variations in the class (as shown below). 
 
 

To fill the missing values, the patterns of missing values were checked first. The plot indicated that we can assume the values in the training dataset are missing at random while the columns where the missing values occupied over 80% of instances were removed in the last step. Even though the PCA didn’t apply to the training set, it was used as a robust imputation method to fill all the NAs.  Unfortunately, the Random Forest imputation were not used due to the limitation of computational capacity. 

The imputed data were ready to train. Now the dataset still contained relatively high dimensions (143).

# Model Fitting and Result Comparison
Here, three classifiers were selected: Naïve Bayes, Random Forest and CART. Given the imbalanced classification, tree-based models were picked up as they are expected to perform better.  It is widely accepted that resampling tends to be an effective way to handle the problem, the Random Forest was used with SMOTE Boosting, a modern approach to oversample underrepresented cases. On the other hand, the cost-sensitive CART was chosen as an example to try penalized models. Additionally, Naïve Bayes served as a probabilistic method and, more important, a baseline to compare with.
## Naive Bayes
Confusion Matrix and Statistics
          Reference
Prediction     0     1
         0 15090    40
         1   535   335
                                          
               Accuracy : 0.9641          
                 95% CI : (0.9611, 0.9669)
    No Information Rate : 0.9766          
    P-Value [Acc > NIR] : 1               
                  Kappa : 0.5225          
 Mcnemar's Test P-Value : <2e-16          
The random forest was initially attempted on the original training dataset. To obtain the best parameters, the grid search was used with 5-fold cross validation. Although the accuracy was significantly improved than that by Naive Bayes, there were too much false negatives in the confusion matrix caused by the imbalanced training set. 

## try Random Forest on the original training set
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 15607   111
         1    18   264
                                          
               Accuracy : 0.9919          
                 95% CI : (0.9904, 0.9933)
    No Information Rate : 0.9766          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.7996          
 Mcnemar's Test P-Value : 5.489e-16  

In this case, the SMOTE boosting was taken. 
## SMOTE boosting
Afterward, the outcomes considerably improved.
Random Forest 

7000 samples
 142 predictor
   2 classes: '0', '1' 

No pre-processing
Resampling: Cross-Validated (5 fold) 
Summary of sample sizes: 5600, 5600, 5600, 5600, 5600 
Resampling results across tuning parameters:

  mtry  Accuracy   Kappa    
   5    0.9737143  0.9464392
   6    0.9732857  0.9455551
   7    0.9728571  0.9446885
   8    0.9735714  0.9461429
   9    0.9734286  0.9458518
  10    0.9737143  0.9464352
  11    0.9734286  0.9458530
  12    0.9740000  0.9470253

The final values used for the model were mtry = 12, splitrule = gini and min.node.size = 1.

Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 15099    15
         1   526   360
                                          
               Accuracy : 0.9662          
                 95% CI : (0.9633, 0.9689)
    No Information Rate : 0.9766          
    P-Value [Acc > NIR] : 1               
                                          
                  Kappa : 0.5564          
 Mcnemar's Test P-Value : <2e-16 

12 predictors were actually involved in the model. To simplify the model, the importance of variables were considered. The top 12 important variables were plotted as below.
 
Since the importance of top 4 predictors seems predominant in the case, an attempt was given on mtry = 4, which came up with a satisfactory result.
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 15051    15
         1   574   360
                                          
               Accuracy : 0.9632          
                 95% CI : (0.9602, 0.9661)
    No Information Rate : 0.9766          
    P-Value [Acc > NIR] : 1               
                                          
                  Kappa : 0.5345          
 Mcnemar's Test P-Value : <2e-16   

Cost sensitive CART was trained with a loss matrix from the total cost matrix. It reduced the number of false negative cases with a cost of more false positive cases. Therefore, the overall accuracy was lower than others.
## Cost Sensitive CART
Confusion Matrix and Statistics
          Reference
Prediction     0     1
         0 14842    16
         1   783   359                      
               Accuracy : 0.9501          
                 95% CI : (0.9466, 0.9534)
    No Information Rate : 0.9766          
    P-Value [Acc > NIR] : 1               
                                          
                  Kappa : 0.454           
 Mcnemar's Test P-Value : <2e-16     

As shown on the ROC plot, the area under ROC curve maximized when the Random Forest with SMOTE Boosting was adopted. Also, the total cost minimized with the same classifier. Therefore, the Random Forest with SMOTE Boosting is the best model to perform the classification in the dataset.
