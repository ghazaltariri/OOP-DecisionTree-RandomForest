# Decision Tree


# RandomForest:

Wisdom of the Crowd.
Random Forests vs Bagging(Bootstrap AGGregatING) Alone:
* Both utilize bootstrap samples
* RF also split on a feature taken from a random subset of all features

In Bagging trees, a very strongly predictive feature will often show up at the top node, 
making trees look similar and making their predictions more correlated.

In RF, the predictive feature will only be available a portion of the time, 
allowing moderately strong predictors a chance

Suggested subset sizes, for p features a hyper-parameter that we tune:
* Classification: d = √p
* Regression: d = p/3

min_sample_leaf:
* start with None and try others
