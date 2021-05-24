The mnist_train.py script takes two parameters.
- model_name: str. One of {'decision_tree', 'rf', 'log_reg'}.
- cv: int. Number of cross_validation folds.

Examples:

1. >> python3 mnist_train.py --model_name decision_tree --cv 3
   Model: decision_tree, mean_crossvalscore: 0.85925
   Model: decision_tree, accuracy_score: 0.8759
   
2. >> python3 mnist_train.py --model_name rf --cv 3
   Model: rf, mean_crossvalscore: 0.9643
   Model: rf, accuracy_score: 0.9706

