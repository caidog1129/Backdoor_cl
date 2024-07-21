# Learning-Backdoors-for-Mixed-Integer-Linear-Programs-with-Contrastive-Learning

Paper[https://arxiv.org/pdf/2401.10467]

## Procedure
1. Use generator to generate 300 instances, 200 for training, 100 for testing, store the instance dir in train.txt, test.txt. Each instance is a dir with instance file inside the dir
2. Collect backdoor using backdoor_search/backdoor_search.py and evaluate backdoor using backdoor_search/backdoor_evaluate.py instance_wise(parallel)
3. Create backdoor_dataset folder, inside create train and valid folder. Create log folder. Then train the model use train.py
4. Use evaluate.py to evaluate(parallel), then use jupyter notebook to summarize and visualize
