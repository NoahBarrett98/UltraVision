# hparam search dnet 169

+------------------------+------------+----------------------+-------------+----------+------------+----------+----------------------+
| Trial name             | status     | loc                  |          lr |     loss |   accuracy |      auc |   training_iteration |
|------------------------+------------+----------------------+-------------+----------+------------+----------+----------------------|
| train_func_47555_00000 | TERMINATED | 129.173.66.120:14056 | 0.000347917 | 0.172496 |   0.957924 | 0.991641 |                    1 |
| train_func_47555_00001 | TERMINATED | 129.173.66.120:14054 | 0.000189441 | 0.222819 |   0.941094 | 0.987064 |                    1 |
| train_func_47555_00002 | TERMINATED | 129.173.66.120:14047 | 0.0822593   | 0.43099  |   0.852735 | 0.973725 |                    1 |
| train_func_47555_00003 | TERMINATED | 129.173.66.120:14057 | 0.000421432 | 0.197459 |   0.949509 | 0.989936 |                    1 |
| train_func_47555_00004 | TERMINATED | 129.173.66.120:14055 | 0.0546405   | 0.343973 |   0.890603 | 0.979839 |                    1 |
| train_func_47555_00005 | TERMINATED | 129.173.66.120:14051 | 0.000334665 | 0.190233 |   0.952314 | 0.990352 |                    1 |
| train_func_47555_00006 | TERMINATED | 129.173.66.120:14052 | 0.000266718 | 0.241115 |   0.934081 | 0.988201 |                    1 |
| train_func_47555_00007 | TERMINATED | 129.173.66.120:14048 | 0.00012549  | 0.191621 |   0.935484 | 0.991592 |                    1 |
| train_func_47555_00008 | TERMINATED | 129.173.66.120:14053 | 0.00828149  | 0.216721 |   0.969144 | 0.992888 |                    1 |
| train_func_47555_00009 | TERMINATED | 129.173.66.120:14049 | 0.00288665  | 0.199703 |   0.960729 | 0.991747 |                    1 |
+------------------------+------------+----------------------+-------------+----------+------------+----------+----------------------+

(ImplicitFunc pid=14049) Confusion Matrix:
(ImplicitFunc pid=14049) [[240   0   4   1   4   5]
(ImplicitFunc pid=14049)  [  1 113   0   0   0   0]
(ImplicitFunc pid=14049)  [  4   0  31   0   0   1]
(ImplicitFunc pid=14049)  [  4   0   0 153   0   0]
(ImplicitFunc pid=14049)  [  4   0   0 153   0   0]
(ImplicitFunc pid=14049)  [  1   0   0   0  48   0]
(ImplicitFunc pid=14049)  [  3   0   0   0   0 100]]
2021-11-30 15:58:32,242 INFO tune.py:630 -- Total run time: 7566.07 seconds (7565.82 seconds for the tuning loop).
Best trial config: {'lr': 0.00034791706104942197}
Best trial final validation loss: 0.17249637635667686

results for evaluating with validation loss ^^^

when evaluating on test images, it performs quite poorly
I want to see if applying the same augmentations to the test data as val data will solve this issue

its clearly not as the validation set uses the testing augmentation - so no augmentation

I am going to try a manual split, not using the dataset split
