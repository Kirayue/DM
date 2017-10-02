svm-scale -s range train_data > train_data.scale
svm-scale -r range test_data > test_data.scale
svm-train -s 3 train_data.scale model
svm-predict test_data.scale model test
cut -d ' ' -f 1 test_data > test_ans
