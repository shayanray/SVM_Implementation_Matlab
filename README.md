# Task
Implement the following SVM methods from scratch using Matlab.

## Experiments:

Implement Kernel SVM in Matlab using Quadratic Programming

Implement Multiclass SVM in Matlab  using Stochastic Gradient Descent

Use the above Multiclass SVM with SGD for activity recognition 

SVM  in Matlab for object detection

## Results

Linear Kernel SVM; C = 0.1

1. Train accuracy: 97.24%

2. Test accuracy: 94.55%

3. Training Objective Value of SVM: 352

4. Validation Objective Value of SVM: 347

5. Number of Support Vectors: 339

6. Confusion Matrix: In output folder 'confusionMatrix0.1.csv'


Linear Kernel SVM; C = 10

1. Train accuracy: 100%

2. Test accuracy: 100%

3. Training Objective Value of SVM: 362

4. Validation Objective Value of SVM: 367

5. Number of Support Vectors: 123

6. Confusion Matrix: In output folder 'confusionMatrix10.csv'


MultiClass_SVM/stochastic_svm.m

a) eta0 = 1, eta1 = 100, C = 0.1

• C = 0.1

• eta0 = 1

• eta1 = 100

• num_epochs = 2000

• train_accuracy = 0.9641

• train_obj_val = 349

• valdn_accuracy = 0.9591

• valdn_obj_val = 352


b) eta0 = 1, eta1 = 100, C = 10

• C = 10

• eta0 = 1

• eta1 = 100

• num_epochs = 2000

• train_accuracy = 0.9945

• train_obj_val = 360

• valdn_accuracy = 0.9918

• valdn_obj_val = 364


For best accuracy the hyper-parameters I have tuned and finally used are as follows:

• C = 0.05

• eta0 = 1

• eta1 = 100

Other observed results are:

• train_accuracy = 1.0

• train_obj_val = 7930

• valdn_accuracy = 1.0

• valdn_obj_val = 2120

• sumW_train = 0.25764

• sumW_valdn = 0.0059

## Implementation
Experiments mentioned above have been implemented from scratch using Matlab