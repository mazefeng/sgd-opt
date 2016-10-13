A comparasion among different variant of Gradient Descent algorithm, based on the MNIST hand-written digit recognition dataset.

The algorithm list include the following:
1. Stochastic Gradient Descent(SGD)
2. Momentum
3. Nesterov Accelerated Gradient(NAG)
4. Adagrad
5. Adadelta
6. RMSprop
7. Adam

All of the 7 algorithms are described in the blog post <An overview of gradient descent optimization algorithms>

The MNIST dataset contains 60,000 samples for training and 10,000 for validating;

The dataset is naturally divide into 10 classes corresponding to digit 0 to 9
The amount of samples for each class is well-balanced.
Each digit is scale into a 28p28 gray scale image

and we use a traditional no-so-deep 3 layer neural network as classifier, with 25 hidden units

Here are the training and validating accuracy for each algorithm, with 30 epoches and 100 mini-batch:

SGD: 95.36% vs 94.06%
Momentum: 97.515% vs 94.88%
NAG: 97.466667% vs 94.33%
Adagrad: 96.165% vs 93.95%
Adadelta: 94.651667% vs 93.84%
RMSprop: 96.345% vs 94.51%
Adam: 96.541667% vs 94.12%







