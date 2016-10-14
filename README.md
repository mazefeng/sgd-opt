A Comparison Among Different Variants of Gradient Descent Algorithm
==

###Introduction
***
This script implements and visualizes the performance the following algorithms, based on the MNIST hand-written digit recognition dataset:

- Stochastic Gradient Descent(SGD)
- Momentum
- Nesterov Accelerated Gradient(NAG)
- Adagrada
- Adadelta
- RMSprop
- Adam

All the detail of the algorithms are described in the blog post [An overview of gradient descent optimization algorithms](http://sebastianruder.com/optimizing-gradient-descent/) by `Sebastian Ruder`

###Dataset
***
The MNIST dataset contains `60,000` samples for training and `10,000` for validating. It is naturally divided into `10 classes` corresponding to digit 0 to 9. The amount of samples for each class is well-balanced. Each digit image is pre-processed and rescaled into a 28\*28 gray scale array, ranging from 0 to 255

###Model
***
A traditional 3 layer neural network is adopted, with `28\*28 input units, 25 hidden units and a softmax output layer`

###Performance
***
Here are the training and validating accuracy of each algorithm, with `30 epochs` and `100 mini-batch`:

- SGD: `95.36% vs 94.06%`
- Momentum: `97.52% vs 94.88%`
- NAG: `97.47% vs 94.33%`
- Adagrad: `96.17% vs 93.95%`
- Adadelta: `94.65% vs 93.84%`
- RMSprop: `96.35% vs 94.51%`
- Adam: `96.54% vs 94.12%`

###Visualization
***
Here is the visualization of cost decreasing w.r.t each mini-batch within the first 10 epochs: 

![image](https://github.com/mazefeng/sgd-opt/blob/master/image.png)

###Conclusion
***
The variants of gradient descent algorithm can be roughly divided into 2 types: `Momentum-like SGD` and `Adaptive learning rate SGD`. 

As mentioned in the blog post `Adaptive learning rate SGD` is suitable for large-scale sparse optimization problem (e.g, predict CTR). While in this case, data is not sparse,  `Momentum-like SGD` significantly outperforms the others.


