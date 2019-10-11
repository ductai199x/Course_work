- [Deep Learning](#deep-learning)
  - [Backpropagation to train multilayer architectures](#backpropagation-to-train-multilayer-architectures)
  - [Convolutional neural networks](#convolutional-neural-networks)
  - [Distributed representations and language processing](#distributed-representations-and-language-processing)
  - [Recurrent neural networks](#recurrent-neural-networks)

# Deep Learning
**LeCun, Bengio & Geoffrey**

https://www.nature.com/articles/nature14539


## Backpropagation to train multilayer architectures
- Multilayer architectures can be trained by simple stochastic gradient descent
- As long as the modules are relatively smooth functions of their inputs and of their internal weights, one can compute gradients using the backpropagation procedure.
- The backpropagation procedure to compute the `gradient of an objective function with respect to the weights of a multilayer stack of modules` (a.k.a $J(w)$) is nothing more than a practical application of the chain rule for derivatives.
- The derivative (or gradient) of the objective with respect to the input of a module (a.k.a $\frac{\partial J(w)}{\partial w}$) can be computed by working backwards from the gradient with respect to the output of that module (or the input of the subsequent module) 
- At present, the most popular non-linear function is the rectified linear unit (ReLU), which is simply the half-wave rectifier $f(z) = max(z, 0)$
- The hidden layers can be seen as distorting the input in a non-linear way so that categories become linearly separable by the last layer
- The objective in learning each layer of feature detectors was to be able to reconstruct or model the activities of feature detectors (or raw inputs) in the layer below. By 'pre-training' several layers of progressively more complex feature detectors using this reconstruction objective, the weights of a deep network could be initialized to sensible values.
- For smaller data sets, unsupervised pre-training helps to prevent overfitting40, leading to significantly better generalization when the number of labelled examples is small, or in a transfer setting where we have lots of examples for some 'source' tasks but very few for some 'target' tasks.

## Convolutional neural networks
- There are four key ideas behind ConvNets that take advantage of the properties of natural signals: local connections, shared weights, pooling and the use of many layers.
- The architecture of a typical ConvNet is structured as a series of stages. The first few stages are composed of two types of layers: convolutional layers and pooling layers.
- Units in a convolutional layer are organized in feature maps, within which each unit is connected to local patches in the feature maps of the previous layer through a set of weights called a filter bank. The result of this local weighted sum is then passed through a non-linearity such as a ReLU. All units in a feature map share the same filter bank. Different feature maps in a layer use different filter banks. 


## Distributed representations and language processing
- Deep-learning theory shows that deep nets have two different exponential advantages over classic learning algorithms that do not use distributed representations21. Both of these advantages arise from the power of composition and depend on the underlying data-generating distribution having an appropriate componential structure40. First, learning distributed representations enable generalization to new combinations of the values of learned features beyond those seen during training (for example, 2n combinations are possible with n binary features)68,69. Second, composing layers of representation in a deep net brings the potential for another exponential advantage70 (exponential in the depth).


## Recurrent neural networks
- RNNs process an input sequence one element at a time, maintaining in their hidden units a 'state vector' that implicitly contains information about the history of all the past elements of the sequence. 