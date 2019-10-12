# TOC
- [TOC](#toc)
- [Deep Learning](#deep-learning)
  - [Backpropagation to train multilayer architectures](#backpropagation-to-train-multilayer-architectures)
  - [Convolutional neural networks](#convolutional-neural-networks)
  - [Distributed representations and language processing](#distributed-representations-and-language-processing)
  - [Recurrent neural networks](#recurrent-neural-networks)
  - [The future of deep learning](#the-future-of-deep-learning)
- [Introduction to Spiking Neural Networks](#introduction-to-spiking-neural-networks)
  - [Abstract](#abstract)
  - [Introduction](#introduction)

# Deep Learning
**LeCun, Bengio & Geoffrey**

https://www.nature.com/articles/nature14539

[TOC](#toc)
## Backpropagation to train multilayer architectures 
- Multilayer architectures can be trained by simple stochastic gradient descent
- As long as the modules are relatively smooth functions of their inputs and of their internal weights, one can compute gradients using the backpropagation procedure.
- The backpropagation procedure to compute the `gradient of an objective function with respect to the weights of a multilayer stack of modules` (a.k.a $J(w)$) is nothing more than a practical application of the chain rule for derivatives.
- The derivative (or gradient) of the objective with respect to the input of a module (a.k.a $\frac{\partial J(w)}{\partial w}$) can be computed by working backwards from the gradient with respect to the output of that module (or the input of the subsequent module) 
- At present, the most popular non-linear function is the rectified linear unit (ReLU), which is simply the half-wave rectifier $f(z) = max(z, 0)$
- The hidden layers can be seen as distorting the input in a non-linear way so that categories become linearly separable by the last layer
- The objective in learning each layer of feature detectors was to be able to reconstruct or model the activities of feature detectors (or raw inputs) in the layer below. By 'pre-training' several layers of progressively more complex feature detectors using this reconstruction objective, the weights of a deep network could be initialized to sensible values.
- For smaller data sets, unsupervised pre-training helps to prevent overfitting40, leading to significantly better generalization when the number of labelled examples is small, or in a transfer setting where we have lots of examples for some 'source' tasks but very few for some 'target' tasks.

[TOC](#toc)
## Convolutional neural networks
- There are four key ideas behind ConvNets that take advantage of the properties of natural signals: local connections, shared weights, pooling and the use of many layers.
- The architecture of a typical ConvNet is structured as a series of stages. The first few stages are composed of two types of layers: convolutional layers and pooling layers.
- Units in a convolutional layer are organized in feature maps, within which each unit is connected to local patches in the feature maps of the previous layer through a set of weights called a filter bank. The result of this local weighted sum is then passed through a non-linearity such as a ReLU. All units in a feature map share the same filter bank. Different feature maps in a layer use different filter banks. 

[TOC](#toc)
## Distributed representations and language processing
- Deep-learning theory shows that deep nets have two different exponential advantages over classic learning algorithms that do not use distributed representations21. Both of these advantages arise from the power of composition and depend on the underlying data-generating distribution having an appropriate componential structure40. First, learning distributed representations enable generalization to new combinations of the values of learned features beyond those seen during training (for example, 2n combinations are possible with n binary features)68,69. Second, composing layers of representation in a deep net brings the potential for another exponential advantage70 (exponential in the depth).

[TOC](#toc)
## Recurrent neural networks
- RNNs process an input sequence one element at a time, maintaining in their hidden units a 'state vector' that implicitly contains information about the history of all the past elements of the sequence. 
- RNNs are very powerful dynamic systems, but training them has proved to be problematic because the backpropagated gradients either grow or shrink at each time step, so over many time steps they typically explode or vanish
- RNNs, once unfolded in time (Fig. 5), can be seen as very deep feedforward networks in which all the layers share the same weights. Although their main purpose is to learn long-term dependencies, theoretical and empirical evidence shows that it is difficult to learn to store information for very long
- augment the network with an explicit memory. The first proposal of this kind is the long short-term memory (LSTM) networks that use special hidden units, the natural behaviour of which is to remember inputs for a long time79. A special unit called the memory cell acts like an accumulator or a gated leaky neuron: it has a connection to itself at the next time step that has a weight of one, so it copies its own real-valued state and accumulates the external signal, but this self-connection is multiplicatively gated by another unit that learns to decide when to clear the content of the memory.

[TOC](#toc)
## The future of deep learning
- systems that are trained end-to-end and combine ConvNets with RNNs that use reinforcement learning to decide where to look. Systems combining deep learning and reinforcement learning are in their infancy, but they already outperform passive vision systems99 at classification tasks and produce impressive results in learning to play many different video games
- Ultimately, major progress in artificial intelligence will come about through systems that combine representation learning with complex reasoning. Although deep learning and simple reasoning have been used for speech and handwriting recognition for a long time, new paradigms are needed to replace rule-based manipulation of symbolic expressions by operations on large vectors

# Introduction to Spiking Neural Networks

## Abstract
- Spiking Neural Networks (SNNs) are distributed trainable systems whose computing elements, or neu-rons, are characterized by internal analog dynamics and by digital and sparse synaptic communications.The sparsity of the synaptic spiking inputs and the corresponding event-driven nature of neural processingcan be leveraged by hardware implementations that have demonstrated significant energy reductions ascompared  to  conventional  Artificial  Neural  Networks  (ANNs).  Most  existing  training  algorithms  forSNNs have been designed either for biological plausibility or through conversion from pre-trained ANNsvia rate encoding. This paper aims at providing an introduction to SNNs by focusing on a probabilisticsignal processing methodology that enables the direct derivation of learning rules leveraging the uniquetime encoding capabilities of SNNs. To this end, the paper adopts discrete-time probabilistic models fornetworked spiking neurons, and it derives supervised and unsupervised learning rules from first principlesby using variational inference. Examples and open research problems are also provided.

## Introduction
- Neurons in the human brain are qualitatively different from those in an ANN: They aredynamicdevices featuring recurrent behavior, rather than static non-linearities; and they process and communicateusingsparse spiking signalsover time, rather than real numbers.
- Spiking Neural Networks (SNNs) have been introduced in the theoretical neuroscience literatureas  networks  of  dynamic  spiking  neurons  [3].  SNNs  have  the  unique  capability  to  process  informationencoded  in  the  timing  of  events,  or  spikes.  Spikes  are  also  used  for  synaptic  communications,  withsynapses delaying and filtering signals before they reach the post-synaptic neuron. Due to the presenceof  synaptic  delays,  neurons  in  an  SNN  can  be  naturally  connected  via  arbitrary  recurrent  topologies,unlike standard multi-layer ANNs or chain-like Recurrent Neural Networks.
- The  energy  spent  by  SNNs  for  learning  and  training  is  essentially  proportional  tothe  number  of  spikes  processed  and  communicated  by  the  neurons,  with  the  energy  per  spike  being  aslow as a few picojoules
- The  SNN  generally  acts  as  a  dynamic  mapping  between  inputs  and  outputs  that  is  defined  by  themodel  parameters,  including,  most  notably,  the  inter-neuron,  synaptic,  weights.  This  mapping  can  bedesigned or trained to carry outinferenceorcontroltasks. When training is enabled, the model parametersare  automatically  adapted  based  on  data  fed  to  the  network  with  the  goal  of  maximizing  a  givenperformance criterion. Training can be carried out in a supervised, unsupervised, or reinforcement learningmanner, depending on the availability of data and feedback signals, as further discussed below. For bothinference/control  and  training,  data  can  be  presented  to  the  SNN  in  abatchmode  -  also  known  asframe-based - or in anon-linemanner