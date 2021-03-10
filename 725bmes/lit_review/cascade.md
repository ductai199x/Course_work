

# How did the researchers in this paper solve the problem

Brain-Computer Interface (BCI) is a popular topic of research due to its huge potential impact to our society. EEG based BCI has been a promising solution due to its ease of use and noninvasiveness. However, raw EEG signals are noisy, highly correlated, and have low spatial resolution. Hence, many methods attempts to construct images out of EEG waves to leverage frequency (spectral) analysis. In this paper, the researcher looked at leveraging the idea of "imagifying" time-domain EEG signals, but building complex representations with Convolutional Neural Networks (CNNs), and learning temporal characteristics with Recurrent Neural Networks (RNNs). 

## "Imagifying"  time-domain EEG signals

Many datasets, including the one used in this paper - PhysioNet EEG Dataset (Goldberger et al. 2000) - only contained a series of numbered waves. Hence, the information about where the waves are recorded is completely lost (as with paper 1). However, this knowledge can be recovered by looking into the device used in the experiment. From the 3 dimensional coordinate of the electrodes, a 2D mapping can be built by projecting them on to a 2D plane. Then, an M by N "image" can be constructed by padding the *null* electrodes with zeros. Then, like an image, each element in the 2D data mesh is normalized by the largest value. Finally a "movie segment" can be created with the sliding window method, bundling S images into 1 segment. The goal of the neural network architecture described below is to predict the human's intention performed during any windowed segment. 

## Convolutional Neural Network

The CNNs used in this paper have a topolgy described by 3 2D layers with the same kernel size of 3 by 3 (with zero-padding to make sure the feature maps have the same size as the input data images) but different number of feature maps: 32, 64, and 128. The output of the 3 convolutional layers is fed into a fully-connected layer of size 1 by 1024. *The researchers did not explain the intuition behind the particular sizing but they mentioned that this layer is essential in helping with convergence and in improving the performance of the whole framework.* Since the movie segment has length S, each images will be fed into its own CNN; meaning there will be S CNNs, creating S sequences of 1 by 1024. Each sequence is a spatial feature representation of the EEG waves collected at a particular time stamp. Therefore, the entire output will describe the spatial feature for the entire movie segment.

## Recurrent Neural Network

The choice for a recurrent neural network architecture in this paper is one called Long-Short Term Memory (LSTM). LSTM was first developed in 1997 for classification tasks on single data points (images) and sequences of data (speech, audio, video). From its birth 23 years ago, LSTM has come a long way and its variants are now used in almost any speech recognition system (Google Voice) and natural language processing (Google Translate). RNNs, especially those trained with back-propagation, are susceptible to the vanishing gradients problem due to finite-precision in computers. However, LSTM can solve the this issue due to the fact that its unit can allow gradients to flow *unchanged*. Also, since the gates inside each LSTM unit are very easily configured, they can be built to resolve the exploding gradients problem as well. The LSTM network in this paper has 2 layers, each layers with S (the number of images in 1 movie segment) units. The output time sequences of each unit in the first layer is fed into the corresponding unit in the second layer as input (Xt). The output of the entire network will be the output of the last unit in the second layer (with respect to the last image in the movie segment of length S). This result is the temporal feature representation of the entire movie segment. 

## Cascading Architecture

The cascading architecture works by connecting the output sequences of the CNNs to the LSTM's first layer's input units. In this fashion, both the spatial and temporal feature of the data is recovered and represented in the LSTM's final output, which is fed into a softmax classifier for calculating multi-class probabilities over different human's intentions in each movie segment. 

In short:

Movie segment -> CNN -> LSTM -> Softmax


## Parallel Architecture

The parallel architecture is different from the cascading one in that the output of the CNNs are not the input to the LSTM; but rather, the input for both networks is the movie segments themselves. In order to produce both the spatial and temporal feature of the data, the output sequences of the CNNs are combined into 1 sequence by simple summation. Then, the final representation is constructed using different fusion methods: concatenation, summation, or a combination of both with an additional fully-connected/convolutional output layer (size = 1 by number of classifying classes) at the end. This terminal transformation is fed into a softmax classifier, similar to the cascading architecture. 

In short:

Movie segment -> CNN -> combined ---|
                                    |--> fusion -> Softmax
Movie segment -> LSTM --------------|

# Results of Paper 2

The results are very promising with 98.31% accuracy for the Cascade model and 98.28% accuracy for the Parallel mode over 5 different human's intentions. The performance of their methods are superior to other cutting edge methods in the field, notably Major and Conrad 2017 (72%), Shenoy, Vinod, and Guan 2015 (82.06%), Pinheiro et al. 2016 (85.05%), Kim et al. 2016 (80.5%), Zhang et al. 2017 (79.4%), and Bashivan et al. 2016 (67.31%). Not only the accuracy number is higher, the 2 models described in this paper requires no data preprocessing steps. Since the code used in this paper is modified from Bashivan et al. 2016 paper, the author drew sharp comparisions along with reasoning. Some interesting ones are summarized below:
- Using spectral feature instead of temporal one incurs data compression due to finite precision, especially with large continuous sampling period.
- Interpolating the raw 64 channel data to a 32 by 32 matrix is a bad idea since it brings in a lot of accumulated noise.
- Solely use 3D CNNs (2D data mesh + 1D time) does not perform nearly as well as 2D CNNs + RNN because CNNs are not good at representing temporal information.
- The fully connected layers are critical components, which helps speeding up convergence and improving performance.
- Physiological activities outside of the intentions being classified affect the quality of the recorded signal (e.g eye blinks).
- The cascade and the parallel models are well suited to low resoluted data because the accuracy does not drop significantly during their real-life studied using only 14 (instead of 64) EEG channels and 128Hz (instead of 160Hz) sampling rate. 


# Why this method is superior to the one in Paper 1

To summarize, the approach described in this paper is superior to the one in Paper 1 because:
- Raw data is used instead of heavily processed data
- Neural networks classifier with large feature space are more resistance to noise, hence, they don't need heavy pre-processing steps such as averaging, band-pass filtering, spectral analyzing, and projecting from high dimension to low dimensions.
- Leveraging CNN to produce tons of spatial features from all frequencies instead of a few in different frequency bands
- Leveraging RNN to extract temporal features instead of only using LDA to exploit a few spatial pattern on specific time stamps in a windowed period.
- Once the CNN+RNN model is computed, the classification can be done in real time. The method in Paper 1 is infeasible to be done real-time due to heavy processing delays and wide windowing periods.



