
1. Neural Network
2. Cascade Model
3. How the cascade model is translated into code
4. Parameters used

The deep neural network models we attempted to evaluate came from Zhang et. al 2018 paper: "Cascade and Parallel Convolutional Recurrent Neural Networks on EEG-based Intention Recognition for Brain Computer Interface". We intended on gauging the performance of both the cascade and the parallel model demonstrated in Zhang's work because the author claimed that his models can work on raw EEG data. However, due to the paper's code not being fully open-sourced, and various technical difficulties during the process of converting the Ofner's dataset to a version which is compatible with the Zhang's models, we only conducted extensive examination of the cascade model. 

This model features a combination of Convolutional Neural Networks (CNNs) and a Long-Short Term Memory (LSTM) Network. The word "cascade" means that the outputs of the CNNs are fed into the LSTM network, whose output is used for classification. Each CNN has a topology described by 3 x 2D layers with the same kernel size of 3 by 3 (with zero-padding to make sure the feature maps have the same size as the input data images) but different number of feature maps: 32, 64, and 128 (Equation 2). The output of the 3 convolutional layers is fed into a fully-connected layer of size 1 by 1024. The number of CNNs in the architecture is determined by the number of 2D data mesh (called frames) in one segment. Since the model we built have 8 frames per segment, each segment has 4 overlapping frames, the CNNs' outputs are 8 sequences of 1 by 1024 (Equation 3). This entire output will describe the spatial feature for the entire segment. On the other hand, the LSTM we built network has 2 layers, each layer with 8 (the number of frames in a segment) units. The output time sequence of each unit in the first layer is fed into the corresponding unit in the second layer as external input. The output of the LSTM network will be the output of the last unit in the second layer (with respect to the frame in segment) (Equation 4). This result represents both the spatial and the temporal features of the entire segment. In order to do classification from this result, we decrease its dimensionality by first feeding it into one fully-connected layer that has 64 neurons (Equation 5). Finally, the output from this layer is connected to another fully-connected layer with 7 neurons (7 classes) with a soft-max activation function for categorical classification (Equation 6 & 7).

$Seg_j = (frame_i, frame_{i+1},..., frame_{i+7})$&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$(\text{eq. 1})$

$f_j = CNN_j(seg), f_j \in \R^{9\times9\times128}$&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$(\text{eq. 2})$

$F_j = Dense(Flat(f_j)), F_j \in \R^{1024}$&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$(\text{eq. 3})$

$H_{lstm} = LSTM(F), H_{lstm} \in \R^{8}$&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$(\text{eq. 4})$

$H' = Dense(H_{lstm}), H' \in \R^{64}$&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$(\text{eq. 5})$

$H = Dense(H'), H \in \R^{7}$&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$(\text{eq. 6})$

$Cascade = Softmax(H)$&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$(\text{eq. 7})$


<!-- create some equations -->

To train the model, we first augment the dataset such that: for every trials in each class, the number of frames N is expanded into M, where M is the final number of frames after applying a sliding window of size = 8 with overlap = 4. Then, the frames for all trials in a class are acculmulated into a big array of size P, for which 0.8*P frames are fed into the neural network described above with a batch size of 64 for training and validation. In addition, the Adam optimizer with a learning rate of 1e-4, and the Categorical Crossentropy loss function are used. Finally, the remaining 20% of the available frames in the dataset are used for testing.




