librnn
==========

Overview
--------

## librnn
librnn has no dependencies for CPU support and only requires CUDA libraries for GPU support.

librnn's main goal is to provide interfaces to easily model with RNNs. Support for various types of RNNs is already included and more will be added. Language modeling and other sequence to sequence models will be added to the examples folder.

## `var<T>`

The `var<T>` class is a wrapper over the data being used by librnn. Some of its semantics are similar to Caffe's Blob, including how it synchronizes data between the CPU and GPU. A `var` is an N-dimensional array that can hold any type of data; e.g., weights, images, derivatives, etc. `var`'s are also overloaded to be able to perform operations between them. As the operations are performed a symbolic graph is produced to automatically calculate the derivatives by using automatic differentiation. As a result, adding new layers in librnn is very simple. If no new operations are needed, one only needs to define the forward pass and librnn will handle the backwards pass automatically similar to other popular machine learning frameowkrs (e.g. torch-autograd, theano, etc.).

## TODO

### Library Features
- [ ] add ability to save and load checkpoints in training
- [ ] speed optimization (https://github.com/tiny-dnn/tiny-dnn/pull/193)
- [ ] random weights init should take doubles, not template typename.

### Library Misc.
- [ ] Reduce memory usage
- [ ] Review tensor implementation (cnhw)
- [ ] Add code for benchmarking
- [ ] Implement tests for the operations forwards/backwards
- [ ] Set up .clang_format (so all imports, namespace, and includes follow same style)
- [ ] include-what-you-use
- [ ] Decide on namespace
- [ ] Run performance tests

### GPU Support
- [ ] Add CUDA code for operations
- [ ] handle moving data between cpu and gpu (SyncedMem)

### Types of RNNs
- [x] RNN
- [x] LSTM
- [ ] GRU
- [ ] Bidirectional RNN
- [ ] Neural Stack Machine
- [ ] Neural Turing Machine
- [ ] [RNN-EM](http://arxiv.org/abs/1506.00195)
- [ ] [Hierarchical Multiscale Recurrent Neural Networks](https://arxiv.org/abs/1609.01704)

### RNN Features
- [ ] Highway Networks
- [ ] Recurrent Highway Networks
- [ ] Multiplicative Integration Within RNNs
- [ ] Recurrent Dropout Without Memory Loss
- [ ] Layer Normalization
- [ ] Layer Normalization
- [ ] LSTM With Multiple Memory Arrays
- [ ] Minimal Gated Unit Recurrent Neural Network
- [ ] GRU Mutants
