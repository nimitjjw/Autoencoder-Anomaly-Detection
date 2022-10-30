# Autoencoder-Anomaly-Detection
Anomaly Detection of ECG data using autoencoders made of LSTM layers

<div align = "justify">

Anomaly detection is the task of determining when something has gone astray from the “norm”. Anomaly detection using neural networks is modeled in an unsupervised / self-supervised manner; as opposed to supervised learning, where there is a one-to-one correspondence between input feature samples and their corresponding output labels. Here, we will use Long Short-Term Memory (LSTM) neural network cells in our autoencoder model. LSTM networks are a sub-type of the more general recurrent neural networks (RNN). A key attribute of recurrent neural networks is their ability to persist information, or cell state, for use later in the network. This makes them particularly well suited for analysis of temporal data that evolves over time. LSTM networks are used in tasks such as speech recognition, text translation and here, in the analysis of sequential sensor readings for anomaly detection.

### Dataset Details
The [dataset](http://timeseriesclassification.com/description.php?Dataset=ECG5000) contains 5,000 Time Series examples (obtained with ECG), each with 140 timesteps. Each sequence corresponds to a single heartbeat from a single patient with congestive heart failure.<br>

Dataset Preview (1 Heartbeat (140 timesteps)):<br>
   1.0000000e+00, -1.1252183e-01, -2.8272038e+00, ……………, 9.2528624e-01, 1.9313742e-01
<br>

The above sequence is the numerical representation of 1 heartbeat of the ECG diagram below:
<br>
We have 5 types of hearbeats (classes):<br>
•	Normal (N)<br>
•	R-on-T Premature Ventricular Contraction (R-on-T PVC)<br>
•	Premature Ventricular Contraction (PVC)<br>
•	Supra-ventricular Premature or Ectopic Beat (SP or EB)<br>
•	Unclassified Beat (UB).<br>
### Model Details<br>
LSTM_AE (input_dim, encoding_dim, h_dims=[], h_activ=torch.nn.Sigmoid(), out_activ=torch.nn.Tanh())
Autoencoder for sequences of vectors which consists of stacked LSTMs. Can be trained on sequences of varying length.<br>

 ![LSTM Autoencoder Architecture](https://github.com/NimitJhunjhunwala/Autoencoder-Anomaly-Detection-/blob/main/LSTM%20Autoencoder%20Architecture.png)
<div align = "center">LSTM Autoencoder Architecture</div>
<br>

Parameters:<br>
•	input_dim (int): Size of each sequence element (vector)<br>
•	encoding_dim (int): Size of the vector encoding<br>
•	h_dims (list, optional (default=[])): List of hidden layer sizes for the encoder<br>
•	h_activ (torch.nn.Module or None, optional (default=torch.nn.Sigmoid())): Activation function to use for hidden layers; if None, no activation function is used<br>

### Processing Details
#### Prepare your data
First, you need to prepare a set of example sequences to train an autoencoder on. This training set should be a list of torch. Tensors, where each tensor has shape [num_elements, *num_features]. So, if each example in your training set is a sequence of 10 5x5 matrices, then each example would be a tensor with shape [10, 5, 5].<br>
#### Training your Model
quick_train(model, train_set, encoding_dim, verbose=False, lr=1e-3, epochs=50, denoise=False, **kwargs)<br>
Lets you train an autoencoder with just one line of code. Useful if you don't want to create your own training loop. Training involves learning a vector encoding of each input sequence, reconstructing the original sequence from the encoding, and calculating the loss (mean-squared error) between the reconstructed input and the original input. The autoencoder weights are updated using the Adam optimizer.<br>

  Parameters:<br>
  •	model (torch.nn.Module): Autoencoder model to train (imported from sequitur.models)<br>
  •	train_set (list): List of sequences (each a torch.Tensor) to train the model on; has shape [num_examples, seq_len, *num_features]<br>
  •	encoding_dim (int): Desired size of the vector encoding<br>
  •	verbose (bool, optional (default=False)): Whether or not to print the loss at each epoch<br>
  •	lr (float, optional (default=1e-3)): Learning rate<br>
  •	epochs (int, optional (default=50)): Number of epochs to train for<br>
  •	**kwargs: Parameters to pass into model when it's instantiated<br>

  Returns:<br>
  •	encoder (torch.nn.Module): Trained encoder model; takes a sequence (as a tensor) as input and returns an encoding of the sequence as a tensor of shape [encoding_dim]<br>
  •	decoder (torch.nn.Module): Trained decoder model; takes an encoding (as a tensor) and returns a decoded sequence<br>
  •	encodings (list): List of tensors corresponding to the final vector encodings of each sequence in the training set<br>
  •	losses (list): List of average MSE values at each epoch<br>

#### Choosing a Threshold
Using the threshold, we can turn the problem into a simple binary classification task:<br>
•	If the reconstruction loss for an example is below the threshold, we’ll classify it as a normal heartbeat<br>
•	Alternatively, if the loss is higher than the threshold, we’ll classify it as an anomaly<br>

### Snippets of output
We have used the normal heartbeats from the test set. 
 
![normal heartbeats threshold evaluation](https://github.com/NimitJhunjhunwala/Autoencoder-Anomaly-Detection-/blob/main/normal%20heartbeats%20threshold%20evaluation.png)
<div align = "center">Normal Heartbeats Threshold Evaluation</div>
<br>
We did the same with the anomaly examples, but their number is much higher. Hence, we used a subset that has the same size as the normal heartbeats. <br><br>

![anomaly heartbeats threshold evaluation](https://github.com/NimitJhunjhunwala/Autoencoder-Anomaly-Detection-/blob/main/anomaly%20heartbeats%20threshold%20evaluation.png) 
<div align = "center">Anomaly Heartbeats Threshold Evaluation</div>
<br>
The count of the correct prediction comes out to be 142/145 in both the cases (normal and anomalies).

</div>
