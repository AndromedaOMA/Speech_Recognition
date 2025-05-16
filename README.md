<h1 align="center">Hi 👋, here we have my Speech Recognition project</h1>
<h3 align="center">Developed and trained a Convolutional ResNet & Bidirectional GRU for Speech Recognition task!</h3>


## Table Of Content
* [About Project](#project)
* [Architecture](#architecture)
* [Dataset](#dataset)
* [Getting Started](#getting-started)

--------------------------------------------------------------------------------
<h1 id="project" align="left">🤖 About Project</h1>

The main objective of this project is to recognize speech and provide transcripts associated with speech in real time. The architecture is compact but efficient, leveraging a well-structured dataset provided by the **TorchAudio** library.

---

<h1 id="architecture" align="left">🧠 Architecture</h1>

The architecture of the **Speech Recognition** model is quite simple, but effective. Thus, it presents a first 2D convolutional layer, followed by a linear one that will transfer a block of Residual Convolutional layers, then a Bidirectional GRU and last but not least the final, linear, classification layers.

The **Residual Convolutional Block** in turn features two 2D Convolutional layers and a Residual connection between the beginning and the end. Before each convolution layer there is a dropout regularization and a GELU activation function

The **Bidirectional GRU** block consists of a normalization layer, a multi-layer gated recurrent unit (GRU) RNN, and a Dropout regularizer.

Finally, the combination of all layers forms the Speech Recognition model.

---

<h1 id="dataset" align="left">📄 Dataset</h1>

Dataset provided via the **TorchAudio** library. The labeled dataset is **LIBRISPEECH**: where the input is given by audio sequences with a sampling rate of 16k, and the target is given by transcript texts associated with the audio sequences.

Next, the dataset was transformed into a Mel Spectogram so that the model, which contains convolutional Residual layers, could process them efficiently. Before loading and injecting the dataset into the model, we will pad each sample in the set to the calculated maximum size.

---

<h1 id="getting-started" align="left">🚀 Getting Started</h1>

1. Clone the repository:
``` git clone git@github.com:AndromedaOMA/Speech_Recognition.git ```
2. Have fun!

---

> 📝 **Note**:  
> By completing this project, we emphasized our knowledge of Deep Learning in order to develop and complete the bachelor's project that solves the Speech Enhancement task.

* [Table Of Content](#table-of-content)
