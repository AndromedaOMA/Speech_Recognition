<h1 align="center">Hi 👋, here we have my Speech Recognition project</h1>
<h3 align="center">Developed and trained a Convolutional ResNet & Bidirectional GRU for Speech Recognition task!</h3>


## Table Of Content
* [About Project](#project)
* [Architecture Overview](#architecture)
* [Dataset](#dataset)
* [Getting Started](#getting-started)

--------------------------------------------------------------------------------
<h1 id="project" align="left">🤖 About Project</h1>

The main objective of this project is to recognize speech and provide transcripts associated with speech in real time. The architecture is compact but efficient, leveraging a well-structured dataset provided by the **TorchAudio** library.

---

<h1 id="architecture" align="left">🧠 Architecture Overview</h1>

The **SpeechRecognitionModel** is a deep neural architecture designed to perform **automatic speech recognition (ASR)**. It processes 2D audio features, typically spectrograms or log-Mel filterbanks, and converts them into class predictions such as character sequences, phonemes, or subword units. The model is composed of a convolutional front-end, a series of residual convolutional layers, multiple bidirectional recurrent layers, and a classification head.

The model begins by receiving an input tensor of shape **(batch_size, 1, input_features, time_steps)**, where **1** is the number of audio channels, **input_features** typically represents the number of frequency bins, and **time_steps** indicates how many audio frames are processed. The first layer is a basic convolutional layer that increases the number of channels from 1 to 32 while preserving the temporal and frequency dimensions, thanks to padding.

Following the initial convolution, the model applies a sequence of **residual convolutional blocks** (ResidualCNN). Each ResidualCNN block contains layer normalization, GELU activation, dropout, and convolutional operations. These blocks also use residual connections to help preserve information across layers and stabilize training. The number of residual blocks is configurable and defined in the training configuration.

After the residual CNN stack, the output is reshaped into a 3D tensor to prepare it for the recurrent layers. This reshaping involves flattening the spatial dimensions (channels × frequency) while keeping the time dimension intact. A linear layer is applied to project this combined feature vector to a size suitable for recurrent processing.

The next stage includes multiple **Bidirectional GRU (Gated Recurrent Unit) layers**. These layers are wrapped in layer normalization, followed by GELU activations and dropout. Each GRU layer processes the sequence bidirectionally, capturing both past and future context for each time step, which is crucial in understanding the temporal nature of speech. The number of GRU layers and their hidden size are also configurable.

The final part of the model is a **classifier** that takes the output of the last GRU layer and maps it to a target dimension. The classifier is composed of a linear layer that reduces the feature size by half, followed by GELU activation and dropout for regularization, and a final linear layer that maps to the number of output classes, such as characters, phonemes, or word pieces.

Overall, the **SpeechRecognitionModel** combines the spatial feature extraction power of CNNs, the temporal modeling capabilities of bidirectional GRUs, and a lightweight classifier head to produce accurate, context-aware predictions for speech recognition tasks. Its design is modular and highly configurable, making it suitable for a range of ASR datasets and applications.

---

<h1 id="dataset" align="left">📄 Dataset</h1>

Dataset provided via the **TorchAudio** library. The labeled dataset is **LIBRISPEECH**: where the input is given by audio sequences with a sampling rate of 16k, and the target is given by transcript texts associated with the audio sequences.

Next, the dataset was transformed into a Mel Spectogram so that the model, which contains convolutional Residual layers, could process them efficiently. Before loading and injecting the dataset into the model, we will pad each sample in the set to the calculated maximum size.

---

<h1 id="getting-started" align="left">🚀 Getting Started</h1>

1. Clone the repository: ``git clone git@github.com:AndromedaOMA/Speech_Recognition.git``
4. Have fun!

---

> 📝 **Note**:  
> By completing this project, I emphasized my knowledge of Deep Learning in order to develop and complete the bachelor's project that solves the Speech Enhancement task.

* [Table Of Content](#table-of-content)
