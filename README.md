# NeuralLab — Deep Learning Playground

A compact educational project covering neural-network fundamentals, computer vision,
sequence models, attention, autoencoders, GANs, and transfer learning.

## Setup (Windows PowerShell)
```powershell
cd D:\NeuralLab
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```
Python 3.10–3.12 is recommended. The first run downloads MNIST and may download
pretrained ResNet weights if transfer learning is run with pretrained weights enabled.

## Run from the command line
```powershell
python -m experiments.mnist_mlp --epochs 2
python -m experiments.mnist_cnn --epochs 2
python -m experiments.sentiment_lstm --epochs 5 --kind lstm
python -m experiments.sentiment_transformer --epochs 5
python -m experiments.autoencoder --epochs 2
python -m experiments.simple_gan --epochs 2
python -m experiments.transfer_learning --epochs 2 --pretrained 0
```

## Learning path
1. `fundamentals/`: NumPy perceptron, activations, losses, manual backpropagation and optimizers.
2. `mnist_mlp.py`: dense neural network, dropout and classification.
3. `mnist_cnn.py`: convolution, filters, pooling and feature maps.
4. `sentiment_lstm.py`: embeddings, RNN, LSTM and GRU.
5. `sentiment_transformer.py`: positional encoding and self-attention.
6. `autoencoder.py`: encoder, latent representation and decoder.
7. `simple_gan.py`: generator/discriminator adversarial training.
8. `transfer_learning.py`: pretrained ResNet18 feature extraction and fine-tuning.

## Notes
- Sentiment data is a tiny synthetic teaching dataset, not a real-world benchmark.
- GAN training can be unstable; generated images are a demonstration.
- Transfer learning uses MNIST resized to fit ResNet18; it is intentionally educational.
- This project is for learning, not production use.
