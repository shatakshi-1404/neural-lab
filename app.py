import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="NeuralLab", page_icon="🧠", layout="wide")
st.title("🧠 NeuralLab")
st.caption("A hands-on playground for deep learning fundamentals and architectures.")
st.info("Start with Fundamentals, then try the experiments. First runs may download datasets.")

page = st.sidebar.selectbox("Choose a module", [
    "Overview", "Fundamentals", "MNIST — MLP", "MNIST — CNN",
    "Sentiment — RNN / LSTM / GRU", "Sentiment — Transformer",
    "Autoencoder", "GAN", "Transfer Learning"
])

if page == "Overview":
    st.subheader("Learning roadmap")
    columns = st.columns(4)
    cards = [
        ("01", "Fundamentals", "Neurons, activations, losses and backpropagation"),
        ("02", "Computer vision", "MLP, CNN, convolution and pooling"),
        ("03", "Sequence models", "Embeddings, RNN, LSTM, GRU and attention"),
        ("04", "Generative models", "Autoencoders and GANs")
    ]
    for column, (number, title, description) in zip(columns, cards):
        column.metric(title, number)
        column.write(description)
    st.markdown("**Suggested order:** Fundamentals → MLP → CNN → RNN/LSTM → Transformer → Autoencoder → GAN → Transfer Learning.")

elif page == "Fundamentals":
    import numpy as np
    from fundamentals.neural_network import TinyMLP
    from fundamentals.activations import sigmoid, relu, tanh

    st.subheader("NumPy fundamentals")
    x = np.linspace(-5, 5, 200)
    fig, ax = plt.subplots()
    ax.plot(x, sigmoid(x), label="Sigmoid")
    ax.plot(x, relu(x), label="ReLU")
    ax.plot(x, tanh(x), label="Tanh")
    ax.legend()
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)

    st.write("Train a tiny neural network on XOR using manually implemented backpropagation.")
    epochs = st.slider("Training epochs", 500, 10000, 4000, 500)
    if st.button("Train NumPy XOR network"):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 1, 1, 0])
        network = TinyMLP(input_size=2, hidden_size=8)
        network.fit(X, y, epochs)
        st.dataframe({
            "input": [str(row) for row in X],
            "actual": y,
            "predicted": network.predict(X),
            "probability": network.predict_proba(X).round(3)
        })
        fig, ax = plt.subplots()
        ax.plot(network.loss_history)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Binary cross-entropy")
        st.pyplot(fig)
        plt.close(fig)

elif page in ["MNIST — MLP", "MNIST — CNN"]:
    from experiments.mnist_mlp import run as run_mlp
    from experiments.mnist_cnn import run as run_cnn
    st.subheader(page)
    epochs = st.slider("Epochs", 1, 5, 2)
    if st.button("Train model"):
        with st.spinner("Training; MNIST may download on the first run..."):
            model, history = run_mlp(epochs) if page.endswith("MLP") else run_cnn(epochs)
        col1, col2 = st.columns(2)
        col1.metric("Final training loss", f"{history['train_loss'][-1]:.4f}")
        col2.metric("Test accuracy", f"{history['test_accuracy'][-1]*100:.2f}%")
        fig, ax = plt.subplots()
        ax.plot(history["train_loss"], marker="o")
        ax.set_title("Training loss")
        ax.set_xlabel("Epoch")
        st.pyplot(fig)
        plt.close(fig)
        if hasattr(model, "features"):
            st.write("CNN feature extractor")
            st.code(str(model.features))

elif page == "Sentiment — RNN / LSTM / GRU":
    from experiments.sentiment_lstm import run
    st.subheader("Sequence models for sentiment")
    kind = st.selectbox("Architecture", ["rnn", "lstm", "gru"])
    epochs = st.slider("Epochs", 1, 15, 5)
    st.caption("Uses a small synthetic teaching dataset; metrics are not a real-world benchmark.")
    if st.button("Train sequence model"):
        with st.spinner("Training..."):
            model, vocab, history = run(epochs, kind)
        st.metric("Validation accuracy", f"{history['val_accuracy'][-1]*100:.1f}%")
        st.line_chart(history["train_loss"])

elif page == "Sentiment — Transformer":
    from experiments.sentiment_transformer import run
    st.subheader("Transformer encoder")
    st.write("Includes token embeddings, positional encoding, multi-head self-attention and feed-forward layers.")
    epochs = st.slider("Epochs", 1, 15, 5)
    if st.button("Train Transformer"):
        with st.spinner("Training..."):
            model, vocab, history = run(epochs)
        st.metric("Validation accuracy", f"{history['val_accuracy'][-1]*100:.1f}%")
        st.line_chart(history["train_loss"])

elif page == "Autoencoder":
    from experiments.autoencoder import run
    st.subheader("Convolutional autoencoder")
    st.write("Learns a compressed latent representation and reconstructs the input image.")
    epochs = st.slider("Epochs", 1, 5, 2)
    if st.button("Train autoencoder"):
        with st.spinner("Training on MNIST..."):
            model, history = run(epochs)
        st.metric("Final reconstruction MSE", f"{history[-1]:.5f}")
        st.line_chart(history)

elif page == "GAN":
    from experiments.simple_gan import run
    st.subheader("Generative adversarial network")
    st.write("The generator creates digit-like images; the discriminator distinguishes real from generated images.")
    epochs = st.slider("Epochs", 1, 5, 2)
    if st.button("Train GAN"):
        with st.spinner("GAN training can take time and may be unstable..."):
            generator, discriminator, history = run(epochs)
        st.success("Generated examples saved as gan_samples.png.")
        st.line_chart(history)
        st.image("gan_samples.png")

elif page == "Transfer Learning":
    from experiments.transfer_learning import run
    st.subheader("Transfer learning with ResNet18")
    st.write("Freezes a ResNet18 feature extractor and trains a new classification head on MNIST.")
    epochs = st.slider("Epochs", 1, 5, 2)
    pretrained = st.checkbox("Download and use pretrained ImageNet weights", value=False)
    if st.button("Train transfer-learning model"):
        with st.spinner("Training..."):
            model, history = run(epochs, pretrained)
        st.metric("Test accuracy", f"{history['test_accuracy'][-1]*100:.2f}%")
        st.line_chart(history["train_loss"])
