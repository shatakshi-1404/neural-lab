import matplotlib.pyplot as plt

def plot_history(values, title="Training history", ylabel="Loss"):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(values, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig
