import matplotlib.pyplot as plt
import torch

def visualize_sample(sample, show=True):
    images, labels = sample
    
    if isinstance(images, torch.Tensor):
        images = images.numpy()

    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()
    
    num_images = images.shape[0]
    
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 3, 3))
    
    if num_images == 1:
        axes = [axes]
        labels = [labels]

    for i, (img, ax) in enumerate(zip(images, axes)):
        ax.imshow(img.squeeze(), cmap='gray')
        ax.axis('off')
        ax.set_title(f"{labels[i]}")
    
    plt.tight_layout()

    if show:
        fig.show()

    return fig, axes
