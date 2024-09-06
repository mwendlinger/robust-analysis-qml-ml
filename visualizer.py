from matplotlib import pyplot as plt
import numpy as np


labels_list = ['\u2212', '\u002b', '\uA714', '\u02E7']

def show_images(X, Y):
    num = 10
    images = X[:num]
    labels = Y[:num]
    num_row = 2
    num_col = 5# plot images
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
    for i in range(num):
        ax = axes[i//num_col, i%num_col]
        ax.imshow(images[i], cmap='gray')
        ax.set_title('Label: {}'.format(labels_list[labels[i]]))
    plt.tight_layout()
    plt.show()

def show_images_pred(X, Y, pred, saved=False, name=None):
    predic = [np.argmax(p) for p in pred]
    num = 10
    images = X[:num]
    labels = Y[:num]
    num_row = 2
    num_col = 5
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
    for i in range(num):
        ax = axes[i//num_col, i%num_col]
        ax.imshow(images[i], cmap='gray')
        ax.set_title('Label: {}/ {}'.format(labels_list[labels[i]], labels_list[predic[i]]))
    plt.tight_layout()
    if saved:
      plt.savefig(f"{name}.png")
    else:
      plt.show()

def show_images_heatmaps(X, Y, pred, mask=None, saved=False, alpha=0.5, name=None, only_img=False, switched=False, eps=1.0):
    predic = [np.argmax(p) for p in pred]
    num = 10
    images = X[:num]
    labels = Y[:num]
    num_row = 2
    num_col = 5
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
    for i in range(num):
        ax = axes[i//num_col, i%num_col]
        if only_img:
            if switched and labels[i] == 0:
                ax.imshow(images[i], cmap='coolwarm_r', vmin=-eps, vmax=eps)
            else:
                ax.imshow(images[i], cmap='coolwarm', vmin=-eps, vmax=eps)
                
            ax.set_title('Label: {}/ {}'.format(labels_list[labels[i]], labels_list[predic[i]]))
        else:    
            ax.imshow(images[i], cmap='gray')
            ax.set_title('Label: {}/ {}'.format(labels_list[labels[i]], labels_list[predic[i]]))
            if switched and labels[i] == 0:
                ax.imshow(mask[i], 'seismic_r', alpha=alpha, vmin=-eps, vmax=eps)
            else:
                ax.imshow(mask[i], 'seismic', alpha=alpha, vmin=-eps, vmax=eps)
    plt.tight_layout()
    if saved:
      plt.savefig(f"{name}.png")
    else:
      plt.show()


def show_single(X):

    plt.imshow(X, cmap='gray')
    plt.show()
    
def show_single_heatmap(X):

    plt.imshow(X, cmap='coolwarm', vmin=-1.0, vmax=1.0)
    plt.show()