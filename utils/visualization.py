import random
import matplotlib.pyplot as plt
import textwrap
from PIL import Image
from data.data_loader import *
import math
import numpy as np

def show_sample_images_with_formatted_captions():
    train, val, test = image_data_loader()
    captions = text_data_loader()
    tokenized_caption, _, __ = build_tokenizer_and_vocab(captions)

    sample_keys = random.sample(list(train.keys()), 6)

    def plot_images(data_dict, captions_dict, title_suffix):
        cols = 3
        rows = (len(sample_keys) + cols - 1) // cols
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 4.5))
        axs = axs.flatten()

        for i, image_name in enumerate(sample_keys):
            image = data_dict[image_name]

            image = image.resize((250, 160)) if isinstance(image, Image.Image) else image

            axs[i].imshow(image)
            axs[i].axis('off')

            if image_name in captions_dict:
                data = captions_dict[image_name]

                if isinstance(data[0], str) and " " in data[0]:
                    caption_text = " ".join([line.strip() for line in data])
                    wrapped_caption = "\n".join(textwrap.wrap(caption_text, width=60))
                elif isinstance(data[0], list): 
                    lines = []
                    for idx, tokens in enumerate(data):
                        clean_tokens = [t for t in tokens if t not in ("<start>", "<end>") and not t.endswith("</w>")]
                        sentence = " ".join(clean_tokens)
                        wrapped = textwrap.wrap(sentence, width=60)
                        numbered = f"{idx+1}. " + "\n   ".join(wrapped)
                        lines.append(numbered)
                    wrapped_caption = "\n".join(lines)
                else:
                    wrapped_caption = "Invalid caption format."
            else:
                wrapped_caption = "No captions found."


            axs[i].imshow(image, extent=[0.05, 0.95, 0.15, 0.95])
            axs[i].axis('off')

            full_caption = f"{image_name}\n{wrapped_caption}"
            axs[i].text(
                0.5, -0.05, full_caption,  
                fontsize=7,
                ha='center',
                va='top',
                transform=axs[i].transAxes,
                wrap=True,
                clip_on=False  
            )


        for j in range(i + 1, len(axs)):
            axs[j].axis('off')

        plt.suptitle(title_suffix.replace("🖼", ""), fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    plot_images(train, captions, "Sample Images with Captions")
    plot_images(train, tokenized_caption, "Sample Images with Tokenized Captions")

# show_sample_images_with_formatted_captions()



import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import math

def visualize_attention(
    image_path,
    model,
    transform,
    vocab,
    idx_to_word,
    max_len=20,
    n_cols=8,
    figsize=(16, 5)
):
    model.eval()
    device = next(model.parameters()).device

    # Load image (raw for display, transformed for model)
    raw_img = Image.open(image_path).convert("RGB")
    proc_img = transform(raw_img).to(device)

    # Generate caption + attention maps
    seq, alphas = model.caption_with_attention(proc_img, vocab, max_len=max_len)
    words = [idx_to_word[idx] for idx in seq]

    # Start plotting
    T = len(words)
    n_rows = math.ceil((T + 1) / n_cols)
    fig = plt.figure(figsize=figsize)
    plt.axis('off')

    # Plot original image
    ax = fig.add_subplot(n_rows, n_cols, 1)
    ax.imshow(raw_img)
    ax.set_title("Input")
    ax.axis('off')

    # Plot attention overlays
    for t, (word, alpha) in enumerate(zip(words, alphas), start=2):
        ax = fig.add_subplot(n_rows, n_cols, t)

        # Reshape attention and upscale
        alpha_np = alpha.detach().numpy()
        sz = int(np.sqrt(alpha_np.shape[0]))
        alpha_np = alpha_np.reshape(sz, sz)

        # Normalize + sqrt for contrast
        alpha_np = alpha_np - alpha_np.min()
        alpha_np = alpha_np / (alpha_np.max() + 1e-8)
        alpha_np = np.sqrt(alpha_np)

        # Resize using PIL (anti-aliasing)
        alpha_img = Image.fromarray((alpha_np * 255).astype(np.uint8)).resize(
            raw_img.size, resample=Image.BILINEAR)
        alpha_np_resized = np.array(alpha_img).astype(np.float32) / 255.0

        # Overlay: background + grayscale attention
        ax.imshow(raw_img)
        ax.imshow(alpha_np_resized, cmap='gray', alpha=0.5)  # adjust alpha for balance
        ax.set_title(word)
        ax.axis('off')


    plt.tight_layout()
    plt.show()
