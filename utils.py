import torch
import umap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

def visualize(image_features, text_features, model_name, reducer):
    data = np.concatenate([image_features, text_features], 0)
    labels = np.concatenate([np.zeros(len(image_features), dtype=np.int8), np.ones(len(text_features), dtype = np.int8)])

    if reducer == 'tsne':
        tsne = TSNE(n_components=2, verbose=1, perplexity=100, n_iter=1000)
        embedding = tsne.fit_transform(data)
    
    elif reducer == 'umap':
        umap_reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = umap_reducer.fit_transform(data)
    
    else:
        raise NotImplementedError


    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red']
    classes = ['image_embeddings', 'text_embeddings']

    for i in range(2):
        plt.scatter(embedding[labels == i, 0], embedding[labels == i, 1], c=colors[i], label=classes[i], s=10)

    for i in range(len(image_features)):
        plt.plot([embedding[i, 0], embedding[len(image_features)+i, 0]], [embedding[i, 1], embedding[len(image_features)+i, 1]], c='black', alpha=0.1)
        
    plt.title(f'Modality Gap in {model_name} using {reducer}')
    plt.xlabel(f'{reducer} 1')
    plt.ylabel(f'{reducer} 2')
    plt.legend()

    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    plt.savefig(f'outputs/{model_name}_{reducer}.png')
    plt.show()

def compute_similarity(img_feats, text_feats):
    cosine_sim = img_feats @ text_feats.T
    cosine_sim = torch.diag(cosine_sim)
    mean_sim = torch.mean(cosine_sim)
    
    return mean_sim

def compute_euclidean_dist(img_feats, text_feats):
    D = np.array(img_feats) - np.array(text_feats)
    D_squared = D ** 2
    sum_sq_diff = np.sum(D_squared, axis=1)
    euclidean_dist = np.sqrt(sum_sq_diff)
    mean_dist = np.mean(euclidean_dist)

    return mean_dist

def get_llava_prompt(text, processor):
    conversation = [
        {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": text},
            ],
        },
    ]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    return text_prompt

def normalize(feat):
    feat /= feat.norm(dim = -1, keepdim = True)
    feat = feat[:,1:,:].mean(dim = 1)
    feat =feat.cpu().numpy()[0]

    return feat
