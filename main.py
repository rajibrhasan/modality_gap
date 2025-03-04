import os
import argparse
import torch
from dataset import get_coco, get_coco_data_by_supercategory
from huggingface_hub import login
from dotenv import load_dotenv
from extract_features import get_features, get_features_llava
from utils import compute_euclidean_dist, compute_similarity, visualize
from models import get_model_and_processor
import pandas as pd
import numpy as np

def process_by_category(args, device, data_dict, model, processor, tokenizer, after_llm = False):
    category_image_features = {}
    category_text_features = {}
    for cat_name, data in data_dict.items():
        if args.model_name == 'LLaVa':
            image_feats, text_feats = get_features_llava( model, processor, device, data[0], data[1], args.batch_size, after_llm)
        
        else:
            image_feats, text_feats = get_features(model, processor, tokenizer, device, data[0], data[1], args.batch_size)

        category_image_features[cat_name] = image_feats
        category_text_features[cat_name] = text_feats
    
    table = []
    supercategories = category_image_features.keys()
    for key1 in supercategories:
        row = []
        for key2 in supercategories:
            print(key1, '->', key2)
            image_features = category_image_features[key1]
            text_features = category_text_features[key2]
            pairwise_distances = torch.cdist(image_features, text_features, p=2)
            row.append(round(pairwise_distances.mean().item(), 3))
        table.append(row)

    df = pd.DataFrame(np.array(table), columns=supercategories)
    df.index = supercategories

    file_name = 'outputs/results_'+ ('after_lm' if after_llm else '')+'.csv'
    df.to_csv(file_name)


def main(args):
    login(token = os.getenv('HF_TOKEN'))
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor, tokenizer = get_model_and_processor(args.model_name, device, args.quantize)

    if args.by_category:
        if args.dataset_name == 'coco':
            data_dict = get_coco_data_by_supercategory(args.root_dir, args.num_ex)
        
        else:
            raise NotImplementedError
        
        process_by_category(args, device, data_dict, model, processor, tokenizer, False)
        process_by_category(args, device, data_dict, model, processor, tokenizer, True)

        
    else:
        if args.dataset_name == 'coco':
            data = get_coco(args.root_dir, args.num_ex)
        else:
            raise NotImplementedError
        
        image_features, text_features = get_features(model, processor, tokenizer, device, data[0], data[1], args.batch_size)
        print(f'Mean cosine simialrity: {compute_similarity(image_features, text_features): 0.3f}')
        print(f'Mean euclidean distance: {compute_euclidean_dist(image_features, text_features): 0.3f}')

        visualize(image_features, text_features, args.model_name, args.reducer)


if __name__ == '__main__':
    load_dotenv()
    parser = argparse.ArgumentParser(description="Arguments for modality gap visualization")
    parser.add_argument("--model_name", type=str, default="CLIP", help="Name of the model")
    parser.add_argument("--dataset_name", type=str, default="coco", help="Name of the dataset")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory for the dataset")
    parser.add_argument("--quantize", action="store_true", help="Load quantized model (default: False)")
    parser.add_argument("--reducer", type=str, default="tsne", help="Dimensionality reduction method")
    parser.add_argument("--num_ex", type=int, default=512, help="Number of examples to visualize")
    parser.add_argument("--by_category", action="store_true", help="Category wise modality gap")
    parser.add_argument("--batch_size", type = int, default=8, help="Category wise modality gap")

    args = parser.parse_args()
    print(args.quantize)

    main(args)
