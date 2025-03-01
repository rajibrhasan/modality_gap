import os
import argparse
from dataset import get_coco
from huggingface_hub import login
from extract_features import get_features
from utils import compute_euclidean_dist, compute_similarity, visualize

def main(args):
    login(token = os.getenv('hf_token'))

    if args.dataset_name == 'coco':
        images, texts = get_coco(args.root_dir, args.num_ex)

    else:
        raise NotImplementedError
    
    image_features, text_features = get_features(args.model_name, images, texts)
    print(f'Mean cosine simialrity: {compute_similarity(image_features, text_features): 0.3f}')
    print(f'Mean euclidean distance: {compute_euclidean_dist(image_features, text_features): 0.3f}')

    visualize(image_features, text_features, args.model_name, args.reducer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Arguments for modality gap visualization")
    parser.add_argument("--model_name", type=str, default="CLIP", help="Name of the model")
    parser.add_argument("--dataset_name", type = str, default="coco", help="Name of the dataset")
    parser.add_argument("--root_dir", type = str, required=True, help='Root directory for the dataset')
    parser.add_argument("--quantize", type = bool, action = 'store_true', help='To load quantize model')
    parser.add_argument("--reducer", type=str, default='tsne', help='Name of the dimensionality reduction method')
    parser.add_argument('--num_ex', type = int, default = 500, help = 'Number of examples to visualize')
    args = parser.parse_args()

    main(args)
