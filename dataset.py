import json
import os
import random
from pycocotools.coco import COCO

def get_coco(root_dir, num_ex):
    data = json.load(open(f'{root_dir}/annotations/captions_val2017.json'))
    id2file = {item['id']: f"{root_dir}/{item['coco_url'].replace('http://images.cocodataset.org/', '')}" for item in data['images']}
    id2caption = {item['image_id']: item['caption'] for item in data['annotations']}
    file2caption = {id2file[id]: id2caption[id] for id in id2caption}

    images, captions = [], []
    for file in file2caption:
        images.append(file)
        captions.append(file2caption[file])
    

    if len(images) > num_ex:
        random_idxs = random.sample(range(len(images)), num_ex)
        new_images = [images[i] for i in random_idxs]
        new_captions = [captions[i] for i in random_idxs]

        return new_images, new_captions
    
    return images, captions


def load_images_and_captions(img_ids, root_dir, coco_instances, coco_captions):
    """Loads images and corresponding captions from COCO dataset."""
    images, captions = [], []
    
    for img_id in img_ids:
        image = coco_instances.loadImgs([img_id])[0]
        image_path = f"{root_dir}/{image['coco_url'].replace('http://images.cocodataset.org/', '')}"
        images.append(image_path)
        
        # Get Captions
        caption_ids = coco_captions.getAnnIds(imgIds=[img_id])
        anns = coco_captions.loadAnns(caption_ids)
        if anns:
            captions.append(random.choice(anns)["caption"])

    return images, captions


def get_coco_data_by_supercategory(root_dir, num_ex):
    
    instance_path = f'{root_dir}/annotations/instances_train2017.json'
    caption_path = f'{root_dir}/annotations/captions_train2017.json'
    coco_instances = COCO(instance_path)
    coco_captions = COCO(caption_path)
    categories = coco_instances.loadCats(coco_instances.getCatIds())
    supercategories = list(set(cat['supercategory'] for cat in categories))

    data_dict = {}
    all_ids = set()

    for sup_cat in supercategories:
        # Get category IDs for the current supercategory
        category_ids = [cat['id'] for cat in categories if cat['supercategory'] == sup_cat]
        
        # Get all image IDs for the selected category IDs
        image_ids = set()
        for cat_id in category_ids:
            image_ids.update(coco_instances.getImgIds(catIds=cat_id))
        
        # Remove already selected IDs to avoid duplicates
        image_ids = list(image_ids - all_ids)
        
        # Sample a subset (up to max_per_class)
        sampled_image_ids = random.sample(image_ids, min(num_ex, len(image_ids)))
        
        # Update tracking variables
        all_ids.update(sampled_image_ids)
        images, captions = load_images_and_captions(sampled_image_ids, root_dir, coco_instances, coco_captions)
        data_dict[sup_cat] = (images, captions)
    
    return data_dict

    
