import json
import random

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

    
