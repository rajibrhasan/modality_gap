import torch
from PIL import Image
from models import get_model_and_processor
from utils import get_llava_prompt, normalize


def get_features_llava(model, processor, device, images, texts, batch_size, after_llm = False):
    image_features, text_features = [], []
    image_seq_len = 576
    num_batches = len(images) // batch_size

    for i in range(num_batches):
        current_ind = i*batch_size
        batch_images = [Image.open(images[j]) for j in range(current_ind, current_ind+batch_size)]

        if after_llm:
            batch_captions = [texts[j]+"<image>" for j in range(current_ind, current_ind+batch_size) ]
        else:
            batch_captions = [texts[j] for j in range(current_ind, current_ind+batch_size) ]
        
        with torch.no_grad():
            inputs = processor(images=batch_images, text = batch_captions, padding=True, truncation=True, return_tensors="pt").to(device)
            if after_llm:
                output = model(**inputs, output_hidden_states = True)
                last_hidden_state = output.hidden_states[-1]
                text_seq_len = last_hidden_state.shape[1] - image_seq_len
                text_feat, img_feat = torch.split(last_hidden_state, [text_seq_len,image_seq_len], dim = 1)

            else:
                img_feat = model.vision_tower(inputs.pixel_values)
                img_feat = model.multi_modal_projector(img_feat.last_hidden_state)
                text_feat = model.language_model.model.embed_tokens(inputs.input_ids)

            
            img_feat /= img_feat.norm(dim = -1, keepdim = True)
            img_feat = img_feat[:,1:,:].mean(dim = 1)
        
           
            text_feat /= text_feat.norm(dim = -1, keepdim = True)
            text_feat = text_feat[:,1:,:].mean(dim = 1)

            image_features.append(img_feat.cpu())
            text_features.append(text_feat.cpu())

    image_features = torch.cat(image_features, dim = 0)
    text_features = torch.cat(text_features, dim = 0)

    return image_features, text_features
    

def get_features(model, processor, tokenizer, device, images, texts, batch_size):
   
    image_features, text_features = [], []
    num_batches = len(images) // batch_size

    for i in range(num_batches):
        current_ind = i*batch_size
        batch_images = [Image.open(images[j]).convert('RGB') for j in range(current_ind, current_ind+batch_size)]
        batch_captions = [texts[j] for j in range(current_ind, current_ind+batch_size) ]
        
        with torch.no_grad():
            img_inputs = processor(images = batch_images, return_tensors = "pt").to(device)
            text_inputs = tokenizer(batch_captions, padding = True, return_tensors = "pt").to(device)
            img_feat = model.get_image_features(**img_inputs)
            text_feat = model.get_text_features(**text_inputs)

            
            img_feat /= img_feat.norm(dim = -1, keepdim = True)
            text_feat /= text_feat.norm(dim = -1, keepdim = True)
                
            image_features.append(img_feat.cpu())
            text_features.append(text_feat.cpu())

    image_features = torch.cat(image_features, dim = 0)
    text_features = torch.cat(text_features, dim = 0)

    return image_features, text_features
    