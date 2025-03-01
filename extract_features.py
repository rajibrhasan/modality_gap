import torch
from PIL import Image
from models import get_model_and_processor
from utils import get_llava_prompt, normalize

def get_features(model_name, images, texts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor, tokenizer = get_model_and_processor(model_name)
    model.eval()
    model.to(device)

    image_features, text_features = [], []

    for i in range(len(images)):
        image = Image.open(images[i]).convert('RGB')
        text = texts[i]
        
        with torch.no_grad():
            
            if model_name == 'LLaVa':
                inputs = processor(images=image, text = text, return_tensors="pt").to(device)
                img_feat = model.vision_tower(inputs.pixel_values)
                img_feat = model.multi_modal_projector(img_feat.last_hidden_state)
                text_feat = model.language_model.model.embed_tokens(inputs.input_ids)
            
            else:
                img_inputs = processor(images = image, return_tensors = "pt").to(device)
                text_inputs = tokenizer([text], padding = True, return_tensors = "pt").to(device)
                img_feat = model.get_image_features(**img_inputs)
                text_feat = model.get_text_features(**text_inputs)
            
            if i == 0:
                print(f'Image Embedding Shape:{img_feat.shape}')
                print(f'Text Embedding Shape: {text_feat.shape}')
                

            img_feat = normalize(img_feat)
            text_feat = normalize(text_feat)
            image_features.append(img_feat)
            text_features.append(text_feat)

    image_features = torch.tensor(image_features)
    text_features = torch.tensor(text_features)

    return image_features, text_features

def get_features_after_llm(model_name, model_id, images, captions):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor, tokenizer = get_model_and_processor(model_name, model_id)

    model.eval()
    model.to(device)

    image_features, text_features = [], []
    image_seq_len = 577

    for i in range(len(images)):
        image = Image.open(images[i]).convert('RGB')
        text = get_llava_prompt(captions[i], processor)
        inputs = processor(text = text,images = image, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model(**inputs, output_hidden_states = True)
            last_hidden_state = output.hidden_states[-1]
            text_seq_len = last_hidden_state.shape[1] - image_seq_len
            text_feat, img_feat = torch.split(last_hidden_state, [text_seq_len,image_seq_len], dim = 1)

            img_feat = normalize(img_feat)
            text_feat = normalize(text_feat)
           
            image_features.append(img_feat)
            text_features.append(text_feat)

    image_features = torch.tensor(image_features)
    text_features = torch.tensor(text_features)

    return image_features, text_features
    