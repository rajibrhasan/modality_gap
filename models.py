import torch
from huggingface_hub import login
from transformers import AutoProcessor, AutoTokenizer, SiglipModel, CLIPModel, BlipModel, Blip2Model, AutoModel,  LlavaForConditionalGeneration, BitsAndBytesConfig

model_dict = {
    'CLIP': {
        'model_id':'openai/clip-vit-base-patch32',
        'model': CLIPModel,
    },

    'BLIP':{
        'model_id': 'Salesforce/blip-image-captioning-base',
        'model': BlipModel
    },

    'BLIP2':{
        'model_id': 'Salesforce/blip2-opt-2.7b',
        'model': Blip2Model
    },

    'SIGLIP':{
        'model_id': 'google/siglip-base-patch16-224',
        'model': SiglipModel
    },

    'LLaVa':{
        'model_id': 'llava-hf/llava-1.5-7b-hf',
        'model': LlavaForConditionalGeneration
    }   
}

def get_model_and_processor(model_name, device, quantize = False):
    model_cls = model_dict[model_name]['model']
    model_id = model_dict[model_name]['model_id']

    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = None

    if quantize:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        model = model_cls.from_pretrained(model_id, quantization_config=quantization_config)

    else:
        model = model_cls.from_pretrained(model_id)

    if model_name != 'LLaVa':
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    model.eval()
    model.to(device)

    return model, processor, tokenizer
    
    
    

    


    

