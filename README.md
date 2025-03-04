# Modality Gap Visualization in Vision-Language Models (VLMs)
## Overview
This project aims to visualize and analyze the modality gap that exists in popular Vision-Language Models (VLMs), such as CLIP, BLIP, BLIP-2, SigLIP, and LLaVA. The modality gap refers to the differences in feature representations between visual and textual embeddings, which can impact model alignment and performance.

## Usage
### Clone the repository
git clone https://github.com/rajibrhasan/ModalityGap.git 

### Install required libraries
pip install -r requirements.txt

### Download Multimodal Dataset
To analyze the modality gap, you need a dataset containing paired image-text samples. We use the MS COCO dataset for this purpose. 


### Run
Once the dataset is downloaded and dependencies are installed, you can start the visualization process by running:<br />
bash scripts/llava.sh

### Argument Description
+ model_name:	Specifies which Vision-Language Model (VLM) to use for extracting embeddings. Options include CLIP, BLIP, BLIP-2, SigLIP, and LLaVA. In this case, LLaVA is selected.<br />
+ dataset_name:	The name of the dataset used for evaluation. Here, coco refers to the MS COCO dataset, a widely used dataset for multimodal learning.<br />
+ root_dir:	The directory where the dataset is stored. coco2017 refers to the COCO 2017 dataset folder.<br />
+ reducer:	The dimensionality reduction method used to visualize the modality gap. Here, UMAP (Uniform Manifold Approximation and Projection) is used to reduce high-dimensional embeddings into a lower-dimensional space for visualization. Other options might include tsne.<br />
+ by_category:	A flag that enables per-category visualization. When set, embeddings are grouped and visualized according to their respective object categories in the dataset.<br />
+ batch_size:	The number of samples processed in each batch during inference. A value of 8 means the model will process 8 images at a time. Increasing this value can speed up processing but requires more memory.<br />
+ num_ex:	The total number of examples to be processed from the dataset. Here, 512 examples will be used for visualization.<br />


