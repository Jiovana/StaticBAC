**huggingface**	https://huggingface.co/models		requires HF console authentication							
* meta-llama/Llama-3.2-1B	python create_meta.py --model meta-llama/Llama-3.2-1B --out_dir ../models8/llama-32 --source hf	 					requires repo owner consent			
* FacebookAI/roberta-large	python create_meta.py --model FacebookAI/roberta-large --out_dir ../models8/roberta --source hf	 								
* google-t5/t5-base	python create_meta.py --model google-t5/t5-base --out_dir ../models8/google-t5 --source hf  									
* google-bert/bert-base-uncased	python create_meta.py --model google-bert/bert-base-uncased --out_dir ../models8/google-bert --source hf  								
* openai-community/gpt2	python create_meta.py --model openai-community/gpt2 --out_dir ../models8/gpt2 --source hf  									
* openai-community/openai-gpt	python create_meta.py --model openai-community/openai-gpt --out_dir ../models8/gpt --source hf	 								
										
**torchvision**	https://docs.pytorch.org/vision/main/models.html									
* vit_b_16	python create_meta.py --model vit_b_16 --out_dir ../models8/vit_b_16 --source torchvision --weights ViT_B_16_Weights.IMAGENET1K_V1  							
* vgg19	python create_meta.py --model vgg19 --out_dir ../models8/vgg19 --source torchvision --weights VGG19_Weights.IMAGENET1K_V  								
* swin_b	python create_meta.py --model swin_b --out_dir ../models8/swin_b --source torchvision --weights Swin_B_Weights.IMAGENET1K_V1  								
* resnet50	python create_meta.py --model resnet50 --out_dir ../models8/resnet50 --source torchvision --weights ResNet50_Weights.IMAGENET1K_V1  							
* mobilenet_v3_large	python create_meta.py --model mobilenet_v3_large --out_dir ../models8/mobilenet_v3_large --source torchvision --weights MobileNet_V3_Large_Weights.IMAGENET1K_V1  
* inception_v3	python create_meta.py --model inception_v3 --out_dir ../models8/inception_v3 --source torchvision --weights Inception_V3_Weights.IMAGENET1K_V1  						
* efficientnet_b7	python create_meta.py --model efficientnet_b7 --out_dir ../models8/efficientnet_b7 --source torchvision --weights EfficientNet_B7_Weights.IMAGENET1K_V1	 								
										