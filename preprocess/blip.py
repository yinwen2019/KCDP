import os
import sys
import json
import time
import argparse

# print("当前的工作目录：", os.getcwd())
# print("python搜索模块的路径集合", sys.path)
# curPath = os.path.abspath(os.path.dirname(__file__))
# rootPath = os.path.split(curPath)[0]
# sys.path.append(rootPath)
# print("python搜索模块的路径集合", sys.path)

sys.path.append("..")
from dataset import SERCaption, Emoset, Emo8, EmotionROI
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from tqdm import tqdm
from PIL import Image
import requests


class BLIPInterface:
    def __init__(self, blip_model_url, caption_dataset, device):
        self.device = device
        self.blip_model_url = blip_model_url
        self.model = None
        self.processor = None
        self.blip_generation_param_dict = {
            'num_beams': 5,
            'repetition_penalty': 5.0,
            'max_new_tokens': 50,
            'min_new_tokens': 30,
        }

        self.caption_dataset = caption_dataset
        self.caption_dataset_name = caption_dataset.__class__.__name__ + '_' + caption_dataset.get_mode()

        self.out_path = "captions/"
        self.captions_file = os.path.join(self.out_path, self.caption_dataset_name + '_captions.json')

    def init_blip_model(self):
        # Takes a while to load
        st = time.time()
        print(f'Loading InstructBlipProcessor...from {self.blip_model_url}')
        self.processor = InstructBlipProcessor.from_pretrained(self.blip_model_url)
        print(f'Loading InstructBlipForConditionalGeneration... from {self.blip_model_url}(can take a minute)')
        self.model = InstructBlipForConditionalGeneration.from_pretrained(self.blip_model_url)
        print(f'Loaded InstructBlip in {time.time() - st} seconds')

        self.model.to(self.device)
        self.model.eval()

    def __call__(self, batch_size=128, img_name_dict=None, profiling=False):
        # init
        self.init_blip_model()

        image_file_captions = []
        num_images = len(self.caption_dataset)
        with torch.no_grad():
            for i in tqdm(range(0, num_images, batch_size)):
                image_batch_indices = list(range(i, min(i + batch_size, num_images)))
                image_batch = [self.caption_dataset.get_image(j, return_img_path=True) for j in image_batch_indices]
                image_batch, image_paths = zip(*image_batch)
                prompt = "What is the information of subject in the picture? What is the facial expression and the body movement of the subject in the picture?What is the emotional meaning of the image? Please answer these question with a descriptive sentence."
                prompts = [prompt] * len(image_batch_indices)
                inputs = self.processor(image_batch, text=prompts, return_tensors="pt").to(self.device)
                generated_ids = self.model.generate(**inputs,
                                                    num_beams=self.blip_generation_param_dict['num_beams'],
                                                    max_new_tokens=self.blip_generation_param_dict['max_new_tokens'],
                                                    min_new_tokens=self.blip_generation_param_dict['min_new_tokens'],
                                                    repetition_penalty=self.blip_generation_param_dict[
                                                        'repetition_penalty'])
                generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

                for j, (img_file, caption) in enumerate(zip(image_paths, generated_texts)):
                    image_file_captions.append(caption.strip())

        with open(self.captions_file, 'w') as f:
            json.dump(image_file_captions, f, indent=4)

        return image_file_captions


def example():
    model = InstructBlipForConditionalGeneration.from_pretrained("/home/user/xxx/PLM/instructblip-flan-t5-xl")
    processor = InstructBlipProcessor.from_pretrained("/home/user/xxx/PLM/instructblip-flan-t5-xl")

    # model = InstructBlipForConditionalGeneration.from_pretrained("/home/user/xxx/PLM/instructblip-vicuna-7b")
    # processor = InstructBlipProcessor.from_pretrained("/home/user/xxx/PLM/instructblip-vicuna-7b")

    device = "cuda:7" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    #url = "/home/user/xxx/MultiModal/TGCA_PVT/dataset/SER30K/Images/cute-ghost/sticker_8.jpg"#abnormal value
    #url = "/home/user/xxx/MultiModal/TGCA_PVT/dataset/SER30K/Images/09-panda-bear/sticker_22.jpg"  # shift1 s
    #url = "/home/user/xxx/MultiModal/Emoset/image/contentment/contentment_03339.jpg"  # shift1 e
    url = "/home/user/xxx/MultiModal/Emoset/image/contentment/contentment_04518.jpg"  # section 3
    #url = "/home/user/xxx/MultiModal/TGCA_PVT/dataset/SER30K/Images/09-panda-bear-1/sticker_3.jpg"  # section 3

    # url = "/home/user/xxx/MultiModal/TGCA_PVT/dataset/SER30K/Images/03-panda-bear/sticker_30.jpg"  # a group of six white panda bears
    #url = "/home/user/xxx/MultiModal/TGCA_PVT/dataset/SER30K/Images/03-panda-bear/sticker_27.jpg"#toilet
    # url = "/home/user/xxx/MultiModal/TGCA_PVT/dataset/SER30K/Images/06-panda-bear/sticker_30.jpg"  # bath
    # url = "/home/user/xxx/MultiModal/TGCA_PVT/dataset/SER30K/Images/07-panda-bear/sticker_8.jpg"#hug
    #url = "/home/user/xxx/MultiModal/Emoset/image/awe/awe_00009.jpg"#awe
    image = Image.open(url).convert("RGB")
    # prompt = "What is the information of subject in the picture? What is the facial expression and the body movement of the subject in the picture?"
    prompt = "What is the information of subject in the picture? What is the facial expression and the body movement of the subject in the picture?What is the emotional meaning of the image? Please answer these question with a descriptive sentence."
    # prompt = "What is the information of subject in the picture? What is the facial expression and the body movement of the subject in the picture?What is the emotional meaning of the image? Please answer these question with triples."
    # prompt = "Generate a detailed description for the image.Focus on the emotional meaning in the image."
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        num_beams=5,
        max_new_tokens=50,
        min_new_tokens=30,
        repetition_penalty=5.0,
    )

    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    print(generated_text)


if __name__ == "__main__":
    #example()

    parser = argparse.ArgumentParser(description='Caption a dataset using BLIP2')
    parser.add_argument('--blip_model_url', default='/home/user/xxx/PLM/instructblip-flan-t5-xl',
                        help='BLIP model path')
    parser.add_argument('--caption_dataset', default='EmotionROI', help='dataset name:Emoset,SERCaption,Emo8,EmotionROI')
    parser.add_argument('--mode', default='test', help='dataset mode')
    args = parser.parse_args()
    device = "cuda:6" if torch.cuda.is_available() else "cpu"
    # init dataset
    dataset = None
    preprocess_path = '/home/user/xxx/MultiModal/TGCA_PVT/preprocess'
    if args.caption_dataset == 'SERCaption':
        dataset = SERCaption(preprocess_path=preprocess_path, mode=args.mode)
    elif args.caption_dataset == 'Emoset':
        dataset = Emoset(preprocess_path=preprocess_path, mode=args.mode)
    elif args.caption_dataset == 'Emo8':
        dataset = Emo8(preprocess_path=preprocess_path, mode=args.mode)
    elif args.caption_dataset == 'EmotionROI':
        dataset = EmotionROI(preprocess_path=preprocess_path, mode=args.mode)

    dataset_captions = dataset.get_captions()
    if len(dataset_captions) == 0:
        # init BLIP
        blip = BLIPInterface(args.blip_model_url, dataset, device)
        # start inference
        captions_list = blip(batch_size=16)
        print("Captions get it!")
    else:
        print("Captions had already been processed")
