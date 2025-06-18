import argparse
import os
import json
import sys

from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction
from tqdm import tqdm
sys.path.append("..")
from dataset import SERCaption, Emoset, Emo8, EmotionROI


class SRLInterface:
    def __init__(self, predictor_path, dataset):
        self.predictor = Predictor.from_path(predictor_path)
        self.dataset = dataset

        self.caption_dataset_name = self.dataset.__class__.__name__ + '_' + self.dataset.get_mode()
        self.out_path = "triples/"
        self.triples_file = os.path.join(self.out_path, self.caption_dataset_name + '_triples.json')

    def __call__(self):
        # load image captions
        captions = self.dataset.get_captions()
        if len(captions) == 0:
            raise ValueError("No captions found in the dataset. Please check the input data.")
        # init
        triples = []
        for i in tqdm(range(len(captions))):
            predictions = self.predictor.predict(sentence=captions[i])
            triple_set = []
            words = predictions['words']
            for verb in predictions['verbs']:
                triple = extract_triple(verb['tags'], words)
                triple_set.append(triple) if triple is not None else None

            triples.append(triple_set)

        with open(self.triples_file, 'w') as f:
            json.dump(triples, f, indent=4)
        return triples


def example():
    print("openie")
    # 加载预训练的 SRL 模型
    predictor = Predictor.from_path("/home/user/xxx/PLM/openie-model.2020.03.26.tar.gz")

    # 输入文本
    #text = "chicken pointing at the camera with his head tilted to the side and his body motioning like he's making a wish."
    text = "rc logo and information of subject in the picture. facial expression and body movement of subject in the picture is excited. emotional meaning of the image is happy"

    # 使用 OpenIE 进行关系抽取
    predictions = predictor.predict(sentence=text)
    triple_set = []
    words = predictions['words']
    # 打印提取的三元组结果
    for verb in predictions['verbs']:
        print(f"Verb: {verb['verb']}")
        print("Description:", verb['description'])
        print("--------")
        triple = extract_triple(verb['tags'], words)

        triple_set.append(triple) if triple is not None else None

    # print final triple_set
    for triple in triple_set:
        print(f"Subject: {triple['subject']}")
        print(f"Verb: {triple['verb']}")
        print(f"Object: {triple['object']}")
        print("-" * 20)
    return None


def extract_triple(tags, words):
    triple = {
        'subject': "",
        'verb': "",
        'object': ""
    }
    argm_word = None
    for i, tag in enumerate(tags):
        if tag == 'B-ARG0':
            triple['subject'] += words[i]
            # load I-ARG0
            for j in range(i + 1, len(tags)):
                if tags[j] == 'I-ARG0':
                    triple['subject'] += ' ' + words[j]
                else:
                    break
        elif tag == 'B-V':
            triple['verb'] += words[i]
            # load I-V
            for j in range(i + 1, len(tags)):
                if tags[j] == 'I-V':
                    triple['verb'] += ' ' + words[j]
                else:
                    break
        elif tag == 'B-ARG1':
            if triple['subject'] == '':
                triple['subject'] += words[i]
                # load I-ARG1
                for j in range(i + 1, len(tags)):
                    if tags[j] == 'I-ARG1':
                        triple['subject'] += ' ' + words[j]
                    else:
                        break
            else:
                triple['object'] += words[i]
                # load I-ARG1
                for j in range(i + 1, len(tags)):
                    if tags[j] == 'I-ARG1':
                        triple['object'] += ' ' + words[j]
                    else:
                        break
        elif tag == 'B-ARG2':
            triple['object'] += words[i]
            # load I-ARG1 or I-ARG2
            for j in range(i + 1, len(tags)):
                if tags[j] == 'I-ARG2':
                    triple['object'] += ' ' + words[j]
                else:
                    break
        elif tag.startswith('B-ARGM'):
            if argm_word is None:  # 只取第一个 B-ARGM
                argm_word = words[i]
                for j in range(i + 1, len(tags)):
                    if tags[j].startswith('I-ARGM'):
                        argm_word += ' ' + words[j]
                    else:
                        break
            else:
                pass

    # cheak triple, add argm words if it didn't have a object
    if not triple['object'] and argm_word:
        triple['object'] = argm_word

    if triple['subject'] and triple['verb'] and triple['object']:
        return triple
    else:
        return None


if __name__ == "__main__":
    #example()

    parser = argparse.ArgumentParser(description='Generate triples for a dataset using SRL model')
    parser.add_argument('--srl_model_url', default='/home/user/xxx/PLM/openie-model.2020.03.26.tar.gz',
                        help='SRL model path')
    parser.add_argument('--dataset', default='EmotionROI', help='dataset name:[Emoset,SERCaption,Emo8,EmotionROI]')
    parser.add_argument('--mode', default='test', help='dataset mode')
    args = parser.parse_args()

    # init dataset
    dataset = None
    preprocess_path = '/home/user/xxx/MultiModal/TGCA_PVT/preprocess'
    if args.dataset == 'SERCaption':
        dataset = SERCaption(preprocess_path=preprocess_path, mode=args.mode)
    elif args.dataset == 'Emoset':
        dataset = Emoset(preprocess_path=preprocess_path, mode=args.mode)
    elif args.dataset == 'Emo8':
        dataset = Emo8(preprocess_path=preprocess_path, mode=args.mode)
    elif args.dataset == 'EmotionROI':
        dataset = EmotionROI(preprocess_path=preprocess_path, mode=args.mode)

    dataset_triples = dataset.get_triples()
    if len(dataset_triples) == 0:
        # init BLIP
        srl = SRLInterface(args.srl_model_url, dataset)
        # start inference
        triples_list = srl()
        print("Triples get it!")
    else:
        print("Triples had already been processed")
