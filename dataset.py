# Code based on the Pyramid Vision Transformer
# https://github.com/whai362/PVT
# Licensed under the Apache License, Version 2.0
import json
import os
import re
from os.path import join
import numpy as np
import scipy
from scipy import io
import imageio
from torch.utils.data import Dataset
#import scipy.misc
from PIL import Image
from tqdm import tqdm

from torchvision.transforms import InterpolationMode
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from transformers import BertTokenizer

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from mcloader import ClassificationDataset

label_dict = {
    "excitement": 0,
    "awe": 0,
    "surprise": 0,
    "contentment": 1,
    "amusement": 1,
    "joy": 1,
    "disgust": 2,
    "fear": 3,
    "sadness": 4,
    "sad": 4,
    "anger": 5,
}


## Bert text encoding
class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    # unique_id = 0
    line = input_line  #reader.readline()
    # if not line:
    #     break
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(
        InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    # unique_id += 1
    return examples


def _truncate_seq_pair(tokens_a, tokens_b, param):
    pass


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


class FI:
    def __init__(self, root, mode='train', data_len=None, transform=None):
        self.root = root
        self.mode = mode
        self.nb_classes = 8  # EmotionROI:6  FI:8
        self.transform = transform

        with open(join(r'./other_dataset/FI/emotion_dataset/', fr'{self.mode}3.json'), 'r') as f:
            annos = json.load(f)
        self.annos = annos

        self.imgs = [os.path.join(self.root, anno[0]) for anno in
                     tqdm(self.annos[:data_len])]
        self.labels = [int(anno[1]) for anno in self.annos][:data_len]
        self.imgnames = [anno[0] for anno in self.annos]
        self.query_len = 30
        bert_model = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, index):
        img_path, target, imgname = self.imgs[index], self.labels[index], self.imgnames[index]
        img_path = img_path.replace("\\", "/")
        img = imageio.imread(img_path)
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)
        topic_exam = read_examples(imgname, index)
        topic_fea = convert_examples_to_features(
            examples=topic_exam, seq_length=self.query_len, tokenizer=self.tokenizer)
        topic_id = topic_fea[0].input_ids
        topic_mask = topic_fea[0].input_mask

        return img, target, np.array(topic_id, dtype=int), np.array(topic_mask, dtype=int)


class FI6:
    def __init__(self,  preprocess_path, mode='train', data_len=None, transform=None):
        self.dataset_path = '/home/user/xxx/MultiModal/FI'
        self.preprocess_path = preprocess_path
        self.mode = mode
        self.transform = transform

        self.img_paths = []
        self.captions = []
        self.labels = []
        self.triples = []

        with open(join(r'./other_dataset/FI/emotion_dataset/', fr'{self.mode}3.json'), 'r') as f:
            annos = json.load(f)
        self.annos = annos

        for anno in tqdm(self.annos[:data_len]):
            self.img_paths.append(os.path.join(self.dataset_path, anno[0]))
            self.labels.append(int(anno[1]))

        # load BLIP's captions
        self.captions_file = os.path.join(self.preprocess_path, 'captions/FI_' + self.mode + '_captions.json')
        try:
            with open(self.captions_file, 'r') as f:
                image_file_captions = json.load(f)
                print(f'Initialized {len(image_file_captions)} captions')
                self.captions = image_file_captions
        except FileNotFoundError:
            print(f'Cant initialize captions, cause it is None. Please check if use BLIP inference.')

        # init SRL's triples
        self.triples_file = os.path.join(self.preprocess_path,
                                         'triples/FI_' + self.mode + '_triples.json')
        try:
            with open(self.triples_file, 'r') as f:
                image_triples = json.load(f)
                print(f'Initialized {len(image_triples)} triples')
                self.triples = image_triples
        except FileNotFoundError:
            print(f'Cant initialize triples, cause it is None. Please check if use SRL inference.')

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, index):
        img_path, target = self.img_paths[index], self.labels[index]
        img_path = img_path.replace("\\", "/")
        img = imageio.imread(img_path)
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, target


class FI6Aligned_1(Dataset):
    def __init__(self, preprocess_path, mode='train', data_len=None):
        self.dataset_path = '/home/user/xxx/MultiModal/FI'
        self.preprocess_path = preprocess_path
        self.mode = mode
        self.transform = build_transform_da(self.mode)

        self.img_paths = []
        self.labels = []
        self.idxs = []
        self.captions = []
        to_6labels = {
            '0': 1,
            '1': 5,
            '2': 0,
            '3': 1,
            '4': 2,
            '5': 0,
            '6': 3,
            '7': 4,
        }
        with open(join(r'./other_dataset/FI/emotion_dataset/', fr'{self.mode}3.json'), 'r') as f:
            annos = json.load(f)
        self.annos = annos

        for idx, anno in enumerate(self.annos[:data_len]):
            self.img_paths.append(os.path.join(self.dataset_path, anno[0]))
            self.labels.append(to_6labels[str(anno[1])])
            self.idxs.append(idx)

        # load BLIP's captions
        self.captions_file = os.path.join(self.preprocess_path, 'captions/Emo8_' + self.mode + '_captions.json')
        try:
            with open(self.captions_file, 'r') as f:
                image_file_captions = json.load(f)
                print(f'Initialized {len(image_file_captions)} captions')
                self.captions = image_file_captions
        except FileNotFoundError:
            print(f'Cant initialize captions, cause it is None. Please check if use BLIP inference.')

        print(f'Initialized {len(self.idxs)} Emo8 from Emoset dataset')

    def __getitem__(self, index):
        img_path, target = self.img_paths[index], self.labels[index]
        idx, caption = self.idxs[index], self.captions[index]
        return img_path, target, idx, caption

    def __len__(self):
        return len(self.idxs)


class SER:
    def __init__(self, root, mode='train', data_len=None, transform=None):
        self.root = root
        self.mode = mode
        self.nb_classes = 7
        self.transform = transform

        with open(join(root, 'Annotations', 'image-level', f'{self.mode}.json'), 'r') as f:
            annos = json.load(f)
        self.annos = annos['annotations']

        # self.imgs = [scipy.misc.imread(os.path.join(self.dataset_url, 'Images', anno['topic'], anno['file_name'])) for anno in
        #                     tqdm(self.annos[:data_len])]
        self.imgs = [os.path.join(self.root, 'Images', anno['topic'], anno['file_name']) for anno in
                     tqdm(self.annos[:data_len])]
        self.topic = [anno['topic'] for anno in self.annos]
        self.labels = [int(anno['anno'] - 1) for anno in self.annos][:data_len]
        self.imgnames = [join(anno['topic'], anno['file_name']) for anno in self.annos]
        self.query_len = 30
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    def __getitem__(self, index):
        img_path, target, imgname, topic = self.imgs[index], self.labels[index], self.imgnames[index], self.topic[index]
        img = imageio.imread(img_path)
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)

        ## enocde the topic
        #topic = topic.rstrip('0123456789')
        topic_exam = read_examples(topic, index)
        topic_fea = convert_examples_to_features(
            examples=topic_exam, seq_length=self.query_len, tokenizer=self.tokenizer)
        topic_id = topic_fea[0].input_ids
        topic_mask = topic_fea[0].input_mask

        return img, target, np.array(topic_id, dtype=int), np.array(topic_mask, dtype=int)  # , imgname

    def __len__(self):
        return len(self.annos)


class SER_Full():
    def __init__(self, root, mode='train', data_len=None, transform=None, max_query_len=30,
                 bert_model='/home/user/xxx/PLM/bert-base-uncased'):
        self.root = root
        self.mode = mode
        self.nb_classes = 7
        self.transform = transform
        self.query_len = max_query_len  # 30
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)

        with open(join(root, 'Annotations', 'image-level', f'{self.mode}.json'), 'r') as f:
            annos = json.load(f)
        self.annos = annos['annotations']

        self.imgs = [os.path.join(self.root, 'Images', anno['topic'], anno['file_name']) for anno in
                     tqdm(self.annos[:data_len])]
        self.topic = [anno['topic'] for anno in self.annos]
        self.labels = [int(anno['anno'] - 1) for anno in self.annos][:data_len]
        self.imgnames = [join(anno['topic'], anno['file_name']) for anno in self.annos]
        self.sentences = [anno['text'] for anno in self.annos]

    def __getitem__(self, index):
        # Read Sample
        img_path, target, imgname, sentence, topic = self.imgs[index], self.labels[index], self.imgnames[index], \
            self.sentences[index], self.topic[index]
        img = imageio.imread(img_path)
        # Read Image
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)
        # Read Sentence
        sentence = sentence.lower()
        ## encode sentence to bert input
        examples = read_examples(sentence, index)
        features = convert_examples_to_features(
            examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
        word_id = features[0].input_ids
        word_mask = features[0].input_mask
        #word_split = features[0].tokens[1:-1]

        ## enocde the topic
        #topic = topic.rstrip('0123456789')
        topic_exam = read_examples(topic, index)
        topic_fea = convert_examples_to_features(
            examples=topic_exam, seq_length=self.query_len, tokenizer=self.tokenizer)
        topic_id = topic_fea[0].input_ids
        topic_mask = topic_fea[0].input_mask

        return img, target, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), np.array(topic_id,
                                                                                                   dtype=int), np.array(
            topic_mask, dtype=int)  #, imgname

    def __len__(self):
        return len(self.annos)


class SERCaption(Dataset):
    def __init__(self, preprocess_path, mode='train', data_len=None, transform=None):
        self.dataset_path = '/home/user/xxx/MultiModal/TGCA_PVT/dataset/SER30K'
        self.preprocess_path = preprocess_path
        self.mode = mode
        self.transform = transform

        self.img_paths = []
        self.captions = []
        self.labels = []
        self.triples = []

        # load annotations
        with open(join(self.dataset_path, 'Annotations', 'image-level', f'{self.mode}.json'), 'r') as f:
            annos = json.load(f)
        self.annos = annos['annotations']

        for anno in tqdm(self.annos[:data_len]):
            self.img_paths.append(os.path.join(self.dataset_path, 'Images', anno['topic'], anno['file_name']))
            self.labels.append(int(anno['anno'] - 1))

        # init BLIP's captions
        self.captions_file = os.path.join(self.preprocess_path, 'captions/SERCaption_' + self.mode + '_captions.json')
        try:
            with open(self.captions_file, 'r') as f:
                image_captions = json.load(f)
                print(f'Initialized {len(image_captions)} captions')
                self.captions = image_captions
        except FileNotFoundError:
            print(f'Cant initialize captions, cause it is None. Please check if use BLIP inference.')

        # init SRL's triples
        self.triples_file = os.path.join(self.preprocess_path,
                                         'triples/SERCaption_' + self.mode + '_triples.json')
        try:
            with open(self.triples_file, 'r') as f:
                image_triples = json.load(f)
                print(f'Initialized {len(image_triples)} triples')
                self.triples = image_triples
        except FileNotFoundError:
            print(f'Cant initialize triples, cause it is None. Please check if use SRL inference.')

    def __getitem__(self, index):
        img_path, target = self.img_paths[index], self.labels[index]
        img = imageio.imread(img_path)
        caption = self.captions[index]
        triple = self.triples[index]
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, target, caption, triple

    def __len__(self):
        return len(self.annos)

    def get_image(self, idx, return_img_path=False):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")

        ret_list = [image]

        if return_img_path:
            ret_list.append(img_path)

        return ret_list

    def get_mode(self):
        return self.mode

    def get_img_paths(self):
        return self.img_paths

    def get_captions(self):
        return self.captions

    def get_triples(self):
        return self.triples

    def get_labels(self):
        return self.labels


class SERCaptionAligned(Dataset):
    def __init__(self, root, mode='train', data_len=None, transform=None, max_query_len=30,
                 bert_model='/home/user/xxx/PLM/bert-base-uncased'):
        self.root = root
        self.mode = mode
        self.nb_classes = 6
        self.transform = transform
        self.query_len = max_query_len  # 30
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)

        with open(join(root, 'Annotations', 'image-level', f'{self.mode}.json'), 'r') as f:
            annos = json.load(f)
        self.annos = annos['annotations']

        self.imgs = [os.path.join(self.root, 'Images', anno['topic'], anno['file_name']) for anno in
                     tqdm(self.annos[:data_len])]
        self.topic = [anno['topic'] for anno in self.annos]
        self.labels = [int(anno['anno'] - 1) for anno in self.annos][:data_len]
        self.imgnames = [join(anno['topic'], anno['file_name']) for anno in self.annos]
        self.sentences = [anno['text'] for anno in self.annos]

        self.imgs_filter = []
        self.labels_filter = []
        self.imgnames_filter = []
        self.sentences_filter = []
        self.topic_filter = []

        for i, label in enumerate(self.labels):
            if label == 6:
                continue
            self.imgs_filter.append(self.imgs[i])
            self.labels_filter.append(label)
            self.imgnames_filter.append(self.imgnames[i])
            self.sentences_filter.append(self.sentences[i])
            self.topic_filter.append(self.topic[i])

    def __getitem__(self, index):
        # Read Sample
        img_path, target, imgname, sentence, topic = self.imgs_filter[index], self.labels_filter[index], self.imgnames[index], \
            self.sentences_filter[index], self.topic_filter[index]
        img = imageio.imread(img_path)
        # Read Image
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)
        # Read Sentence
        sentence = sentence.lower()
        ## encode sentence to bert input
        examples = read_examples(sentence, index)
        features = convert_examples_to_features(
            examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
        word_id = features[0].input_ids
        word_mask = features[0].input_mask
        # word_split = features[0].tokens[1:-1]

        ## enocde the topic
        # topic = topic.rstrip('0123456789')
        topic_exam = read_examples(topic, index)
        topic_fea = convert_examples_to_features(
            examples=topic_exam, seq_length=self.query_len, tokenizer=self.tokenizer)
        topic_id = topic_fea[0].input_ids
        topic_mask = topic_fea[0].input_mask

        return img, target, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), np.array(topic_id,
                                                                                                   dtype=int), np.array(
            topic_mask, dtype=int)  # , imgname

    def __len__(self):
        return len(self.labels_filter)


class SERCaptionAligned_1(Dataset):
    def __init__(self, preprocess_path, mode='train', data_len=None):
        self.dataset_path = '/home/user/xxx/MultiModal/TGCA_PVT/dataset/SER30K'
        self.preprocess_path = preprocess_path
        self.mode = mode

        self.img_paths = []
        self.labels = []
        self.captions = []
        self.idxs = []
        self.img_paths_filter = []
        self.labels_filter = []
        self.captions_filter = []

        # load annotations
        with open(join(self.dataset_path, 'Annotations', 'image-level', f'{self.mode}.json'), 'r') as f:
            annos = json.load(f)
        self.annos = annos['annotations']
        for anno in self.annos[:data_len]:
            self.img_paths.append(os.path.join(self.dataset_path, 'Images', anno['topic'], anno['file_name']))
            self.labels.append(int(anno['anno'] - 1))

        # load BLIP's captions
        self.captions_file = os.path.join(self.preprocess_path, 'captions/SERCaption_' + self.mode + '_captions.json')
        try:
            with open(self.captions_file, 'r') as f:
                image_file_captions = json.load(f)
                print(f'Initialized {len(image_file_captions)} captions')
                self.captions = image_file_captions
        except FileNotFoundError:
            print(f'Cant initialize captions, cause it is None. Please check if use BLIP inference.')

        for i, label in enumerate(self.labels):
            if label == 6:
                continue
            self.img_paths_filter.append(self.img_paths[i])
            self.labels_filter.append(label)
            self.captions_filter.append(self.captions[i])
            self.idxs.append(i)
        print(f'Initialized {len(self.idxs)} samples from SER30K dataset')

    def __getitem__(self, index):
        img_path, target = self.img_paths_filter[index], self.labels_filter[index]
        idx, caption = self.idxs[index], self.captions_filter[index]
        return img_path, target, idx, caption

    def __len__(self):
        return len(self.idxs)


class Emoset(Dataset):
    def __init__(self, preprocess_path, mode='train', data_len=None, transform=None):
        self.dataset_path = '/home/user/xxx/MultiModal/Emoset'
        self.preprocess_path = preprocess_path
        self.mode = mode
        self.transform = transform

        self.img_paths = []
        self.captions = []
        self.labels = []
        self.triples = []

        # load annotations
        with open(join(self.dataset_path, f'{self.mode}.json'), 'r') as f:
            annos = json.load(f)
        self.annos = annos

        for anno in tqdm(self.annos[:data_len]):
            self.img_paths.append(os.path.join(self.dataset_path, anno[1]))
            self.labels.append(anno[0])

        # load BLIP's captions
        self.captions_file = os.path.join(self.preprocess_path, 'captions/Emoset_' + self.mode + '_captions.json')
        try:
            with open(self.captions_file, 'r') as f:
                image_file_captions = json.load(f)
                print(f'Initialized {len(image_file_captions)} captions')
                self.captions = image_file_captions
        except FileNotFoundError:
            print(f'Cant initialize captions, cause it is None. Please check if use BLIP inference.')

        # init SRL's triples
        self.triples_file = os.path.join(self.preprocess_path,
                                         'triples/Emoset_' + self.mode + '_triples.json')
        try:
            with open(self.triples_file, 'r') as f:
                image_triples = json.load(f)
                print(f'Initialized {len(image_triples)} triples')
                self.triples = image_triples
        except FileNotFoundError:
            print(f'Cant initialize triples, cause it is None. Please check if use SRL inference.')

        #

    def __getitem__(self, index):
        img_path, target = self.img_paths[index], self.labels[index]
        img = imageio.imread(img_path)
        caption = self.captions[index]
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, target, caption

    def __len__(self):
        return len(self.annos)

    def get_image(self, idx, return_img_path=False):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")

        ret_list = [image]

        if return_img_path:
            ret_list.append(img_path)

        return ret_list

    def get_mode(self):
        return self.mode

    def get_img_paths(self):
        return self.img_paths

    def get_captions(self):
        return self.captions

    def get_triples(self):
        return self.triples

    def get_labels(self):
        return self.labels


class EmosetAligned(Dataset):
    def __init__(self, root, mode='train', data_len=None, transform=None, max_query_len=30,
                 bert_model='/home/user/xxx/PLM/bert-base-uncased'):
        self.root = root
        self.mode = mode
        self.nb_classes = 6
        self.transform = transform
        self.query_len = max_query_len  # 30
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.imgs = []
        self.labels = []
        self.imgnames = []
        self.sentences = []
        self.topic = []

        # load annotations
        with open(join(self.root, f'{self.mode}.json'), 'r') as f:
            annos = json.load(f)
        self.annos = annos

        for anno in tqdm(self.annos[:data_len]):
            self.imgs.append(os.path.join(self.root, anno[1]))
            self.imgnames.append(anno[1])
            self.labels.append(int(label_dict[anno[0]]))
            with open(join(self.root, anno[2]), 'r') as f:
                js = json.load(f)
                if 'object' in js:
                    self.sentences.append(js['object'][0])
                    self.topic.append(js['object'][0])
                else:
                    self.sentences.append('')
                    self.topic.append('')

    def __getitem__(self, index):
        # Read Sample
        img_path, target, imgname, sentence, topic = self.imgs[index], self.labels[index], self.imgnames[
            index], \
            self.sentences[index], self.topic[index]
        img = imageio.imread(img_path)
        # Read Image
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)
        # Read Sentence
        sentence = sentence.lower()
        ## encode sentence to bert input
        examples = read_examples(sentence, index)
        features = convert_examples_to_features(
            examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
        word_id = features[0].input_ids
        word_mask = features[0].input_mask
        # word_split = features[0].tokens[1:-1]

        ## enocde the topic
        # topic = topic.rstrip('0123456789')
        topic_exam = read_examples(topic, index)
        topic_fea = convert_examples_to_features(
            examples=topic_exam, seq_length=self.query_len, tokenizer=self.tokenizer)
        topic_id = topic_fea[0].input_ids
        topic_mask = topic_fea[0].input_mask

        return img, target, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), np.array(topic_id,
                                                                                                   dtype=int), np.array(
            topic_mask, dtype=int)  # , imgname

    def __len__(self):
        return len(self.labels)


class EmosetAligned_1(Dataset):
    def __init__(self, preprocess_path, mode='train', data_len=None):
        self.dataset_path = '/home/user/xxx/MultiModal/Emoset'
        self.preprocess_path = preprocess_path
        self.mode = mode
        self.transform = build_transform_da(self.mode)

        self.img_paths = []
        self.labels = []
        self.idxs = []
        self.captions = []

        # load annotations
        with open(join(self.dataset_path, f'{self.mode}.json'), 'r') as f:
            annos = json.load(f)
        self.annos = annos

        for i, anno in enumerate(self.annos[:data_len]):
            self.img_paths.append(os.path.join(self.dataset_path, anno[1]))
            self.labels.append(int(label_dict[anno[0]]))
            self.idxs.append(i)

        # load BLIP's captions
        self.captions_file = os.path.join(self.preprocess_path, 'captions/Emoset_' + self.mode + '_captions.json')
        try:
            with open(self.captions_file, 'r') as f:
                image_file_captions = json.load(f)
                print(f'Initialized {len(image_file_captions)} captions')
                self.captions = image_file_captions
        except FileNotFoundError:
            print(f'Cant initialize captions, cause it is None. Please check if use BLIP inference.')

        print(f'Initialized {len(self.idxs)} samples from Emoset dataset')

    def __getitem__(self, index):
        img_path, target = self.img_paths[index], self.labels[index]
        idx, caption = self.idxs[index], self.captions[index]
        return img_path, target, idx, caption

    def __len__(self):
        return len(self.idxs)


class Emo8(Dataset):
    def __init__(self, preprocess_path, mode='train', data_len=None, transform=None):
        self.dataset_path = '/home/user/xxx/MultiModal/Emo8'
        self.preprocess_path = preprocess_path
        self.mode = mode
        self.transform = transform

        self.img_paths = []
        self.captions = []
        self.labels = []
        self.triples = []

        for idx, class_name in enumerate(os.listdir(os.path.join(self.dataset_path, self.mode))):
            class_dir = os.path.join(self.dataset_path, self.mode, class_name)
            if os.path.isdir(class_dir):
                y = label_dict[class_name]
                # 遍历每个类别文件夹下的图片
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    self.img_paths.append(img_path)
                    self.labels.append(y)

        # load BLIP's captions
        self.captions_file = os.path.join(self.preprocess_path, 'captions/Emo8_' + self.mode + '_captions.json')
        try:
            with open(self.captions_file, 'r') as f:
                image_file_captions = json.load(f)
                print(f'Initialized {len(image_file_captions)} captions')
                self.captions = image_file_captions
        except FileNotFoundError:
            print(f'Cant initialize captions, cause it is None. Please check if use BLIP inference.')

        # init SRL's triples
        self.triples_file = os.path.join(self.preprocess_path,
                                         'triples/Emo8_' + self.mode + '_triples.json')
        try:
            with open(self.triples_file, 'r') as f:
                image_triples = json.load(f)
                print(f'Initialized {len(image_triples)} triples')
                self.triples = image_triples
        except FileNotFoundError:
            print(f'Cant initialize triples, cause it is None. Please check if use SRL inference.')

    def __getitem__(self, index):
        img_path, target = self.img_paths[index], self.labels[index]
        img = imageio.imread(img_path)
        caption = self.captions[index]
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, target, caption

    def __len__(self):
        return len(self.labels)

    def get_image(self, idx, return_img_path=False):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")

        ret_list = [image]

        if return_img_path:
            ret_list.append(img_path)

        return ret_list

    def get_mode(self):
        return self.mode

    def get_img_paths(self):
        return self.img_paths

    def get_captions(self):
        return self.captions

    def get_triples(self):
        return self.triples

    def get_labels(self):
        return self.labels


class Emo8Aligned_1(Dataset):
    def __init__(self, preprocess_path, mode='train', data_len=None):
        self.dataset_path = '/home/user/xxx/MultiModal/Emo8'
        self.preprocess_path = preprocess_path
        self.mode = mode
        self.transform = build_transform_da(self.mode)

        self.img_paths = []
        self.labels = []
        self.idxs = []
        self.captions = []

        idx = 0
        for idx, class_name in enumerate(os.listdir(os.path.join(self.dataset_path, self.mode))):
            class_dir = os.path.join(self.dataset_path, self.mode, class_name)
            if os.path.isdir(class_dir):
                y = label_dict[class_name]
                # 遍历每个类别文件夹下的图片
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    self.img_paths.append(img_path)
                    self.labels.append(y)
                    self.idxs.append(idx)
                    idx += 1

        # load BLIP's captions
        self.captions_file = os.path.join(self.preprocess_path, 'captions/Emo8_' + self.mode + '_captions.json')
        try:
            with open(self.captions_file, 'r') as f:
                image_file_captions = json.load(f)
                print(f'Initialized {len(image_file_captions)} captions')
                self.captions = image_file_captions
        except FileNotFoundError:
            print(f'Cant initialize captions, cause it is None. Please check if use BLIP inference.')

        print(f'Initialized {len(self.idxs)} Emo8 from Emoset dataset')

    def __getitem__(self, index):
        img_path, target = self.img_paths[index], self.labels[index]
        idx, caption = self.idxs[index], self.captions[index]
        return img_path, target, idx, caption

    def __len__(self):
        return len(self.idxs)


class EmotionROI(Dataset):
    def __init__(self, preprocess_path, mode='train', data_len=None, transform=None):
        self.dataset_path = '/home/user/xxx/MultiModal/EmotionROI'
        self.preprocess_path = preprocess_path
        self.mode = mode
        self.transform = transform

        self.img_paths = []
        self.captions = []
        self.labels = []
        self.triples = []

        with open(join(self.dataset_path, 'training_testing_split', f'{self.mode}ing.txt'), 'r') as f:
            for line in f:
                class_name, img_name = line.strip().split('/')
                img_path = os.path.join(self.dataset_path, 'images', class_name, img_name)
                self.img_paths.append(img_path)
                self.labels.append(label_dict[class_name])

        # load BLIP's captions
        self.captions_file = os.path.join(self.preprocess_path, 'captions/EmotionROI_' + self.mode + '_captions.json')
        try:
            with open(self.captions_file, 'r') as f:
                image_file_captions = json.load(f)
                print(f'Initialized {len(image_file_captions)} captions')
                self.captions = image_file_captions
        except FileNotFoundError:
            print(f'Cant initialize captions, cause it is None. Please check if use BLIP inference.')

        # init SRL's triples
        self.triples_file = os.path.join(self.preprocess_path,
                                         'triples/EmotionROI_' + self.mode + '_triples.json')
        try:
            with open(self.triples_file, 'r') as f:
                image_triples = json.load(f)
                print(f'Initialized {len(image_triples)} triples')
                self.triples = image_triples
        except FileNotFoundError:
            print(f'Cant initialize triples, cause it is None. Please check if use SRL inference.')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path, target = self.img_paths[index], self.labels[index]
        img = imageio.imread(img_path)
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, target  # , imgname

    def get_image(self, idx, return_img_path=False):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")

        ret_list = [image]

        if return_img_path:
            ret_list.append(img_path)

        return ret_list

    def get_mode(self):
        return self.mode

    def get_img_paths(self):
        return self.img_paths

    def get_captions(self):
        return self.captions

    def get_triples(self):
        return self.triples

    def get_labels(self):
        return self.labels


class EmotionROIAligned_1(Dataset):
    def __init__(self, preprocess_path, mode='train', data_len=None):
        self.dataset_path = '/home/user/xxx/MultiModal/EmotionROI'
        self.preprocess_path = preprocess_path
        self.mode = mode
        self.transform = build_transform_da(self.mode)

        self.img_paths = []
        self.labels = []
        self.idxs = []
        self.captions = []

        idx = 0
        with open(join(self.dataset_path, 'training_testing_split', f'{self.mode}ing.txt'), 'r') as f:
            for line in f:
                class_name, img_name = line.strip().split('/')
                img_path = os.path.join(self.dataset_path, 'images', class_name, img_name)
                self.img_paths.append(img_path)
                self.labels.append(label_dict[class_name])
                self.idxs.append(idx)
                idx += 1

        # load BLIP's captions
        self.captions_file = os.path.join(self.preprocess_path, 'captions/EmotionROI_' + self.mode + '_captions.json')
        try:
            with open(self.captions_file, 'r') as f:
                image_file_captions = json.load(f)
                print(f'Initialized {len(image_file_captions)} captions')
                self.captions = image_file_captions
        except FileNotFoundError:
            print(f'Cant initialize captions, cause it is None. Please check if use BLIP inference.')

        print(f'Initialized {len(self.idxs)} Emo8 from Emoset dataset')

    def __getitem__(self, index):
        img_path, target = self.img_paths[index], self.labels[index]
        idx, caption = self.idxs[index], self.captions[index]
        return img_path, target, idx, caption

    def __len__(self):
        return len(self.idxs)



def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    if args.dataset == 'SER':
        dataset = SER_Full(root=args.data_path, mode='train' if is_train else 'test', transform=transform)
    elif args.dataset == 'SER_V':
        dataset = SER(root=args.data_path, mode='train' if is_train else 'test', transform=transform)
    elif args.dataset == 'FI':
        dataset = FI(root=args.data_path, mode='train' if is_train else 'test', transform=transform)
    elif args.dataset == 'EmotionROI':
        dataset = EmotionROI(root=args.data_path, mode='train' if is_train else 'test', transform=transform)
    elif args.dataset == 'SER_6':
        dataset = SERCaptionAligned(root='/home/user/xxx/MultiModal/TGCA_PVT/dataset/SER30K', mode='train' if is_train else 'test', transform=transform)
    elif args.dataset == 'Emoset':
        dataset = EmosetAligned(root='/home/user/xxx/MultiModal/Emoset', mode='train' if is_train else 'test', transform=transform,data_len=50000)
    nb_classes = dataset.nb_classes

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
            # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def build_transform_da(mode):
    if mode == 'train':
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=512,
            is_training=True,
            color_jitter=0.1,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0,
            re_mode='pixel',
            re_count=1,
        )
        return transform
    else:
        t = [transforms.Resize((512, 512), interpolation=InterpolationMode.BICUBIC),
             transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)]
        return transforms.Compose(t)


