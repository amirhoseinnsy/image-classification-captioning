import re
from collections import defaultdict, Counter
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
import torch


from PIL import Image
import os

def image_data_loader(image_dir='data/dataset/Images'):
    with open('data/dataset/Flickr_8k.trainImages.txt', 'r') as f:
        train_image_names = [line.strip() for line in f]
    with open('data/dataset/Flickr_8k.testImages.txt', 'r') as f:
        test_image_names = [line.strip() for line in f]
    with open('data/dataset/Flickr_8k.devImages.txt', 'r') as f:
        val_image_names = [line.strip() for line in f]

    train = {}
    val = {}
    test = {}

    for name in train_image_names:
        path = os.path.join(image_dir, name)
        if os.path.exists(path):
            train[name] = Image.open(path).convert('RGB')

    for name in val_image_names:
        path = os.path.join(image_dir, name)
        if os.path.exists(path):
            val[name] = Image.open(path).convert('RGB')

    for name in test_image_names:
        path = os.path.join(image_dir, name)
        if os.path.exists(path):
            test[name] = Image.open(path).convert('RGB')

    return train, val, test


def text_data_loader():
    with open('data/dataset/captions.txt', 'r') as f:
        pic_to_cap_temp = [line.strip().split(",") for line in f]

    pic_to_cap = {}
    for i in pic_to_cap_temp:
        pic = i[0]
        cap = i[1]
        if pic not in pic_to_cap:
            pic_to_cap[pic] = [cap]
        else:
            pic_to_cap[pic].append(cap)

    return pic_to_cap

SPECIALS = ['<pad>', '<unk>', '<start>', '<end>']

def tokenize_words(text):
    return re.findall(r"\w+", text.lower())

def learn_bpe(words, num_merges=1000):
    vocab = defaultdict(int)
    for word in words:
        tokens = ' '.join(list(word)) + ' </w>'
        vocab[tokens] += 1

    merges = []
    for _ in range(num_merges):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[(symbols[i], symbols[i+1])] += freq
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        new_vocab = {}
        bigram = ' '.join(best)
        replacement = ''.join(best)
        for word in vocab:
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = vocab[word]
        vocab = new_vocab
        merges.append(best)
    return merges

def apply_bpe(word, merges):
    word = list(word) + ['</w>']
    i = 0
    while i < len(word) - 1:
        pair = (word[i], word[i+1])
        merged = False
        for merge in merges:
            if pair == merge:
                word[i:i+2] = [''.join(pair)]
                i = max(i - 1, 0)
                merged = True
                break
        if not merged:
            i += 1
    return word

SPECIALS     = ["<unk>", "<pad>", "<start>", "<end>"]
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = range(len(SPECIALS))

def build_tokenizer_and_vocab(captions, top_k_words=5000, bpe_merges=1000):
    all_lines = [line for caps in captions.values() for line in caps]
    all_words = [w for line in all_lines for w in tokenize_words(line)]
    word_freq = Counter(all_words)

    top_words = set(w for w, _ in word_freq.most_common(top_k_words))

    bpe_training_words = [w for w in word_freq if w not in top_words]
    merges = learn_bpe(bpe_training_words, bpe_merges)

    vocab = {tok: idx for idx, tok in enumerate(SPECIALS)}
    next_idx = len(SPECIALS)

    for token in sorted(top_words.union(*[
               apply_bpe(w, merges) for w in bpe_training_words
           ])):
        if token not in vocab:
            vocab[token] = next_idx
            next_idx += 1

    def vocab_fn(tokens):
        return [vocab.get(tok, UNK_IDX) for tok in tokens]

    pic_to_token = {}
    pic_to_ids   = {}

    for img, caps in captions.items():
        pic_to_token[img] = []
        pic_to_ids[img]   = []
        for cap in caps:
            tokens = [ "<start>" ]
            for w in tokenize_words(cap):
                if w in top_words:
                    tokens.append(w)
                else:
                    tokens.extend(apply_bpe(w, merges))
            tokens.append("<end>")

            ids = vocab_fn(tokens)
            pic_to_token[img].append(tokens)
            pic_to_ids[img].append(ids)

    return pic_to_token, pic_to_ids, vocab

def collate_fn(batch, pad_idx):
    images, captions = zip(*batch)
    lengths = [len(cap) for cap in captions]

    caps_padded = pad_sequence(
        [torch.tensor(cap, dtype=torch.long) for cap in captions],
        batch_first=True,
        padding_value=pad_idx
    )

    images_tensor = torch.stack(images)
    return images_tensor, caps_padded 



class Flickr8(Dataset):
    def __init__(self, image_map: dict, caption_map: dict):
        self.image_map = image_map  
        self.samples = []  
        for image_name, caption_list in caption_map.items():
            if image_name in self.image_map:
                for caption in caption_list:
                    self.samples.append((image_name, caption))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
        ])

    def __getitem__(self, index):
        image_name, caption = self.samples[index]
        image = self.transform(self.image_map[image_name])
        caption = torch.tensor(caption, dtype=torch.long)
        return image, caption

    def __len__(self):
        return len(self.samples)
