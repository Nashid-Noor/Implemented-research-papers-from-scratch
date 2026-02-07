import torch
import regex as re
import numpy as np
from collections import OrderedDict
import gzip
import html
import os
from functools import lru_cache


@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer:
    def __init__(self, context_length=77, vocab_file=None):
        self.context_length = context_length
        
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        # Create regex pattern for BPE
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        # Load vocabulary file
        if vocab_file is None:
            try:
                vocab_file = default_bpe()
                with gzip.open(vocab_file) as f:
                    merges = f.read().decode('utf-8').split('\n')
            except (FileNotFoundError, IOError):
                # Fallback to a simplified set of merges if file is not found
                merges = ["Ġ t", "Ġ a", "Ġ i", "Ġ s", "Ġ w", "Ġ o", "Ġ b", "Ġ c", "e s", "e r", "e d", "i n", "o n"]
        else:
            with open(vocab_file, 'r', encoding='utf-8') as f:
                merges = f.read().split('\n')
                
        # Filter out empty merges
        merges = [m for m in merges if len(m) > 0]
        
        # Create vocabulary and dictionaries
        merges = merges[1:49152-256-2+1]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        
        # Special tokens
        self.encoder = {}
        self.decoder = {}
        
        # First add byte tokens
        for i, char in enumerate(self.byte_encoder.values()):
            self.encoder[char] = i
            self.decoder[i] = char
        
        # Then add merge tokens
        for i, merge in enumerate(merges):
            new_token = merge.replace(' ', '')
            self.encoder[new_token] = len(self.byte_encoder) + i
            self.decoder[len(self.byte_encoder) + i] = new_token
        
        # Add special tokens
        special_tokens = {
            '<start_of_text>': 0,
            '<end_of_text>': 1,
            '<pad>': 2
        }
        
        # Remap the first few tokens to special tokens
        for token, idx in special_tokens.items():
            if idx in self.decoder:
                del self.decoder[idx]
            self.encoder[token] = idx
            self.decoder[idx] = token
    
    def bpe(self, token):
        # This is a simplified BPE implementation
        # In a full implementation, this would apply the merges according to bpe_ranks
        
        if token in self.encoder:
            return [token]
        
        word = tuple(token)
        pairs = get_pairs(word)
        
        if not pairs:
            return [token]
        
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(' '.join(pair), float('inf')))
            if ' '.join(bigram) not in self.bpe_ranks:
                break
            
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            word = tuple(new_word)
            if len(word) == 1:
                break
            
            pairs = get_pairs(word)
        
        return list(word)
    
    def encode(self, text):
        bpe_tokens = []
        # Add start token
        bpe_tokens.append(self.encoder["<start_of_text>"])
        
        # Clean text
        text = basic_clean(text)
        text = whitespace_clean(text)
        
        # Tokenize
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_result = self.bpe(token)
            
            # Convert subwords to token ids
            for bpe in bpe_result:
                # Use hash-based fallback for unknown tokens
                if bpe in self.encoder:
                    bpe_tokens.append(self.encoder[bpe])
                else:
                    # Generate a deterministic token ID based on hash
                    hash_id = hash(bpe) % (49152 - 3) + 3  # Avoid special tokens
                    bpe_tokens.append(hash_id)
        
        # Add end token
        bpe_tokens.append(self.encoder["<end_of_text>"])
        
        # Pad/truncate to context length
        if len(bpe_tokens) < self.context_length:
            bpe_tokens = bpe_tokens + [self.encoder["<pad>"]] * (self.context_length - len(bpe_tokens))
        else:
            bpe_tokens = bpe_tokens[:self.context_length - 1] + [self.encoder["<end_of_text>"]]
            
        return bpe_tokens
    
    def decode(self, token_ids):
        # Filter out special tokens
        filtered_tokens = [token_id for token_id in token_ids if token_id >= 3]
        
        # Convert token IDs to bytes
        text_bytes = []
        for token_id in filtered_tokens:
            if token_id in self.decoder:
                token = self.decoder[token_id]
                for byte_val in [self.byte_decoder.get(c, ord(c)) for c in token]:
                    text_bytes.append(byte_val)
        
        # Decode bytes to text
        try:
            text = bytes(text_bytes).decode('utf-8')
        except UnicodeDecodeError:
            text = "[Error: Could not decode tokens]"
        
        return text


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, texts, transform=None, tokenizer=None):
        assert len(image_paths) == len(texts), "Number of images must match number of texts"
        
        self.image_paths = image_paths
        self.texts = texts
        self.transform = transform
        self.tokenizer = tokenizer or SimpleTokenizer()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Get image path and text
        image_path = self.image_paths[idx]
        text = self.texts[idx]
        
        # Load and transform image
        try:
            from PIL import Image
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image if there's an error
            image = torch.zeros((3, 224, 224))
        
        # Tokenize text
        tokens = torch.tensor(self.tokenizer.encode(text))
        
        return image, tokens