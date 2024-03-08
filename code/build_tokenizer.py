from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset
from pathlib import Path

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]
'''
def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency = 2)
        tokenizer.train_from_iterator(get_all_sentences(ds ,lang), trainer= trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer
'''
def train_tokenizer(config):
    ds_raw = load_dataset("adilgupta/cfilt-iitb-en-hi-truncated", split= 'train')
    def fit(config, ds, lang):
        tokenizer_path = Path(config['tokenizer_file'].format(lang))
        if not Path.exists(tokenizer_path):
            tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency = 2)
            print(f'building tokenizer_{lang}....')
            tokenizer.train_from_iterator(get_all_sentences(ds ,lang), trainer= trainer)
            tokenizer.save(str(tokenizer_path))
        else:
            print(f'tokenizer_{lang} already built, so not training a new tokenizer')
    fit(config, ds_raw, config['lang_src'])
    fit(config, ds_raw, config['lang_tgt'])

def get_tokenizer(config, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        raise ImportError(f'cannot find tokenizer_{lang}')
    else:
        print(f'loading tokenizer_{lang}....')
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer
