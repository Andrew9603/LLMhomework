from datasets import load_dataset
from transformers import MarianTokenizer
from torch.utils.data import Dataset
import torch


class IWSLT2017Dataset(Dataset):
    """
    IWSLT2017 数据集加载器，支持 en↔de 翻译任务，使用预训练 tokenizer。

    参数:
        - split: 'train' / 'validation' / 'test'
        - src_lang: 源语言 'en' 或 'de'
        - tgt_lang: 目标语言 'de' 或 'en'
        - tokenizer_name: 预训练 tokenizer 名称，默认使用 Marian en-de
        - max_src_len: 源句子最大长度
        - max_tgt_len: 目标句子最大长度
        - limit: 可选参数，限制数据集大小，便于调试
    """

    def __init__(self, split='train', src_lang='en', tgt_lang='de',
                 tokenizer_name='Helsinki-NLP/opus-mt-en-de',
                 max_src_len=128, max_tgt_len=128, limit=None):
        # 加载数据集
        self.dataset = load_dataset("iwslt2017", f"iwslt2017-{src_lang}-{tgt_lang}", split=split)
        # 加载预训练 tokenizer
        self.tokenizer = MarianTokenizer.from_pretrained(tokenizer_name)
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # 确保 tokenizer 包含特殊 token
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        if self.tokenizer.bos_token is None:
            self.tokenizer.add_special_tokens({'bos_token': '<s>'})
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({'eos_token': '</s>'})

        # 限制数据集大小（调试用）
        if limit is not None:
            self.dataset = self.dataset.select(range(min(limit, len(self.dataset))))

    def __len__(self):
        # 返回数据集大小
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        获取单条样本
        返回字典：
            - src_ids: 源句子 token id
            - src_mask: 源句子 attention mask
            - tgt_input: 解码器输入（右移的目标句子）
            - tgt_labels: 目标句子 token id，用于计算 loss
            - tgt_mask: 目标句子 attention mask
        """
        item = self.dataset[idx]
        src_text = item['translation'][self.src_lang]  # 源句子文本
        tgt_text = item['translation'][self.tgt_lang]  # 目标句子文本

        # 对源句子进行编码
        src_enc = self.tokenizer(src_text,
                                 max_length=self.max_src_len,
                                 truncation=True,
                                 padding='max_length',
                                 return_tensors='pt')

        # 对目标句子进行编码
        tgt_enc = self.tokenizer(tgt_text,
                                 max_length=self.max_tgt_len,
                                 truncation=True,
                                 padding='max_length',
                                 return_tensors='pt')

        # 去掉 batch 维度
        src_ids = src_enc['input_ids'].squeeze(0)  # (max_src_len)
        src_mask = src_enc['attention_mask'].squeeze(0)  # (max_src_len)
        tgt_ids = tgt_enc['input_ids'].squeeze(0)  # (max_tgt_len)
        tgt_mask = tgt_enc['attention_mask'].squeeze(0)

        # 准备解码器输入 (shift right)
        bos = self.tokenizer.bos_token_id
        eos = self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id

        # 创建右移的 tgt_input，第一个 token 为 bos
        tgt_input = torch.zeros_like(tgt_ids)
        tgt_input[0] = bos if bos is not None else tgt_ids[0]
        tgt_input[1:] = tgt_ids[:-1]

        # 返回样本字典
        sample = {
            'src_ids': src_ids,
            'src_mask': src_mask,
            'tgt_input': tgt_input,
            'tgt_labels': tgt_ids,
            'tgt_mask': tgt_mask
        }
        return sample


def collate_fn(batch):
    """
    DataLoader 的 collate_fn，将 list[dict] 转为 batch 张量
    """
    import torch
    src_ids = torch.stack([b['src_ids'] for b in batch])
    src_mask = torch.stack([b['src_mask'] for b in batch])
    tgt_input = torch.stack([b['tgt_input'] for b in batch])
    tgt_labels = torch.stack([b['tgt_labels'] for b in batch])
    tgt_mask = torch.stack([b['tgt_mask'] for b in batch])
    return src_ids, src_mask, tgt_input, tgt_labels, tgt_mask
