# テキストをトークンに変更するクラス
import re
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab # encodeメソッドとdecodeメソッドでアクセスできるように語彙をクラス属性として格納
        self.int_to_str = {i:s for s,i in vocab.items()} # トークンIDを基のテキストトークンにマッピングする逆引き語彙を作成

    def encode(self, text): # 入力テキストをトークンIDに変換
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [# 道の単語を<|unk|>トークンに置き換える
            item if item in self.str_to_int
            else "<|unk|>" for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids): # トークンをテキストに変換
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) # 指定された句読点の前にあるスペースを削除
        return text