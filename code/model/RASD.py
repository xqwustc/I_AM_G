import os.path

import torch
from torch import nn
import torch.nn.functional as F
import sys
sys.path.append('../proj/personalized')
sys.path.append('../proj/personalized/model')
from conf import *
import json
from PIL import Image
import matplotlib.pyplot as plt
from model.data_watcher import *
import math
# from model.CLIP4embeddings import TEXT2CLIP


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        if isinstance(x,list):
            return self.fn(self.norm(x[0]),self.norm(x[1]),**kwargs)
        else:
            return self.fn(self.norm(x), self.norm(input_keys), self.norm(input_values))

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads, dim_head):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.head_dim = num_heads * dim_head

        # Query, Key, Value projections for the input sequences
        self.query = nn.Linear(input_dim, self.head_dim)
        self.key = nn.Linear(input_dim, self.head_dim)
        self.value = nn.Linear(input_dim, self.head_dim)

        # Output projection
        self.out = nn.Linear(self.head_dim, input_dim)

        self.apply(self.initialize_weights)

    def initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            print(f"Initial weights of {module} are set to Xavier Uniform")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, input_queries, input_keys = None, input_values = None):
        if input_keys == None:
            input_keys = input_queries
        if input_values == None:
            input_values = input_keys
        # Project input sequences into query, key, and value vectors
        queries = self.query(input_queries)
        keys = self.key(input_keys)
        values = self.value(input_values)

        # Reshape queries, keys, and values to split heads
        batch_size = queries.shape[0]
        queries = queries.view(batch_size, -1, self.num_heads, self.dim_head)
        keys = keys.view(batch_size, -1, self.num_heads, self.dim_head)
        values = values.view(batch_size, -1, self.num_heads, self.dim_head)

        # Compute attention scores
        attention_scores = torch.einsum('bqhd,bkhd->bhqk', queries, keys)
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.dim_head, dtype=torch.float32))

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Weighted sum of values
        weighted_values = torch.einsum('bhqk,bkhd->bqhd', attention_weights, values)

        # Merge heads
        weighted_values = weighted_values.contiguous().view(batch_size, -1, self.head_dim)

        # Output projection
        outputs = self.out(weighted_values)

        return outputs


class RASD(torch.nn.Module):
    def __init__(self,
                 linear_proj,
                 dataset = 'movielens', # 'movielens', 'POG', 'MIND'
                 embedding_dim=768,
                 num_heads=4,
                 head_dim=192,
                ):
        super(RASD, self).__init__()

        self.linear_proj = linear_proj
        self.data_path = os.path.join(PROJ_PATH, 'data', dataset)


        self.user_history, self.item_set = self.get_user_history(dataset)

        self.attention = PreNorm(
            dim=embedding_dim, fn=MultiHeadAttention(embedding_dim, num_heads, head_dim)
        ).to('cuda')


        self.text_keywords = self.get_keywords(dataset)

        self.text_origin = self.get_texts(dataset)

        self.text_embedding, self.text_names = \
            self.get_text_kr_embedding(self.data_path)

        self.image_embedding, self.image_names = \
            self.get_image_kr_embedding(self.data_path)

    def get_texts(self, dataset_name):
        keywords_path = "text.txt"

        text_origin = {}
        with open(keywords_path, 'r') as f:
            for line in f:
                parts = line.strip().split(':', 1)
                assert len(parts) == 2, "Not complete description!"
                key = parts[0].strip()
                value = parts[1].strip()
                text_origin[key] = value

        return text_origin
    def get_keywords(self, dataset_name):
        keywords_path = 'keywords.txt'

        keywords = {}
        with open(keywords_path, 'r') as f:
            for line in f:
                parts = line.strip().split(':')
                assert len(parts) == 2, "Not complete description!"
                key = parts[0].strip()
                value = parts[1].strip()
                keywords[key] = value.split(',')

        return keywords


    def get_user_history(self,dataset_name = 'movielens'):
        # 获取用户交互数据
        function_name = f'process_{dataset_name}'

        # 在当前模块中查找这个函数
        try:
            # 如果函数存在，调用它
            process_function = globals()[function_name]
            user_history, item_set = process_function()
            return user_history, item_set
        except KeyError:
            # 如果函数不存在，返回错误信息
            return f"No process function found for {dataset_name}"

    def get_image_kr_embedding(self,
                               data_path,
                               k_r=5,
                               show_topk = False):
        # 从图像路径中获取图像embedding
        print('Loading image embeddings...')
        img_embedding_path = os.path.join(data_path, 'image_embedding.pth')
        img_embedding_map = torch.load(img_embedding_path)

        names = []
        embeddings = []
        for name, emb in img_embedding_map.items():
            names.append(name)
            embeddings.append(torch.tensor(emb))

        embeddings_tensor = torch.stack(embeddings).to('cuda')

        batch_size, emb_dim = embeddings_tensor.size()

        # 计算余弦相似度
        # cos = torch.nn.CosineSimilarity(dim=1)
        # similarities = cos(query_embedding, embeddings_tensor)

        tr_embedding = torch.zeros((batch_size, k_r, emb_dim), device='cuda')

        for i in tqdm(range(batch_size)):
            query_embedding = embeddings_tensor[i]

            # 计算余弦相似度
            sim = F.cosine_similarity(query_embedding, embeddings_tensor, dim=-1)

            top_kr_indices = torch.topk(sim, k_r, largest=True).indices

            # different IDs may refer to the same image
            assert top_kr_indices[0] == i or math.isclose(sim[i], 1, abs_tol = 1e-6), "The most similar image should be itself"

            tr_embedding[i] = embeddings_tensor[top_kr_indices]

        tr_embedding = self.linear_proj(tr_embedding.view(-1, emb_dim))
        seq_len, new_emb_dim = tr_embedding.size()[-2],tr_embedding.size()[-1]

        tr_embedding = tr_embedding.view(batch_size, k_r, seq_len, new_emb_dim)

        print('Image embeddings loaded.')
        return tr_embedding, names

    def get_text_kr_embedding(self,
                              data_path,
                              k_r=5):
        # 从图像路径中获取文本embedding
        print('Loading text embeddings...')
        text_embedding_path = os.path.join(data_path, 'text_embedding.pth')
        text_embedding_map = torch.load(text_embedding_path)

        names = []
        embeddings = []
        for name, emb in text_embedding_map.items():
            names.append(name)
            embeddings.append(emb)

        embeddings_tensor = torch.stack(embeddings).to('cuda')
        batch_size, seq_len, emb_dim = embeddings_tensor.size()

        # 假设query_embedding是您要查询的embedding，也需要转到CUDA上
        # index = names.index('356')
        # query_embedding = embeddings_tensor[index]

        # 计算余弦相似度
        # cos = torch.nn.CosineSimilarity(dim=1)
        # similarities = cos(query_embedding.flatten(), embeddings_tensor.flatten(start_dim=1))

        tr_embedding = torch.zeros((batch_size, k_r, seq_len, emb_dim), device ='cuda')
        for i in tqdm(range(batch_size)):
            query_embedding = embeddings_tensor[i]

            # 计算余弦相似度
            sim = F.cosine_similarity(query_embedding.flatten(), embeddings_tensor.flatten(start_dim=1), dim = -1)

            top_kr_indices = torch.topk(sim, k_r, largest=True).indices

            assert top_kr_indices[0] == i or math.isclose(sim[i], 1, abs_tol = 1e-6), \
                "The most similar item should be itself"

            tr_embedding[i] = embeddings_tensor[top_kr_indices]

        print('Text embeddings loaded.')
        return tr_embedding, names

    def forward(self, text_kr_embedding, image_kr_embedding):
        '''
            Input: # [batch, k_r, 77, 768] and [batch, k_r, 4, 768]
            First text intra attention, then image intra attention.
            Finally use text embedding as query and image embedding as key and value to get the final output
        '''

        batch, k_r, seq_len, emb_dim = text_kr_embedding.size()
        # Text intra-attention
        text_kr_embedding = torch.flatten(text_kr_embedding, start_dim=1, end_dim=2)
        text_attended = self.attention(text_kr_embedding)

        # Image intra-attention
        # image_kr_embedding = torch.flatten(image_kr_embedding, start_dim=1, end_dim=2)

        # Cross-attention: text as query, image as key and value
        output = self.attention(text_attended, input_keys=image_attended, input_values=image_attended)

        return output.view(batch, k_r, seq_len, emb_dim)[:,0,:,:]  # .squeeze(1)

if __name__ == '__main__':
    model = RASD()