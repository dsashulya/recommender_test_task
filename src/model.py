from typing import NoReturn

import torch
from torch import nn


class UserEmbedding(nn.Module):
    def __init__(self, 
                 n_users: int, 
                 embed_dim: int = 128, 
                 hidden_dim: int = 256
                ) -> NoReturn:
        super().__init__()
        self.embedding = nn.Embedding(n_users, embed_dim)
        self.sequential = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
    def forward(self, user_id: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(user_id).squeeze(1)
        return self.sequential(emb)
    
    
class ItemEmbedding(nn.Module):
    def __init__(self, 
                 n_items: int, 
                 n_genres: int, 
                 n_tags: int, 
                 id_embed_dim: int = 64, 
                 genres_embed_dim: int = 32, 
                 tags_embed_dim: int = 32, 
                 hidden_dim: int = 256
                ) -> NoReturn:
        super().__init__()
        self.id_embedding = nn.Embedding(n_items, id_embed_dim)
        self.genre_embedding = nn.Embedding(n_genres, genres_embed_dim)
        self.tag_embedding = nn.Embedding(n_tags, tags_embed_dim)
        
        embed_dim = id_embed_dim + genres_embed_dim + tags_embed_dim
        self.sequential = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
    def forward(self, 
                item_id: torch.Tensor, 
                genres: torch.Tensor, 
                tags: torch.Tensor,
               ) -> torch.Tensor:
        id_emb = self.id_embedding(item_id)
        genre_emb = self.genre_embedding(genres)
        tag_emb = self.tag_embedding(tags)
        x = torch.cat([
            id_emb.squeeze(1), 
            genre_emb.mean(dim=1), 
            tag_emb.mean(dim=1)
        ], dim=-1)
        return self.sequential(x)

    
class MovieClassifier(nn.Module):
    def __init__(self, 
                 n_users: int, 
                 n_items: int,
                 n_genres: int, 
                 n_tags: int, 
                 user_id_embed_dim: int = 128, 
                 hidden_dim_users: int = 256,
                 item_id_embed_dim: int = 64,
                 genres_embed_dim: int = 32, 
                 tags_embed_dim: int = 32, 
                 hidden_dim_items: int = 256
                ) -> NoReturn:
        super().__init__()
        self.user_emb = UserEmbedding(n_users, 
                                      user_id_embed_dim, 
                                      hidden_dim_users)
        self.item_emb = ItemEmbedding(n_items,
                                      n_genres, 
                                      n_tags,
                                      item_id_embed_dim, 
                                      genres_embed_dim, 
                                      tags_embed_dim, 
                                      hidden_dim_items)
        ff_dim = user_id_embed_dim + item_id_embed_dim + genres_embed_dim + tags_embed_dim
        self.sequential = nn.Sequential(
            nn.Linear(ff_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, 
                user_id: torch.Tensor,
                item_id: torch.Tensor, 
                genres: torch.Tensor, 
                tags: torch.Tensor,
               ) -> torch.Tensor:
        user_emb = self.user_emb(user_id)
        item_emb = self.item_emb(item_id, genres, tags)
        x = self.sequential(torch.cat([user_emb, item_emb], dim=-1))
        return self.sigmoid(x)