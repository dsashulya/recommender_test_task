import os
import pickle

from typing import List, NoReturn, Tuple, Union

import numpy as np
import pandas as pd
from torch import Tensor, float32, long, tensor
from tqdm import tqdm

class Columns:
    user_id = 'userId'
    item_id = 'movieId'
    timestamp = 'timestamp'
    rating = 'rating'
    genres = 'genres'
    tag = 'tag'
    tag_id = 'tagId'
    relevance = 'relevance'
    

class RatingsDataset:
    def __init__(self, 
                 ratings: pd.DataFrame, 
                 genres: List[List[int]], 
                 tags: List[List[int]], 
                 genres_to_ids: dict,
                 idx_to_uid: dict,
                 idx_to_iid: dict,
                 idx_to_tid: dict,
                 rating_cutoff: float = 3.5
                ) -> NoReturn:
        self.ratings = ratings
        self.genres = genres
        self.tags = tags
        self.genres_to_ids = genres_to_ids
        
        self.rating_cutoff = rating_cutoff
        
        self.idx_to_uid = idx_to_uid
        self.idx_to_iid = idx_to_iid
        self.idx_to_tid = idx_to_tid
        
        self.uid_to_idx = {v: k for k, v in self.idx_to_uid.items()}
        self.iid_to_idx = {v: k for k, v in self.idx_to_iid.items()}
        self.tid_to_idx = {v: k for k, v in self.idx_to_tid.items()}
    
    def __getitem__(self, idx: int) -> Tensor:
        user, item, rat, timestamp = self.ratings.iloc[idx]
        user_id = self.uid_to_idx[int(user)]
        item_id = self.iid_to_idx[int(item)]
        rating = 1 if rat > self.rating_cutoff else 0
        
        gen = self.genres[item_id]
        tags_out = self.tags[item_id]
        
        tags_out = tensor(tags_out, dtype=long)
        gen = tensor(gen, dtype=long)
        user_id = tensor([user_id], dtype=long)
        item_id = tensor([item_id], dtype=long)
        rating = tensor([rating], dtype=float32)
        return user_id, item_id, gen, tags_out, rating
    
    def __len__(self):
        return self.ratings.shape[0]
    
    @property
    def user_count(self):
        return self.ratings[Columns.user_id].unique().shape[0]
    
    @property
    def item_count(self):
        return self.ratings[Columns.item_id].unique().shape[0]
    
    @property
    def genre_count(self):
        return len(self.genres_to_ids)
    
    @property
    def tag_count(self):
        return len(self.idx_to_tid)
    

def get_train_test_split_timestamp(ratings: pd.DataFrame, 
                                   train: float = 0.7, 
                                   q: float = 0.5
                                  ) -> float:
    """
    get each user's \train\ quantile timestamp value
    get \q\ quantile of the above (0.5 is median)
    runs for about 1-1.5 min
    return: split timestamp
    """
    n_users = ratings[Columns.user_id].unique().shape[0]
    quantiles = ratings.sort_values([Columns.user_id, Columns.timestamp], ascending=True)\
                .groupby(Columns.user_id)[Columns.timestamp]\
                .quantile(q=train)
    quantiles = sorted(quantiles.tolist())
    return quantiles[int(n_users * q) - 1]


def train_test_split(ratings: pd.DataFrame, 
                     timestamp: Union[float, int], 
                     remove_not_in_train: bool = False
                    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    remove_not_in_train: if True, removes from test set users not present in train set
    (use False for the val/test split)
    return: train, test df
    """
    train = ratings[ratings[Columns.timestamp] < timestamp]
    test = ratings[ratings[Columns.timestamp] >= timestamp]
    if remove_not_in_train:
        train_users = train[Columns.user_id].unique()
        train_items = train[Columns.item_id].unique()
        test = test[test[Columns.user_id].isin(train_users)]
        test = test[test[Columns.item_id].isin(train_items)]
    return train, test


def preprocess_tags(item: int,
                   tags_time: pd.DataFrame,
                   tag_ids: pd.DataFrame,
                   tag_scores: pd.DataFrame,
                   tid_to_idx: dict
                   ) -> Tuple[List[int], List[int]]:
    tags = tags_time[tags_time[Columns.item_id] == item][Columns.tag].tolist()
    tags_out, relevance = [], []
    for tag in tags:
        tag_id = tag_ids[tag_ids[Columns.tag] == str(tag).lower()][Columns.tag_id]
        if tag_id.shape[0] == 1:  # some tags are not present in the tag list
            tag_id = tag_id.item()
#             scores = tag_scores[tag_scores[Columns.item_id] == item]
#             relevance.append(scores[scores[Columns.tag_id] == tag_id][Columns.relevance].item())
            tags_out.append(tid_to_idx[tag_id])
#     relevance = np.array(relevance) / sum(relevance)
    return tags_out#, relevance.tolist()


def preprocess_genres(item: int,
                     genres: pd.DataFrame,
                     genres_to_ids: dict
                     ) -> List[int]:
    gen = genres[genres[Columns.item_id] == item][Columns.genres]
    gen = gen.item().strip().split('|')
    genres_out = []
    for genre in gen:
        genres_out.append(genres_to_ids[genre])
    return genres_out


def preprocess_tags_genres(dataset: RatingsDataset,
                            idx_to_iid: dict,
                           tags_time: pd.DataFrame,
                           tag_ids: pd.DataFrame,
                           tag_scores: pd.DataFrame,
                           tid_to_idx: dict,
                            genres: pd.DataFrame,
                            genres_to_ids: dict,
                            path_to_data: str,
                            dataset_type: str = 'train',
                           ) -> NoReturn:
    tags_by_idx = []
    relevance_by_idx = []
    genres_by_idx = []
    items = dataset[Columns.item_id].unique().shape[0]
    for idx in tqdm(range(items),
                    position=0,
                    leave=True,
                    desc='Processing tags'):
        item_id = idx_to_iid[idx]
        tags_out = preprocess_tags(item_id, tags_time, tag_ids, tag_scores, tid_to_idx)
        genres_by_idx.append(preprocess_genres(item_id, genres, genres_to_ids))
        tags_by_idx.append(tags_out)
#         relevance_by_idx.append(relevance)
    
    # saving preprocessed tags and genres
    with open(os.path.join(path_to_data, f'{dataset_type}_tags.pkl'), 'wb') as tags_file:
        pickle.dump(tags_by_idx, tags_file)
#     with open(os.path.join(path_to_data, f'{dataset_type}_relevance.pkl'), 'wb') as relevance_file:
#         pickle.dump(relevance_by_idx, relevance_file)
    with open(os.path.join(path_to_data, f'{dataset_type}_genres.pkl'), 'wb') as genres_file:
        pickle.dump(genres_by_idx, genres_file)
