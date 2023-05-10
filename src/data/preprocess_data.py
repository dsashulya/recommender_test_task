import argparse
import os
import pickle

from typing import NoReturn

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import Columns, get_train_test_split_timestamp, preprocess_tags_genres, train_test_split


RATINGS_FILE = 'ratings.csv'
TAG_SCORES_FILE = 'genome-scores.csv'
TAG_IDS_FILE = 'genome-tags.csv'
TAG_TIMES_FILE = 'tags.csv'
GENRES_FILE = 'movies.csv'


def parse_args(parser: argparse.ArgumentParser) -> NoReturn:
    parser.add_argument('-t', '--train_size',
                        help='Ratio of ratings to add to the train set (results will be approximate)',
                        type=float,
                        default=0.7)
    parser.add_argument('-v', '--val_size',
                        help='Ratio of ratings to add to the val set  after train is made(results will be approximate)',
                        type=float,
                        default=0.5)
    parser.add_argument('-q', '--timestamp_quantile',
                        help='Quantile of all timestamps at --train_size for each user (results will be approximate)',
                        type=float,
                        default=0.7)
    parser.add_argument('-r', '--remove_not_in_train',
                        help='Whether to remove from test (& val) set the users not present in train set',
                        type=lambda x: bool(x),
                        default=1)
    parser.add_argument('-d', '--path_to_data',
                        help='Directory where data .csv files are located',
                        type=str,
                        default='ml-latest')
    parser.add_argument('-o', '--output_dir',
                        help='Directory where output data files are stored',
                        type=str,
                        default='ml-latest')


def main(args):
    print("Reading data files")
    ratings = pd.read_csv(os.path.join(args.path_to_data, RATINGS_FILE))
    tag_ids = pd.read_csv(os.path.join(args.path_to_data, TAG_IDS_FILE))
    genres = pd.read_csv(os.path.join(args.path_to_data, GENRES_FILE))
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Searching for train TIMESTAMP (takes around a minute)")
    timestamp_train = get_train_test_split_timestamp(ratings, args.train_size, args.timestamp_quantile)
    train, test = train_test_split(ratings, timestamp_train, remove_not_in_train=args.remove_not_in_train)
    
    print("Searching for val TIMESTAMP")
    timestamp_val = get_train_test_split_timestamp(test, args.val_size, args.timestamp_quantile)
    val, test = train_test_split(test, timestamp_val, remove_not_in_train=False)
    
    print(f"With remove_not_in_train set to {args.remove_not_in_train} the split is:")
    print(f"TRAIN {train.shape[0] / ratings.shape[0] * 100:.1f}%")
    print(f"VAL {val.shape[0] / ratings.shape[0] * 100:.1f}%")
    print(f"TEST {test.shape[0] / ratings.shape[0] * 100:.1f}%")
    
    # saving dataframes
    train.to_csv(os.path.join(args.output_dir, 'train.csv'), index=False)
    val.to_csv(os.path.join(args.output_dir, 'val.csv'), index=False)
    test.to_csv(os.path.join(args.output_dir, 'test.csv'), index=False)
    
    print("Reindexing users, items and tags")
    idx_to_uid = {i: u_id for i, u_id in enumerate(train[Columns.user_id].unique())}
    idx_to_iid = {i: i_id for i, i_id in enumerate(train[Columns.item_id].unique())}
    idx_to_tid = {i: t_id for i, t_id in enumerate(tag_ids[Columns.tag_id].unique())}
    
    # saving indexing dicts
    with open(os.path.join(args.output_dir, 'idx_to_uid.pkl'), 'wb') as uid_file:
        pickle.dump(idx_to_uid, uid_file)
    with open(os.path.join(args.output_dir, 'idx_to_iid.pkl'), 'wb') as iid_file:
        pickle.dump(idx_to_iid, iid_file)
    with open(os.path.join(args.output_dir, 'idx_to_tid.pkl'), 'wb') as tid_file:
        pickle.dump(idx_to_uid, tid_file)
        
    tags_time = pd.read_csv(os.path.join(args.path_to_data, TAG_TIMES_FILE))
    tag_scores = pd.read_csv(os.path.join(args.path_to_data, TAG_SCORES_FILE))
    tid_to_idx = {v: k for k, v in idx_to_tid.items()}
    
    # CUTOFF
    tags_time = tags_time[tags_time[Columns.timestamp] < timestamp_train]
    
    print("Processing tags and genres")
    # genres to ids
    genre = []
    for g in genres['genres']:
        genre.extend(g.strip().split('|'))
    genres_to_ids = {}
    for i, g in enumerate(np.unique(genre)):
        genres_to_ids[g] = i
        
    with open(os.path.join(args.output_dir, 'genres_to_ids.pkl'), 'wb') as genres_file:
        pickle.dump(genres_to_ids, genres_file)
        
    # preprocess and save
    preprocess_tags_genres(train, 
                           idx_to_iid,
                           tags_time,
                           tag_ids,
                           tag_scores,
                           tid_to_idx,
                           genres, 
                           genres_to_ids, 
                           args.output_dir, 
                           'train')
    preprocess_tags_genres(val, 
                           idx_to_iid,
                           tags_time,
                           tag_ids,
                           tag_scores,
                           tid_to_idx,
                           genres, 
                           genres_to_ids, 
                           args.output_dir, 
                           'val')
    preprocess_tags_genres(test, 
                           idx_to_iid,
                           tags_time,
                           tag_ids,
                           tag_scores,
                           tid_to_idx,
                           genres, 
                           genres_to_ids, 
                           args.output_dir, 
                           'test')
    print("FINISHED")
    print(f"All data can be found at {args.output_dir}")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parse_args(parser)
    args = parser.parse_args()
    main(args)
    