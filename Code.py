# %%
# import required modules
import random
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import model_selection, metrics, preprocessing
import copy

import torch
from torch import nn, optim, Tensor

from collections import defaultdict

from torch_geometric.utils import structured_negative_sampling
from torch_geometric.data import download_url, extract_zip
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
import torch.nn.functional as F

# %%
movie_path = './movies.csv'
rating_path = './ratings.csv'

# %%
rating_df = pd.read_csv(rating_path)

# %%
rating_df.head()

# %%
lbl_user = preprocessing.LabelEncoder()
lbl_movie = preprocessing.LabelEncoder()

rating_df.userId = lbl_user.fit_transform(rating_df.userId.values)
rating_df.movieId = lbl_movie.fit_transform(rating_df.movieId.values)

# %%
print(rating_df.userId.max())
print(rating_df.movieId.max())

# %%
# load edges between users and movies
def load_edge_csv(df, 
                  src_index_col, 
                  dst_index_col,  
                  link_index_col, 
                  rating_threshold=3.5):
    
    edge_index = None
    src = [user_id for user_id in  df['userId']]
    
    num_users = len(df['userId'].unique())

    dst = [(movie_id) for movie_id in df['movieId']]
    
    link_vals = df[link_index_col].values

    edge_attr = torch.from_numpy(df[link_index_col].values).view(-1, 1).to(torch.long) >= rating_threshold

    edge_values = []

    edge_index = [[], []]
    for i in range(edge_attr.shape[0]):
        if edge_attr[i]:
            edge_index[0].append(src[i])
            edge_index[1].append(dst[i])
            edge_values.append(link_vals[i])

                
    return edge_index, edge_values

# %%
edge_index, edge_values = load_edge_csv(
    rating_df,
    src_index_col='userId',
    dst_index_col='movieId',
    link_index_col='rating',
    rating_threshold=1 
)

# %%
edge_index = torch.LongTensor(edge_index) 
edge_values = torch.tensor(edge_values)

print(edge_index)
print(edge_index.size())

print(edge_values)
print(edge_values.size())

# %%
num_users = len(rating_df['userId'].unique())
num_movies = len(rating_df['movieId'].unique())

print(f"num_users {num_users}, num_movies {num_movies}")

# %%
def convert_r_mat_edge_index_to_adj_mat_edge_index(input_edge_index, input_edge_values):
    R = torch.zeros((num_users, num_movies))
    for i in range(len(input_edge_index[0])):
        row_idx = input_edge_index[0][i]
        col_idx = input_edge_index[1][i]
        R[row_idx][col_idx] = input_edge_values[i] 

    R_transpose = torch.transpose(R, 0, 1)
    
    
    adj_mat = torch.zeros((num_users + num_movies , num_users + num_movies))
    adj_mat[: num_users, num_users :] = R.clone()
    adj_mat[num_users :, : num_users] = R_transpose.clone()
    
    adj_mat_coo = adj_mat.to_sparse_coo()
    adj_mat_coo_indices = adj_mat_coo.indices()
    adj_mat_coo_values = adj_mat_coo.values()
    return adj_mat_coo_indices, adj_mat_coo_values

# %%
def convert_adj_mat_edge_index_to_r_mat_edge_index(input_edge_index, input_edge_values):    

    row_indices, col_indices = input_edge_index

    user_movie_mask = (row_indices < num_users) & (col_indices >= num_users)
    
    r_mat_edge_index = torch.stack([row_indices[user_movie_mask], col_indices[user_movie_mask] - num_users])
    r_mat_edge_values = input_edge_values[user_movie_mask]

    return r_mat_edge_index, r_mat_edge_values


# %%
num_interactions = edge_index.shape[1]
all_indices = [i for i in range(num_interactions)]

train_indices, test_indices = train_test_split(all_indices, 
                                               test_size=0.2, 
                                               random_state=1)

val_indices, test_indices = train_test_split(test_indices, 
                                             test_size=0.5, 
                                             random_state=1)

# %%
train_edge_index = edge_index[:, train_indices]
train_edge_value = edge_values[train_indices]

val_edge_index = edge_index[:, val_indices]
val_edge_value = edge_values[val_indices]

test_edge_index = edge_index[:, test_indices]
test_edge_value = edge_values[test_indices]

# %%
print(f"num_users {num_users}, num_movies {num_movies}, num_interactions {num_interactions}")
print(f"train_edge_index {train_edge_index}")
print((num_users + num_movies))
print(torch.unique(train_edge_index[0]).size())
print(torch.unique(train_edge_index[1]).size())

print(test_edge_value)
print(test_edge_value.size())

# %%
train_edge_index, train_edge_values  = convert_r_mat_edge_index_to_adj_mat_edge_index(train_edge_index, train_edge_value)
val_edge_index, val_edge_values = convert_r_mat_edge_index_to_adj_mat_edge_index(val_edge_index, val_edge_value)
test_edge_index, test_edge_values = convert_r_mat_edge_index_to_adj_mat_edge_index(test_edge_index, test_edge_value)

# %%
print(train_edge_index)
print(train_edge_index.size())
print(val_edge_index)
print(val_edge_index.size())
print(test_edge_index)
print(test_edge_index.size())

print(f"\n train_edge_values: \n {train_edge_values} \n {train_edge_values.size()}")
print(f"\n val_edge_values: \n {val_edge_values} \n {val_edge_values.size()}")
print(f"\n test_edge_values: \n {test_edge_values} \n {test_edge_values.size()}")

# %%
class LightGCN(MessagePassing):

    def __init__(self, num_users, num_items, embedding_dim=64, K=3, add_self_loops=False, dropout_rate=0.1):

        super().__init__()
        self.dropout_rate = dropout_rate
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.K = K
        self.add_self_loops = add_self_loops

        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim) 
        
        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim) 

        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)
        
        self.out = nn.Linear(embedding_dim + embedding_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, edge_index: Tensor, edge_values: Tensor):
        
        edge_index_norm = gcn_norm(edge_index=edge_index, add_self_loops=self.add_self_loops)

        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight]) 

        embs = [emb_0] 
        emb_k = emb_0 
        
        for i in range(self.K):
            emb_k = self.propagate(edge_index=edge_index_norm[0], x=emb_k, norm=edge_index_norm[1])
            embs.append(emb_k)
            
        embs = torch.stack(embs, dim=1)
        
        emb_final = torch.mean(embs, dim=1)
        
        users_emb_final, items_emb_final = torch.split(emb_final, [self.num_users, self.num_items]) 


        r_mat_edge_index, _ = convert_adj_mat_edge_index_to_r_mat_edge_index(edge_index, edge_values)
        
        src, dest =  r_mat_edge_index[0], r_mat_edge_index[1]
        
        user_embeds = users_emb_final[src]
        item_embeds = items_emb_final[dest]
        
        output = torch.cat([user_embeds, item_embeds], dim=1)
        
        output = self.out(output)
        return output
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

layers = 1 


# %%
model = LightGCN(num_users=num_users, 
                 num_items=num_movies, 
                 K=layers)

# %%
# define constants
ITERATIONS = 10000
EPOCHS = 5

BATCH_SIZE = 32

LR = 1e-3
ITERS_PER_EVAL = 1000
ITERS_PER_LR_DECAY = 200
K = 10
LAMBDA = 1e-6


# %%
print(f"BATCH_SIZE {BATCH_SIZE}")

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}.")


model = model.to(device)
model.train()

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.01)

scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)




loss_func = nn.MSELoss()


# %%
def get_recall_at_k(input_edge_index, 
                     input_edge_values, 
                     pred_ratings, 
                     k=10, 
                     threshold=3.5):
    with torch.no_grad():
        user_item_rating_list = defaultdict(list)

        for i in range(len(input_edge_index[0])):
            src = input_edge_index[0][i].item()
            dest = input_edge_index[1][i].item()
            true_rating = input_edge_values[i].item()
            pred_rating = pred_ratings[i].item()

            user_item_rating_list[src].append((pred_rating, true_rating))

        recalls = dict()
        precisions = dict()

        for user_id, user_ratings in user_item_rating_list.items():
            user_ratings.sort(key=lambda x: x[0], reverse=True)

            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
            
            n_rel_and_rec_k = sum(
                ((true_r >= threshold) and (est >= threshold))
                for (est, true_r) in user_ratings[:k]
            )

            precisions[user_id] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
            recalls[user_id] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

        overall_recall = sum(rec for rec in recalls.values()) / len(recalls)
        overall_precision = sum(prec for prec in precisions.values()) / len(precisions)

        return overall_recall, overall_precision
    

# %%

from sklearn.metrics import roc_auc_score
from sklearn.metrics import ndcg_score
from collections import defaultdict

def get_auc_and_ndcg(input_edge_index, 
                     input_edge_values, 
                     pred_ratings, 
                     k=10, 
                     threshold=3.5):
    with torch.no_grad():
        user_item_rating_list = defaultdict(list)
        y_true = []
        y_scores = []

        for i in range(len(input_edge_index[0])):
            src = input_edge_index[0][i].item()
            dest = input_edge_index[1][i].item()
            true_rating = input_edge_values[i].item()
            pred_rating = pred_ratings[i].item()

            user_item_rating_list[src].append((pred_rating, true_rating))

            # Convert true ratings to binary (0 or 1) based on the threshold
            binary_true_rating = 1 if true_rating >= threshold else 0
            y_true.append(binary_true_rating)
            y_scores.append(pred_rating)

        # Calculate AUC
        auc = roc_auc_score(y_true, y_scores)

        # Calculate NDCG
        y_true_ndcg = np.array([y_true])  # Convert to a 2D array
        y_scores_ndcg = np.array([y_scores])  # Convert to a 2D array
        computed_ndcg = ndcg_score(y_true_ndcg, y_scores_ndcg, k=k)

        return auc, computed_ndcg


# %%
r_mat_train_edge_index, r_mat_train_edge_values = convert_adj_mat_edge_index_to_r_mat_edge_index(train_edge_index, train_edge_values)
r_mat_val_edge_index, r_mat_val_edge_values = convert_adj_mat_edge_index_to_r_mat_edge_index(val_edge_index, val_edge_values)
r_mat_test_edge_index, r_mat_test_edge_values = convert_adj_mat_edge_index_to_r_mat_edge_index(test_edge_index, test_edge_values)





# %%
r_mat_train_edge_index = r_mat_train_edge_index.to(device)
r_mat_train_edge_values = r_mat_train_edge_values.to(device)
r_mat_val_edge_index = r_mat_val_edge_index.to(device)
r_mat_val_edge_values = r_mat_val_edge_values.to(device)
r_mat_test_edge_index = r_mat_test_edge_index.to(device)
r_mat_test_edge_values = r_mat_test_edge_values.to(device)

edge_index = edge_index.to(device)
edge_values = edge_values.to(device)
train_edge_index = train_edge_index.to(device)
train_edge_values = train_edge_values.to(device)
val_edge_index = val_edge_index.to(device)
val_edge_values = val_edge_values.to(device)
test_edge_index = test_edge_index.to(device)
test_edge_values = test_edge_values.to(device)

# %%
# training
train_losses = []
val_losses = []
val_recall_at_ks = []

for _ in range(EPOCHS):

    for iter in tqdm(range(ITERATIONS)):
        
        pred_ratings = model.forward(train_edge_index, train_edge_values)
        
        train_loss = loss_func(pred_ratings, r_mat_train_edge_values.view(-1,1))    

            
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if iter % ITERS_PER_EVAL == 0:
            model.eval()

            with torch.no_grad():
                val_pred_ratings = model.forward(val_edge_index, val_edge_values)
            
                val_loss = loss_func(val_pred_ratings, r_mat_val_edge_values.view(-1,1)).sum()
                
                recall_at_k, precision_at_k = get_recall_at_k(r_mat_val_edge_index, 
                                                            r_mat_val_edge_values, 
                                                            val_pred_ratings, 
                                                            k = 20
                                                            )
        
                    
                val_recall_at_ks.append(round(recall_at_k, 5))
                train_losses.append(train_loss.item())
                val_losses.append(val_loss.item())
            
            print(f"[Iteration {iter}/{ITERATIONS}], train_loss: {round(train_loss.item(), 7)}")

            model.train()

        if iter % ITERS_PER_LR_DECAY == 0 and iter != 0:
            scheduler.step()

# %%
iters = [iter * ITERS_PER_EVAL for iter in range(len(train_losses))]
plt.plot(iters, train_losses, label='train')
#plt.plot(iters, val_losses, label='validation')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('training and validation loss curves')
plt.legend()
plt.show()

# %%
f2 = plt.figure()
plt.plot(iters, val_recall_at_ks, label='recall_at_k')
plt.xlabel('iteration')
plt.ylabel('recall_at_k')
plt.title('recall_at_k curves')
plt.show()

# %%
model.eval()
with torch.no_grad():
    pred_ratings = model.forward(test_edge_index, test_edge_values)
    recall_at_k, precision_at_k = get_recall_at_k(r_mat_test_edge_index, 
                                                  r_mat_test_edge_values, 
                                                  pred_ratings, 50)
    print(f"recall_at_k {round(recall_at_k, 5)}, precision_at_k {round(precision_at_k, 5)}")
    auc_score, ndcg_score = get_auc_and_ndcg(r_mat_test_edge_index, r_mat_test_edge_values,  pred_ratings)
    print(f"AUC: {auc_score}, NDCG: {ndcg_score}")


# %%
model_path = 'model2.pth'
torch.save(model.state_dict(), model_path)

# %%
# model_path = 'model2.pth'
# model.load_state_dict(torch.load(model_path))


