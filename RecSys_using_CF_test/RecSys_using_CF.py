#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import time
from surprise import KNNBasic
from surprise import Dataset
from surprise import KNNBaseline
from surprise import reader, accuracy
import math
from sklearn.model_selection import train_test_split


# In[8]:


#load dữ liệu vào
#interactions_full_check = pd.read_csv('course_user.csv')


# In[9]:


class RecSys_CF(object):
    def __init__(self, interactions_full_check):
        self.interactions_full_check = interactions_full_check
    
    #hàm tính strength point
    def smooth_user_preference(self,x):
        return math.log(1+float(x), 2)
    
    #hàm lấy ra dữ liệu tương ứng với app id
    def get_app_id(self,a_id):
        interactions_full_df = self.interactions_full_check.groupby(['app_id','user_id', 'item_id'])['rate'].sum().apply(self.smooth_user_preference).reset_index()
        self.dataset = interactions_full_df[interactions_full_df.app_id == a_id].iloc[:, 1:]
        return self.dataset
    
    # chuyển dữ liệu sang form theo thư viện surprise
    def data_form(self, a_id):
        self.get_app_id(a_id)
        r_s = reader.Reader(rating_scale=(0.1, 15))
        data_train_df = Dataset.load_from_df(self.dataset, r_s)
        self.data_train = data_train_df.build_full_trainset()
       
    #các thuật toán gợi ý theo user-user và item_item
    def model(self):
        bsl_options = {'method': 'sgd',
               'learning_rate': .00005,
               }
        sim_options_uuCF={'name': 'pearson_baseline', 'user_based': True, 'shrinkage':100}
        sim_options_iiCF = {'name': 'pearson_baseline', 'user_based': False, 'shrinkage':100}
        
        #sim_options_uuCF={'name': 'cosine', 'user_based': True}
        #sim_options_iiCF = {'name': 'cosine', 'user_based': False}
        self.algo_iiCF = KNNBaseline(bsl_options=bsl_options,sim_options=sim_options_iiCF, k=10)
        self.algo_uuCF = KNNBaseline(bsl_options=bsl_options,sim_options=sim_options_uuCF, k=10)
        
    #fit model
    def fit(self, a_id):
        self.data_form(a_id)
        self.model()
        self.algo_iiCF.fit(self.data_train)
        self.algo_uuCF.fit(self.data_train)
        
    #đưa ra k items gợi ý cho 1 user, k mặc định bằng 5
    def recommend_items_to_user(self, user_id, k=5):
        full_items_id = self.dataset.item_id.unique()#danh sách các tất cả các itemid
        items_rated = self.dataset[self.dataset.user_id == user_id].item_id.values # các item được rate bởi user
        items_not_rated = list(set(full_items_id) - set(items_rated))#các item chưa được rate bởi user
        #lấy danh sách k items gợi ý cho user
        list_rate = []
        for i in items_not_rated:
            _, _, _, r, _ = self.algo_iiCF.predict(user_id, i) # dự đoán rating user cho 1 item
            list_rate.append(r) # thêm vào list_rate
        idx = np.argsort(list_rate)[::-1][:k] # vị trí k items có điểm dự đoán cao nhất
        items_recommended = list(np.array(items_not_rated)[idx]) # k items được gợi ý
        #return {user_id: items_recommended}
        return {user_id : items_recommended}

    #đưa ra k users gợi ý cho 1 item, k mặc định bằng 5
    #các bước tương tự như trong hàm recommend_items_to_user
    def recommend_users_to_item(self, item_id, k=5):
        full_users_id = self.dataset.user_id.unique()
        users_rate = self.dataset[self.dataset.item_id == item_id].user_id.values
        users_not_rate = list(set(full_users_id) - set(users_rate))
        list_rate = []
        for i in users_not_rate:
            _, _, _, r, _ = self.algo_iiCF.predict(i, item_id)
            list_rate.append(r)
        idx = np.argsort(list_rate)[::-1][:k]
        users_recommended = list(np.array(users_not_rate)[idx])
        #return {item_id: users_recommended}
        return {item_id : users_recommended}
        
    #đưa ra k users gợi ý cho 1 user, k mặc định bằng 5
    def recommend_users_to_user(self, user_id, k=5):
        user_iid = self.data_train.to_inner_uid(user_id) # chuyển hóa raw user_id của dataset gốc thành user_id dùng trong model
        users_iid_recommended = self.algo_uuCF.get_neighbors(user_iid, k) # lấy ra k users giống nhất với user
        #lấy danh sách k users gợi ý cho user
        users_recommended = []
        for i in users_iid_recommended:
            convert_uid = self.data_train.to_raw_uid(i) #chuyển hóa user_id trong model sang user_id gốc
            users_recommended.append(convert_uid)
        return {user_id: users_recommended}
    
    #đưa ra k items gợi ý cho 1 item, k mặc định bằng 5
    #các bước tương tự như trong hàm users_to_user
    def recommend_items_to_item(self, item_id, k=5):
        item_iid = self.data_train.to_inner_iid(item_id)
        items_iid_recommended = self.algo_iiCF.get_neighbors(item_iid, k)
        items_recommended = []
        for i in items_iid_recommended:
            convert_uid = self.data_train.to_raw_iid(i)
            items_recommended.append(convert_uid)
        return {item_id: items_recommended}


# # In[10]:
#
#
# Model = RecSys_using_CF(interactions_full_check=interactions_full_check)
#
#
# # In[11]:
#
#
# t1 = time.time()
# Model.fit(a_id='1')
# t2 = time.time()
# t2 - t1
#
#
# # In[12]:
#
#
# Model.recommend_users_to_item(item_id='3')
#
#
# # In[13]:
#
#
# Model.recommend_items_to_item(item_id='3', k=10)
#
#
# # In[14]:
#
#
# Model.recommend_items_to_user(user_id='1', k=10)
#
#
# # In[15]:
#
#
# Model.recommend_users_to_user(user_id='1', k=10)
#
#
# # In[16]:
#
#
# # users_test = X_test[X_test.app_id == "1"].user_id.unique()
# # items_test = X_test[X_test.app_id == "1"].item_id.unique()
#
#
# # In[ ]:
#
#
# # count_item_rated_user = []
# # for i in range(len(users_test)):
# #     count_item_rated_user.append(X_test[X_test.app_id == "1"].groupby("user_id").get_group(users_test[i]).item_id.unique())
# #
# # count_user_rate_item = []
# # for i in range(len(items_test)):
# #     count_user_rate_item.append(X_test[X_test.app_id == "1"].groupby("item_id").get_group(items_test[i]).user_id.unique())
#
#
#
# # In[ ]:
#
#
# # t1 = time.time()
# # chose = [10,20,30]
# # for j in range(1):
# #     recall = 0
# #     for i in range(1407):
# #         user_true = set(count_user_rate_item[i]) - (set(count_user_rate_item[i]) - set(Model.recommend_users_to_item(item_id=items_test[i],k=chose[j])))
# #         recall += len(user_true)/len(count_user_rate_item[i])
# #     print(recall/len(items_test))
# #
# # t2 = time.time()
# # t2 - t1
#
#
# # In[ ]:
#
#
# # t1 = time.time()
# # recall = 0
# # for i in range(len(users_test)):
# #     user_true = set(count_item_rated_user[i]) - (set(count_item_rated_user[i]) - set(Model.recommend_items_to_user(user_id=users_test[i],k=30)))
# #     recall += len(user_true)/len(count_item_rated_user[i])
# # print(recall/len(users_test))
# #
# # t2 = time.time()
# # t2 - t1
#
#
# # In[ ]:
#
#
#
#
