import collections
import os
import numpy as np
np.random.seed(555)

'''
train_data/eval_data/test_data-> list  [[user, item, rating]..]

ripple_set-> dict:{user: list}    ripple_set[user] -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]

'''
def load_data(args):
    train_data, eval_data, test_data, user_history_dict = load_rating(args)
    n_entity, n_relation, kg = load_kg(args)   # kg ： { head：[tail, relation] ...}
    ripple_set = get_ripple_set(args, kg, user_history_dict)
    return train_data, eval_data, test_data, n_entity, n_relation, ripple_set


def load_rating(args):
    """说明
    
    Keyword arguments:
    argument -- description
    Return: 返回只有正样本的 train_data, eval_data, test_data, user_history_dict
    """
    
    print('reading rating file ...')

    # reading rating file
    # rating_file = '../data/' + args.dataset + '/ratings_final'
    rating_file = '../../process/ratings'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)
        np.save(rating_file + '.npy', rating_np)

    # n_user = len(set(rating_np[:, 0]))
    # n_item = len(set(rating_np[:, 1]))
    return dataset_split(rating_np)


def dataset_split(rating_np):
    """sumary_line
    
    Keyword arguments:
    rating_np -- 所有用户评分数据的二维数组
    Return: return_description
    """
    
    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]  # 拿到具体多少个评分数据

    eval_indices = np.random.choice(n_ratings, size=int(n_ratings * eval_ratio), replace=False) 
    left = set(range(n_ratings)) - set(eval_indices)  # 计算总数-验证集数 
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    # print(len(train_indices), len(eval_indices), len(test_indices))

    # traverse training data, only keeping the users with positive ratings
    # 遍历训练数据，只保留具有正面评分的用户。
    user_history_dict = dict()
    for i in train_indices:
        user = rating_np[i][0]
        item = rating_np[i][1]
        rating = rating_np[i][2]
        if rating == 1:
            if user not in user_history_dict:
                user_history_dict[user] = []
            user_history_dict[user].append(item) # 仅仅存储训练集中用户所对应的正样本

    train_indices = [i for i in train_indices if rating_np[i][0] in user_history_dict]
    eval_indices = [i for i in eval_indices if rating_np[i][0] in user_history_dict]
    test_indices = [i for i in test_indices if rating_np[i][0] in user_history_dict]
    # print(len(train_indices), len(eval_indices), len(test_indices))

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data, eval_data, test_data, user_history_dict

# 加载kg_final文件并存为npy文件并传入到kg = construct_kg(kg_np)变为{ head：[tail, relation] }
def load_kg(args):
    print('reading KG file ...')

    # reading kg file
    # kg_file = '../data/' + args.dataset + '/kg_final'
    kg_file = '../../process/triples'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int32)
        np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2])) # 求并集后算个数
    n_relation = len(set(kg_np[:, 1])) 

    kg = construct_kg(kg_np)

    return n_entity, n_relation, kg


def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    kg = collections.defaultdict(list)  # 字典{key: list}
    for head, relation, tail in kg_np:
        kg[head].append((tail, relation))
    return kg


# kg -> dict {head：[tail, relation]}
# user_history_dict -> {user: [item1, item2]]}
def get_ripple_set(args, kg, user_history_dict):
    """sumary_line
    Keyword arguments:
    argument -- description
    Return: ripple_set
    # ripple_set[user] -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
    """
    
    print('constructing ripple set ...')
    
    ripple_set = collections.defaultdict(list)

    for user in user_history_dict:
        for h in range(args.n_hop):  # 拿到最大跳数
            memories_h = []
            memories_r = []
            memories_t = []

            if h == 0:  # seed
                tails_of_last_hop = user_history_dict[user]  # tails_of_last_hop是一个list
            else:
                tails_of_last_hop = ripple_set[user][-1][2]  # 是以上一跳hop中的t作为头节点

            for entity in tails_of_last_hop:    # 遍历每个交互的物品
                for tail_and_relation in kg[entity]:
                    memories_h.append(entity)
                    memories_r.append(tail_and_relation[1])
                    memories_t.append(tail_and_relation[0])

            # if the current ripple set of the given user is empty, we simply copy the ripple set of the last hop here
            # 如果当前给定用户的Ripple集是空的，我们简单地复制上一跳的Ripple集。
            # this won't happen for h = 0, because only the items that appear in the KG have been selected
            # 当h=0时，这不会发生，因为只选择了KG中出现的项。
            # this only happens on 154 users in Book-Crossing dataset (since both BX dataset and the KG are sparse)
            # 这只发生在Book-Crossing数据集中的154个用户上(因为BX数据集和KG都是稀疏的)
            # 这部分主要说2个事情，第一个是设置每一跳采用多少个邻居信息，这里默认是32
            # 随机选取32个结点，当replace为false时，取32个不同的数字，如果replace为True，则可以选取相同的数字
            if len(memories_h) == 0:
                ripple_set[user].append(ripple_set[user][-1]) # 以上一跳结果作为这一跳的结果
            else:
                # sample a fixed-size 1-hop memory for each user
                replace = len(memories_h) < args.n_memory  # 如果节点数量小于32（给定最大数量） 随机挑32个
                indices = np.random.choice(len(memories_h), size=args.n_memory, replace=replace)   # replace=replace: 这是可选参数，它确定在选择时是否允许重复
                memories_h = [memories_h[i] for i in indices]
                memories_r = [memories_r[i] for i in indices]
                memories_t = [memories_t[i] for i in indices]
                ripple_set[user].append((memories_h, memories_r, memories_t))
                # 最后返回的每一个Set都有n_memory个节点（不够就重复， 但是可能会有没有选到的）

    return ripple_set
