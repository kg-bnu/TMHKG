import pandas as pd
import gzip
import numpy as np
import json
import tqdm
import random
import collections
import time

random.seed(2020)


# --- 辅助函数 (保持不变) ---
def read_user_list(path):
    """读取用户ID映射文件。返回: dict{org_id: remap_id}"""
    lines = open(path + "user_list.txt", "r").readlines()
    user_dict = dict()
    for idx, line in enumerate(lines):
        if idx == 0:
            continue
        l = line.strip()
        tmp = l.split()
        user_dict[tmp[0]] = tmp[1]
    return user_dict


def read_poi_list(path):
    """读取POI ID映射文件。返回: dict{org_id: remap_id}"""
    lines = open(path + "poi_list.txt", "r").readlines()
    poi_dict = dict()
    for idx, line in enumerate(lines):
        if idx == 0:
            continue
        l = line.strip()
        tmp = l.split()
        poi_dict[tmp[0]] = str(idx - 1)
    return poi_dict


def merge_train_vali_test(path):
    """合并训练、验证和测试集，得到每个用户的完整交互历史。"""
    print("Merging train/valid/test files...")
    user_poi_dict = {}
    try:
        with open(path + "train.txt", "r") as f_train, open(
            path + "valid.txt", "r"
        ) as f_valid, open(path + "test.txt", "r") as f_test:

            for l in f_train.readlines() + f_valid.readlines() + f_test.readlines():
                parts = [int(i) for i in l.strip().split()]
                if not parts:
                    continue
                user_id, poi_ids = parts[0], parts[1:]
                if user_id not in user_poi_dict:
                    user_poi_dict[user_id] = set()
                user_poi_dict[user_id].update(poi_ids)

        user_poi_dict_list = {str(k): list(v) for k, v in user_poi_dict.items()}
        with open(path + "user_pois_all.json", "w") as f:
            json.dump(user_poi_dict_list, f)
        print("Merged user interactions saved to user_pois_all.json.")
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure train.txt, valid.txt, and test.txt exist.")


# --- 核心功能：创建元学习场景 ---


def construct_test_scenario(path):
    """
    【重大修改】
    根据交互数量将用户和POI划分为“冷启动”和“已有”，并构建不同的测试场景。
    这与您论文中的描述完全一致。
    """
    print("Constructing meta-learning scenarios based on interaction counts...")

    with open(path + "user_pois_all.json", "r") as f:
        user_poi_all = json.load(f)

    # 1. 计算每个用户和POI的交互数量
    user_interaction_counts = {
        int(u): len(p_list) for u, p_list in user_poi_all.items()
    }

    poi_interaction_counts = collections.defaultdict(int)
    for p_list in user_poi_all.values():
        for poi in p_list:
            poi_interaction_counts[poi] += 1

    # 2. 根据交互数量排序
    sorted_users = sorted(user_interaction_counts.items(), key=lambda x: x[1])
    sorted_pois = sorted(poi_interaction_counts.items(), key=lambda x: x[1])

    # 3. 按8:2的比例划分
    split_ratio = 0.8
    # 交互数量最少的20%为冷启动用户/POI
    cold_start_users = [
        u for u, count in sorted_users[: int((1 - split_ratio) * len(sorted_users))]
    ]
    warm_users = [
        u for u, count in sorted_users[int((1 - split_ratio) * len(sorted_users)) :]
    ]

    cold_start_pois = [
        p for p, count in sorted_pois[: int((1 - split_ratio) * len(sorted_pois))]
    ]
    warm_pois = [
        p for p, count in sorted_pois[int((1 - split_ratio) * len(sorted_pois)) :]
    ]

    print(
        f"Total users: {len(sorted_users)}, Cold-start users: {len(cold_start_users)}"
    )
    print(f"Total POIs: {len(sorted_pois)}, Cold-start POIs: {len(cold_start_pois)}")

    # 4. 构建场景
    scenarios = {
        "meta_training": {},
        "warm_up": {},
        "user_cold": {},
        "poi_cold": {},
        "user_poi_cold": {},
    }

    for u_str, p_list in user_poi_all.items():
        u = int(u_str)
        p_set = set(p_list)
        if u in cold_start_users:
            # 冷启动用户的交互：与已有POI的交互
            scenarios["user_cold"][u] = list(p_set & set(warm_pois))
            # 用户和POI双冷启动：冷启动用户与冷启动POI的交互
            scenarios["user_poi_cold"][u] = list(p_set & set(cold_start_pois))
        elif u in warm_users:
            # POI冷启动：已有用户与冷启动POI的交互
            scenarios["poi_cold"][u] = list(p_set & set(cold_start_pois))
            # 元训练数据：已有用户与已有POI的交互
            scenarios["meta_training"][u] = list(p_set & set(warm_pois))

    # 从meta_training中划分出一部分作为warm_up
    warm_up_keys = random.sample(
        list(scenarios["meta_training"].keys()),
        k=int(0.1 * len(scenarios["meta_training"])),
    )
    for key in warm_up_keys:
        scenarios["warm_up"][key] = scenarios["meta_training"].pop(key)

    for s_name, s_data in scenarios.items():
        # 过滤掉没有交互的条目
        filtered_data = {k: v for k, v in s_data.items() if v}
        with open(path + f"test_scenario/{s_name}.json", "w") as f:
            json.dump(filtered_data, f)
    print("Meta-learning scenarios constructed and saved.")


def support_query_set(path, states):
    """为每个场景将其用户交互划分为support集和query集。(此函数逻辑不变)"""
    print("Splitting scenarios into support/query sets...")
    path_test = path + "test_scenario/"
    for s in states:
        path_json = path_test + s + ".json"
        try:
            with open(path_json, "r") as f:
                scenario = json.load(f)
        except FileNotFoundError:
            print(f"Warning: {path_json} not found, skipping.")
            continue

        with open(path_test + s + "_support.txt", "w") as support_txt, open(
            path_test + s + "_query.txt", "w"
        ) as query_txt:

            for u, p_list in scenario.items():
                if len(p_list) >= 13 and len(p_list) <= 100:
                    random.shuffle(p_list)
                    support = p_list[:-10]
                    query = p_list[-10:]
                    support_txt.write(f"{u} {' '.join(map(str, support))}\n")
                    query_txt.write(f"{u} {' '.join(map(str, query))}\n")
    print("Support/query sets created.")


def create_poi_category_file(path, poi_map):
    """【需要您来实现】为 MemoryAwareTaskSampler 创建 poi_cate.txt 文件。"""
    print("Placeholder function: Creating poi_cate.txt...")
    num_categories = 50  # 假设Yelp有50个类别
    with open(path + "poi_cate.txt", "w") as f:
        for org_id, remap_id in poi_map.items():
            category_id = random.randint(0, num_categories - 1)
            f.write(f"{remap_id} {category_id}\n")
    print(
        "poi_cate.txt created with placeholder data. Please implement with real data."
    )


if __name__ == "__main__":
    # --- 主执行流程 ---
    # 1. 设置数据集
    # dataset = "yelp"
    dataset = "nycrestraunt"
    path = f"./datasets/{dataset}/"

    # 【注意】: 新的 `construct_test_scenario` 不再需要时间戳文件。
    # 因此，可以注释掉或删除 first_reach_* 函数的调用。
    # if dataset == "yelp":
    #     first_reach_yelp(path)
    # elif dataset == "nycrestraunt":
    #     first_reach_nycrestraunt(path)

    # 2. 合并所有交互
    merge_train_vali_test(path)

    # 3. 构建元学习场景 (使用新的、基于交互数量的方法)
    construct_test_scenario(path)

    # 4. 划分Support/Query集
    states = ["meta_training", "warm_up", "user_cold", "poi_cold", "user_poi_cold"]
    support_query_set(path, states)

    # 5. 创建POI类别文件
    poi_map = read_poi_list(path + "poi_list.txt")
    create_poi_category_file(path, poi_map)

    print(f"\nData construction for {dataset} finished!")
