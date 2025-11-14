import numpy as np
from tqdm import tqdm
import networkx as nx
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")


class DataLoader:

    def __init__(self, args, state="cold_start"):
        self.args = args
        self.dataset = args.dataset
        self.path = args.data_path + self.dataset + "/"

        self._load_dataset_stats()

        self.user_history_dict = self._load_user_history(self.path + "train.txt")

        self.support_tasks = self._read_task_file(
            self.path + f"test_scenario/{state}_support.txt"
        )
        self.query_tasks = self._read_task_file(
            self.path + f"test_scenario/{state}_query.txt"
        )

        self.triplets, self.n_entities, self.n_relations = self._read_triplets(
            self.path + "kg_final.txt"
        )
        self.n_nodes = self.n_entities + self.n_users

        self.poi_category_map = self._load_poi_categories(self.path + "item_cate.txt")

        self.graph = self._build_graph()

    def _load_dataset_stats(self):
        user_item_stats = {
            "yelp": {"users": 45919, "items": 45538},
            "NYCRestaurant": {"users": 3112, "items": 3298},
        }
        stats = user_item_stats.get(self.dataset)
        if stats:
            self.n_users = stats["users"]
            self.n_items = stats["items"]
        else:
            raise ValueError(f"Dataset stats for {self.dataset} not found.")

    def _load_user_history(self, file_name):

        print("Loading user interaction history...")
        user_history_dict = defaultdict(list)
        with open(file_name, "r") as f:
            for line in f.readlines():
                parts = [int(i) for i in line.strip().split(" ")]
                user_id, item_ids = parts[0], parts[1:]
                user_history_dict[user_id].extend(item_ids)
        return user_history_dict

    def _read_task_file(self, file_name):

        task_dict = defaultdict(list)
        with open(file_name, "r") as f:
            for line in f.readlines():
                parts = [int(i) for i in line.strip().split(" ")]
                user_id, item_ids = parts[0], parts[1:]
                task_dict[user_id].extend(list(set(item_ids)))
        return task_dict

    def _read_triplets(self, file_name):

        print("Loading knowledge graph triplets...")
        can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
        can_triplets_np = np.unique(can_triplets_np, axis=0)

        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1

        n_entities = (
            max(np.max(can_triplets_np[:, 0]), np.max(can_triplets_np[:, 2])) + 1
        )
        n_relations = np.max(can_triplets_np[:, 1]) + 1

        return can_triplets_np, n_entities, n_relations

    def _load_poi_categories(self, file_name):

        print("Loading POI category map...")
        poi_category_map = {}
        with open(file_name, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 2:
                    poi_id, cat_id = int(parts[0]), int(parts[1])
                    poi_category_map[poi_id] = cat_id
        return poi_category_map

    def _build_graph(self):

        print("Building the graph...")
        ckg_graph = nx.MultiDiGraph()

        for u_id, items in self.user_history_dict.items():
            for i_id in items:

                ckg_graph.add_edge(u_id, i_id + self.n_users, key=0)

        for h_id, r_id, t_id in tqdm(
            self.triplets, ascii=True, desc="Loading KG edges"
        ):

            ckg_graph.add_edge(h_id + self.n_users, t_id + self.n_users, key=r_id)

        return ckg_graph

    def get_data_for_sampler(self):

        return {
            "n_users": self.n_users,
            "user_history_dict": self.user_history_dict,
            "poi_category_map": self.poi_category_map,
        }

    def get_data_for_model(self):

        return {
            "n_users": self.n_users,
            "n_items": self.n_items,
            "n_entities": self.n_entities,
            "n_nodes": self.n_nodes,
            "n_relations": self.n_relations,
            "graph": self.graph,
        }

    def get_task_data(self):

        return self.support_tasks, self.query_tasks
