import torch
import numpy as np
from scipy.stats import entropy


class MemoryAwareTaskSampler:

    def __init__(
        self,
        n_users,
        user_graph_embeddings,
        user_history_dict,
        poi_category_map,
        device,
    ):

        self.n_users = n_users
        self.user_graph_embeddings = user_graph_embeddings.to(device)
        self.user_history_dict = user_history_dict
        self.poi_category_map = poi_category_map
        self.device = device

        self.selection_memory = torch.zeros(n_users, device=self.device)

        self.mean_selected_embedding = torch.mean(self.user_graph_embeddings, dim=0)

        self.behavioral_entropies = self._precompute_all_entropies()

    def _precompute_all_entropies(self):

        all_entropies = []

        all_categories = sorted(list(set(self.poi_category_map.values())))
        n_categories = len(all_categories)
        category_to_idx = {cat_id: i for i, cat_id in enumerate(all_categories)}

        print("Pre-computing behavioral entropies for all users...")
        for user_id in range(self.n_users):
            history = self.user_history_dict.get(user_id, [])
            if not history:
                all_entropies.append(0.0)
                continue

            category_counts = np.zeros(n_categories)
            for poi_id in history:
                category_id = self.poi_category_map.get(poi_id)
                if category_id is not None:
                    cat_idx = category_to_idx.get(category_id)
                    if cat_idx is not None:
                        category_counts[cat_idx] += 1

            if np.sum(category_counts) > 0:
                category_dist = category_counts / np.sum(category_counts)

                user_entropy = entropy(category_dist, base=2)
                all_entropies.append(user_entropy)
            else:
                all_entropies.append(0.0)

        print("Entropy computation finished.")
        return torch.tensor(all_entropies, device=self.device, dtype=torch.float)

    def _normalize_scores(self, scores):

        min_val = torch.min(scores)
        max_val = torch.max(scores)
        if max_val > min_val:
            return (scores - min_val) / (max_val - min_val)
        return torch.zeros_like(scores)

    def sample_tasks(self, batch_size, replace=False):

        with torch.no_grad():

            structural_scores = torch.norm(
                self.user_graph_embeddings - self.mean_selected_embedding, dim=1
            )

            entropy_scores = self.behavioral_entropies

            struct_norm = self._normalize_scores(structural_scores)
            entropy_norm = self._normalize_scores(entropy_scores)

            gamma = 0.5
            composite_scores = struct_norm * entropy_norm

            adjusted_scores = composite_scores / (
                1 + torch.log1p(self.selection_memory)
            )

            adjusted_scores = torch.clamp(adjusted_scores, min=1e-9)
            probabilities = adjusted_scores / torch.sum(adjusted_scores)

            p_numpy = probabilities.cpu().numpy()
            sampled_user_indices = np.random.choice(
                self.n_users, size=batch_size, replace=replace, p=p_numpy
            )

            self.selection_memory[sampled_user_indices] += 1

            selected_embeddings = self.user_graph_embeddings[sampled_user_indices]
            current_batch_mean_emb = torch.mean(selected_embeddings, dim=0)

            update_rate = 0.1
            self.mean_selected_embedding = (
                1 - update_rate
            ) * self.mean_selected_embedding + update_rate * current_batch_mean_emb

        return sampled_user_indices
