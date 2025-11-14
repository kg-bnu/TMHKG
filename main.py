# environment:Python >= 3.8 ;PyTorch 1.7+; cuda 11.1+
# python construct_data.py
# python main.py --dataset yelp --use_meta_model False


import torch
import numpy as np
import random
from time import time
from tqdm import tqdm
import scipy.sparse as sp

# 导入我们新的模块
from utility.parser_TMHKG import parse_args
from utility.data_loader import DataLoader
from model.TMHKG import Recommender
from utility.scheduler import MemoryAwareTaskSampler
from utility.evaluate import test
from utility.helper import early_stopping
from collections import defaultdict


def build_interaction_matrix(n_users, n_items, user_history_dict):
    """
    根据用户历史构建稀疏的、标准化的交互矩阵。
    """
    rows, cols = [], []
    for user_id, items in user_history_dict.items():
        for item_id in items:
            rows.append(user_id)
            cols.append(item_id)

    vals = np.ones(len(rows))
    mat = sp.coo_matrix((vals, (rows, cols)), shape=(n_users, n_items))

    # D^{-1}A normalization
    rowsum = np.array(mat.sum(1))
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat_inv = sp.diags(d_inv)

    norm_adj = d_mat_inv.dot(mat)
    coo = norm_adj.tocoo()

    i = torch.LongTensor([coo.row, coo.col])
    v = torch.from_numpy(coo.data).float()
    return torch.sparse.FloatTensor(i, v, coo.shape).to(device)


def build_batch(user_ids, task_data, user_history, n_items, max_history_len):
    """
    为给定的用户ID构建一个batch。
    这个函数会进行负采样，并处理用户历史的填充。
    """
    batch = defaultdict(list)
    for user_id in user_ids:
        pos_items = task_data.get(user_id, [])
        if not pos_items:
            continue

        # 随机选择一个正样本
        pos_item = random.choice(pos_items)

        # 负采样
        while True:
            neg_item = random.randint(0, n_items - 1)
            # 确保负样本不在用户的全部历史中
            if neg_item not in user_history.get(user_id, []):
                break

        # 获取并处理用户历史序列 (截断或填充)
        history = user_history.get(user_id, [])
        if len(history) > max_history_len:
            history = history[-max_history_len:]
        else:
            # 使用 n_items 作为一个特殊的 padding ID
            history = history + [n_items] * (max_history_len - len(history))

        batch["users"].append(user_id)
        batch["pos_items"].append(pos_item)
        batch["neg_items"].append(neg_item)
        batch["history"].append(history)

    # 转换为PyTorch张量
    for key in batch:
        batch[key] = torch.LongTensor(batch[key]).to(device)

    return batch


if __name__ == "__main__":
    """固定随机种子"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """读取参数"""
    global args, device
    args = parse_args()
    device = (
        torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
    )

    """1. 使用新的 DataLoader 加载数据"""
    print("Loading data...")
    # 加载元训练数据
    meta_loader = DataLoader(args, state="meta_training")
    meta_model_data = meta_loader.get_data_for_model()
    meta_sampler_data = meta_loader.get_data_for_sampler()
    support_tasks, query_tasks = meta_loader.get_task_data()

    # 加载用于微调和测试的冷启动场景数据
    cold_loader = DataLoader(args, state=args.cold_scenario)
    cold_model_data = cold_loader.get_data_for_model()
    cold_support_tasks, cold_query_tasks = cold_loader.get_task_data()
    # 注意：在评估时，我们需要所有用户的历史记录来生成嵌入
    all_user_history = meta_loader.user_history_dict

    """2. 初始化 Recommender 模型"""
    print("Initializing model...")
    model = Recommender(
        data_config=meta_model_data,
        args_config=args,
        graph=meta_model_data["graph"],
        user_pre_embed=None,
        item_pre_embed=None,
    ).to(device)

    """3. 初始化 MemoryAwareTaskSampler"""
    print("Initializing task sampler...")
    # Sampler 需要模型的初始用户嵌入
    initial_user_embeds = model.all_embed[: meta_model_data["n_users"]].detach()
    sampler = MemoryAwareTaskSampler(
        n_users=meta_sampler_data["n_users"],
        user_graph_embeddings=initial_user_embeds,
        user_history_dict=meta_sampler_data["user_history_dict"],
        poi_category_map=meta_sampler_data["poi_category_map"],
        device=device,
    )

    """定义优化器"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    """元训练阶段"""
    if not args.use_meta_model:
        print("Start meta-training...")
        # 为模型设置交互矩阵
        model.interact_mat = build_interaction_matrix(
            meta_model_data["n_users"], meta_model_data["n_items"], all_user_history
        )

        for epoch in range(args.epoch):  # 使用主训练的 epoch 数
            model.train()
            total_meta_loss = 0.0
            n_batches = int(meta_model_data["n_users"] / args.meta_batch_size)

            with tqdm(
                total=n_batches, desc=f"Epoch {epoch+1}/{args.epoch}", unit="batch"
            ) as pbar:
                for _ in range(n_batches):
                    # 1. 使用新的采样器采样一批任务（用户）
                    sampled_users = sampler.sample_tasks(
                        batch_size=args.meta_batch_size
                    )

                    # 2. 为采样的用户构建 support 和 query batch
                    support_batch = build_batch(
                        sampled_users,
                        support_tasks,
                        all_user_history,
                        meta_model_data["n_items"],
                        args.max_history_len,
                    )
                    query_batch = build_batch(
                        sampled_users,
                        query_tasks,
                        all_user_history,
                        meta_model_data["n_items"],
                        args.max_history_len,
                    )

                    if (
                        not support_batch["users"].numel()
                        or not query_batch["users"].numel()
                    ):
                        continue

                    # 3. 执行元学习更新
                    optimizer.zero_grad()
                    meta_loss = model.forward_meta(support_batch, query_batch)
                    meta_loss.backward()
                    optimizer.step()

                    total_meta_loss += meta_loss.item()
                    pbar.set_postfix(loss=meta_loss.item())
                    pbar.update(1)

            print(
                f"Epoch {epoch+1} finished. Average meta-loss: {total_meta_loss / n_batches:.4f}"
            )

        if args.save:
            print("Saving meta-trained model...")
            torch.save(
                model.state_dict(),
                args.out_dir + "meta_model_" + args.dataset + ".ckpt",
            )

    else:
        print("Loading pre-trained meta-model...")
        model.load_state_dict(
            torch.load("./model_para/meta_model_{}.ckpt".format(args.dataset))
        )

    """微调和评估阶段"""
    print(f"Start fine-tuning and evaluation on '{args.cold_scenario}' scenario...")
    # 为模型设置冷启动场景的交互矩阵
    model.interact_mat = build_interaction_matrix(
        cold_model_data["n_users"],
        cold_model_data["n_items"],
        cold_loader.user_history_dict,
    )

    # 重设优化器学习率
    for g in optimizer.param_groups:
        g["lr"] = args.fine_tune_lr

    cur_best_pre_0 = 0
    stopping_step = 0

    # 微调用户是冷启动场景中的所有用户
    finetune_users = list(cold_support_tasks.keys())

    for epoch in range(args.fine_tune_epoch):
        model.train()
        total_finetune_loss = 0.0

        # 打乱微调用户
        random.shuffle(finetune_users)
        n_batches = int(len(finetune_users) / args.fine_tune_batch_size)

        with tqdm(
            total=n_batches,
            desc=f"Fine-tune Epoch {epoch+1}/{args.fine_tune_epoch}",
            unit="batch",
        ) as pbar:
            for i in range(n_batches):
                batch_users = finetune_users[
                    i * args.fine_tune_batch_size : (i + 1) * args.fine_tune_batch_size
                ]

                # 在冷启动场景的 support set 上进行微调
                finetune_batch = build_batch(
                    batch_users,
                    cold_support_tasks,
                    all_user_history,
                    cold_model_data["n_items"],
                    args.max_history_len,
                )

                if not finetune_batch["users"].numel():
                    continue

                optimizer.zero_grad()
                # 直接调用 forward 进行标准训练
                loss = model.forward(finetune_batch)
                loss.backward()
                optimizer.step()

                total_finetune_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

        print(
            f"Fine-tune Epoch {epoch+1} finished. Average loss: {total_finetune_loss / n_batches:.4f}"
        )

        # 每隔一定周期进行评估
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print("Evaluating...")
            model.eval()
            with torch.no_grad():
                # 准备评估所需的所有用户历史数据
                # 注意：这里的 all_user_history 需要是一个 tensor
                # 我们需要一个函数来将字典转换为填充好的 tensor
                history_list = []
                for u in range(cold_model_data["n_users"]):
                    h = all_user_history.get(u, [])
                    if len(h) > args.max_history_len:
                        h = h[-args.max_history_len :]
                    else:
                        h = h + [cold_model_data["n_items"]] * (
                            args.max_history_len - len(h)
                        )
                    history_list.append(h)
                all_history_tensor = torch.LongTensor(history_list).to(device)

                # 在冷启动场景的 query set (即测试集) 上进行评估
                ret = test(
                    model,
                    cold_support_tasks,
                    cold_query_tasks,
                    cold_model_data,
                    all_history_tensor,
                )

            print(
                f"Epoch {epoch+1} Test Results: Recall@{args.K}={ret['recall']}, NDCG@{args.K}={ret['ndcg']}"
            )

            cur_best_pre_0, stopping_step, should_stop = early_stopping(
                ret["recall"][0],
                cur_best_pre_0,
                stopping_step,
                expected_order="acc",
                flag_step=10,
            )
            if should_stop:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    print(f"Finished. Best Recall@20: {cur_best_pre_0:.4f}")
