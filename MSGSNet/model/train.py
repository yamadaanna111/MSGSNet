import os
import random
import torch
import argparse
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from MSGSNet import MultiScaleGraphSeqNet
from torchmetrics.classification import Accuracy, AUROC
from sklearn.metrics import average_precision_score
import pandas as pd
import platform
import cpuinfo
from tqdm.auto import tqdm
import yaml
import time


seed = 42
hidden_channels = 128
lr = 1e-4
KS = [5, 10, 15]  # 三个尺度，顺序固定


# ---------- seed / device ----------
def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def identify_device():
    so = platform.system()
    if so == "Darwin":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        dev_name = cpuinfo.get_cpu_info()["brand_raw"]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dev_name = torch.cuda.get_device_name() if device.type == "cuda" else cpuinfo.get_cpu_info()["brand_raw"]
    return device, dev_name


# ---------- dataset list ----------
def get_dataset_list(base_path="/mnt/share/wzy/GCBLANE/Datasets/165 ChIP-seq/"):
    fileslist_path = os.path.join(base_path, "FILESLIST.txt")
    with open(fileslist_path, 'r') as f:
        datasets = [line.strip() for line in f if line.strip()]
    valid = []
    for d in datasets:
        if os.path.isdir(os.path.join(base_path, d)):
            valid.append(d)
        else:
            print(f"Warning: dataset dir missing: {d}")
    return valid


# ---------- load saved pickle graphs for a single k ----------
def load_graph_pickle(k, dataset_name, split="train"):
    path = f"/mnt/share/wzy/GCBLANE/experiments/{dataset_name}/{k}_mer/{split}_dataset_{k}.pickle"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


# ---------- load SCE features ----------
def get_sce(dataset_name, Test=False):
    """
    从 .npz 文件中加载 SCE 序列特征
    返回：
        sce_features: list[np.ndarray]
        seq_labels: list[int]
    """
    split = "test" if Test else "train"
    path = f"/mnt/share/wzy/GCBLANE/experiments/{dataset_name}/SCE/{split}_sce.npz"  # 修改为npz文件

    if not os.path.exists(path):
        raise FileNotFoundError(f"SCE 文件不存在: {path}")

    # 加载npz文件
    sce_data = np.load(path, allow_pickle=True)
    X = sce_data['X']
    y = sce_data['y']

    #确保每个元素是统一的 float32 数组
    if X.dtype == np.object_:
        X = np.stack([np.array(x, dtype=np.float32) for x in X])
    else:
        X = X.astype(np.float32)

    y = np.array(y, dtype=np.int64)

    # 转为 list（与上层兼容）
    sce_features = [x for x in X]
    seq_labels = [int(label) for label in y]

    assert len(sce_features) == len(seq_labels)

    return sce_features, seq_labels


# ---------- 创建SCE数据对象 ----------
class SCEData(Data):
    """将SCE特征包装成PyG Data对象"""

    def __init__(self, sce_feature, y=None):
        super().__init__()
        self.sce_feature = torch.tensor(sce_feature, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor([y], dtype=torch.long)
        else:
            self.y = None

    def __inc__(self, key, value, *args, **kwargs):
        """确保batch时不会错误地增加索引"""
        if key == 'sce_feature':
            return 0
        return super().__inc__(key, value, *args, **kwargs)

# ---------- load multi-scale graphs and SCE features ----------
def load_multiscale_train_data(dataset_name):
    lists = {}
    lengths = []
    for k in KS:
        lst = load_graph_pickle(k, dataset_name, split="train")
        lists[k] = lst
        lengths.append(len(lst))

    # 加载SCE特征
    sce_features, sce_labels = get_sce(dataset_name, Test=False)
    lengths.append(len(sce_features))

    # sanity: lengths equal
    if not all(l == lengths[0] for l in lengths):
        raise ValueError("Different number of samples between ks and SCE for dataset %s: %s" % (dataset_name, lengths))
    num_samples = lengths[0]

    # ensure labels一致（以 k=10 的 labels 为准）
    labels = [int(d.y) for d in lists[10]]

    # 创建SCE数据对象列表
    sce_data_list = []
    for i in range(num_samples):
        sce_data = SCEData(sce_features[i], y=labels[i])
        sce_data_list.append(sce_data)

    for k in KS:
        for i, d in enumerate(lists[k]):
            d.y = int(labels[i])  # overwrite label to ensure alignment

    return lists[5], lists[10], lists[15], sce_features, labels, num_samples


def load_multiscale_test_data(dataset_name):
    lists = {}
    lengths = []
    for k in KS:
        lst = load_graph_pickle(k, dataset_name, split="test")
        lists[k] = lst
        lengths.append(len(lst))

    # 加载SCE特征
    sce_features, sce_labels = get_sce(dataset_name, Test=True)
    lengths.append(len(sce_features))

    if not all(l == lengths[0] for l in lengths):
        raise ValueError("Different number of samples between ks and SCE for dataset %s: %s" % (dataset_name, lengths))

    labels = [int(d.y) for d in lists[10]]

    # 创建SCE数据对象列表
    sce_data_list = []
    for i in range(len(sce_features)):
        sce_data = SCEData(sce_features[i], y=labels[i])
        sce_data_list.append(sce_data)

    for k in KS:
        for i, d in enumerate(lists[k]):
            d.y = int(labels[i])

    return lists[5], lists[10], lists[15], sce_features


# ---------- MultiScaleDataset wrapper (包含SCE) ----------
class MultiScaleDataset(torch.utils.data.Dataset):
    """
    Stores four parallel lists: graph data (k5,k10,k15) and SCE features
    __getitem__(i) returns tuple (data_k5_i, data_k10_i, data_k15_i, sce_feature_i)
    """

    def __init__(self, list5, list10, list15, sce_features):
        assert len(list5) == len(list10) == len(list15) == len(sce_features)
        self.l5 = list5
        self.l10 = list10
        self.l15 = list15
        self.sce_features = sce_features

    def __len__(self):
        return len(self.l5)

    def __getitem__(self, idx):
        sce_tensor = torch.tensor(self.sce_features[idx], dtype=torch.float32)
        return self.l5[idx], self.l10[idx], self.l15[idx], sce_tensor


# ---------- collate_fn: batch -> three batched graph objects + SCE tensor ----------
def multis_collate(batch):
    # batch: list of tuples (d5,d10,d15,sce)
    bat5 = Batch.from_data_list([t[0] for t in batch])
    bat10 = Batch.from_data_list([t[1] for t in batch])
    bat15 = Batch.from_data_list([t[2] for t in batch])
    sce_tensor = torch.stack([t[3] for t in batch], dim=0)
    return bat5, bat10, bat15, sce_tensor


# ---------- compute_metrics  ----------
def compute_metrics_local(y_true, y_pred, y_prob, nclasses, f):
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.from_numpy(np.array(y_true))
    if not isinstance(y_prob, torch.Tensor):
        y_prob = torch.from_numpy(np.array(y_prob))
    y_true = y_true.clone().detach().to(torch.long)
    y_prob_t = y_prob.clone().detach().to(torch.float32)
    accuracy = Accuracy(task="binary" if nclasses == 2 else "multiclass", num_classes=nclasses)
    roc = AUROC(task="binary" if nclasses == 2 else "multiclass", num_classes=nclasses)

    if nclasses == 2:
        prob_pos = y_prob_t[:, 1] if y_prob_t.ndim == 2 else y_prob_t
        acc = accuracy(torch.tensor(y_pred), y_true).item()
        roc_auc = roc(prob_pos, y_true).item()
        try:
            pr_auc = average_precision_score(y_true.cpu().numpy(), prob_pos.cpu().numpy())
        except Exception:
            pr_auc = 0.0
    else:
        acc = accuracy(torch.tensor(y_pred), y_true).item()
        roc_auc = roc(y_prob_t, y_true).item()
        pr_auc = 0.0
    print("Accuracy:", acc, "\nROC-AUC:", roc_auc, "\nPR-AUC:", pr_auc)
    return [f, acc, roc_auc, pr_auc]


# ---------- save_metrics / save_report ----------
def save_metrics(metrics, k, dataset_name, test=False):
    if not metrics:
        print("Warning: No metrics to save!")
        return

    prefix = "test" if test else "train"
    columns = ["Fold", "Accuracy", "ROC-AUC", "PR-AUC"]

    if isinstance(metrics[0], (int, float, str)):
        metrics = [metrics]

    data = pd.DataFrame(metrics, columns=columns)
    path = f"../experiments/{dataset_name}/{prefix}_metrics_multiscale_sce_{k}.csv"  # 修改文件名
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data.to_csv(path, index=False)

    yaml_data = {}
    for metric in ["Accuracy", "ROC-AUC", "PR-AUC"]:
        values = data[metric].values
        avg = np.mean(values)
        sd = np.std(values)
        yaml_data[metric] = {"Mean": float(avg), "Standard Deviation": float(sd)}

    path_yaml = f"../experiments/{dataset_name}/{prefix}_results_multiscale_sce_{k}.yaml"  # 修改文件名
    with open(path_yaml, "w") as f:
        yaml.dump(yaml_data, f)

    if test:
        eval_dir = f"../experiments/{dataset_name}"
        os.makedirs(eval_dir, exist_ok=True)
        eval_path = os.path.join(eval_dir, "run2.csv")  # 修改文件名

        df = pd.DataFrame({
            "Dataset": [dataset_name],
            "Accuracy": [yaml_data["Accuracy"]["Mean"]],
            "ROC-AUC": [yaml_data["ROC-AUC"]["Mean"]],
            "PR-AUC": [yaml_data["PR-AUC"]["Mean"]]
        })

        # if os.path.exists(eval_path):
        #     df_old = pd.read_csv(eval_path)
        #     df = pd.concat([df_old, df], ignore_index=True)

        df.to_csv(eval_path, index=False)
        print(f"\n测试集平均结果已保存至: {eval_path}")

    print(f"\n===== {prefix.upper()} SET RESULTS =====")
    for metric, stats in yaml_data.items():
        print(f"AVG. {metric} = {stats['Mean']:.4f} SD = {stats['Standard Deviation']:.4f}")


def save_report(train_time, test_time, k, epochs, devname, dataset_name):
    path = f"../experiments/{dataset_name}/times_multiscale_sce_{k}.csv"  # 修改文件名
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame([(1, float(train_time), float(test_time))], columns=["Fold", "Training time", "Testing time"])
    df.to_csv(path, index=False)
    yaml_data = {
        "Device": devname,
        "Training Time": {"Mean": float(df["Training time"].mean()), "SD": float(df["Training time"].std())},
        "Testing Time": {"Mean": float(df["Testing time"].mean()), "SD": float(df["Testing time"].std())},
        "Epochs": int(epochs)
    }
    with open(f"../experiments/{dataset_name}/times_avg_multiscale_sce_{k}.yaml", "w") as f:  # 修改文件名
        yaml.dump(yaml_data, f)


# ---------- training / predict ----------
def train_net(device, net, trainloader, valloader, epochs, dataset_name, lr, weight_decay=1e-5):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    net = net.to(device)

    patience = 20
    best_val_auc = 0.0
    best_val_aupr = 0.0
    best_score = 0.0
    best_state = None
    early_cnt = 0

    start = time.time()
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0.0
        step = 0
        for bat5, bat10, bat15, sce_x in trainloader:  # 修改：接收SCE输入
            bat5 = bat5.to(device)
            bat10 = bat10.to(device)
            bat15 = bat15.to(device)
            sce_x = sce_x.to(device)  # 添加SCE数据到设备

            logits = net(bat5, bat10, bat15, sce_x)

            labels = bat5.y
            loss = criterion(logits, labels)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step += 1
        epoch_loss /= max(1, step)

        # validation
        net.eval()
        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for bat5, bat10, bat15, sce_x in valloader:  # 修改：接收SCE输入
                bat5 = bat5.to(device)
                bat10 = bat10.to(device)
                bat15 = bat15.to(device)
                sce_x = sce_x.to(device)  # 添加SCE数据到设备

                # 修改：验证时不需要对比损失
                logits = net(bat5, bat10, bat15, sce_x)
                proba = torch.softmax(logits, dim=1).cpu().numpy()
                pred = logits.argmax(dim=1).cpu().numpy()
                labels = bat5.y.cpu().numpy()
                y_true.extend(labels.tolist())
                y_pred.extend(pred.tolist())
                y_prob.extend(proba.tolist())

        val_metrics = compute_metrics_local(y_true, y_pred, np.array(y_prob), 2, "val")
        val_roc_auc = val_metrics[2]
        val_pr_auc = val_metrics[3]
        score = val_roc_auc + val_pr_auc
        print(
            f"Epoch {epoch + 1}/{epochs} | Loss {epoch_loss:.4f} | Val ROC {val_roc_auc:.4f} PR {val_pr_auc:.4f} Score {score:.4f}")

        if score > best_score:
            best_score = score
            best_val_auc = val_roc_auc
            best_val_aupr = val_pr_auc
            best_state = net.state_dict().copy()
            early_cnt = 0
            os.makedirs(f"../experiments/{dataset_name}/model", exist_ok=True)
            torch.save({
                'model_state_dict': best_state,
                'epoch': epoch,
                'score': best_score,
                'best_val_auc': best_val_auc,
                'best_val_aupr': best_val_aupr,
            }, f"../experiments/{dataset_name}/model/best_model.pt")
        else:
            early_cnt += 1
        if early_cnt >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}, best ROC-AUC: {best_val_auc:.4f}, best PR-AUC: {best_val_aupr:.4f}")
            break

    end = time.time()
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, end - start, val_roc_auc, val_pr_auc, best_score


def predict(device, net, loader, return_repr=False):
    net.eval()
    y_true, y_pred, y_prob = [], [], []
    z_all = []

    start = time.time()
    with torch.no_grad():
        for bat5, bat10, bat15, sce_x in loader:  # 修改：接收SCE输入
            bat5 = bat5.to(device)
            bat10 = bat10.to(device)
            bat15 = bat15.to(device)
            sce_x = sce_x.to(device)  # 添加SCE数据到设备
            logits = net(bat5, bat10, bat15, sce_x)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            pred = logits.argmax(dim=1).cpu().numpy()
            labels = bat5.y.cpu().numpy()

            y_true.extend(labels.tolist())
            y_pred.extend(pred.tolist())
            y_prob.extend(proba.tolist())

    elapsed = time.time() - start
    return np.array(y_true), np.array(y_pred), np.array(y_prob), elapsed


# ---------- main per-dataset processing ----------
def process_single_dataset(device, devname, dataset_name, epochs, test_size=0.2):
    print(f"\nProcessing dataset: {dataset_name}")
    try:
        # load multiscale train lists and SCE features
        list5, list10, list15, sce_features, labels, num_samples = load_multiscale_train_data(dataset_name)

        # dynamic batch size
        if num_samples < 20000:
            batch_size = 64
        elif num_samples < 60000:
            batch_size = 128
        else:
            batch_size = 256

        print(f"Dataset {dataset_name} has {num_samples} samples, using batch size {batch_size}")

        nclasses = 2  # binary
        test_size = 0.2

        # split indices
        train_idx, val_idx = train_test_split(range(num_samples), test_size=test_size, stratify=labels,
                                              random_state=seed)

        # create datasets and loaders (包含SCE)
        train_ds = MultiScaleDataset([list5[i] for i in train_idx],
                                     [list10[i] for i in train_idx],
                                     [list15[i] for i in train_idx],
                                     [sce_features[i] for i in train_idx])
        val_ds = MultiScaleDataset([list5[i] for i in val_idx],
                                   [list10[i] for i in val_idx],
                                   [list15[i] for i in val_idx],
                                   [sce_features[i] for i in val_idx])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=multis_collate)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=multis_collate)

        # init model
        in5 = list5[0].x.shape[1]
        in10 = list10[0].x.shape[1]
        in15 = list15[0].x.shape[1]

        # 创建新模型
        net = MultiScaleGraphSeqNet(
            in_channels_list=[in5, in10, in15],
            hidden_channels=hidden_channels,
            seq_input_dim=21,  # SCE特征维度
            seq_cnn_hidden=128,
            seq_lstm_hidden=64,
            fusion_embed_dim=256,
            fusion_heads=8,
            num_classes=2,
            dropout=0.5,
        )
        net.to(device)

        # train
        net, train_time, best_roc_auc, best_pr_auc, best_score = train_net(device, net, train_loader, val_loader,
                                                                           epochs, dataset_name, lr)

        # validation metrics
        y_true, y_pred, y_prob, _ = predict(device, net, val_loader)
        train_metrics = compute_metrics_local(y_true, y_pred, y_prob, 2, "Train")

        # load test set
        test5, test10, test15, test_sce = load_multiscale_test_data(dataset_name)
        test_ds = MultiScaleDataset(test5, test10, test15, test_sce)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=multis_collate)

        y_true, y_pred, y_prob, test_time = predict(device, net, test_loader)
        test_metrics = compute_metrics_local(y_true, y_pred, y_prob, 2, "Test")

        # save metrics and report
        save_metrics([train_metrics], k=KS[1], dataset_name=dataset_name, test=False)
        save_metrics([test_metrics], k=KS[1], dataset_name=dataset_name, test=True)
        save_report(train_time, test_time, k=KS[1], epochs=epochs, devname=devname, dataset_name=dataset_name)

        print(f"Dataset {dataset_name} done. Test ROC {test_metrics[2]:.4f} PR {test_metrics[3]:.4f}")

    except Exception as e:
        print(f"Error processing {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        os.makedirs(f"../experiments/{dataset_name}", exist_ok=True)
        with open(f"../experiments/{dataset_name}/error_log.txt", "w") as f:
            f.write(str(e))


# ---------- main ----------
def main():
    set_seed(seed)
    device, devname = identify_device()
    print("Using %s - %s" % (device, devname))

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, required=True, help="Number of epochs")
    args = parser.parse_args()

    epochs = args.epochs

    dataset_list = get_dataset_list()
    print(f"Found {len(dataset_list)} datasets")

    for ds in tqdm(dataset_list, desc="Datasets"):
        process_single_dataset(device, devname, ds, epochs)


if __name__ == "__main__":

    main()
