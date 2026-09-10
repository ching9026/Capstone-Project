import json
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from torch import nn
from tqdm import tqdm
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 設定中文字體
font = FontProperties(fname="nesscary/Noto_Sans_TC/NotoSansTC-VariableFont_wght.ttf")

# 定義情緒類別（包含"看不出情緒"）
EMOTION_CATEGORIES = ["愉悅", "悲傷", "憤怒", "厭惡", "恐懼", "看不出情緒"]


class EarlyStopping:
    """Early stopping 機制，用於防止過擬合"""

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


def read_json_file(file_path):
    """讀取JSON文件並返回訓練和測試數據"""
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        data = json.load(file)

    train_data = []
    test_data = []

    # 處理訓練數據
    for comment_id, comment_data in data['train_data'].items():
        train_data.append({
            'id': comment_id,
            'comment': comment_data['comment'],
            'emotions': comment_data['emotions']
        })

    # 處理測試數據
    if 'test_data' in data:
        for comment_id, comment_data in data['test_data'].items():
            test_data.append({
                'id': comment_id,
                'comment': comment_data['comment'],
                'emotions': comment_data['emotions']
            })

    return train_data, test_data


def process_data(data, shuffle=True):
    """處理數據，生成統一的情緒分類數據集"""
    processed_data = []

    for item in data:
        # 檢查是否包含指定的情緒類別
        specific_emotions = [e for e in item['emotions'] if e in EMOTION_CATEGORIES[:5]]  # 排除"看不出情緒"

        if len(specific_emotions) > 0:
            emotion = specific_emotions[0]  # 使用第一個找到的特定情緒
        else:
            emotion = "看不出情緒"

        processed_data.append({
            'comment': item['comment'],
            'emotion': emotion,
            'id': item['id']
        })

    # 轉換為DataFrame並去重
    df = pd.DataFrame(processed_data).drop_duplicates(subset=['comment'])

    # 如果需要打亂數據
    if shuffle:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


class BERTClassifier(nn.Module):
    """基於BERT的分類器"""

    def __init__(self, output_dim, pretrained_name='bert-base-chinese'):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_name)
        self.mlp = nn.Linear(768, output_dim)

    def forward(self, tokens_X):
        res = self.bert(**tokens_X)
        return self.mlp(res[1])


def evaluate(net, comments_data, labels_data, device, batch_size, tokenizer, loss_fn=None):
    """評估模型性能"""
    net.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    batch_count = 0

    with torch.no_grad():
        for i in range(0, len(comments_data), batch_size):
            batch_comments = comments_data[i: min(i + batch_size, len(comments_data))]
            batch_labels = labels_data[i: min(i + batch_size, len(comments_data))]

            tokens_X = tokenizer(batch_comments, padding=True, truncation=True, return_tensors='pt').to(device)
            y = torch.tensor(batch_labels).to(device)

            res = net(tokens_X)
            pred = res.argmax(axis=1)

            if loss_fn is not None:
                loss = loss_fn(res, y)
                total_loss += loss.item()
                batch_count += 1

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / batch_count if batch_count > 0 else None
    return all_preds, all_labels, avg_loss


def train_model(net, tokenizer, loss_fn, optimizer, train_comments, train_labels,
                test_comments, test_labels, device, epochs, all_labels, model_save_path, batch_size=16):
    """訓練模型"""
    best_f1 = 0
    best_model_state = None
    train_losses = []
    test_losses = []
    early_stopping = EarlyStopping(patience=10, min_delta=1e-4)

    for epoch in range(epochs):
        net.train()
        total_loss = 0
        batch_count = 0

        progress_bar = tqdm(range(0, len(train_comments), batch_size), desc=f"Epoch {epoch + 1}/{epochs}")
        for i in progress_bar:
            batch_comments = train_comments[i: min(i + batch_size, len(train_comments))]
            batch_labels = train_labels[i: min(i + batch_size, len(train_comments))]

            tokens_X = tokenizer(batch_comments, padding=True, truncation=True, return_tensors='pt').to(device)
            y = torch.tensor(batch_labels).to(device)

            optimizer.zero_grad()
            res = net(tokens_X)
            loss = loss_fn(res, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1
            progress_bar.set_postfix({'train_loss': total_loss / batch_count})

        avg_train_loss = total_loss / batch_count
        train_losses.append(avg_train_loss)

        test_preds, test_true, test_loss = evaluate(net, test_comments, test_labels, device, batch_size, tokenizer,
                                                    loss_fn)
        test_losses.append(test_loss)

        # 計算F1分數並保存最佳模型
        test_report = classification_report(test_true, test_preds, target_names=all_labels, output_dict=True)
        current_f1 = test_report['weighted avg']['f1-score']

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_model_state = net.state_dict()
            torch.save(best_model_state, model_save_path)
            print(f"Saved best model with F1 score: {best_f1:.4f}")

        print(
            f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}, F1: {current_f1:.4f}")

        if early_stopping(test_loss):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    net.load_state_dict(best_model_state)
    return net, train_losses, test_losses


def plot_distribution(train_data, test_data, filename):
    """繪製數據分布圖"""
    plt.figure(figsize=(15, 6))

    # 訓練集分布
    plt.subplot(1, 2, 1)
    train_counts = train_data['emotion'].value_counts()
    train_bars = plt.bar(range(len(train_counts)), train_counts.values, alpha=0.8)

    for bar in train_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom', fontproperties=font)

    plt.xlabel('情緒類別', fontproperties=font)
    plt.ylabel('數量', fontproperties=font)
    plt.title('訓練集分布', fontproperties=font)
    plt.xticks(range(len(train_counts)), train_counts.index, rotation=45, ha='right', fontproperties=font)

    # 測試集分布
    plt.subplot(1, 2, 2)
    test_counts = test_data['emotion'].value_counts()
    test_bars = plt.bar(range(len(test_counts)), test_counts.values, alpha=0.8)

    for bar in test_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom', fontproperties=font)

    plt.xlabel('情緒類別', fontproperties=font)
    plt.ylabel('數量', fontproperties=font)
    plt.title('測試集分布', fontproperties=font)
    plt.xticks(range(len(test_counts)), test_counts.index, rotation=45, ha='right', fontproperties=font)

    plt.suptitle('情緒分類數據分布', fontproperties=font, fontsize=16)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_training_history(train_losses, test_losses, filename):
    """繪製訓練歷史圖"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-', label='訓練損失', alpha=0.7)
    plt.plot(epochs, test_losses, 'r-', label='測試損失', alpha=0.7)

    plt.xlabel('訓練週期', fontproperties=font)
    plt.ylabel('損失值', fontproperties=font)
    plt.title('訓練和測試損失歷史', fontproperties=font)
    plt.legend(prop=font)
    plt.grid(True)

    plt.savefig(filename)
    plt.close()


def main():
    """主函數，執行情緒分類流程"""
    # 讀取數據
    file_path = '結果/ptt/train_test_split_ptt(調整後).json'
    raw_train_data, raw_test_data = read_json_file(file_path)

    # 處理數據
    train_data = process_data(raw_train_data, shuffle=True)
    test_data = process_data(raw_test_data, shuffle=True)

    # 打印數據統計
    print("數據統計：")
    print(f"訓練集大小: {len(train_data)}")
    print(f"測試集大小: {len(test_data)}")
    print("\n訓練集情緒分布：")
    print(train_data['emotion'].value_counts())

    # 繪製數據分布圖
    plot_distribution(train_data, test_data, 'emotion_distribution.png')

    # 設置設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    # 初始化分詞器
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # 準備訓練數據
    label_to_id = {label: i for i, label in enumerate(EMOTION_CATEGORIES)}

    train_comments = list(train_data['comment'])
    test_comments = list(test_data['comment'])
    train_labels = [label_to_id[label] for label in train_data['emotion']]
    test_labels = [label_to_id[label] for label in test_data['emotion']]

    # 初始化和訓練模型
    classifier = BERTClassifier(output_dim=len(EMOTION_CATEGORIES)).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    classifier, train_losses, test_losses = train_model(
        classifier, tokenizer, loss_fn, optimizer,
        train_comments, train_labels,
        test_comments, test_labels,
        device, epochs=400, all_labels=EMOTION_CATEGORIES,
        model_save_path='best_emotion_classifier.pth',
        batch_size=16
    )

    # 繪製訓練歷史
    plot_training_history(train_losses, test_losses, 'training_history.png')

    # 評估並保存結果
    classifier.load_state_dict(torch.load('best_emotion_classifier.pth', map_location=device))
    preds, true_labels, _ = evaluate(classifier, test_comments, test_labels, device, 64, tokenizer)

    # 保存分類報告
    with open('classification_results.txt', 'w', encoding='utf-8') as f:
        f.write("情緒分類結果：\n")
        f.write(classification_report(true_labels, preds, target_names=EMOTION_CATEGORIES, digits=4))


if __name__ == "__main__":
    main()