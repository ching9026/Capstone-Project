# 🎬 PTT Movie Review Emotion Analysis

## PTT 電影觀後感情緒分析

以 PTT 電影相關討論為資料來源，建立一套從 **網路爬蟲、資料前處理、人工標記、中文斷詞、資料切分，到 BERT 情緒分類模型訓練與評估** 的完整 NLP Pipeline。

本專題主要研究如何從 PTT 使用者留言中辨識情緒，並比較不同情緒分類策略，包括：

* **直接六分類（Direct 6-Class Classification）**
* **二階段情緒分類（Two-Stage Emotion Classification）**

---

## 📌 Project Overview

PTT 留言通常包含大量：

* 網路用語
* 簡短句子
* 推文與連續發言
* URL
* 重複字元
* 無明顯情緒的內容
* 諷刺或非自身情緒描述

因此本專題建立完整的資料處理與模型訓練流程，將原始 PTT 留言轉換成可供深度學習模型使用的情緒分類資料。

整體流程：

```text
PTT
 │
 ▼
PTT Web Crawler
 │
 ▼
Raw JSON
 │
 ▼
Comment Preprocessing
 │
 ├─ 合併留言
 ├─ 建立留言 ID
 ├─ 移除 URL
 ├─ 移除洗版文字
 └─ 句子切割
 │
 ▼
Manual Emotion Annotation
 │
 ▼
CKIP Word Segmentation
 │
 ▼
Train / Test Partition
 │
 ▼
BERT Emotion Classification
 │
 ├─ Direct 6-Class
 │
 └─ Two-Stage Classification
 │
 ▼
Evaluation
```

---

# ✨ Main Features

## 1. PTT Web Crawler

使用修改過的 PTT Web Crawler 蒐集文章與留言資料。

主要取得：

* Article ID
* Article title
* Board
* Author
* Push / Boo / Neutral
* User comment
* Comment timestamp

本專題中的 crawler 參考並修改自：

https://github.com/jwlin/ptt-web-crawler

相關程式：

```text
ptt爬蟲/
```

---

# 2. Data Preprocessing

PTT 原始留言在送入模型前會經過多階段前處理。

相關程式：

```text
資料處理/
├── 合併發言後增加id.py
├── 將句子優先排序.py
├── 將發言句子切割.py
├── 將電影重新命名.py
├── 標記.py
├── 相近時間發言合併.py
└── 統計總情緒.py
```

### Sentence Cleaning

資料前處理包含：

* 移除 HTTP / HTTPS URL
* 移除大量重複字元
* 依標點符號切割留言
* 處理 PTT 使用者的連續留言
* 建立 comment ID
* 移除重複資料

句子切割優先順序：

```text
。？！.
   ↓
，；：,;
   ↓
Whitespace
```

藉此將較長的 PTT 留言拆分成適合情緒分析的文字單位。

---

# 3. Manual Emotion Annotation Tool

本專題自行利用 **Pygame** 製作情緒資料標記 GUI。

可以人工閱讀 PTT 留言並點擊對應的情緒標籤。

標記類別包含：

| 中文    | Emotion           |
| ----- | ----------------- |
| 愉悅    | Joy               |
| 悲傷    | Sadness           |
| 憤怒    | Anger             |
| 厭惡    | Disgust           |
| 恐懼    | Fear              |
| 諷刺    | Satire            |
| 看不出情緒 | Neutral / Unclear |
| 非自我情緒 | Non-self Emotion  |

相關程式：

```text
資料處理/標記.py
```

GUI 同時支援：

* 上一頁 / 下一頁
* 每頁顯示多筆留言
* 多標籤紀錄
* 保存標記進度
* 從上次標記位置繼續

---

# 4. Chinese Word Segmentation

部分資料處理流程使用 **CKIPTagger** 進行繁體中文斷詞。

相關程式：

```text
訓練/斷詞.py
```

CKIP 斷詞後：

```json
{
    "comment": "這部電影真的很好看",
    "emotions": ["愉悅"],
    "segmented": [
        "這部",
        "電影",
        "真的",
        "很",
        "好看"
    ]
}
```

---

# 5. Similarity-aware Dataset Partition

為了整理訓練資料，本專題會先將同一情緒中的留言轉成詞頻向量，再計算留言間的文字相似度。

使用：

```text
Cosine Similarity
        +
Dice Similarity
        ↓
Average Similarity
```

程式會找出：

```text
similarity > 0.7
```

的留言組合。

接著將資料分配到 **10 個 partitions**：

```text
Partition 1
Partition 2
...
Partition 10
```

最後使用：

```text
Partition 1 ~ 9 → Training Set
Partition 10    → Test Set
```

形成約：

```text
Training : Test
    9    :   1
```

的資料切分。

相關程式：

```text
訓練/訓練集分割.py
```

同時會輸出：

```text
partitions_ptt_info.xlsx
similarities_results/
train_test_split_ptt.json
```

方便分析各 partition 的資料與情緒分布。

---

# 🧠 Emotion Classification Models

本專題使用：

```text
bert-base-chinese
```

作為文字 Encoder。

基本架構：

```text
PTT Comment
      │
      ▼
BERT Tokenizer
      │
      ▼
Chinese BERT
      │
      ▼
768-dimensional Representation
      │
      ▼
Linear Layer
      │
      ▼
Emotion Prediction
```

---

# Model 1 — Direct Six-Class Classification

第一種方法直接將留言分類為六種情緒：

```text
愉悅
悲傷
憤怒
厭惡
恐懼
看不出情緒
```

模型：

```text
Chinese BERT
     │
     ▼
Linear(768 → 6)
     │
     ▼
6-Class Emotion
```

相關程式：

```text
訓練/直接六分類訓練.py
```

---

# Model 2 — Two-Stage Emotion Classification

另一個方法採用兩階段分類。

## Stage 1 — Emotion Detection

先判斷留言是否具有可辨識的情緒：

```text
Comment
   │
   ▼
BERT
   │
   ▼
有情緒 / 看不出情緒
```

---

## Stage 2 — Emotion Classification

若 Stage 1 判斷為：

```text
有情緒
```

則再進一步分類為：

```text
愉悅
悲傷
憤怒
厭惡
恐懼
```

完整架構：

```text
                    ┌─────────────┐
                    │ PTT Comment │
                    └──────┬──────┘
                           │
                           ▼
                     ┌──────────┐
                     │   BERT   │
                     └────┬─────┘
                          │
                          ▼
              ┌─────────────────────┐
              │ Emotion / No Emotion│
              └──────────┬──────────┘
                         │
             ┌───────────┴──────────┐
             │                      │
             ▼                      ▼
       No Emotion                Emotion
                                    │
                                    ▼
                               ┌─────────┐
                               │  BERT   │
                               └────┬────┘
                                    │
                                    ▼
                        ┌─────────────────────┐
                        │ Joy                │
                        │ Sadness            │
                        │ Anger              │
                        │ Disgust            │
                        │ Fear               │
                        └─────────────────────┘
```

相關程式：

```text
訓練/兩階段判斷.py
訓練/兩階段判斷合併.py
```

---

# 🏋️ Training Strategy

訓練流程包含：

* Adam Optimizer
* Cross Entropy Loss
* Early Stopping
* GPU / CPU 自動偵測
* Best Model Saving
* Weighted F1 Score evaluation

主要設定：

```text
Pretrained Model : bert-base-chinese
Learning Rate    : 1e-5
Batch Size       : 16
Max Epoch        : 400
Early Stopping   : patience = 10
```

模型會根據 **Weighted F1 Score** 保存最佳權重。

---

# 📊 Evaluation

模型評估使用：

```python
sklearn.metrics.classification_report
```

包含：

* Precision
* Recall
* F1-score
* Support
* Weighted Average

訓練完成後會產生：

```text
classification_results.txt

best_binary_classifier.pth
best_emotion_classifier.pth

binary_training_history.png
emotion_training_history.png
```

---

# 📂 Repository Structure

```text
Capstone-Project/
│
├── README.md
│
├── 專題報告.docx
├── 專題海報.pdf
├── 電影觀後感情緒分析ppt.pptx
│
├── ptt爬蟲/
│   ├── PttWebCrawler/
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   └── crawler.py
│   ├── requirements.txt
│   ├── setup.py
│   └── test.py
│
├── 資料處理/
│   ├── 合併發言後增加id.py
│   ├── 將句子優先排序.py
│   ├── 將發言句子切割.py
│   ├── 將電影重新命名.py
│   ├── 標記.py
│   ├── 相近時間發言合併.py
│   └── 統計總情緒.py
│
└── 訓練/
    ├── 人工調整partition後合併.py
    ├── 兩階段判斷.py
    ├── 兩階段判斷合併.py
    ├── 合併所有電影資料.py
    ├── 斷詞.py
    ├── 直接六分類訓練.py
    └── 訓練集分割.py
```

---

# ⚙️ Environment Setup

Clone repository：

```bash
git clone https://github.com/ching9026/Capstone-Project.git
cd Capstone-Project
```

建議建立 Python virtual environment：

```bash
python -m venv .venv
```

Windows：

```bash
.venv\Scripts\activate
```

Linux / macOS：

```bash
source .venv/bin/activate
```

安裝套件：

```bash
pip install -r requirements.txt
```

主要 Python dependencies：

```text
torch
transformers
pandas
scikit-learn
matplotlib
tqdm
pygame
ckiptagger
beautifulsoup4
requests
openpyxl
```

---

# 🔤 CKIPTagger Setup

斷詞流程需要下載 CKIPTagger model data。

程式預設路徑：

```text
nesscary/data/
```

另外，為了正常顯示繁體中文字，需要準備：

```text
Noto Sans TC
```

例如：

```text
nesscary/
├── data/
│   └── ...
│
└── Noto_Sans_TC/
    └── NotoSansTC-VariableFont_wght.ttf
```

如果放置在其他位置，請修改程式中的：

```python
FontProperties(...)
```

或：

```python
pygame.font.Font(...)
```

路徑。

---

# 🚀 Example Workflow

完整實驗可概念化為：

```text
1. Crawl PTT
        ↓
2. Merge Comments
        ↓
3. Assign Comment ID
        ↓
4. Clean / Split Comments
        ↓
5. Manual Emotion Annotation
        ↓
6. CKIP Word Segmentation
        ↓
7. Dataset Partition
        ↓
8. Train BERT
        ↓
9. Evaluate Model
```

由於目前部分 script 使用實驗時的相對路徑，實際執行前請確認各 Python 檔案中的：

```python
input_dir
output_dir
file_path
```

是否符合本機資料位置。

---

# 📄 Project Documents

Repository 中亦提供完整專題資料：

### 專題報告

```text
專題報告.docx
```

### 專題海報

```text
專題海報.pdf
```

### Project Presentation

```text
電影觀後感情緒分析ppt.pptx
```

---

# 🛠️ Tech Stack

| Category         | Technology                |
| ---------------- | ------------------------- |
| Language         | Python                    |
| Deep Learning    | PyTorch                   |
| NLP Model        | BERT                      |
| Transformers     | Hugging Face Transformers |
| Chinese NLP      | CKIPTagger                |
| Data Processing  | Pandas                    |
| Machine Learning | Scikit-learn              |
| Visualization    | Matplotlib                |
| Annotation GUI   | Pygame                    |
| Web Crawling     | BeautifulSoup / Requests  |
| Data Format      | JSON                      |

---

# 🔮 Future Improvements

目前專案仍有一些可以進一步改善的方向：

1. 將 hard-coded file path 改為 command-line arguments。
2. 將資料處理流程整合成統一 Pipeline。
3. 建立 `config.yaml` 管理模型與資料路徑。
4. 增加 `requirements.txt` 與 reproducible environment。
5. 加入 confusion matrix。
6. 增加 Macro-F1 與 per-class F1 分析。
7. 比較不同中文 pretrained language models。
8. 建立 inference script，輸入一句留言即可預測情緒。
9. 建立 Streamlit / Gradio Demo。
10. 將訓練與推論程式模組化。

例如未來可加入：

```bash
python predict.py --text "這部電影真的超好看"
```

輸出：

```text
Prediction : 愉悅
Confidence : 0.94
```

---

# ⚠️ Notes

本 repository 中部分程式為專題研究與實驗過程所使用，因此不同 script 的輸入／輸出資料夾可能不同。

若希望重新執行完整 Pipeline，建議依據：

```text
Crawler
→ Preprocessing
→ Annotation
→ CKIP
→ Dataset Split
→ Training
→ Evaluation
```

依序整理資料。

---

# 🙏 Acknowledgements

PTT crawler 部分參考：

**jwlin/ptt-web-crawler**

https://github.com/jwlin/ptt-web-crawler

並依本專題需求進行修改，以取得與分析 PTT 電影相關留言資料。

---

# 📚 Project

**PTT Movie Review Emotion Analysis**

A capstone project exploring Traditional Chinese social-media emotion classification using:

**PTT + NLP + BERT + PyTorch**
