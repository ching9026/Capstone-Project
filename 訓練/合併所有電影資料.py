import os
import json

input_dir = "PTT"  # 替換為你的輸入目錄路徑
output_dir = "merge_after"  # 替換為你的輸出目錄路徑
output_file = "combined_output.json"  # 最終合併後的輸出文件名

# 確認輸出目錄是否存在，如果不存在則創建它
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

all_data = []  # 用於存儲所有文件的數據

# 遍歷輸入目錄中的所有檔案
for file_name in os.listdir(input_dir):
    if file_name.endswith(".json"):  # 僅處理 .json 檔案
        input_file_path = os.path.join(input_dir, file_name)  # 完整的輸入檔案路徑

        # 讀取並處理 JSON 檔案
        with open(input_file_path, "r", encoding="utf-8-sig") as file:
            json_data = json.load(file)  # 載入 JSON 資料

        # 直接將每個項目加入到總數據中
        for item in json_data:
                data_item = {
                    "id": item["id"],
                    "comment": item["comment"],
                    "emotions": item["emotions"]
                }
                all_data.append(data_item)

# 將所有資料寫入單一輸出文件
output_file_path = os.path.join(output_dir, output_file)
with open(output_file_path, "w", encoding="utf-8-sig") as file:
    json.dump(all_data, file, ensure_ascii=False, indent=4)

print(f"處理完成。所有數據已合併到 {output_file_path}")