import json
import os
from collections import Counter

input_dir = "output_results1"  # 銜接merged_content的output
output_file = "all_combined_emotion_stats.json"  # 合併後的輸出文件名

# 用於存儲所有文件的情緒統計
total_emotion_counter = Counter()
all_file_stats = {}

for file_name in os.listdir(input_dir):
    if file_name.endswith(".json"):
        input_file_path = os.path.join(input_dir, file_name)

        with open(input_file_path, "r", encoding="utf-8-sig") as file:
            comments = json.load(file)

            # 統計各情緒出現次數
            file_emotion_counter = Counter()
            for comment in comments:
                emotions = comment.get("emotions", [])
                file_emotion_counter.update(emotions)

            # 將當前文件的統計結果加入總計數器
            total_emotion_counter.update(file_emotion_counter)

            # 對當前文件的情緒統計進行排序
            sorted_file_emotions = dict(sorted(file_emotion_counter.items(), key=lambda x: x[1]))
            all_file_stats[file_name] = sorted_file_emotions

# 準備輸出數據
output_data = {
    "file_stats": all_file_stats,
    "total_stats": dict(sorted(total_emotion_counter.items(), key=lambda x: x[1]))
}

# 將結果寫入輸出文件
output_file_path = output_file
with open(output_file_path, "w", encoding="utf-8") as out_file:
    json.dump(output_data, out_file, ensure_ascii=False, indent=4)

print("所有檔案處理完成。")
print(f"合併結果已保存至: {output_file_path}")