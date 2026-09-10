import json
import os

input_dir = "結果檔/將發言句子切割"  # 銜接merged_content的output
output_dir = "結果檔/將發言句子排序優先"  # 拿來分析各情緒字的統計，並開始標記

def load_mood_words(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        return json.load(file)

def calculate_emotion_scores(comment, mood_words):
    scores = {emotion: 0 for emotion in mood_words.keys()}

    for emotion, words in mood_words.items():
        for word in words:
            if word in comment:
                scores[emotion] += 1

    return scores

def rank_comments_by_emotion(comments, mood_words):
    ranked_comments = {}
    for emotion in mood_words.keys():
        scored_comments = []
        for comment in comments:
            scores = calculate_emotion_scores(comment, mood_words)
            total_score = scores[emotion]
            scored_comments.append((comment, total_score, scores))

        # Sort comments by total score in descending order
        sorted_comments = sorted(scored_comments, key=lambda x: -x[1])
        ranked_comments[emotion] = [comment for comment, _, _ in sorted_comments]

    return ranked_comments

def reorganize_comments(ranked_comments):
    emotions = ['joy', 'sadness', 'anger', 'fear', 'disgust']
    reorganized = []
    used_comments = set()
    max_length = max(len(ranked_comments[emotion]) for emotion in emotions)

    for i in range(max_length):
        for emotion in emotions:
            if i < len(ranked_comments[emotion]):
                comment = ranked_comments[emotion][i]
                if comment not in used_comments:
                    reorganized.append(comment)
                    used_comments.add(comment)

    return reorganized

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(input_dir):
    print(f"{input_dir} 不存在")
else:
    mood_words = load_mood_words("nesscary/mood_word_1.0.json")  # 載入情緒詞

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".json"):
            input_file_path = os.path.join(input_dir, file_name)

            with open(input_file_path, "r", encoding="utf-8-sig") as file:
                comments = json.load(file)

            # 對評論進行情感分析和排序
            ranked_comments = rank_comments_by_emotion(comments, mood_words)

            # 重新組織評論
            reorganized_comments = reorganize_comments(ranked_comments)

            # 寫入結果
            output_file_path = os.path.join(output_dir, file_name)
            with open(output_file_path, "w", encoding="utf-8") as file:
                json.dump(reorganized_comments, file, ensure_ascii=False, indent=4)

    print("處理完成，結果已寫入 ranked_sentence 目錄")