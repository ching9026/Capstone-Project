import os

movie_mapping = {
    "M003": "M001",
    "M004": "M002",
    "M005": "M003",
    "M006": "M004",
    "M007": "M005",
    "M008": "M006",
    "M009": "M007",
    "M010": "M008",
    "M011": "M009",
    "M012": "M010",
    "M013": "M011",
    "M014": "M012",
    "M015": "M013",
    "M016": "M014",
    "M017": "M015",
    "M018": "M016",
    "M020": "M018",
    "M021": "M019",
    "M022": "M020",
    "M023": "M021",
    "M024": "M022",
    "M025": "M023", "M026": "M024", "M027": "M025",
    "M028": "M026", "M029": "M027", "M030": "M028", "M031": "M029",
    "M032": "M030", "M034": "M032", "M035": "M033",
    "M036": "M034", "M037": "M035", "M038": "M036", "M039": "M037",
    "M040": "M038", "M041": "M039", "M042": "M040", "M043": "M041",
    "M045": "M043", "M047": "M045",
    "M048": "M046", "M050": "M048", "M051": "M049",
    "M052": "M050", "M053": "M051", "M055": "M053",
    "M056": "M054", "M057": "M055", "M058": "M056", "M059": "M057",
    "M060": "M058", "M061": "M059", "M063": "M061",
    "M064": "M062", "M066": "M064", "M067": "M065",
    "M068": "M066", "M070": "M068", "M071": "M069",
    "M072": "M070", "M073": "M071", "M074": "M072", "M075": "M073",
    "M076": "M074", "M077": "M075", "M078": "M076", "M079": "M077",
    "M080": "M078", "M081": "M079", "M082": "M080", "M083": "M081",
    "M084": "M082", "M085": "M083", "M086": "M084", "M087": "M085",
    "M088": "M086", "M089": "M087", "M090": "M088", "M091": "M089",
    "M092": "M090", "M094": "M092",
    "M096": "M094", "M097": "M095", "M098": "M096", "M099": "M097",
    "M100": "M098", "M101": "M099", "M102": "M100", "M103": "M101",
    "M104": "M102", "M106": "M104", "M107": "M105",
    "M108": "M106", "M109": "M107"
}





input_dir = "暫時用不到/test"  #先將爬下來的資料轉名稱
output_dir = "暫時用不到/test1" #銜接到merge_content(new)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(input_dir):
    print(f"{input_dir} 不存在")
else:
    for file_name in os.listdir(input_dir):
        print(file_name)
        movie_name = os.path.splitext(file_name)[0]

        if movie_name in movie_mapping:
            movie_code = movie_mapping[movie_name]
            new_file_name = f"{movie_code}.json"
            input_file_path = os.path.join(input_dir, file_name)
            output_file_path = os.path.join(output_dir, new_file_name)

            # 重命名文件
            os.rename(input_file_path, output_file_path)
            print(f"已重命名 {file_name} 為 {new_file_name}")
        else:
            print(f"電影 {movie_name} 沒有對應的編號")
