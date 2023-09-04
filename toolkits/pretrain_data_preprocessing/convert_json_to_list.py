import json
json_file_path = ['wudao_train.json', 'wudao_valid.json']
for path in json_file_path:
    b = []
    with open (path,encoding='utf-8') as json_file:
        for line in json_file.readlines():
            dict=json.loads(line)
            b.append(dict)
    with open(path,'w',encoding='utf-8') as file_obj:
        json.dump(b,file_obj,ensure_ascii=False, indent=4)
