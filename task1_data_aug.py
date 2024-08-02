import json
import random
from nltk.corpus import wordnet

# 读取JSON文件
with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)


# 定义同义词替换函数
def synonym_replacement(sentence, n):
    words = sentence.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    random.shuffle(random_word_list)
    num_replaced = 0

    for random_word in random_word_list:
        synonyms = wordnet.synsets(random_word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    return ' '.join(new_words)


# 定义随机插入函数
def random_insertion(sentence, n):
    words = sentence.split()
    new_words = words.copy()

    for _ in range(n):
        add_word(new_words)

    return ' '.join(new_words)


def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) == 0:
        random_word = new_words[random.randint(0, len(new_words) - 1)]
        synonyms = wordnet.synsets(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0].lemmas()[0].name()
    random_idx = random.randint(0, len(new_words) - 1)
    new_words.insert(random_idx, random_synonym)


# 定义随机交换函数
def random_swap(sentence, n):
    words = sentence.split()
    new_words = words.copy()

    for _ in range(n):
        new_words = swap_word(new_words)

    return ' '.join(new_words)


def swap_word(new_words):
    idx1 = random.randint(0, len(new_words) - 1)
    idx2 = idx1
    counter = 0
    while idx2 == idx1:
        idx2 = random.randint(0, len(new_words) - 1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return new_words


# 定义随机删除函数
def random_deletion(sentence, p):
    words = sentence.split()
    if len(words) == 1:
        return sentence

    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    if len(new_words) == 0:
        return words[random.randint(0, len(words) - 1)]

    return ' '.join(new_words)


# 应用数据增强方法
augmented_data = []
for item in data:
    content = item['content']

    # 原始数据
    augmented_data.append(item)

    # 同义词替换
    new_content = synonym_replacement(content, 2)
    if new_content != content:
        augmented_data.append({'content': new_content, 'summary': item['summary']})

    # 随机插入
    new_content = random_insertion(content, 2)
    if new_content != content:
        augmented_data.append({'content': new_content, 'summary': item['summary']})

    # 随机交换
    new_content = random_swap(content, 2)
    if new_content != content:
        augmented_data.append({'content': new_content, 'summary': item['summary']})

    # 随机删除
    new_content = random_deletion(content, 0.2)
    if new_content != content:
        augmented_data.append({'content': new_content, 'summary': item['summary']})

# 保存增强后的数据
with open('augmented_data.json', 'w', encoding='utf-8') as f:
    json.dump(augmented_data, f, ensure_ascii=False, indent=4)

print(f"原始数据条数: {len(data)}")
print(f"增强后数据条数: {len(augmented_data)}")
