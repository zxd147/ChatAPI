import time


def split_text(text, min_tokens, max_tokens, sentences):
    # 如果文本长度不超过80个字，则不需要断句
    if len(text) <= max_tokens:
        if len(text) != 0:
            sentences.append(text)
        return sentences
    # 如果文本长度在80到100个字之间，则在40-80个字之间寻找分割的符号
    elif max_tokens < len(text) <= max_tokens + min_tokens:
        cut_text = text[max_tokens // 2:max_tokens]
        cut_punctuation = get_punctuation(cut_text)
        # 在大于40个字的第一个标点符号处分割文本
        index = max_tokens // 2
        found_index = False
        while index <= min_tokens+max_tokens and not found_index:
            index += 1
            found_index = text[index] in cut_punctuation
        if not found_index:
            index = max_tokens
        # 按照找到的位置分割文本
        first_sentence = text[:index + 1]
        second_sentence = text[index + 1:]
        sentences.append(first_sentence)
        sentences = split_text(second_sentence, min_tokens, max_tokens, sentences)
        return sentences

    else:
        # 如果文本长度超过100个字，则在40-80个字之间寻找分割的符号
        cut_text = text[max_tokens // 2:max_tokens]
        cut_punctuation = get_punctuation(cut_text)
        # print(cut_punctuation)
        # 在小于80个字的最后一个标点符号处分割文本
        index = max_tokens
        found_index = False
        while index > 0 and not found_index:
            index -= 1
            found_index = text[index] in cut_punctuation
        if not found_index:
            index = max_tokens
            while index <= min_tokens+max_tokens and not found_index:
                index += 1
                found_index = text[index] in cut_punctuation
        if not found_index:
            index = max_tokens
        # 按照找到的位置分割文本
        first_sentence = text[:index + 1]
        second_sentence = text[index + 1:]
        sentences.append(first_sentence)
        sentences = split_text(second_sentence, min_tokens, max_tokens, sentences)
        return sentences


def get_punctuation(text):
    punctuation = [':', '：', '。', '!', '?', '。', '！', '？', ';', '；', ',', '，', '、']
    text_set = set(text)
    # 计算文本集合和标点符号集合的交集
    if text_set.intersection(set(punctuation[0:2])):
        return punctuation[0:2]
    elif text_set.intersection(set(punctuation[2:8])):
        return punctuation[2:8]
    elif text_set.intersection(set(punctuation[8:10])):
        return punctuation[8:10]
    else:
        # 返回交集是否为空
        return punctuation


if __name__ == "__main__":
    # 测试
    text = (
        "在夜幕降临之际星星闪烁着微光月亮静静地悬挂在天空."
        "一阵微风拂过树叶沙沙作响似乎说着未来的故事!"
        "远处传来了狗的吠叫声,一只猫悄悄地溜进了黑暗的角落，"
        "突然闪电划破了夜空随之而来的是雷声的轰鸣！"
        "雨滴敲打着窗户节奏有序而明快在这样的夜晚任何故事都有可能发生任何梦想都有可能实现。"
    )
    min_tokens = 20
    max_tokens = 80
    start_time = time.perf_counter()
    sentences = split_text(text, min_tokens, max_tokens, sentences=[])
    end_time = time.perf_counter()

    print(sentences)
    print(f"Execution time: {end_time - start_time} seconds")


