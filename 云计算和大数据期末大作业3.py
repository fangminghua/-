import os
import re
import pandas as pd
import numpy as np
import time
from typing import List, Dict, Tuple, Any
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pyLDAvis
import pyLDAvis.gensim_models
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from nltk.sentiment import SentimentIntensityAnalyzer  # 新增：VADER情感分析

# 指定NLTK资源路径
nltk_data_path = "./nltk_data"
nltk.data.path.append(nltk_data_path)

# 下载VADER情感分析词典（如果不存在）
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True, download_dir=nltk_data_path)


# 数据加载与预处理
def load_data(posts_path: str, sample_size: int = 5000) -> Tuple[List[str], List[int]]:
    """加载数据并调整样本量（默认为5000条）"""
    try:
        posts = pd.read_csv(posts_path, sep='\t', header=0)
        print(f"[数据加载] 原始数据共{len(posts)}条新闻")

        # 检查是否有足够的样本
        if len(posts) < sample_size:
            print(f"[数据加载] 警告：原始数据仅{len(posts)}条，不足{sample_size}条，将全部使用")
            sample_size = len(posts)

        # 随机抽样并确保真假新闻比例
        true_posts = posts[posts.iloc[:, -1] == 'true']
        fake_posts = posts[posts.iloc[:, -1] == 'fake']

        # 按比例抽样（如果有足够的真假样本）
        if len(true_posts) > 0 and len(fake_posts) > 0:
            true_ratio = len(true_posts) / len(posts)
            true_sample_size = max(1, int(sample_size * true_ratio))  # 至少保留1条
            fake_sample_size = sample_size - true_sample_size

            if len(true_posts) >= true_sample_size and len(fake_posts) >= fake_sample_size:
                sampled_true = true_posts.sample(n=true_sample_size, random_state=42)
                sampled_fake = fake_posts.sample(n=fake_sample_size, random_state=42)
                sampled_posts = pd.concat([sampled_true, sampled_fake]).sample(frac=1, random_state=42)
                print(
                    f"[数据加载] 从{posts_path}按比例抽取了{sample_size}条新闻（真:{true_sample_size}, 假:{fake_sample_size}）")
            else:
                # 如果某类样本不足，按实际比例抽样
                sampled_posts = posts.sample(n=sample_size, random_state=42)
                true_count = (sampled_posts.iloc[:, -1] == 'true').sum()
                print(
                    f"[数据加载] 从{posts_path}抽取了{sample_size}条新闻（真:{true_count}, 假:{sample_size - true_count}）")
        else:
            # 如果只有一类样本，直接抽样
            sampled_posts = posts.sample(n=sample_size, random_state=42)
            true_count = (sampled_posts.iloc[:, -1] == 'true').sum()
            print(f"[数据加载] 从{posts_path}抽取了{sample_size}条新闻（真:{true_count}, 假:{sample_size - true_count}）")

        texts = sampled_posts['post_text'].tolist()
        true_labels = (sampled_posts.iloc[:, -1] == 'true').astype(int).tolist()
        return texts, true_labels
    except Exception as e:
        print(f"[数据加载] 失败，使用模拟数据: {e}")
        # 模拟数据增加到50条，确保有更多样化的情感和标签组合
        return [
            "Breaking: Scientists discover new planet (TRUE)",  # 积极，真
            "Stock market crashes: Economic fears (TRUE)",  # 消极，真
            "Aliens invade Earth! (FAKE)",  # 积极，假
            "Tech company announces fake product (FAKE)",  # 消极，假
            "Healthy eating leads to better life (TRUE)",  # 积极，真
            "Political unrest causes chaos (FAKE)",  # 消极，假
            "Healthcare bill passes with major changes (TRUE)",  # 积极，真
            "Climate change protests turn violent (FAKE)",  # 消极，假
            "Space station celebrates 20 years (TRUE)",  # 积极，真
            "Sports team wins championship fraudulently (FAKE)",  # 消极，假
            "Amazing breakthrough in cancer research (TRUE)",  # 积极，真
            "Terrorist attack kills dozens (TRUE)",  # 消极，真
            "Celebrity wedding sparks joy (TRUE)",  # 积极，真
            "Fake celebrity death hoax spreads (FAKE)",  # 消极，假
            "New study links sugar to heart disease (TRUE)",  # 消极，真
            "Conspiracy theory about moon landing (FAKE)",  # 消极，假
            "Renewable energy investments surge (TRUE)",  # 积极，真
            "Fake product review scam exposed (FAKE)",  # 消极，假
            "Peace agreement signed between warring nations (TRUE)",  # 积极，真
            "Fake news about election rigging (FAKE)",  # 消极，假
            "Incredible photos from space mission (TRUE)",  # 积极，真
            "Natural disaster destroys homes (TRUE)",  # 消极，真
            "Positive economic growth reported (TRUE)",  # 积极，真
            "Fake miracle cure advertised (FAKE)",  # 消极，假
            "Major tech company launches innovative product (TRUE)",  # 积极，真
            "False accusation against public figure (FAKE)",  # 消极，假
            "New park opens, community celebrates (TRUE)",  # 积极，真
            "Fake charity scam uncovered (FAKE)",  # 消极，假
            "Scientific consensus on climate change (TRUE)",  # 消极，真
            "Fake social media trend spreads misinformation (FAKE)",  # 消极，假
            "Astronomers discover potentially habitable planet (TRUE)",  # 积极，真
            "Scandal rocks political party (TRUE)",  # 消极，真
            "Successful vaccine trial results (TRUE)",  # 积极，真
            "Fake job offer scam targets job seekers (FAKE)",  # 消极，假
            "Local business thrives despite challenges (TRUE)",  # 积极，真
            "False claim about food safety (FAKE)",  # 消极，假
            "Olympic athletes break records (TRUE)",  # 积极，真
            "Fake news website spreads propaganda (FAKE)",  # 消极，假
            "Medical breakthrough saves lives (TRUE)",  # 积极，真
            "False report about celebrity feud (FAKE)",  # 消极，假
            "Environmental cleanup project succeeds (TRUE)",  # 积极，真
            "Fake customer testimonial used in ad (FAKE)",  # 消极，假
            "Historic peace talks begin (TRUE)",  # 积极，真
            "False information about emergency (FAKE)",  # 消极，假
            "Artificial intelligence makes medical diagnosis (TRUE)",  # 积极，真
            "Fake review bombards competitor's product (FAKE)",  # 消极，假
        ], [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0]


def preprocess_texts(texts: List[str]) -> List[List[str]]:
    """文本预处理：移除URL、特殊字符、停用词，并进行词形还原"""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    processed = []
    for i, text in enumerate(texts):
        # 移除URL（社交媒体文本常见）
        text = re.sub(r'http\S+', '', text)
        # 移除特殊字符和数字
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        words = text.split()
        filtered = [lemmatizer.lemmatize(word) for word in words
                    if word not in stop_words and len(word) > 2]
        processed.append(filtered)
        # 每1000条打印一条日志
        if i % 1000 == 0 and i < 5000:
            print(f"[预处理] 已处理{i}条新闻，示例关键词: {filtered[:5]}...")
    return processed


# LDA模型构建
def build_lda_model(processed_texts: List[List[str]], num_topics: int = 5) -> Tuple[
    LdaModel, Dictionary, List[List[Tuple[int, float]]]]:
    """构建LDA主题模型，增加主题数到5个以提高区分度"""
    print(f"[LDA] 开始构建模型，主题数={num_topics}，文档数={len(processed_texts)}")
    dictionary = Dictionary(processed_texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5)  # 调整参数，过滤罕见词和高频词
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    # 训练LDA模型（增加passes提高精度）
    start_time = time.time()
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=10,
        alpha='auto',
        eta='auto',
        random_state=42
    )
    print(f"[LDA] 模型训练完成，耗时{time.time() - start_time:.2f}秒")

    # 打印主题关键词
    for i in range(num_topics):
        topic_words = lda_model.show_topic(i, topn=5)
        print(f"  主题{i + 1}关键词: {', '.join([w for w, _ in topic_words])}")

    return lda_model, dictionary, corpus


# 增强版情感分析（使用VADER）
def analyze_emotion(text: str) -> str:
    """使用NLTK的VADER进行情感分析（比简单词典更准确）"""
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    compound = scores['compound']  # 综合情感得分（-1到1）

    if compound > 0.1:
        return "积极"
    elif compound < -0.1:
        return "消极"
    else:
        return "中性"


# MemoryBank实现（增强版）
class MemoryBank:
    def __init__(self):
        self.memories = []
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None

    def add_memory(self, text: str, emotion: str, topic: List[float], label: int) -> Dict[str, Any]:
        memory_entry = {
            "text": text,
            "emotion": emotion,
            "topic": topic,
            "label": label,
            "timestamp": time.time(),
            "strength": 1.0
        }
        self.memories.append(memory_entry)
        # 每1000条记忆打印一条日志
        if len(self.memories) % 1000 == 0:
            print(f"[MemoryBank] 已添加{len(self.memories)}条记忆")
        return memory_entry

    def _update_index(self):
        """更新向量索引（每100条记忆更新一次，避免频繁更新）"""
        if len(self.memories) % 100 == 0:
            texts = [
                f"文本: {mem['text']}\n情感: {mem['emotion']}\n主题: {mem['topic']}\n标签: {mem['label']}"
                for mem in self.memories
            ]
            self.vectorstore = FAISS.from_texts(texts, self.embeddings)
            print(f"[MemoryBank] 向量索引已更新，当前记忆数={len(self.memories)}")

    def retrieve_relevant_memories(self, current_text: str, current_emotion: str, current_topic: List[float],
                                   k: int = 5) -> List[Dict[str, Any]]:
        """检索相关记忆（增加k值到5，提高召回率）"""
        # 每1000条记忆更新一次索引
        if len(self.memories) > 0 and len(self.memories) % 1000 == 0:
            self._update_index()

        query = f"文本: {current_text}\n情感: {current_emotion}\n主题: {current_topic}"
        if self.vectorstore is None:
            return []

        # 检索相关记忆（增加k值）
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        relevant_memories = []

        for doc, score in docs_and_scores:
            memory_text = doc.page_content
            text = re.search(r"文本: (.*)\n情感:", memory_text, re.DOTALL).group(1).strip()
            emotion = re.search(r"情感: (.*)\n主题:", memory_text).group(1).strip()
            topic_match = re.search(r"主题: \[([^\]]+)\]", memory_text)
            topic = [float(x) for x in topic_match.group(1).split(",")] if topic_match else [0.0] * 10
            label = int(re.search(r"标签: (\d)", memory_text).group(1))

            for mem in self.memories:
                if mem["text"] == text and mem["emotion"] == emotion:
                    relevant_memories.append(mem)
                    break

        # 更新记忆强度（时间衰减）
        current_time = time.time()
        for mem in relevant_memories:
            time_elapsed = current_time - mem["timestamp"]
            mem["strength"] = np.exp(-time_elapsed / (60 * 60 * 24))  # 按天衰减

        return sorted(relevant_memories, key=lambda x: x["strength"], reverse=True)


# 增强版预测函数（增加主题相似度权重）
def predict_with_memory_bank(text: str, emotion: str, topic: List[float], memory_bank: MemoryBank) -> int:
    """基于MemoryBank的预测（增加主题相似度权重）"""
    relevant_memories = memory_bank.retrieve_relevant_memories(text, emotion, topic)

    # 如果没有相关记忆，返回情感分析结果
    if not relevant_memories:
        return 1 if emotion == "积极" else (0 if emotion == "消极" else -1)  # 中性返回-1表示不确定

    # 计算主题相似度（欧氏距离的倒数）
    topic_similarities = []
    for mem in relevant_memories:
        # 计算欧氏距离
        distance = np.sqrt(np.sum((np.array(topic) - np.array(mem["topic"])) ** 2))
        # 转换为相似度（距离越小，相似度越高）
        similarity = 1 / (1 + distance)  # 避免除以0
        topic_similarities.append(similarity)

    # 综合权重（情感权重 + 主题权重）
    emotion_weights = [mem["strength"] for mem in relevant_memories]
    topic_weights = [s * 0.5 for s in topic_similarities]  # 主题权重占50%
    combined_weights = [e + t for e, t in zip(emotion_weights, topic_weights)]

    # 加权投票
    label_votes = [mem["label"] for mem in relevant_memories]
    weighted_label = sum(l * w for l, w in zip(label_votes, combined_weights)) / sum(combined_weights)

    # 输出预测依据（每100条输出一次）
    global prediction_counter
    prediction_counter += 1
    if prediction_counter % 100 == 0:
        print(f"[预测依据] 新闻#{prediction_counter}: 情感={emotion}, 主题={[round(t, 2) for t in topic[:3]]}...")
        print(f"  检索到{len(relevant_memories)}条相关记忆，加权标签={weighted_label:.2f}")

    return 1 if weighted_label > 0.5 else 0


# 初始化预测计数器
prediction_counter = 0


# 准确率计算
def calculate_accuracy(predictions: List[int], true_labels: List[int]) -> Dict[str, float]:
    """计算准确率（区分真新闻和假新闻）"""
    valid_indices = [i for i, pred in enumerate(predictions) if pred != -1]
    if not valid_indices:
        return {"Accuracy": 0.0, "Accuracy_fake": 0.0, "Accuracy_true": 0.0}

    valid_preds = [predictions[i] for i in valid_indices]
    valid_trues = [true_labels[i] for i in valid_indices]

    # 总准确率
    total_correct = sum(p == t for p, t in zip(valid_preds, valid_trues))
    accuracy = total_correct / len(valid_preds)

    # 假新闻准确率
    fake_indices = [i for i, t in enumerate(valid_trues) if t == 0]
    fake_correct = sum(valid_preds[i] == 0 for i in fake_indices)
    accuracy_fake = fake_correct / len(fake_indices) if fake_indices else 0.0

    # 真新闻准确率
    true_indices = [i for i, t in enumerate(valid_trues) if t == 1]
    true_correct = sum(valid_preds[i] == 1 for i in true_indices)
    accuracy_true = true_correct / len(true_indices) if true_indices else 0.0

    # 打印详细结果
    print(f"[准确率统计] 总样本数={len(valid_preds)}, 真新闻={len(true_indices)}, 假新闻={len(fake_indices)}")
    print(f"  预测正确: {total_correct}/{len(valid_preds)} ({accuracy:.2%})")
    print(f"  真新闻预测正确: {true_correct}/{len(true_indices)} ({accuracy_true:.2%})")
    print(f"  假新闻预测正确: {fake_correct}/{len(fake_indices)} ({accuracy_fake:.2%})")

    return {
        "Accuracy": round(accuracy, 4),
        "Accuracy_fake": round(accuracy_fake, 4),
        "Accuracy_true": round(accuracy_true, 4)
    }


# 主函数
def main():
    print("=== 开始执行第三问：基于MemoryBank的真假新闻预测 ===")
    posts_path = r"C:\Users\86182\Desktop\twitter_dataset\devset\posts.txt"

    # 加载数据（默认5000条）
    print("\n=== 数据加载 ===")
    texts, true_labels = load_data(posts_path)
    print(f"  实际加载新闻数: {len(texts)}")

    # 文本预处理
    print("\n=== 文本预处理 ===")
    processed_texts = preprocess_texts(texts)

    # LDA模型构建
    print("\n=== LDA模型构建 ===")
    lda_model, dictionary, corpus = build_lda_model(processed_texts)

    # 主题特征提取
    print("\n=== 主题特征提取 ===")
    topic_features = []
    for i, doc in enumerate(corpus):
        topic_dist = [0.0] * lda_model.num_topics
        for topic_id, prob in lda_model[doc]:
            topic_dist[topic_id] = prob
        topic_features.append(topic_dist)

        # 每1000条打印一条日志
        if i % 1000 == 0:
            print(f"  已处理{i}条新闻，主题分布示例: {[round(p, 2) for p in topic_dist[:3]]}...")

    # 情感分析
    print("\n=== 情感分析 ===")
    emotions = [analyze_emotion(text) for text in texts]

    # 统计情感分布
    emotion_counts = {
        "积极": sum(1 for e in emotions if e == "积极"),
        "消极": sum(1 for e in emotions if e == "消极"),
        "中性": sum(1 for e in emotions if e == "中性")
    }
    print(f"  情感分布: 积极={emotion_counts['积极']}, 消极={emotion_counts['消极']}, 中性={emotion_counts['中性']}")

    # 打印前5条新闻的情感
    for i in range(min(5, len(texts))):
        print(f"  新闻{i + 1}: {texts[i][:50]}..., 情感={emotions[i]}")

    # 基础预测（仅情感分析）
    print("\n=== 基础预测（仅情感分析） ===")
    base_predictions = []
    for i, emotion in enumerate(emotions):
        # 积极=真，消极=假，中性=随机（避免全部预测为假）
        if emotion == "积极":
            pred = 1
        elif emotion == "消极":
            pred = 0
        else:
            # 中性情感随机预测（50%概率）
            pred = np.random.choice([0, 1], p=[0.5, 0.5])
        base_predictions.append(pred)

    base_acc = calculate_accuracy(base_predictions, true_labels)
    print(f"  基础准确率: {base_acc['Accuracy']:.4f}")
    print(f"  真新闻准确率: {base_acc['Accuracy_true']:.4f}")
    print(f"  假新闻准确率: {base_acc['Accuracy_fake']:.4f}")

    # MemoryBank预测
    print("\n=== 使用MemoryBank预测 ===")
    memory_bank = MemoryBank()
    mb_predictions = []

    # 每100条打印一条日志
    log_interval = min(100, len(texts) // 10)  # 至少打印10条日志

    for i, (text, emotion, topic) in enumerate(zip(texts, emotions, topic_features)):
        # 添加记忆
        memory_bank.add_memory(text, emotion, topic, true_labels[i])

        # 每1000条更新一次索引
        if i > 0 and i % 1000 == 0:
            memory_bank._update_index()

        # 从第2条开始预测（第1条没有历史记忆）
        if i > 0:
            prediction = predict_with_memory_bank(text, emotion, topic, memory_bank)
            mb_predictions.append(prediction)

            # 每log_interval条打印一条预测结果
            if i % log_interval == 0:
                is_correct = "✅" if prediction == true_labels[i] else "❌"
                print(f"  新闻{i + 1}预测: {prediction}, 真实标签: {true_labels[i]}, {is_correct}")
        else:
            # 第1条使用基础预测结果
            mb_predictions.append(base_predictions[i])

    # 计算MemoryBank准确率
    mb_acc = calculate_accuracy(mb_predictions, true_labels)
    print(f"  MemoryBank准确率: {mb_acc['Accuracy']:.4f}")
    print(f"  真新闻准确率: {mb_acc['Accuracy_true']:.4f}")
    print(f"  假新闻准确率: {mb_acc['Accuracy_fake']:.4f}")

    # 准确率提升对比
    print("\n=== 准确率提升对比 ===")
    accuracy_improvement = mb_acc['Accuracy'] - base_acc['Accuracy']
    print(f"  总准确率提升: {accuracy_improvement:.4f}")
    print(f"  假新闻准确率提升: {mb_acc['Accuracy_fake'] - base_acc['Accuracy_fake']:.4f}")
    print(f"  真新闻准确率提升: {mb_acc['Accuracy_true'] - base_acc['Accuracy_true']:.4f}")

    # 保存结果
    result_df = pd.DataFrame({
        'text': texts,
        'true_label': true_labels,
        'emotion': emotions,
        'base_prediction': base_predictions,
        'mb_prediction': mb_predictions
    })
    result_df.to_csv('prediction_results.csv', index=False)
    print("\n=== 结果保存 ===")
    print("  预测结果已保存至prediction_results.csv")


if __name__ == "__main__":
    main()