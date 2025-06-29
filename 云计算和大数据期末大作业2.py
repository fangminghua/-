import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

# 指定NLTK资源路径（手动下载的资源）
nltk_data_path = "./nltk_data"
nltk.data.path.append(nltk_data_path)


def load_data(posts_path, sample_size=10):
    """从第一问数据源加载10条新闻"""
    try:
        df = pd.read_csv(posts_path, sep='\t', header=0)
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            print(f"从{posts_path}随机抽取了{sample_size}条新闻")
        else:
            print(f"数据源仅{len(df)}条新闻，全部使用")
        return df['post_text'].tolist()
    except Exception as e:
        print(f"数据加载失败，使用模拟数据: {e}")
        return [
            "Breaking news: Scientists discover new planet",
            "Stock market crashes due to economic fears",
            "Hurricane approaching Florida coast",
            "New study shows exercise benefits",
            "Tech company announces new product",
            "Political unrest in multiple countries",
            "Healthcare bill passes in Congress",
            "Climate change protests worldwide",
            "Space station celebrates 20 years",
            "Sports team wins championship"
        ]


def preprocess_texts(texts):
    """文本预处理：分词、清洗、去停用词、词形还原"""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    processed = []
    for text in texts:
        # 转为小写并去除非字母字符
        text = re.sub(r'[^a-z\s]', '', text.lower())
        # 分词
        words = text.split()
        # 去停用词和短单词，词形还原
        filtered = [lemmatizer.lemmatize(word) for word in words
                    if word not in stop_words and len(word) > 2]
        processed.append(filtered)
    return processed


def build_lda_model(processed_texts, num_topics=3):
    """构建LDA模型"""
    dictionary = Dictionary(processed_texts)
    dictionary.filter_extremes(no_below=1, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=5,
        random_state=42
    )
    return lda_model, dictionary, corpus


def visualize_topics(lda_model, dictionary, corpus):
    """可视化主题分析结果"""
    # 1. pyLDAvis交互图
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, 'lda_visualization.html')
    print("已生成pyLDAvis交互图：lda_visualization.html")

    # 2. 词云图
    os.makedirs('wordclouds', exist_ok=True)
    for topic_id in range(lda_model.num_topics):
        topic_words = lda_model.show_topic(topic_id, topn=15)
        word_dict = {word: weight for word, weight in topic_words}
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
        ).generate_from_frequencies(word_dict)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'主题 {topic_id + 1} 关键词云')
        plt.axis('off')
        plt.savefig(f'wordclouds/topic_{topic_id + 1}.png', dpi=300)
    print("已生成各主题词云图，保存至wordclouds文件夹")


def analyze_topics_with_llm(lda_model, num_topics=3):
    """结合大模型分析各主题的内容（模拟实现）"""
    print("\n===== 主题内容大模型分析 =====")
    for topic_id in range(num_topics):
        topic_words = lda_model.show_topic(topic_id, topn=5)
        keywords = ", ".join([word for word, _ in topic_words])

        # 模拟大模型分析
        if "scientist" in keywords or "discover" in keywords:
            analysis = "该主题围绕科学发现与技术创新，涉及天文学、医学等领域的最新研究成果与突破。"
        elif "market" in keywords or "economic" in keywords:
            analysis = "此主题聚焦经济与金融动态，包括股市波动、政策影响及市场趋势分析。"
        elif "hurricane" in keywords or "climate" in keywords:
            analysis = "主题关注自然环境与气候变化，涵盖自然灾害预警、环保行动及生态研究。"
        else:
            analysis = "主题与社会文化活动相关，包括体育赛事、文化庆典及公众活动等。"

        print(f"主题 {topic_id + 1} (关键词: {keywords}):")
        print(f"  大模型分析: {analysis}")


def main():
    # 第一问数据源路径（需替换为实际路径）
    posts_path = r"C:\Users\86182\Desktop\twitter_dataset\devset\posts.txt"

    print("===== 开始第二问：基于大模型的Twitter主题分析 =====")
    # 1. 数据准备
    print("\n--- 数据准备 ---")
    texts = load_data(posts_path)

    # 2. 文本预处理
    print("\n--- 文本预处理 ---")
    processed_texts = preprocess_texts(texts)
    for i, text in enumerate(processed_texts[:3]):
        print(f"预处理后新闻 {i + 1}: {text[:5]}...")

    # 3. 模型构建与训练
    print("\n--- LDA模型构建 ---")
    lda_model, dictionary, corpus = build_lda_model(processed_texts, num_topics=3)
    print("LDA模型训练完成，各主题关键词：")
    for topic_id in range(3):
        print(f"  主题 {topic_id + 1}: {lda_model.show_topic(topic_id, topn=3)}")

    # 4. 可视化分析
    print("\n--- 可视化生成 ---")
    visualize_topics(lda_model, dictionary, corpus)

    # 5. 结合大模型分析主题
    analyze_topics_with_llm(lda_model)
    print("\n===== 主题分析完成 =====")


if __name__ == "__main__":
    main()