#Импортирует модуль для работы с регулярными выражениями
import re
#Импортирует модуль для работы с регулярными выражениями
from collections import Counter
import matplotlib.pyplot as plt
import gensim
from gensim.models import Word2Vec
#функция, которая принимает на вход текст и возвращает список токенов
def tokenize(text):
    text = text.lower()
    #удаление лишних символов
    text = re.sub(r'[^a-zA-Z ^0-9]', '', str(text))
    return text.split()

#количество слов
def word_counter(tokens):
    # создаем экземпляр класса
    word_counts = Counter()
    word_counts.update(tokens)
    return word_counts

def sents(text):
    text = text.lower()
    s = text.split('.')
    ans = list()
    for i in range(len(s)):
        ans.append(tokenize(s[i]))
    return ans

sample_text = ("There is growing interest in using language models to generate text for practical applications. "
               "We find that current large language models are significantly undertrained, a consequence of the recent "
               "focus on scaling language models whilst keeping the amount of training data constant. "
               "Natural language processing tasks, such as question answering, machine translation, reading comprehension, "
               "and summarization, are typically approached with supervised learning on taskspecific datasets. "
               "We demonstrate that language models begin to learn these tasks without any explicit supervision "
               "when trained on a new dataset of millions of webpages. ")
tokens = tokenize(sample_text)
print(tokens)
word_count = word_counter(tokens)
word_count = sorted(word_count.items(), key=lambda item: item[1],reverse=True)
print(word_count)

for word, count in word_count[:6]:
    plt.bar(word, count)
plt.title('Most common words')
plt.xlabel('Word')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

sentences = sents(sample_text)
print(sentences)
model = gensim.models.Word2Vec(sentences, vector_size = 2, window = 5, min_count = 1)
model.train(sentences, total_examples=model.corpus_count, epochs=30, report_delay=1)
print("word vector of 'tasks':",model.wv['tasks'])
print("3 most similar words to 'tasks':",model.wv.most_similar('tasks', topn=3))
print("similarity between 'tasks' and 'text':",model.wv.similarity('tasks','text'))
print("difference between 'tasks' and 'text':",model.wv.distance('tasks', 'text'))
print("similar word to 'tasks' from 'such, learn, question':", model.wv.most_similar_to_given("tasks", ["such", "learn", "question"]))
print("predict conrext to' question':", model.predict_output_word( ["question"],3))