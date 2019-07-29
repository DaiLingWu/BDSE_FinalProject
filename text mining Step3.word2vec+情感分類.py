import logging
from gensim.models import word2vec
import pandas as pd
import numpy as np

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus('/Users/wudailing/Desktop/text8')  # 加载语料
model = word2vec.Word2Vec(sentences, size=200)       # 训练skip-gram模型; 默认window=5
#model.save('text8model')

#model=word2vec.Word2Vec.load('text8model')


#positive_text_list建立
file=open('/Users/wudailing/Desktop/positive-words.txt')
positive_text_list=[]
for line in file:
    line_new=line.strip(line[-1])
    positive_text_list.append(line_new)


#建立positive_text_word2vec_dict，positive_text_list裡每個詞的向量
positive_text_word2vec_dict={}
for txt in positive_text_list:
    if txt in model:
        positive_text_word2vec_dict[txt]=model[txt]
    else:
        continue

#建立sentiment dataframe
positive_text_word2vec_df=pd.DataFrame(positive_text_word2vec_dict).T
idx=positive_text_word2vec_df.index
np_one=np.ones((1,1433),dtype=int)
df_one=pd.DataFrame(np_one).T
df_one.index=idx
df_one.columns=['sentiment']
positive_text_word2vec_df=pd.concat([positive_text_word2vec_df,df_one],axis=1)



#negative_text_list建立
file1=open('/Users/wudailing/Desktop/negative-words.txt',encoding = "ISO-8859-1")
negative_text_list=[]
for line in file1:
    line_new=line.strip(line[-1])
    negative_text_list.append(line_new)


#建立negative_text_word2vec_dict，negatove_text_list裡每個詞的向量
negative_text_word2vec_dict={}
for txt in negative_text_list:
    if txt in model:
        negative_text_word2vec_dict[txt]=model[txt]
    else:
        continue

#建立sentiment dataframe
negative_text_word2vec_df=pd.DataFrame(negative_text_word2vec_dict).T
idx1=negative_text_word2vec_df.index
np_minusone=np.full((1,3030),-1)
df_minusone=pd.DataFrame(np_minusone).T
df_minusone.index=idx1
df_minusone.columns=['sentiment']
negative_text_word2vec_df=pd.concat([negative_text_word2vec_df,df_minusone],axis=1)


training_data=pd.concat([positive_text_word2vec_df,negative_text_word2vec_df])

train_x=training_data.iloc[:,:-1]
train_y=training_data.iloc[:,-1]


#關聯詞set
#建立各個關聯詞的vector

relation_word = pd.read_csv('/Users/wudailing/Desktop/relation_word.csv',encoding = "ISO-8859-1")
words = relation_word['relationword']
unique = set(list(words.str.lower().values))
len(unique) #831

relation_dict = {}
for word in unique:
    if word in model:
        relation_dict.update({word : model[word]})
    else:
        continue
df_relationword=pd.DataFrame.from_dict(relation_dict,  orient='index')
test_x=df_relationword
idx=test_x.index


#隨機森林做關聯詞的情緒分類+1 、-1
from sklearn.ensemble import RandomForestClassifier
forest= RandomForestClassifier(n_estimators=100)
model=forest.fit(train_x,train_y)
pred_y=model.predict(test_x)
pred_y=pd.DataFrame(pred_y)
pred_y.index=idx

df_pred_y=pd.concat([test_x,pred_y],axis=1)
df_pred_y.to_csv('sentimental_output',encoding='utf-8')
pred_y.to_csv('sentimental_model',encoding='utf-8')