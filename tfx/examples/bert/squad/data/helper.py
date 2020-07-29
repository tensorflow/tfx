import tensorflow_datasets as tfds
import pandas as pd

squad = tfds.load('squad')
train = squad['train']
df = pd.DataFrame(tfds.as_numpy(train))
df['context'] = df['context'].str.decode('utf-8')
df['question'] = df['question'].str.decode('utf-8')
df = pd.concat([df.drop('answers', axis=1), df['answers'].apply(pd.Series)], axis=1)
df['answer_start'] = df['answer_start'].apply(lambda x: x[0])
df['text'] = df['text'].apply(lambda x: x[0]).str.decode('utf-8')
df = df.drop('id', axis=1)
df = df.drop('title', axis=1)
df.to_csv('train.csv', index=False)
