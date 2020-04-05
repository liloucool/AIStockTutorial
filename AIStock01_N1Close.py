#!/usr/bin/env python
# coding: utf-8

# In[180]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# #表示直接在浏览器中显示matplotlib图表
# get_ipython().run_line_magic('matplotlib', 'inline')


file=r'C:/Users/admin/Desktop/TestData/000001_day.csv'
df=pd.read_csv(file)
print(df.tail())
# 处理涨跌字段
df['tomorrow']=df['close'].shift(-1)
# df['tomorrow'] = df.apply(lambda y: (1 if y['tomorrow']>y['close'] else 0), axis=1)
# df['tomorrow'] = df.apply(lambda y: (100*(y['tomorrow']/y['close']-1)), axis=1)
print(df.tail())

testCount=int(len(df)*0.8) #训练数量
cols=['open','close','high','low']
#训练集
x_train=df[cols][:testCount]
y_train=df['tomorrow'][:testCount]
# 测试集
x_test=df[cols][testCount:]
y_test=df['tomorrow'][testCount:]

print(x_train.tail(),y_train.tail(),x_test.tail(),y_test.tail())

model = tf.keras.models.Sequential([  
  tf.keras.layers.Dense(6*4*5,input_shape=(4,), activation='relu'),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(1)
])

adam=tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=adam,
              loss='mse',
              metrics=['accuracy'])

history=model.fit(x_train, y_train, epochs=500,validation_data=(x_test, y_test))
model.evaluate(x_test,  y_test, verbose=2)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[179]:


pred = model.predict(x_test)
r=pd.DataFrame(x_test,columns=cols)
r=r.reset_index(drop = True)
r['实际值']=pd.DataFrame(y_test.values)
r['预测值']=pd.DataFrame(pred)
print(r)
r.to_csv('predictResult.csv')


# In[187]:


plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
def ShowResImage(df,title=''):
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.plot(df['预测值'].values, 'b', label=r'预测值')
    # b is for "solid blue line"
    plt.plot(df['实际值'].values,  'g', label=u'实际值')
    plt.plot(df['close'].values,  'r', label=u'当日收盘价格')
    plt.title(title+' Training and Result')
    plt.xlabel('时间')
    plt.ylabel('价格')
    plt.legend()
    plt.show()
ShowResImage(r)
ShowResImage(r[:100])


# In[ ]:




