#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#表示直接在浏览器中显示matplotlib图表
get_ipython().run_line_magic('matplotlib', 'inline')


file=r'C:/Users/admin/Desktop/TestData/000001_day.csv'
df1=pd.read_csv(file)
# 处理涨跌字段
df1['tomorrow']=df1['close'].shift(-1)

nm_scaler= MinMaxScaler()
df = nm_scaler.fit_transform(df1[['open','close','high','low','volume','tomorrow']])
print(df)
df=pd.DataFrame(df)
df.columns=['open','close','high','low','volume','tomorrow']

# df['tomorrow'] = df.apply(lambda y: (1 if y['tomorrow']>y['close'] else 0), axis=1)
# df['tomorrow'] = df.apply(lambda y: (100*(y['tomorrow']/y['close']-1)), axis=1)
print(df.tail())





testCount=int(len(df)*0.8) #训练数量
cols=['open','close','high','low','volume']
#训练集
x_train=df[cols][:testCount]

print(x_train)


y_train=df['tomorrow'][:testCount]
# 测试集
x_test=df[cols][testCount:]

x_test=pd.DataFrame(x_test)
print(x_test)

y_test=df['tomorrow'][testCount:]

print(x_train.tail(),y_train.tail(),x_test.tail(),y_test.tail())

model = tf.keras.models.Sequential([  
  tf.keras.layers.Dense(6*4*5,input_shape=(5,), activation='relu'),
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


# In[25]:


# history=model.fit(x_train, y_train, epochs=100,validation_data=(x_test, y_test))
print(x_test)
pred = model.predict(x_test)
r=pd.DataFrame(x_test.values,columns=cols)
r1=r.copy()
r1['实际值']=pd.DataFrame(y_test.values)
r1=nm_scaler.inverse_transform(r1)
r1=pd.DataFrame(r1)
print(r1)

r2=r.copy()
r2['预测值']=pd.DataFrame(pred)
r2=nm_scaler.inverse_transform(r2)
r2=pd.DataFrame(r2)
print(r2)
r1['预测值']=r2[5]
r1.columns=['open','close','high','low','volume','实际值','预测值']
print(r1)

r1.to_csv('predictResult4.csv')


# In[26]:



dfRes=pd.read_csv('predictResult4.csv')
print(dfRes)
dfRes.head()

# 计算预测值与实际值偏差百分比
dfRes['percent'] = dfRes.apply(lambda y: (100*(y['预测值']/y['实际值']-1)), axis=1)
# 预测准确率
dfRes1=dfRes[(dfRes['percent']<1) & (dfRes['percent']>-1)]
print('1%以内正确率：',len(dfRes1)/len(dfRes))

dfRes3=dfRes[(dfRes['percent']<3) & (dfRes['percent']>-3)]
print('3%以内正确率：',len(dfRes3)/len(dfRes))

dfRes5=dfRes[(dfRes['percent']<5) & (dfRes['percent']>-5)]
print('5%以内正确率：',len(dfRes5)/len(dfRes))

dfRes10=dfRes[(dfRes['percent']<10) & (dfRes['percent']>-10)]
print('10%以内正确率：',len(dfRes10)/len(dfRes))

# 1%以内正确率： 0.4370477568740955
# 3%以内正确率： 0.8523878437047757
# 5%以内正确率： 0.9573082489146165
# 10%以内正确率： 0.9949348769898697


# In[ ]:





# In[ ]:




