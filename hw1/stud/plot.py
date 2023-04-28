import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
y_true= np.load('/content/labels_s.npy',allow_pickle='TRUE')
y_pred= np.load('/content/predictions_s.npy',allow_pickle='TRUE')


y=[]
y_p=[]

for t in y_true:
   y = y + t 

for k in y_pred:
   y_p = y_p + k




import seaborn as sns
cf=confusion_matrix(y, y_p)


# Normalise
cmn = cf.astype('float') / cf.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cmn, annot=True, fmt='.2f')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)