
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import sys
from pandas import DataFrame
#input_String="cleaning2/3451"
input_String=sys.argv[1]
#output_String=

flow=pd.read_csv(input_String+"/flow.tsv",sep="\t", header=None)
probability=pd.read_csv(input_String+"/prob.tsv",sep="\t", header=None)


# In[2]:

if(len(flow.columns)==1):
    flow.columns = ["flow1"]
    probability.columns = ["p1"]
elif(len(flow.columns)==2):
    flow.columns = ["flow1","flow2"]
    probability.columns = ["p1","p2"]
elif(len(flow.columns)==3):
    flow.columns = ["flow1","flow2","flow3"]
    probability.columns = ["p1","p2","p3"]
elif(len(flow.columns)==4):
    flow.columns = ["flow1","flow2","flow3","flow4"]
    probability.columns = ["p1","p2","p3","p4"]
elif(len(flow.columns)==5):
    flow.columns = ["flow1","flow2","flow3","flow4","flow5"]
    probability.columns = ["p1","p2","p3","p4","p5"]
elif(len(flow.columns)==6):
    flow.columns = ["flow1","flow2","flow3","flow4","flow5","flow6"]
    probability.columns = ["p1","p2","p3","p4","p5","p6"]


# In[3]:

df = pd.concat([flow, probability], axis=1)


# In[3]:




# In[4]:

df_filtered= df.dropna()
for i in range(len(flow.columns)):
    flow_n='flow'+str(i+1)
    df_filtered= df_filtered[df_filtered[flow_n]>=0]


# In[5]:

from pandas import *
flow_lanes=[]
for i in range(len(flow.columns)):
    flow_lane1 = df_filtered['flow'+str(i+1)].tolist()
    flow_lanes.append(flow_lane1)
#flow_lane2 = df['flow2'].tolist()
#flow_lane3 = df['flow3'].tolist()
len(flow_lanes[0])


# In[6]:

#flow_lane1= [x for x in flow_lane1 if str(x) != 'NAN']
#flow_lane2= [x for x in flow_lane2 if str(x) != 'NAN']
len(df)


# In[6]:




# In[7]:

import numpy as np


# In[8]:

#X = np.array(flow_lane1)
#Y = np.array(flow_lane2)


# In[9]:

from sklearn import linear_model
reg = linear_model.LinearRegression()


# In[18]:

predicted1=[]
confidence1=[]
if(len(flow.columns)!=1):
    for i in range(len(flow.columns)):
        if(i==(len(flow.columns)-1)):
            df1= df_filtered[['flow'+str(1)]]
            df2=df[['flow'+str(1)]]
        else:
            df1= df_filtered[['flow'+str(i+2)]]
            df2=df[['flow'+str(i+2)]]
        X= df1.as_matrix()
        X_to_predict=df2.as_matrix()
        Y= flow_lanes[i]
        reg.fit (X, Y)
        coefficient=reg.coef_
        interception=reg.intercept_
        predicted_lane1=[]
        #predicted_lane1=reg.predict(X_to_predict)
        for j in range(len(X_to_predict)):
            predicted_lane1.extend(X_to_predict[j]*coefficient+interception)
        predicted1.append(predicted_lane1)
else:
    predicted_lane1=[]
    confidence1_flow1=[]
    df1= df_filtered[['flow'+str(1)]]
    df2=df[['flow'+str(1)]]
    X= df1.as_matrix()
    X_to_predict=df2.as_matrix()
    Y= flow_lanes[i]
    for j in range(len(X_to_predict)):
        predicted_lane1.extend(X_to_predict[j])
        confidence1_flow1.append(0)
    predicted1.append(predicted_lane1)
    confidence1.append(confidence1_flow1)


# In[20]:

#print reg.coef_
#print reg.intercept_


# In[20]:




# In[20]:




# In[20]:




# In[20]:




# In[21]:


if(len(flow.columns)!=1):
    for i in range(len(flow.columns)):
        if(i==(len(flow.columns)-1)):
            df2=df['p'+str(1)]
            confidence1_flow1=df2.tolist()
        else:
            df2=df['p'+str(i+2)]
            confidence1_flow1=df2.tolist()
        confidence1.append(confidence1_flow1)
        print "Method 1 success"


# In[21]:




#### # Method 2

                #predicted2_flow1 = []

#p1 = df['p1'].tolist()
#p2 = df['p2'].tolist()
#p3 = df['p3'].tolist()
                
# In[21]:




# In[22]:

predicted2=[]
for i in range(len(flow.columns)):
    predicted2_flow1 = []
    index_p='p'+str(i+1)
    p1=probability[index_p].tolist()
    f1=flow["flow"+str(i+1)].tolist()
    for j in range(1, len(probability)-1):
        
        if (p1[j-1]+p1[j+1])!=0 :
            w1=p1[j-1]/(p1[j-1]+p1[j+1])
            w2=p1[j+1]/(p1[j-1]+p1[j+1])
            predicted2_flow1.append((w1)*f1[j-1] + (w2)*f1[j+1])
        else:
            predicted2_flow1.append(0)

    var=f1[1]
    predicted2_flow1.insert(0,var) # add first element
    predicted2_flow1.append(f1[len(p1)-2]) #add last element
    predicted2.append(predicted2_flow1)
    print "predicted2  success"


# In[35]:


#p1=probability[index_p]
type(predicted2)


# In[24]:

confidence2=[]
for i in range(len(flow.columns)):
    confidence2_flow1= []
    index_p='p'+str(i+1)
    p1=probability[index_p].tolist()
    for j in range(1, len(p1)-1):
        confidence2_flow1.append(min(p1[j-1],p1[j+1]))
    
    confidence2_flow1.insert(0,p1[1])
    confidence2_flow1.append(p1[len(p1)-2])
    confidence2.append(confidence2_flow1)
    print "confidence2  success"


# In[25]:

len(confidence2_flow1)


# In[36]:

confidence2


# In[25]:




# In[25]:




# In[26]:

# Method 3


# In[27]:

predicted3=[]
confidence3=[]
for i in range(len(flow.columns)):
    predicted3_flow1 = []
    confidence3_flow1= []
    index_p='p'+str(i+1)
    p1=probability[index_p].tolist()
    f1=flow["flow"+str(i+1)].tolist()
    for j in range(0, len(p1)):
        predicted3_flow1.append (f1[j])
        confidence3_flow1.append (p1[j])
    confidence3.append(confidence3_flow1)
    predicted3.append(predicted3_flow1)
    print "predicted3  success"


# In[27]:




# In[28]:

# Step 4: Merge


# In[37]:

merge_flow=[]
for i in range(len(flow.columns)):
    merge_flow1 = [] # one flow only
    for j in range(0, len(confidence1[i])):
        if (confidence1[i][j]+confidence2[i][j]+confidence2[i][j])!=0:
            w1= confidence1[i][j]/(confidence1[i][j]+confidence2[i][j]+confidence3[i][j])
            w2= confidence2[i][j]/(confidence1[i][j]+confidence2[i][j]+confidence3[i][j])
            w3= confidence3[i][j]/(confidence1[i][j]+confidence2[i][j]+confidence3[i][j])
            result=w1* predicted1[i][j]+ w2*  predicted2[i][j]+ w3*  predicted3[i][j]
            if(result>=0):
                merge_flow1.append(result)
            else:
                merge_flow1.append(-1)
        else: 
            merge_flow1.append(-1)
    merge_flow.append(merge_flow1)
    print "merge_flow  success"


# In[38]:

type(merge_flow[0][0])


# In[39]:

#import csv
#fileObj = open("sample.csv", "wb")
#csv_file = csv.writer(fileObj) 
#input_String="cleaning2/3232"
folder=input_String[-4:]


# In[40]:

outputfile=folder+".flow.txt"


# In[41]:

with open(outputfile, "w") as fh:
    for j in range(0, len(merge_flow[0])):
        for i in range(0,len(merge_flow)):
            if(merge_flow[i][j]==-1):
                fh.write("")
            else:
                fh.write(str(merge_flow[i][j]))
            if(i==(len(merge_flow)-1)):
                fh.write('\n')
            else:
                fh.write('\t')


# In[2]:




# In[ ]:



