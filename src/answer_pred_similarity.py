import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

def calculation_(n1, n2, pred, gt):
    
    df = pd.concat([pred, gt], axis=0)
    df.reset_index(inplace=True, drop=True)
    
    ingredient_li = list(df.columns[1:])
    ingredient_length = len(ingredient_li)
    
    col_li = []
    for col in range(1, len(ingredient_li)):

        if ((df.iloc[n1, col] == 0 and df.iloc[n2, col] > 0) or (df.iloc[n1, col] > 1 and df.iloc[n2, col] == 0)):
            col_li.append(ingredient_li[col])
            
    l = list(set(ingredient_li) - set(col_li))
        
    cnt = 0
    for col in l:
        if (df.iloc[n1][col] >= 35) and (df.iloc[n2][col] >= 35): # case1) 예측, 정답 함량이 모두 35인 경우
            cnt+=1
        elif (10 <= df.iloc[n1][col] <= 35) and (10 <= df.iloc[n2][col] <= 35): # case2) 예측과 정답이 10과 35 사이인 경우
            cnt+=1
        elif (0 <= df.iloc[n1][col] <= 10) and (0 <= df.iloc[n2][col] <= 10): # case3) 예측과 정답이 0과 10 사이인 경우
            cnt+=1
        elif (df.iloc[n1][col] == 0) and (df.iloc[n2][col] ==0): # case4) 예측과 정답이 모두 0인 경우 
            cnt+=1
            
    result = cnt / ingredient_length
    return result
    
if __name__ == "__main__":
    
    pred = pd.read_csv("/Users/hwangjaesung/Desktop/재성/src/bluerecipe_code/predict_2.csv", index_col=0)
    gt = pd.read_csv("/Users/hwangjaesung/Desktop/재성/src/bluerecipe_code/label.csv", index_col=0)
    
    
    calculation_(0, 3, pred, gt)

        
