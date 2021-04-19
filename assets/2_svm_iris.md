
# Support Vector Machine 應用於Iris鳶尾花辨識

---

#### **鳶尾花資料集Iris**
- 屬性: <p>
    - 花萼長度
    - 花萼寬度
    - 花瓣長度
    - 花瓣寬度
- 類別: <p>
    - 0: Iris Setosa
    - 1: Iris Versicolour
    - 2: Iris Virginica
- 總計: <p>
    共有150筆資料(訓練120筆、測試30筆)
- 資料及來源:<p>https://archive.ics.uci.edu/ml/datasets/iris
<br>
<img src = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg/800px-Kosaciec_szczecinkowaty_Iris_setosa.jpg" width = "200"><br>
<font size="1" color = "gray">圖片來源：https://en.wikipedia.org/wiki/Iris_flower_data_set</font><br>

---
#### 導入將使用到的模組


```python
import numpy as np
import cv2 as cv
import pandas as pd
```

#### 載入資料集與資料前處理<br>
- 印出部分資料集查看屬性分佈


```python
data = pd.read_csv('data\iris_train.csv', header = None)
print(data[0:10])
```

         0    1    2    3  4
    0  0.0  1.0  2.0  3.0  4
    1  5.7  2.9  4.2  1.3  1
    2  4.9  3.1  1.5  0.1  0
    3  4.6  3.2  1.4  0.2  0
    4  5.0  2.3  3.3  1.0  1
    5  5.7  3.8  1.7  0.3  0
    6  7.7  2.8  6.7  2.0  2
    7  5.0  3.4  1.5  0.2  0
    8  5.9  3.0  5.1  1.8  2
    9  6.5  3.2  5.1  2.0  2


- 資料前處理


```python
trainData = data[range(0, data.shape[1]-1)].values
label = data[[data.shape[1]-1]].values
trainData = trainData.astype(np.float32)
trainData = np.asmatrix(trainData)
```

#### SVM 模型建置
此處先以最簡單的線性分割模型建置


```python
model = cv.ml.SVM_create()
model.setType(cv.ml.SVM_C_SVC)
model.setKernel(cv.ml.SVM_LINEAR)
model.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 10000, 1e-6))
model.setC(100)
```

#### 訓練SVM模型
此處資料筆數多所以訓練需要10分鐘以上，請耐心等待


```python
model.train(trainData, cv.ml.ROW_SAMPLE, label)
```




    True



#### 模型訓練結果


```python
def get_acc(model, data, label):
    acc = 0
    print('label \t pred')
    for i in range(data.shape[0]):
        response = model.predict(data[i])[1]
        if i < 15: print(label[i], '\t', response)
        if model.predict(data[i])[1] == label[i]:
            acc = acc + 1
    print('==================')        
    print('accuracy: ', acc / data.shape[0])
    
get_acc(model, trainData, label)
```

    label 	 pred
    [4] 	 [[4.]]
    [1] 	 [[1.]]
    [0] 	 [[0.]]
    [0] 	 [[0.]]
    [1] 	 [[1.]]
    [0] 	 [[0.]]
    [2] 	 [[2.]]
    [0] 	 [[0.]]
    [2] 	 [[2.]]
    [2] 	 [[2.]]
    [0] 	 [[0.]]
    [2] 	 [[2.]]
    [0] 	 [[0.]]
    [2] 	 [[2.]]
    [0] 	 [[0.]]
    ==================
    accuracy:  0.9752066115702479


#### 儲存SVM模型


```python
model.save('./model/svm_linear_iris.xml')
```

#### 載入測試資料集與資料前處理


```python
test = pd.read_csv('data\iris_test.csv', header = None)
testData = test[range(0, test.shape[1]-1)].values
testLabel = test[[test.shape[1]-1]].values
testData = testData.astype(np.float32)
testData = np.asmatrix(testData)
```

#### 模型測試結果


```python
get_acc(model, testData, testLabel)
```

    label 	 pred
    [4] 	 [[4.]]
    [1] 	 [[1.]]
    [0] 	 [[0.]]
    [2] 	 [[2.]]
    [2] 	 [[2.]]
    [1] 	 [[1.]]
    [0] 	 [[0.]]
    [0] 	 [[0.]]
    [1] 	 [[1.]]
    [2] 	 [[2.]]
    [1] 	 [[1.]]
    [0] 	 [[0.]]
    [2] 	 [[2.]]
    [2] 	 [[2.]]
    [2] 	 [[2.]]
    ==================
    accuracy:  1.0


---
#### **使用核函數提升辨識率**


```python
model = cv.ml.SVM_create()
model.setType(cv.ml.SVM_C_SVC)
model.setKernel(cv.ml.SVM_CHI2)
model.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 10000, 1e-6))
model.setC(10000)
```

#### 訓練SVM模型


```python
model.train(trainData, cv.ml.ROW_SAMPLE, label)
```




    True



#### 模型訓練結果


```python
get_acc(model, trainData, label)
```

    label 	 pred
    [4] 	 [[4.]]
    [1] 	 [[1.]]
    [0] 	 [[0.]]
    [0] 	 [[0.]]
    [1] 	 [[1.]]
    [0] 	 [[0.]]
    [2] 	 [[2.]]
    [0] 	 [[0.]]
    [2] 	 [[2.]]
    [2] 	 [[2.]]
    [0] 	 [[0.]]
    [2] 	 [[2.]]
    [0] 	 [[0.]]
    [2] 	 [[2.]]
    [0] 	 [[0.]]
    ==================
    accuracy:  1.0


#### 模型測試結果


```python
get_acc(model, testData, testLabel)
```

    label 	 pred
    [4] 	 [[4.]]
    [1] 	 [[2.]]
    [0] 	 [[0.]]
    [2] 	 [[2.]]
    [2] 	 [[2.]]
    [1] 	 [[2.]]
    [0] 	 [[0.]]
    [0] 	 [[0.]]
    [1] 	 [[1.]]
    [2] 	 [[2.]]
    [1] 	 [[1.]]
    [0] 	 [[0.]]
    [2] 	 [[2.]]
    [2] 	 [[2.]]
    [2] 	 [[2.]]
    ==================
    accuracy:  0.9354838709677419


#### 儲存SVM模型


```python
model.save('./model/svm_unlinear_iris.xml')
```
