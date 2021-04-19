
# Support Vector Machine

---

### **SVM應用於手寫數字辨識**

#### **手寫數字影像資料集MNIST**
- 影像大小: <p>
    28 * 28 * 1 (長 * 寬 * 通道)
- 屬性: <p>
    各點像素值(0~255)
- 類別: <p>
    0~9數字
- 總計:<p>
    - 共有60000筆資料
    - 共有784個屬性(28 * 28 * 1)
- 資料及來源:<p>http://yann.lecun.com/exdb/mnist/
<br>
<img src = "https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2019/02/Plot-of-a-Subset-of-Images-from-the-MNIST-Dataset.png" width = "500"><br>
<font size="1" color = "gray">圖片來源：https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/</font><br>

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
data = pd.read_csv('data\mnist_train.csv', header = None)
print(data[0:10])
```

       0    1    2    3    4    5    6    7    8    9    ...  775  776  777  778  \
    0    5    0    0    0    0    0    0    0    0    0  ...    0    0    0    0   
    1    0    0    0    0    0    0    0    0    0    0  ...    0    0    0    0   
    2    4    0    0    0    0    0    0    0    0    0  ...    0    0    0    0   
    3    1    0    0    0    0    0    0    0    0    0  ...    0    0    0    0   
    4    9    0    0    0    0    0    0    0    0    0  ...    0    0    0    0   
    5    2    0    0    0    0    0    0    0    0    0  ...    0    0    0    0   
    6    1    0    0    0    0    0    0    0    0    0  ...    0    0    0    0   
    7    3    0    0    0    0    0    0    0    0    0  ...    0    0    0    0   
    8    1    0    0    0    0    0    0    0    0    0  ...    0    0    0    0   
    9    4    0    0    0    0    0    0    0    0    0  ...    0    0    0    0   
    
       779  780  781  782  783  784  
    0    0    0    0    0    0    0  
    1    0    0    0    0    0    0  
    2    0    0    0    0    0    0  
    3    0    0    0    0    0    0  
    4    0    0    0    0    0    0  
    5    0    0    0    0    0    0  
    6    0    0    0    0    0    0  
    7    0    0    0    0    0    0  
    8    0    0    0    0    0    0  
    9    0    0    0    0    0    0  
    
    [10 rows x 785 columns]


- 資料前處理


```python
trainData = data[range(1, data.shape[1])].values
label = data[[0]].values
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


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-5-5197275d8305> in <module>
         10     print('accuracy: ', acc / data.shape[0])
         11 
    ---> 12 get_acc(model, trainData, label)
    

    NameError: name 'model' is not defined


#### 儲存SVM模型


```python
model.save('./model/svm_linear_mnist.xml')
```

#### 載入測試資料集與資料前處理


```python
test = pd.read_csv('data\mnist_test.csv', header = None)
testData = test[range(1, test.shape[1])].values
testLabel = test[[0]].values
testData = testData.astype(np.float32)
testData = np.asmatrix(testData)
```

#### 模型測試結果


```python
get_acc(model, testData, testLabel)
```

    label 	 pred
    [7] 	 [[7.]]
    [2] 	 [[2.]]
    [1] 	 [[1.]]
    [0] 	 [[0.]]
    [4] 	 [[4.]]
    [1] 	 [[1.]]
    [4] 	 [[4.]]
    [9] 	 [[9.]]
    [5] 	 [[6.]]
    [9] 	 [[9.]]
    [0] 	 [[0.]]
    [6] 	 [[6.]]
    [9] 	 [[9.]]
    [0] 	 [[0.]]
    [1] 	 [[1.]]
    ==================
    accuracy:  0.8279


---
#### 查看真實影像

---
#### **使用核函數提升辨識率**


```python
model = cv.ml.SVM_create()
model.setType(cv.ml.SVM_C_SVC)
model.setKernel(cv.ml.SVM_CHI2)
model.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 10000, 1e-6))
model.setC(100)
```

#### 訓練SVM模型
此處資料筆數多所以訓練需要10分鐘以上，請耐心等待


```python
model.train(trainData, cv.ml.ROW_SAMPLE, label)
```

#### 模型訓練結果


```python
get_acc(model, trainData, label)
```

#### 模型測試結果


```python
get_acc(model, testData, testLabel)
```
