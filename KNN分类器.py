import random
import matplotlib.pyplot as plt
import pandas # 函数功能：KNN分类器
import numpy
# BMI=体重（公斤）/身高²（米²）
# 体重过轻：BMI<18.5
# 正常范围：18.5≤BMI<24
# 体重过重：BMI≥24
fr=open('dataset1.txt','w+')#a为追加写，r+,w+为覆盖写
for i in range(0,1000):#每次都将文件从0开始读写
    str1 = ''
    str2 = ''
    hight=random.uniform(1.55,1.85)#round(random.uniform(1.55,1.85),2)保留2位小数8
    weight=random.uniform(45,80)
    if(weight/pow(hight,2)<18.5):str2+='light weight'
    if ((weight / pow(hight, 2)) < 24)and(weight / pow(hight, 2) > 18.5):str2+= 'normal'#python不用&&而用and
    if(weight/pow(hight,2)>=24):str2+='heavy weight'
    str1+=str(hight)+"\t"+str(weight)+'\t'+str2+'\n'
    fr.write(str1)
fr.close()
dataset=pandas.read_table('dataset1.txt',sep='\t',header=None)#sep='\t'是分隔符，读入数据
# print(dataset.head())输出前5个
# print(dataset.shape)注意shape(0)与shape[0]有区别,结果为(1000,3)
# print(dataset.info())
Colors=[]#把不同标签用颜色区分
for i in range(dataset.shape[0]):
    m=dataset.iloc[i,-1]  #第i行的第-1个数据
    if m=="light weight":
        Colors.append('yellow')
    if m=="normal":
        Colors.append('green')
    if m=="heavy weight":
        Colors.append('red')
plt.rcParams['font.sans-serif']=['Simhei']#图中字体设置为黑体
pl=plt.figure(figsize=(6,4))   #figsize:指定figure的宽和高，单位为英寸，1英寸≈2.5cm
fig1=pl.add_subplot(111)    #图像1行1列，fig放在位置1
plt.scatter(dataset.iloc[:,0],dataset.iloc[:,1],marker=".",c=Colors)#散点
plt.xlabel('身高')
plt.ylabel('体重')
plt.show()
def stander(dataset):#0-1标准化函数
    min=dataset.min()    #numpy的array.min()  返回结果：1.55  45.01
    max=dataset.max()    #不加其他参数,会返回数组中所有数据中的最大值或最小值
    normset=(dataset-min)/(max-min)
    return normset
#进行0-1标准化

# test
re_dataset=pandas.concat([stander(dataset.iloc[:,:2]),dataset.iloc[:,2]],axis=1)#axis=1实现在横向的连接
def split(dataset,rate=0.9):#划分训练集和测试集
    n=dataset.shape[0]
    m=int(n*rate)
    train=dataset.iloc[:m,:]#前900个数据
    test=dataset.iloc[m:,:]#后100个数据
    test.index=range(test.shape[0])#返回值：RangeIndex(start=0, stop=100, step=1)
    return train,test
train,test=split(re_dataset)
def dataclass(train,test,k):#数据分类
    n=train.shape[1]-1#shape=(900,3),n为除去标签列的列数
    m=test.shape[0]#测试集=(100,3) m=100
    result=[]
    for i in range(m):
        dist=list((((train.iloc[:,:n]-test.iloc[i,:n])**2).sum(1))**5)#曾经在这犯错，计算欧拉距离
        dist_l=pandas.DataFrame({'dist':dist,'labels':(train.iloc[:,n])})
        dr=dist_l.sort_values(by='dist')[:k]
        re=dr.loc[:,'labels'].value_counts()
        result.append(re.index[0])
    result=pandas.Series(result)
    test=pandas.DataFrame(test)
    test.loc[:,'predict']=result#新建一栏predict结果
    # test['predict']=result
    acc=(test.iloc[:,-1]==test.iloc[:,-2]).mean()
    print(f'模型预测准确率为{acc}')
    return test
dataclass(train,test,5)

# test
