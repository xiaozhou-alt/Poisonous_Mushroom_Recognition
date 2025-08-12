# **毒蘑菇的二元预测**



目的是根据蘑菇的物理特征来预测蘑菇是可食用还是有毒。

## 数据集描述

本次比赛的数据集（训练和测试）是由在UCI [Mushroom](https://archive.ics.uci.edu/dataset/73/mushroom)数据集上训练的深度学习模型生成的。特征分布与原始分布接近，但并不完全相同。请随意使用原始数据集作为本次竞赛的一部分，既可以探索差异，也可以看看将原始数据集纳入训练是否可以提高模型性能。

## 文件

- **train.csv** - 训练数据集; 是二进制目标（或`class``e``p`)
- **test.csv** - 测试数据集;您的目标是预测每行的目标`class`
- **sample_submission.csv** - 格式正确的示例提交文件



| Variable Name   | Role    | Type        | Description                                                  | Units | Missing Values |
| --------------- | ------- | ----------- | ------------------------------------------------------------ | ----- | -------------- |
| poisonous       | Target  | Categorical |                                                              |       | no             |
| cap-shape       | Feature | Categorical | bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s         |       | no             |
| cap-surface     | Feature | Categorical | fibrous=f,grooves=g,scaly=y,smooth=s                         |       | no             |
| cap-color       | Feature | Binary      | brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y |       | no             |
| bruises         | Feature | Categorical | bruises=t,no=f                                               |       | no             |
| odor            | Feature | Categorical | almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s |       | no             |
| gill-attachment | Feature | Categorical | attached=a,descending=d,free=f,notched=n                     |       | no             |
| gill-spacing    | Feature | Categorical | close=c,crowded=w,distant=d                                  |       | no             |
| gill-size       | Feature | Categorical | broad=b,narrow=n                                             |       | no             |
| gill-color      | Feature | Categorical | black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y |       | no             |
| stalk-shape              | Feature | Categorical | enlarging=e,tapering=t                                       |      | no   |
| stalk-root               | Feature | Categorical | bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=? |      | yes  |
| stalk-surface-above-ring | Feature | Categorical | fibrous=f,scaly=y,silky=k,smooth=s                           |      | no   |
| stalk-surface-below-ring | Feature | Categorical | fibrous=f,scaly=y,silky=k,smooth=s                           |      | no   |
| stalk-color-above-ring   | Feature | Categorical | brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y |      | no   |
| stalk-color-below-ring   | Feature | Categorical | brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y |      | no   |
| veil-type                | Feature | Binary      | partial=p,universal=u                                        |      | no   |
| veil-color               | Feature | Categorical | brown=n,orange=o,white=w,yellow=y                            |      | no   |
| ring-number              | Feature | Categorical | none=n,one=o,two=t                                           |      | no   |
| ring-type                | Feature | Categorical | cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z |      | no   |
| spore-print-color | Feature | Categorical | black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y |      | no   |
| population        | Feature | Categorical | abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y |      | no   |
| habitat           | Feature | Categorical | grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d |      | no   |



### 评估

使用[马修斯相关系数 （MCC）](https://en.wikipedia.org/wiki/Phi_coefficient) 评估，它通过综合考虑混淆矩阵中的四个关键值：真正例（TP）、真负例（TN）、假正例（FP）和假负例（FN），提供了一个全面的性能评估。

$MCC$ 的公式如下：
$$
MCC = \frac{TP * TN - FP * FN}{\sqrt{(TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)}}
$$




## 提交文件

对于测试集中的每一行，您必须预测目标，观察值是可编辑的 （） 还是有毒的 （）。该文件应包含标头，并具有以下格式：`id``class``e``p`

```
id,class
3116945,e
3116946,p
3116947,e
etc.
```

