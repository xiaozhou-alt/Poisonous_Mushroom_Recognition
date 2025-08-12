# **蘑菇毒性分类项目**



## 项目介绍

本项目使用 CatBoost 梯度提升算法构建了一个蘑菇毒性分类器，能够准确区分可食用(e)和有毒(p)的蘑菇种类。该模型在验证集上达到了高准确率（0.99+），并提供了详细的预测概率输出，可用于野外蘑菇识别辅助决策。



## 数据集说明

数据集包含以下特征：

| Variable Name            | Role    | Type        | Description                                                  | Units | Missing Values |
| ------------------------ | ------- | ----------- | ------------------------------------------------------------ | ----- | -------------- |
| poisonous                | Target  | Categorical |                                                              |       | no             |
| cap-shape                | Feature | Categorical | bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s         |       | no             |
| cap-surface              | Feature | Categorical | fibrous=f,grooves=g,scaly=y,smooth=s                         |       | no             |
| cap-color                | Feature | Binary      | brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y |       | no             |
| bruises                  | Feature | Categorical | bruises=t,no=f                                               |       | no             |
| odor                     | Feature | Categorical | almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s |       | no             |
| gill-attachment          | Feature | Categorical | attached=a,descending=d,free=f,notched=n                     |       | no             |
| gill-spacing             | Feature | Categorical | close=c,crowded=w,distant=d                                  |       | no             |
| gill-size                | Feature | Categorical | broad=b,narrow=n                                             |       | no             |
| gill-color               | Feature | Categorical | black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y |       | no             |
| stalk-shape              | Feature | Categorical | enlarging=e,tapering=t                                       |       | no             |
| stalk-root               | Feature | Categorical | bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=? |       | yes            |
| stalk-surface-above-ring | Feature | Categorical | fibrous=f,scaly=y,silky=k,smooth=s                           |       | no             |
| stalk-surface-below-ring | Feature | Categorical | fibrous=f,scaly=y,silky=k,smooth=s                           |       | no             |
| stalk-color-above-ring   | Feature | Categorical | brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y |       | no             |
| stalk-color-below-ring   | Feature | Categorical | brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y |       | no             |
| veil-type                | Feature | Binary      | partial=p,universal=u                                        |       | no             |
| veil-color               | Feature | Categorical | brown=n,orange=o,white=w,yellow=y                            |       | no             |
| ring-number              | Feature | Categorical | none=n,one=o,two=t                                           |       | no             |
| ring-type                | Feature | Categorical | cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z |       | no             |
| spore-print-color        | Feature | Categorical | black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y |       | no             |
| population               | Feature | Categorical | abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y |       | no             |
| habitat                  | Feature | Categorical | grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d |       |                |

- **train.csv** - 训练数据集; 是二进制目标（或`class``e``p`)
- **test.csv** - 测试数据集;您的目标是预测每行的目标`class`
- **sample_submission.csv** - 格式正确的示例提交文件

### 评估

使用[马修斯相关系数 （MCC）](https://en.wikipedia.org/wiki/Phi_coefficient) 评估，它通过综合考虑混淆矩阵中的四个关键值：真正例（TP）、真负例（TN）、假正例（FP）和假负例（FN），提供了一个全面的性能评估。

$MCC$ 的公式如下：
$$
MCC = \frac{TP * TN - FP * FN}{\sqrt{(TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)}}
$$



## 快速开始

### 环境要求

```bash
pip install -r requirements.txt
```

### 训练模型

```bash
python train.py \
  --train_path /path/to/train.csv \
  --test_path /path/to/test.csv \
  --model_save_path mushroom_model.cbm
```

## 结果展示

### 评估指标

| 指标                    | 值                                |
| :---------------------- | :-------------------------------- |
| **验证集准确率**        | 0.9915                            |
| **马修斯相关系数(MCC)** | 0.9828                            |
| **训练时间**            | 10m57s (Kaggle GPU P100 显存16GB) |

### 随机样本预测结果

![./output/pic/sample.png]()

战绩可查 ∠( ᐛ 」∠)_

![./output/pic/record.png]()

