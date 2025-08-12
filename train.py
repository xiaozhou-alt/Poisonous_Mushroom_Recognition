import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
import random

# 修正后的数据加载函数
def load_data(file_path):
    df = pd.read_csv(file_path)
    
    # 明确指定特征列（排除id和class）
    feature_cols = df.columns.drop(['id', 'class'])
    
    # 数值列：选择数值类型的特征
    num_cols = df[feature_cols].select_dtypes(include=['float', 'int']).columns
    
    # 类别列：选择非数值类型的特征（排除id和class）
    cat_cols = df[feature_cols].select_dtypes(include=['object', 'category']).columns
    
    # 处理缺失值
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # 映射标签
    df['label'] = df['class'].map({'p': 1, 'e': 0})
    return df, num_cols, cat_cols

# 划分数据集
train_df, num_cols, cat_cols = load_data('/kaggle/input/poisonous-mushroom/data/train.csv')
X = train_df[list(num_cols) + list(cat_cols)]
y = train_df['label']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 识别类别特征在数据框中的位置
cat_features_indices = [X.columns.get_loc(col) for col in cat_cols]

print("创建数据池...")

# 创建CatBoost数据池
train_pool = Pool(X_train, y_train, cat_features=cat_features_indices)
val_pool = Pool(X_val, y_val, cat_features=cat_features_indices)

# 检查GPU可用性
try:
    import torch
    device = 'GPU' if torch.cuda.is_available() else 'CPU'
except:
    device = 'CPU'

# 模型配置
model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.05,
    depth=8,
    loss_function='Logloss',
    eval_metric='Accuracy',
    task_type=device,
    early_stopping_rounds=50,
    random_seed=42,
    verbose=100
)

print("开始训练...")

# 训练模型
model.fit(
    train_pool,
    eval_set=val_pool,
    use_best_model=True
)

model.save_model('mushroom_classifier.cbm')
print("模型已保存为: mushroom_classifier.cbm")

# 验证集预测
val_preds = model.predict(X_val)
val_probs = model.predict_proba(X_val)[:, 1]

# 计算MCC
mcc = matthews_corrcoef(y_val, val_preds)
print(f"Validation MCC: {mcc:.4f}")

# 随机选择10个验证样本并保存结果
random_indices = random.sample(range(len(X_val)), min(10, len(X_val)))
sample_results = []

for idx in random_indices:
    sample = X_val.iloc[idx].copy()
    true_class = 'p' if y_val.iloc[idx] == 1 else 'e'
    pred_class = 'p' if val_preds[idx] == 1 else 'e'
    pred_prob = val_probs[idx]
    
    sample_results.append({
        'Index': idx,
        'True Class': true_class,
        'Predicted Class': pred_class,
        'Predicted Probability': f"{pred_prob:.4f}",
        **sample.to_dict()
    })

print("正在保存验证集测试结果...")

# 保存样本结果
with open('/kaggle/working/validation_samples.txt', 'w') as f:
    f.write(f"MCC Score: {mcc:.4f}\n{'='*50}\n")
    for result in sample_results:
        f.write(f"Sample Index: {result['Index']}\n")
        f.write(f"True Class: {result['True Class']}, Predicted: {result['Predicted Class']} (Prob: {result['Predicted Probability']})\n")
        f.write("Features:\n")
        for feature in X.columns:
            f.write(f"  {feature}: {result[feature]}\n")
        f.write('-'*50 + '\n')

# 测试集预测
test_df = pd.read_csv('/kaggle/input/poisonous-mushroom/data/test.csv')

# 处理测试集缺失值
for col in num_cols:
    if col in test_df.columns:
        test_df[col] = test_df[col].fillna(train_df[col].median())

for col in cat_cols:
    if col in test_df.columns:
        test_df[col] = test_df[col].fillna(train_df[col].mode()[0])

# 确保测试集有所有需要的列
missing_cols = set(X.columns) - set(test_df.columns)
for col in missing_cols:
    test_df[col] = train_df[col].mode()[0] if col in cat_cols else train_df[col].median()

# 预测
test_pool = test_df[X.columns]
test_preds = model.predict(test_pool)
test_df['class'] = ['p' if p == 1 else 'e' for p in test_preds]

# 保存预测结果
test_df[['id', 'class']].to_csv('/kaggle/working/submission.csv', index=False)

print("训练完成！结果已保存至 /kaggle/working/submission.csv")