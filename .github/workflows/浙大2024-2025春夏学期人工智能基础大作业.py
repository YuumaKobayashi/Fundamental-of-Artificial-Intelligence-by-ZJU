#写在前面
#这门课虽然一言难尽，但是它的大作业还是很有意思的。我的选题是“用卷积神经网络训练中药的分类模型”，要求如下：
#1、根据提供的20个中药图片数据集设计卷积神经网络；2、将数据集分为训练集和验证集；
#3、训练模型，提供至少3个不同模型参数的组合；4、对模型进行评价，包括准确率、精确率、召回率、F1分数、ROC曲线；
#5、用gradio设计UI界面，可进行一定的交互功能，包括但不限于超参数设置、模型参数设置、训练功能、评价指标显示、数据集加载/选择等。
#我的设计思路就是将每个中药文件按13：2的比例分为训练集和验证集然后进行训练。
#以下是jupyter notebook上拷贝下来的代码。



#注意本设备未安装pytorch,直接在jupyter notebook的编辑界面直接安装
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

#以下是Claude 3.7 sonnet生成的模型选择、训练、评估、测试代码
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# 设置随机种子,确保结果可复现
np.random.seed(42)
tf.random.set_seed(42)

# 数据集路径
dataset_path = r"C:\Users\12518\Desktop\中药数据集"

# 图像参数
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 30

# 获取中药类别
medicine_classes = [folder for folder in os.listdir(dataset_path) 
                   if os.path.isdir(os.path.join(dataset_path, folder))]
num_classes = len(medicine_classes)
print(f"总共发现 {num_classes} 种中药类别: {medicine_classes}")

# 创建训练集和验证集的文件夹
def create_train_val_split(dataset_path, train_ratio=0.866):  # 13:2的比例约等于0.866
    train_dir = os.path.join(os.path.dirname(dataset_path), "中药数据集_train")
    val_dir = os.path.join(os.path.dirname(dataset_path), "中药数据集_val")
    
    # 创建训练和验证目录
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # 遍历每个中药类别
    for medicine in medicine_classes:
        # 创建训练和验证集中的类别目录
        os.makedirs(os.path.join(train_dir, medicine), exist_ok=True)
        os.makedirs(os.path.join(val_dir, medicine), exist_ok=True)
        
        # 获取当前中药的所有图片
        medicine_path = os.path.join(dataset_path, medicine)
        image_files = [f for f in os.listdir(medicine_path) if f.endswith('.jpg') or f.endswith('.png')]
        
        # 随机打乱图片顺序
        np.random.shuffle(image_files)
        
        # 计算训练集图片数量
        num_train = int(len(image_files) * train_ratio)
        
        # 分割为训练集和验证集
        train_files = image_files[:num_train]
        val_files = image_files[num_train:]
        
        # 创建符号链接或复制文件 (此处使用符号链接以节省空间)
        for file in train_files:
            src = os.path.join(medicine_path, file)
            dst = os.path.join(train_dir, medicine, file)
            if not os.path.exists(dst):
                # 在Windows下使用复制而不是符号链接
                import shutil
                shutil.copy2(src, dst)
        
        for file in val_files:
            src = os.path.join(medicine_path, file)
            dst = os.path.join(val_dir, medicine, file)
            if not os.path.exists(dst):
                import shutil
                shutil.copy2(src, dst)
    
    return train_dir, val_dir

# 执行数据集分割
train_dir, val_dir = create_train_val_split(dataset_path)
print(f"训练集路径: {train_dir}")
print(f"验证集路径: {val_dir}")

# 数据增强和预处理
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# 加载训练集和验证集
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# 获取标签映射关系
class_indices = train_generator.class_indices
class_names = list(class_indices.keys())
print("类别对应的标签索引:", class_indices)

# 定义评估函数
def evaluate_model(model, generator, model_name="Model"):
    # 预测
    print(f"\n评估 {model_name} 性能...")
    Y_pred = model.predict(generator, verbose=1)
    y_pred = np.argmax(Y_pred, axis=1)
    
    # 获取真实标签
    y_true = generator.classes
    
    # 计算评估指标
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"{model_name} 评估结果:")
    print(f"准确率 (Accuracy): {acc:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1 分数: {f1:.4f}")
    
    # 生成混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(f'{model_name} 混淆矩阵')
    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.close()
    
    # 生成分类报告
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\n分类报告:")
    print(report)
    
    # 计算和绘制ROC曲线 (一对多方式)
    plt.figure(figsize=(12, 10))
    
    Y_true_one_hot = tf.keras.utils.to_categorical(y_true, num_classes=num_classes)
    
    # 为每个类别绘制ROC曲线
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(Y_true_one_hot[:, i], Y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title(f'{model_name} ROC曲线')
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{model_name}_roc_curve.png')
    plt.close()
    
    # 返回评估结果
    evaluation_results = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return evaluation_results

# ======= 模型1: 基础CNN模型 =======
def create_basic_cnn():
    model = Sequential([
        # 第一个卷积块
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D((2, 2)),
        
        # 第二个卷积块
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # 第三个卷积块
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # 全连接层
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ======= 模型2: 深度CNN模型 =======
def create_deep_cnn():
    model = Sequential([
        # 第一个卷积块
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # 第二个卷积块
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # 第三个卷积块
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # 第四个卷积块
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # 全连接层
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ======= 模型3: 轻量级CNN模型 =======
def create_lightweight_cnn():
    model = Sequential([
        # 第一个卷积块 - 使用更小的卷积核和更少的过滤器
        Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D((2, 2)),
        
        # 第二个卷积块
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # 第三个卷积块
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # 全连接层
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# 定义回调函数
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# 创建结果保存目录
results_dir = "中药分类_模型结果"
os.makedirs(results_dir, exist_ok=True)

# ========= 训练和评估模型 =========

# 训练模型1
print("\n开始训练模型1: 基础CNN模型")
model1 = create_basic_cnn()
model1.summary()

checkpoint1 = ModelCheckpoint(
    os.path.join(results_dir, 'model1_best.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

history1 = model1.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stopping, checkpoint1]
)

# 训练模型2
print("\n开始训练模型2: 深度CNN模型")
model2 = create_deep_cnn()
model2.summary()

checkpoint2 = ModelCheckpoint(
    os.path.join(results_dir, 'model2_best.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

history2 = model2.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stopping, checkpoint2]
)

# 训练模型3
print("\n开始训练模型3: 轻量级CNN模型")
model3 = create_lightweight_cnn()
model3.summary()

checkpoint3 = ModelCheckpoint(
    os.path.join(results_dir, 'model3_best.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

history3 = model3.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stopping, checkpoint3]
)

# 比较三个模型的训练过程
plt.figure(figsize=(12, 10))

# 准确率
plt.subplot(2, 1, 1)
plt.plot(history1.history['accuracy'], label='基础CNN - 训练准确率')
plt.plot(history1.history['val_accuracy'], label='基础CNN - 验证准确率')
plt.plot(history2.history['accuracy'], label='深度CNN - 训练准确率')
plt.plot(history2.history['val_accuracy'], label='深度CNN - 验证准确率')
plt.plot(history3.history['accuracy'], label='轻量级CNN - 训练准确率')
plt.plot(history3.history['val_accuracy'], label='轻量级CNN - 验证准确率')
plt.title('模型准确率对比')
plt.ylabel('准确率')
plt.xlabel('轮次')
plt.legend()

# 损失
plt.subplot(2, 1, 2)
plt.plot(history1.history['loss'], label='基础CNN - 训练损失')
plt.plot(history1.history['val_loss'], label='基础CNN - 验证损失')
plt.plot(history2.history['loss'], label='深度CNN - 训练损失')
plt.plot(history2.history['val_loss'], label='深度CNN - 验证损失')
plt.plot(history3.history['loss'], label='轻量级CNN - 训练损失')
plt.plot(history3.history['val_loss'], label='轻量级CNN - 验证损失')
plt.title('模型损失对比')
plt.ylabel('损失')
plt.xlabel('轮次')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(results_dir, '模型训练对比.png'))
plt.close()

# 评估三个模型
results1 = evaluate_model(model1, validation_generator, model_name="基础CNN模型")
results2 = evaluate_model(model2, validation_generator, model_name="深度CNN模型")
results3 = evaluate_model(model3, validation_generator, model_name="轻量级CNN模型")

# 创建评估结果表格
results_df = pd.DataFrame({
    '模型': ['基础CNN模型', '深度CNN模型', '轻量级CNN模型'],
    '准确率': [results1['accuracy'], results2['accuracy'], results3['accuracy']],
    '精确率': [results1['precision'], results2['precision'], results3['precision']],
    '召回率': [results1['recall'], results2['recall'], results3['recall']],
    'F1分数': [results1['f1_score'], results2['f1_score'], results3['f1_score']]
})

# 保存评估结果
results_df.to_csv(os.path.join(results_dir, '模型评估结果.csv'), index=False)
print("\n模型比较结果:")
print(results_df)

# 可视化评估结果
plt.figure(figsize=(10, 6))
results_df.set_index('模型').plot(kind='bar', figsize=(12, 6))
plt.title('模型性能对比')
plt.ylabel('分数')
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig(os.path.join(results_dir, '模型性能对比.png'))
plt.close()

# 输出总结报告
print("\n==== 中药分类模型训练评估总结 ====")
print(f"数据集包含 {num_classes} 种中药类别")
print(f"每种中药样本数: 75")
print(f"训练集比例: 0.866 (约13:2)")
print(f"图像大小: {IMG_HEIGHT}x{IMG_WIDTH}")
print(f"批次大小: {BATCH_SIZE}")
print(f"最大训练轮次: {EPOCHS}")
print("\n性能最佳的模型:")
best_model_idx = results_df['准确率'].argmax()
print(f"模型: {results_df.iloc[best_model_idx]['模型']}")
print(f"准确率: {results_df.iloc[best_model_idx]['准确率']:.4f}")
print(f"精确率: {results_df.iloc[best_model_idx]['精确率']:.4f}")
print(f"召回率: {results_df.iloc[best_model_idx]['召回率']:.4f}")
print(f"F1分数: {results_df.iloc[best_model_idx]['F1分数']:.4f}")
print("\n所有模型参数和评估结果已保存至: ", results_dir)

#以下是模型的最终UI界面设计代码
# 检查并安装所需库
import sys
import subprocess
import importlib.util

# 检查并安装必要的库
required_packages = ["opencv-python", "tensorflow", "gradio", "seaborn", "scikit-learn"]
for package in required_packages:
    package_name = package.split("==")[0] if "==" in package else package
    # 使用importlib检查库是否已安装
    if importlib.util.find_spec(package_name.replace("-", "_")) is None:
        print(f"安装 {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"{package} 安装完成！")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
# 尝试导入cv2，如果失败则尝试再次安装
try:
    import cv2
except ImportError:
    print("首次导入cv2失败，尝试安装opencv-python...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
    import cv2
    print("cv2成功导入!")
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
import gradio as gr
import matplotlib
matplotlib.use('Agg')  # 设置后端为Agg，防止显示问题

# 设置全局变量
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
DATA_DIR = r"C:\Users\12518\Desktop\中药数据集"

class ChineseMedicineClassifier:
    def __init__(self):
        self.model = None
        self.class_names = []
        self.history = None
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
        self.data_dir = DATA_DIR
        self.model_name = "custom_cnn"
        self.learning_rate = 0.001
        self.batch_size = BATCH_SIZE
        self.epochs = EPOCHS
        self.optimizer_name = "adam"
        self.dropout_rate = 0.5
        self.image_size = IMAGE_SIZE
        self.augmentation = True
        self.early_stopping = True
        self.model_architecture = "custom_cnn"

    def load_data(self):
        """加载数据并准备数据生成器"""
        # 检查数据目录是否存在
        if not os.path.exists(self.data_dir):
            return False, f"数据目录 {self.data_dir} 不存在"
        
        # 获取所有中药类别
        self.class_names = [folder for folder in os.listdir(self.data_dir) 
                           if os.path.isdir(os.path.join(self.data_dir, folder))]
        
        if not self.class_names:
            return False, "未找到有效的中药类别文件夹"
            
        print(f"找到以下中药类别: {self.class_names}")
        
        try:
            # 创建数据增强器
            if self.augmentation:
                train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=20,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest',
                    validation_split=0.2  # 20% 用于验证集
                )
            else:
                train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    validation_split=0.2  # 20% 用于验证集
                )
                
            # 创建训练集生成器
            self.train_generator = train_datagen.flow_from_directory(
                self.data_dir,
                target_size=self.image_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                subset='training',
                shuffle=True
            )
            
            # 创建验证集生成器
            self.val_generator = train_datagen.flow_from_directory(
                self.data_dir,
                target_size=self.image_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                subset='validation',
                shuffle=False
            )
            
            # 创建测试集生成器（用于最终评估）
            test_datagen = ImageDataGenerator(rescale=1./255)
            self.test_generator = test_datagen.flow_from_directory(
                self.data_dir,
                target_size=self.image_size,
                batch_size=1,
                class_mode='categorical',
                shuffle=False,
                subset='validation'  # 使用验证集数据进行测试
            )
            
            return True, f"成功加载数据: 训练集 {self.train_generator.samples} 张图片, 验证集 {self.val_generator.samples} 张图片, 共 {len(self.class_names)} 个类别"
        except Exception as e:
            return False, f"加载数据时发生错误: {str(e)}"

    def create_model(self):
        """创建模型"""
        num_classes = len(self.class_names)
        
        # 根据选择的模型创建不同的架构
        if self.model_architecture == "custom_cnn":
            self.model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(*self.image_size, 3)),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                
                Conv2D(64, (3, 3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                
                Conv2D(128, (3, 3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                
                Flatten(),
                Dense(256, activation='relu'),
                Dropout(self.dropout_rate),
                Dense(num_classes, activation='softmax')
            ])
            
            # 设置优化器
            if self.optimizer_name == "adam":
                optimizer = Adam(learning_rate=self.learning_rate)
            elif self.optimizer_name == "sgd":
                optimizer = SGD(learning_rate=self.learning_rate, momentum=0.9)
            else:
                optimizer = RMSprop(learning_rate=self.learning_rate)
                
            self.model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
        elif self.model_architecture == "vgg16":
            # 使用预训练的VGG16模型
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=(*self.image_size, 3))
            
            # 冻结基础模型的层
            for layer in base_model.layers:
                layer.trainable = False
                
            # 添加自定义分类层
            x = Flatten()(base_model.output)
            x = Dense(512, activation='relu')(x)
            x = Dropout(self.dropout_rate)(x)
            predictions = Dense(num_classes, activation='softmax')(x)
            
            self.model = Model(inputs=base_model.input, outputs=predictions)
            
            # 编译模型
            optimizer = Adam(learning_rate=self.learning_rate)
            self.model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
        elif self.model_architecture == "resnet50":
            # 使用预训练的ResNet50模型
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*self.image_size, 3))
            
            # 冻结基础模型的层
            for layer in base_model.layers:
                layer.trainable = False
                
            # 添加自定义分类层
            x = Flatten()(base_model.output)
            x = Dense(512, activation='relu')(x)
            x = Dropout(self.dropout_rate)(x)
            predictions = Dense(num_classes, activation='softmax')(x)
            
            self.model = Model(inputs=base_model.input, outputs=predictions)
            
            # 编译模型
            optimizer = Adam(learning_rate=self.learning_rate)
            self.model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
        elif self.model_architecture == "mobilenetv2":
            # 使用预训练的MobileNetV2模型
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*self.image_size, 3))
            
            # 冻结基础模型的层
            for layer in base_model.layers:
                layer.trainable = False
                
            # 添加自定义分类层
            x = Flatten()(base_model.output)
            x = Dense(256, activation='relu')(x)
            x = Dropout(self.dropout_rate)(x)
            predictions = Dense(num_classes, activation='softmax')(x)
            
            self.model = Model(inputs=base_model.input, outputs=predictions)
            
            # 编译模型
            optimizer = Adam(learning_rate=self.learning_rate)
            self.model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
        self.model.summary()
        return "模型创建成功"

    def train_model(self):
        """训练模型"""
        if self.model is None:
            return "请先创建模型"
            
        callbacks = []
        
        # 添加早停回调
        if self.early_stopping:
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
            
        # 添加学习率降低回调
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.00001
        )
        callbacks.append(reduce_lr)
        
        # 添加模型检查点保存回调
        checkpoint = ModelCheckpoint(
            f"{self.model_architecture}_model.h5",
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # 训练模型
        self.history = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.train_generator.samples // self.batch_size,
            epochs=self.epochs,
            validation_data=self.val_generator,
            validation_steps=self.val_generator.samples // self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return "模型训练完成"

    def evaluate_model(self):
        """评估模型并返回性能指标图表"""
        if self.model is None or self.history is None:
            return "请先训练模型", None, None, None, None
            
        try:
            # 历史记录图表
            history_fig = plt.figure(figsize=(12, 5))
            
            # 绘制准确率图表
            plt.subplot(1, 2, 1)
            plt.plot(self.history.history['accuracy'])
            plt.plot(self.history.history['val_accuracy'])
            plt.title('模型准确率')
            plt.ylabel('准确率')
            plt.xlabel('轮次')
            plt.legend(['训练集', '验证集'], loc='lower right')
            
            # 绘制损失图表
            plt.subplot(1, 2, 2)
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.title('模型损失')
            plt.ylabel('损失')
            plt.xlabel('轮次')
            plt.legend(['训练集', '验证集'], loc='upper right')
            plt.tight_layout()
            
            # 重置测试生成器
            self.test_generator.reset()
            
            # 预测测试集
            y_pred_prob = self.model.predict(self.test_generator, steps=self.test_generator.samples)
            y_pred = np.argmax(y_pred_prob, axis=1)
            y_true = self.test_generator.classes
            
            # 混淆矩阵
            cm = confusion_matrix(y_true, y_pred)
            cm_fig = plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=self.class_names, 
                        yticklabels=self.class_names)
            plt.title('混淆矩阵')
            plt.ylabel('真实标签')
            plt.xlabel('预测标签')
            plt.tight_layout()
            
            # ROC曲线
            roc_fig = plt.figure(figsize=(10, 8))
            # 多类别ROC曲线
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(len(self.class_names)):
                fpr[i], tpr[i], _ = roc_curve(
                    tf.keras.utils.to_categorical(y_true, num_classes=len(self.class_names))[:, i], 
                    y_pred_prob[:, i]
                )
                roc_auc[i] = auc(fpr[i], tpr[i])
                
            # 绘制所有ROC曲线
            for i in range(len(self.class_names)):
                plt.plot(fpr[i], tpr[i], lw=2,
                         label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.2f})')
                
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('假阳性率')
            plt.ylabel('真阳性率')
            plt.title('各类别ROC曲线')
            plt.legend(loc="lower right")
            plt.tight_layout()
            
            # PR曲线
            pr_fig = plt.figure(figsize=(10, 8))
            # 计算各类别的精确率-召回率曲线
            precision = dict()
            recall = dict()
            
            for i in range(len(self.class_names)):
                precision[i], recall[i], _ = precision_recall_curve(
                    tf.keras.utils.to_categorical(y_true, num_classes=len(self.class_names))[:, i], 
                    y_pred_prob[:, i]
                )
                
            # 绘制所有PR曲线
            for i in range(len(self.class_names)):
                plt.plot(recall[i], precision[i], lw=2,
                         label=f'{self.class_names[i]}')
                
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('召回率')
            plt.ylabel('精确率')
            plt.title('各类别PR曲线')
            plt.legend(loc="lower left")
            plt.tight_layout()
            
            # 获取分类报告
            report = classification_report(y_true, y_pred, target_names=self.class_names)
            
            # 获取整体评估结果
            test_loss, test_acc = self.model.evaluate(self.test_generator, steps=self.test_generator.samples)
            
            evaluation_text = f"模型总体评估:\n测试集损失: {test_loss:.4f}\n测试集准确率: {test_acc:.4f}\n\n分类报告:\n{report}"
            
            return evaluation_text, history_fig, cm_fig, roc_fig, pr_fig
        except Exception as e:
            return f"评估模型时发生错误: {str(e)}", None, None, None, None

    def predict_image(self, image):
        """对单张图片进行预测"""
        if self.model is None:
            return "请先训练模型", None
            
        # 检查图片是否为None
        if image is None:
            return "请先上传图片", None
            
        # 预处理图片
        img = cv2.resize(image, self.image_size)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # 预测
        prediction = self.model.predict(img)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        # 创建结果可视化
        result_fig = plt.figure(figsize=(10, 6))
        
        # 显示原始图片
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('输入图片')
        plt.axis('off')
        
        # 绘制预测结果
        plt.subplot(1, 2, 2)
        plt.bar(self.class_names, prediction[0])
        plt.xticks(rotation=90)
        plt.title('预测结果')
        plt.tight_layout()
        
        result_text = f"预测中药类别: {self.class_names[predicted_class]}\n置信度: {confidence:.2f}"
        
        return result_text, result_fig

    def save_model(self, path="chinese_medicine_model.h5"):
        """保存模型"""
        if self.model is None:
            return "请先创建和训练模型"
            
        self.model.save(path)
        return f"模型已保存至 {path}"

    def load_saved_model(self, path="chinese_medicine_model.h5"):
        """加载保存的模型"""
        if not os.path.exists(path):
            return f"模型文件 {path} 不存在"
            
        self.model = load_model(path)
        return "模型加载成功"

# 创建Gradio界面
def create_gradio_interface():
    # 创建分类器实例
    classifier = ChineseMedicineClassifier()
    
    with gr.Blocks(title="中药分类系统") as app:
        gr.Markdown("# 中药图像分类系统")
        
        with gr.Tab("数据加载"):
            with gr.Row():
                with gr.Column():
                    data_dir_input = gr.Textbox(
                        label="数据目录路径", 
                        value=DATA_DIR,
                        info="包含中药子文件夹的主目录"
                    )
                    load_data_btn = gr.Button("加载数据")
                    load_data_output = gr.Textbox(label="加载状态")
                    
                    def load_data_fn(data_dir):
                        classifier.data_dir = data_dir
                        success, message = classifier.load_data()
                        return message
                    
                    load_data_btn.click(
                        fn=load_data_fn,
                        inputs=[data_dir_input],
                        outputs=[load_data_output]
                    )
        
        with gr.Tab("模型配置"):
            with gr.Row():
                with gr.Column():
                    model_architecture = gr.Dropdown(
                        label="模型架构",
                        choices=["custom_cnn", "vgg16", "resnet50", "mobilenetv2"],
                        value="custom_cnn",
                        info="选择CNN模型架构"
                    )
                    optimizer = gr.Dropdown(
                        label="优化器",
                        choices=["adam", "sgd", "rmsprop"],
                        value="adam",
                        info="选择优化算法"
                    )
                    learning_rate = gr.Slider(
                        label="学习率",
                        minimum=0.0001,
                        maximum=0.01,
                        value=0.001,
                        step=0.0001,
                        info="模型学习率"
                    )
                    dropout_rate = gr.Slider(
                        label="Dropout率",
                        minimum=0.1,
                        maximum=0.9,
                        value=0.5,
                        step=0.1,
                        info="防止过拟合的Dropout比率"
                    )
                
                with gr.Column():
                    batch_size = gr.Slider(
                        label="批次大小",
                        minimum=8,
                        maximum=64,
                        value=32,
                        step=8,
                        info="每批处理的图片数量"
                    )
                    epochs = gr.Slider(
                        label="训练轮次",
                        minimum=5,
                        maximum=50,
                        value=20,
                        step=5,
                        info="训练的总轮次"
                    )
                    image_size = gr.Dropdown(
                        label="图像大小",
                        choices=["(224, 224)", "(160, 160)", "(128, 128)"],
                        value="(224, 224)",
                        info="输入图像尺寸"
                    )
                    data_augmentation = gr.Checkbox(
                        label="使用数据增强",
                        value=True,
                        info="通过图像变换增加数据多样性"
                    )
                    early_stopping = gr.Checkbox(
                        label="使用早停",
                        value=True,
                        info="当验证集性能不再提升时停止训练"
                    )
            
            create_model_btn = gr.Button("创建模型")
            model_output = gr.Textbox(label="模型状态")
            
            def create_model_fn(architecture, optimizer_name, lr, dropout, batch, epochs_num, img_size, augmentation, early_stop):
                classifier.model_architecture = architecture
                classifier.optimizer_name = optimizer_name
                classifier.learning_rate = lr
                classifier.dropout_rate = dropout
                classifier.batch_size = batch
                classifier.epochs = epochs_num
                
                # 处理图像大小
                if img_size == "(224, 224)":
                    classifier.image_size = (224, 224)
                elif img_size == "(160, 160)":
                    classifier.image_size = (160, 160)
                else:
                    classifier.image_size = (128, 128)
                    
                classifier.augmentation = augmentation
                classifier.early_stopping = early_stop
                
                # 重新加载数据以应用新的配置
                classifier.load_data()
                
                return classifier.create_model()
            
            create_model_btn.click(
                fn=create_model_fn,
                inputs=[
                    model_architecture, optimizer, learning_rate, dropout_rate,
                    batch_size, epochs, image_size, data_augmentation, early_stopping
                ],
                outputs=[model_output]
            )
        
        with gr.Tab("模型训练"):
            train_btn = gr.Button("开始训练")
            train_output = gr.Textbox(label="训练状态")
            
            def train_model_fn():
                return classifier.train_model()
                
            train_btn.click(
                fn=train_model_fn,
                inputs=[],
                outputs=[train_output]
            )
        
        with gr.Tab("模型评估"):
            evaluate_btn = gr.Button("评估模型")
            evaluate_output = gr.Textbox(label="评估结果", lines=15)
            history_plot = gr.Plot(label="训练历史")
            confusion_matrix_plot = gr.Plot(label="混淆矩阵")
            roc_plot = gr.Plot(label="ROC曲线")
            pr_plot = gr.Plot(label="精确率-召回率曲线")
            
            def evaluate_model_fn():
                return classifier.evaluate_model()
                
            evaluate_btn.click(
                fn=evaluate_model_fn,
                inputs=[],
                outputs=[evaluate_output, history_plot, confusion_matrix_plot, roc_plot, pr_plot]
            )
        
        with gr.Tab("图片预测"):
            with gr.Row():
                image_input = gr.Image(label="上传图片", type="numpy")
                predict_btn = gr.Button("预测")
            
            with gr.Row():
                predict_output = gr.Textbox(label="预测结果")
                predict_plot = gr.Plot(label="预测可视化")
            
            def predict_image_fn(image):
                return classifier.predict_image(image)
                
            predict_btn.click(
                fn=predict_image_fn,
                inputs=[image_input],
                outputs=[predict_output, predict_plot]
            )
        
        with gr.Tab("模型保存/加载"):
            with gr.Row():
                with gr.Column():
                    save_path = gr.Textbox(label="保存模型路径", value="chinese_medicine_model.h5")
                    save_btn = gr.Button("保存模型")
                    save_output = gr.Textbox(label="保存状态")
                    
                    def save_model_fn(path):
                        return classifier.save_model(path)
                        
                    save_btn.click(
                        fn=save_model_fn,
                        inputs=[save_path],
                        outputs=[save_output]
                    )
                
                with gr.Column():
                    load_path = gr.Textbox(label="加载模型路径", value="chinese_medicine_model.h5")
                    load_btn = gr.Button("加载模型")
                    load_output = gr.Textbox(label="加载状态")
                    
                    def load_model_fn(path):
                        return classifier.load_saved_model(path)
                        
                    load_btn.click(
                        fn=load_model_fn,
                        inputs=[load_path],
                        outputs=[load_output]
                    )
    
    return app

# 运行Gradio应用
if __name__ == "__main__":
    # 检查gradio是否已安装，如果没有则安装
    try:
        import gradio as gr
    except ImportError:
        print("安装 gradio...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio"])
        import gradio as gr
        print("gradio 安装完成!")
    
    app = create_gradio_interface()
    app.launch(share=True)
