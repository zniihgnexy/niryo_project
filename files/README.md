# 基于大语言模型的机器人控制

## 项目概述
本项目分为两个部分：**硬件部分**和**软件部分**。
- **硬件部分** 负责机械臂的控制。
- **软件部分** 使用 **GPT-4** 语言模型生成指令，并将其发送至机械臂控制部分执行。

### **项目结构**
![项目结构](./figures/hard_soft.jpg)

---

## 语言模型部分
我们使用 OpenAI 的 GPT-4 生成机械臂的控制指令。整个语言模型的处理流程如下：

![语言模型流程概述](./figures/pipeline_whole.jpg)

### **阶段 1：拆分指令**
将复杂的自然语言指令拆分成多个小任务。
![阶段 1](./figures/stage1.jpg)

### **阶段 2：清理名称指代**
消除指令中的名称歧义，例如在以下示例中，“queen” 指的是 B6 位置上的棋子。
![阶段 2](./figures/stage2.jpg)

### **阶段 3：清理位置指代**
解析并消除指令中的模糊位置描述，例如“B6 右侧的格子”指的是 B5。
![阶段 3](./figures/stage3.jpg)

### **阶段 4：最终检查**
该阶段用于检查指令的逻辑一致性和整体合理性。
![阶段 4](./figures/stage4.jpg)

---

## **项目环境搭建**

### **先决条件**
- Python 3.8 或更高版本
- mujoco 210（最新版本）
- 不使用 `mujoco-py`，仅使用 `mujoco`
- 推荐使用 `mamba` 进行环境管理

### **安装步骤**
```sh
# 克隆仓库
git clone https://github.com/zniihgnexy/niryo_project.git
cd niryo_project

# 创建虚拟环境
mamba env create -f mamba_mujoco_base.yml

# 激活环境
conda activate mujoco

# 安装依赖
pip install -r requirements.txt
```

---

## **运行模拟演示**

### **运行单指令和多指令模拟**
```sh
python main_simulation.py
python main_simulation_multi.py
```

### **单指令模拟**
输入命令：`move the queen to C2`

机器人会将小绿球（位于 B6 的皇后）移动到 C2。

🎥 [单指令模拟视频](https://github.com/user-attachments/assets/400ba2c1-fa2e-46e6-ac44-b870bc80d0c0)

### **多指令模拟**
输入命令：`move the queen to its further square and move the pawn to its diagonal square`

机器人会将皇后（B6 的球）移动到 C6，并将兵（B3 的球）移动到 C2。

🎥 [多指令模拟视频](https://github.com/user-attachments/assets/71c3d2f9-7d9a-4fef-a84f-e60a670d1be6)

---

## **语言模型 API 说明**
本实验基于 OpenAI 的 GPT-4 语言模型，因此需要 API Key。请在 `llmAPI/api.py` 文件中设置您的 API Key。

🔑 获取 API Key：[OpenAI API Key 页面](https://beta.openai.com/account/api-keys)

⚠️ **当前 API 配置文件尚未上传！**

---

## **项目结构**
```
niryo_project/
├── figures/               # 项目图片
├── llmAPI/                # GPT-4 API 配置
├── scripts/               # 主要代码文件
├── main_simulation.py     # 单指令模拟脚本
├── main_simulation_multi.py # 多指令模拟脚本
├── requirements.txt       # 依赖项
├── mamba_mujoco_base.yml  # 环境配置
├── README.md              # 项目说明
```

📌 **本项目仍在开发中，欢迎贡献和改进！** 🚀
