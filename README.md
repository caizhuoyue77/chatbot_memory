# my-langgraph

## 启动方式

1. 激活 Conda 环境：
    ```bash
    conda activate langgraph
    ```

2. 启动后端服务：
    ```bash
    fastapi dev api.py
    ```

3. 启动前端服务：
    ```bash
    streamlit run chatbot.py
    ```

---

## 开发TODO

~~拟人回复用户~~

~~LLM协助提取记忆~~

~~向量化记忆内容~~

~~记忆存储下来到txt~~

~~获取所有记忆~~

~~每次和所有记忆一起计算相似度~~

~~记忆的搜索~~

~~记忆展示页面前端~~

~~添加记忆时的提示~~

[ ] 对于可以提出2条记忆的对话，提取一个记忆的list

[ ] 针对聊天rate limit还有没钱的错误处理

[ ] 存储提取的记忆到向量数据库

[ ] 向量搜索所有记忆

~~在拟人回复中利用记忆~~

~~对话页面前端~~

[ ] （暂缓）更新记忆

[ ] （暂缓）删除记忆

[ ] docker化全部

---

## 呈现方式

### 页面 1: 聊天页面 + 配置参数页面
- **功能**：
    - 拟人聊天
    - 配置与记忆/聊天相关的参数
    - 显示记忆的变动

### 页面 2: 记忆查询/展示页面
- **功能**：
    - 根据标签 (tag) 搜索记忆
    - 根据关键字搜索记忆
    - 直接展示所有记忆

---

## 接口设计

1. **模型调用接口**
2. **拟人聊天接口**：
    - 输入：用户输入
    - 输出：角色回复 + 是否抽取记忆
    - **备注**：此部分可以用小模型来实现
3. **连贯聊天接口**：
    - 需求：维护当前上下文窗口，可清空上下文
    - **备注**：建议将此接口设计成一个类来处理
4. **记忆抽取接口**：
    - 输入：聊天记录
    - 输出：抽取的记忆
5. **记忆查询接口**：
    - 输入：查询模式、关键字/tag
    - 输出：所有匹配的记忆

---

## 逻辑设计（算法）

### 1. 记忆抽取
- **目标**：判断每句话是否包含用户的偏好信息（如饮食、聊天偏好等）
- **实现方式**：每次生成回答时同时进行用户偏好信息的提取
- **频率**：每句话生成时进行一次记忆抽取
- **示例**：用户喜欢吃鱼

### 2. 记忆更新
- **目标**：更新已有的记忆
- **实现方式**：1.通过相似度匹配，若相似度超过 0.9 则替换旧记忆 2.通过相似度+llm判断来选择是否替换/替换哪一条记忆
- **示例**：更新用户的新偏好（如换了新的饮食习惯）

### 3. 记忆利用
- **目标**：将抽取的记忆融入对话
- **实现方式**：通过记忆与用户当前信息的相似度匹配，将最相关的两到三条记忆添加到对话提示中（prompt）
- **示例**：使用最相关的记忆来增强对话的个性化

---

## 架构设计

1. **实现方式**：
    - 采用面向对象的方式来设计系统
2. **开发流程**：
    - 自底向上开发，从后端开始实现
3. **记忆存储格式**：
    - **字段**：内容、时间戳、tag（关键词）、用户 ID、记忆 ID
4. **技术选型**:
   - **拟人模型**：deepseek（因为便宜，先用着）
   - **前端**：streamlit（因为好看）
   - **后端**：fastapi