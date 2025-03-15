import requests
import streamlit as st
import logging
import base64
# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

##
# from mylogger import logger
from loguru import logger
##

# 设置标题和描述
st.title("💬 你的喜好我都记得")
st.caption("🚀 带有长期记忆的聊天哦")

# 获取最新的记忆数据（从 FastAPI 获取）
def get_memories():
    try:
        logging.info("正在从后端获取记忆数据...")
        response = requests.get("http://backend:8000/memories")
        if response.status_code == 200:
            memories = response.json().get("memories", [])
            logging.info(f"成功获取记忆数据: {memories}")
            return memories
        else:
            error_msg = f"Error: Unable to fetch memories from the backend. Status code: {response.status_code}"
            logging.error(error_msg)
            st.error(error_msg)
            return []
    except requests.exceptions.RequestException as e:
        error_msg = f"Error: {e}"
        logging.error(error_msg)
        st.error(error_msg)
        return []

# 清空所有记忆
def clear_memories():
    try:
        logging.info("正在清空记忆数据...")
        response = requests.delete("http://backend:8000/memories")
        if response.status_code == 200:
            logging.info("记忆已清空")
            st.session_state["memories"] = []  # 清空前端记忆
            st.success("所有记忆已清空")
        else:
            error_msg = f"Error: Unable to clear memories. Status code: {response.status_code}"
            logging.error(error_msg)
            st.error(error_msg)
    except requests.exceptions.RequestException as e:
        error_msg = f"Error: {e}"
        logging.error(error_msg)
        st.error(error_msg)

# 初始化聊天记录和记忆
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "memories" not in st.session_state:
    logging.info("初始化记忆数据...")
    st.session_state["memories"] = get_memories()
if "mem_changed" not in st.session_state:
    st.session_state["mem_changed"] = False
if "cached_images" not in st.session_state:
    st.session_state["cached_images"] = []

# 显示侧边栏的输入选项和记忆
with st.sidebar:
    st.subheader("设置")
    chat_model = st.selectbox("对话模型", ["moonshot-v1-8k", "deepseek-chat"])
    memory_model = st.selectbox("记忆抽取模型", ["moonshot-v1-8k", "deepseek-chat"])
    vlm_model = st.selectbox("图像描述模型", ["moonshot-v1-8k-vision-preview"])
    role_prompt = st.text_area("人设", "请你扮演一个小狗狗和我说话，注意语气可爱、亲密，叫我“主人”，喜欢用emoji", height=100)
    top_k = st.number_input("记忆召回Top K", min_value=1, max_value=5, step=1, value=3)
    memory_threshold = st.number_input("输入记忆阈值", min_value=0.0, max_value=1.0, step=0.01, value=0.6)
    
    st.subheader("记忆库")
    if st.session_state["memories"]:
        st.markdown("### 记忆列表")
        for memory in st.session_state["memories"]:
            st.markdown(f"- {memory}")
    else:
        st.info("暂无记忆")

    # 添加清空记忆按钮
    if st.button("清空所有记忆", type="primary"):
        clear_memories()

# 显示聊天记录
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 添加图片上传组件 by jcj（调整到聊天输入下方）

uploaded_images = st.file_uploader("上传图片（支持PNG）",
                                   type=["png"],
                                   key="image_uploader",
                                   accept_multiple_files=True)
logger.info(f"准备处理上传的图片约为: {len(uploaded_images)}")
# 缓存新上传的图片（移除数量限制）
if uploaded_images:
    # 将新图片加入缓存并记录日志
    new_images = [base64.b64encode(img.getvalue()).decode('utf-8') for img in uploaded_images]
    st.session_state.cached_images = new_images
    logging.info(f"新增 {len(new_images)} 张图片，当前缓存总数：{len(st.session_state.cached_images)}")

# 用户输入
if prompt := st.chat_input():

    # 输入验证
    if not prompt or len(prompt) > 1000:
        error_msg = "输入不能为空或超过1000字符"
        logging.error(error_msg)
        st.error(error_msg)
        st.stop()  # 停止后续代码执行

    # 记录并发送所有缓存图片
    # logger.info(f"准备发送 {len(image_bytes_list)} 张图片到后端")  # 添加日志记录


    logging.info(f"用户输入: {prompt}")
    st.session_state["mem_changed"] = False  # 重置记忆更新状态
    
    # 显示用户输入
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 向自定义 API 发送请求，获取聊天回复
    with st.spinner("正在生成回复..."):  # 显示加载状态
        try:
            # 记录发送的请求数据
            request_data = {
                "content": prompt,
                "chat_model": chat_model,
                "memory_model": memory_model,
                "role_prompt": role_prompt,
                "memory_threshold": memory_threshold,
                "top_k": top_k,
                "image_bytes_list": st.session_state.cached_images,
                "vlm_model": vlm_model
            }

            logging.info(f"正在向后端发送请求。")

            response = requests.post(
                "http://backend:8000/chat",
                json=request_data
            )
            logging.info(f"后端响应状态码: {response.status_code}")

            if response.status_code == 200:
                json_response = response.json()
                logging.info(f"后端响应数据: {json_response}")

                bot_reply = json_response.get("reply", "No response from API.")
                has_mem = json_response.get("has_mem", False)
                
                if has_mem:
                    bot_reply += "\n\n" + "[记忆已更新]"
                    logging.info("检测到记忆更新，正在获取最新记忆数据...")
                    new_memories = get_memories()
                    # 更新记忆（增量更新）
                    if new_memories != st.session_state["memories"]:
                        logging.info(f"记忆已更新，新记忆数据: {new_memories}")
                        st.session_state["memories"] = new_memories
                        st.session_state["mem_changed"] = True
                
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                st.chat_message("assistant").write(bot_reply)
            else:
                error_msg = f"Error: {response.text}"
                logging.error(error_msg)
                st.error(error_msg)
                if st.button("重试"):  # 提供重试按钮
                    logging.info("用户点击重试按钮，重新发送请求...")
                    st.rerun()  # 重新运行当前脚本
        except requests.exceptions.RequestException as e:
            error_msg = f"Error: {e}"
            logging.error(error_msg)
            st.error(error_msg)
            if st.button("重试"):  # 提供重试按钮
                logging.info("用户点击重试按钮，重新发送请求...")
                st.rerun()  # 重新运行当前脚本