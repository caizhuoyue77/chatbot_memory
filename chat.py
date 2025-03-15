from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal, List
import numpy as np
import logging
from llm import call_moonshot_llm, call_deepseek_llm, call_moonshot_vlm  # 假设你已经实现了这些函数
from db import save_to_db, get_all_db, clear_db  # 添加 clear_db 函数
from bge import compute_similarity  # 假设你已经实现了这个函数
from config import MEM_EXTRACTION_PROMPT, MEMORY_USE_PROMPT, ROLEPLAY_PROMPT, MOONSHOT_MODEL, DEEPSEEK_MODEL, MOONSHOT_VLM_MODEL
import base64
import time
# 创建 FastAPI 实例
app = FastAPI()

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", filename='./app.log')

# 定义输入数据结构
class ChatRequest(BaseModel):
    content: str
    chat_model: Literal[MOONSHOT_MODEL, DEEPSEEK_MODEL]
    memory_model: Literal[MOONSHOT_MODEL, DEEPSEEK_MODEL]
    role_prompt: str = ROLEPLAY_PROMPT
    memory_threshold: float = 0.6
    top_k: int = 3
    image_bytes_list: List[str] = []
    vlm_model: Literal[MOONSHOT_VLM_MODEL] = MOONSHOT_VLM_MODEL

def process_and_merge_memory(new_mem: str, threshold: float):
    """处理新记忆条目，合并或替换现有记忆"""
    threshold = 0.8
    existing_mems = get_all_db()
    
    
    logging.info(f"待添加记忆：{new_mem}")
    logging.info(f"现有记忆：{existing_mems}")
    
    if not existing_mems:
        save_to_db([new_mem])
        logging.info(f"记忆库为空，直接添加新记忆：{new_mem}")
        return
    
    # 计算新记忆与所有现有记忆的相似度
    similarities = compute_similarity([new_mem], existing_mems)
    similarities = np.array(similarities).flatten()
    
    logging.info(f"记忆相似度们：{similarities}")
    max_sim_index = np.argmax(similarities)
    max_sim = similarities[max_sim_index]
    
    if max_sim >= threshold:
        # 替换现有记忆
        replace_in_db(max_sim_index, new_mem)
        logging.info(f"替换相似记忆：索引 {max_sim_index}，相似度 {max_sim:.4f}")
    else:
        # 添加新记忆
        save_to_db([new_mem])
        logging.info(f"添加新记忆：{new_mem}")

def replace_in_db(index: int, new_memory: str):
    """替换指定索引的记忆条目"""
    try:
        memories = get_all_db()
        if index < 0 or index >= len(memories):
            logging.error(f"替换索引 {index} 无效，当前记忆数量：{len(memories)}")
            return False
        
        memories[index] = new_memory
        clear_db()          # 清空原有数据库
        save_to_db(memories) # 重新写入修改后的记忆列表
        return True
    except Exception as e:
        logging.error(f"替换记忆时出错：{e}")
        return False

# chat_one() - 处理聊天回复并抽取记忆
def chat_one(
    content: str,
    chat_model: Literal[MOONSHOT_MODEL, DEEPSEEK_MODEL],
    memory_model: Literal[MOONSHOT_MODEL, DEEPSEEK_MODEL],
    role_prompt: str = ROLEPLAY_PROMPT,
    memory_threshold: float = 0.6,
    top_k: int = 3,
    image_bytes_list: List[str] = [],
    vlm_model: Literal[MOONSHOT_VLM_MODEL] = MOONSHOT_VLM_MODEL
):
    try:
        # 记录接收到的参数

        logging.info(f"chat_one 接收到参数: content={content}, chat_model={chat_model}, memory_model={memory_model}, role_prompt={role_prompt}, memory_threshold={memory_threshold}, top_k={top_k}")

        role_prompt = "请你与我对话的时候扮演该角色：" + role_prompt
        logging.info(f"role_prompt: {role_prompt}")

        #### Test VLM
        # 方案1 直接使用VLM获取图片描述，从而后续内容无需更改，多张图对应一个描述，这里为节省Token设定。
            # 可选：1张图1个描述
        if image_bytes_list:
            logging.info("检测到图片输入，正在使用VLM获取图片描述...")
            logging.info(f"for循环中图片个数: {len(image_bytes_list)}")

            image_descriptions = reply_with_VLM(image_bytes_list=image_bytes_list, vlm_model=vlm_model)
            logging.info(image_descriptions)
            content = "以下内容为相关多个图片的描述，假设你可以看到这些图片，根据图片的描述回答我的问题。图片的描述为：" + str(image_descriptions) + "\n" + content
            logging.info(f"content 的内容为{content}")
        ####


        # 获取模型的回复
        logging.info(f"图片个数: {len(image_bytes_list)}")
        reply = reply_with_memory(content, chat_model, memory_model, role_prompt, memory_threshold, top_k)
        logging.info(f"模型回复: {reply}")

        # 抽取记忆
        has_mem, memory = extract_mem(content, memory_model)
        logging.info(f"记忆抽取结果: {'有记忆' if has_mem else '无记忆'}")

        if has_mem:
            # 逐个处理记忆条目（合并或添加）
            for mem in memory:
                process_and_merge_memory(mem, memory_threshold)
            logging.info(f"处理完成的新记忆列表: {memory}")
        else:
            logging.info("没有找到需要保存的记忆")

        return reply, has_mem, memory

    except Exception as e:
        logging.error(f"Error in chat_one: {e}")
        # logging.info(f"图片个数: {len(image_bytes_list)}")
        # logging.error(f"Error in chat_one: {content}")
        # logging.error(f"Error in chat_one: {image_descriptions}")
        return "抱歉，我无法处理这个请求。", False, ""

def reply_with_VLM(
        image_bytes_list: List[str],
        vlm_model: Literal[MOONSHOT_VLM_MODEL]
):
    try:
        # 记录接收到的参数

        logging.info("图像的类型为使用的模型: {vlm_model}, 其中 MOONSHOT_VLM_MODEL 为{MOONSHOT_VLM_MODEL}".format(vlm_model=vlm_model, MOONSHOT_VLM_MODEL=MOONSHOT_VLM_MODEL))
        # 调用 LLM 获取回复
        if vlm_model == MOONSHOT_VLM_MODEL:
            logging.info("使用 Moonshot 模型生成回复")
            reply = call_moonshot_vlm(images=image_bytes_list)
        # 判断返回的内容是否包含错误信息
        if "Error" in reply:
            logging.error(f"LLM API 错误：{reply}")
            return "抱歉，处理您的请求时出现错误。"

        return reply

    except Exception as e:
        logging.error(f"Error in reply_with_memory: {e}")
        return "抱歉，我无法处理这个请求。"
# reply_with_memory() - 根据用户输入和记忆生成回复
def reply_with_memory(
    content: str,
    chat_model: Literal[MOONSHOT_MODEL, DEEPSEEK_MODEL],
    memory_model: Literal[MOONSHOT_MODEL, DEEPSEEK_MODEL],
    role_prompt: str = ROLEPLAY_PROMPT,
    memory_threshold: float = 0.6,
    top_k: int = 3,
):
    try:
        # 记录接收到的参数
        logging.info(f"reply_with_memory 接收到参数: content={content}, chat_model={chat_model}, memory_model={memory_model}, role_prompt={role_prompt}, memory_threshold={memory_threshold}, top_k={top_k}")

        # 获取相关记忆
        memory_results = search_mem(content, top_k, memory_threshold)
        memory = [item[0] for item in memory_results] if memory_results else []

        # 如果没有记忆，返回空字符串
        memory_str = "".join(memory) if memory else ""
        logging.info(f"召回的记忆内容: {memory_str}")

        # 拼接记忆提示词
        content_with_memory = content + MEMORY_USE_PROMPT + memory_str
        logging.info(f"拼接后的输入内容: {content_with_memory}")

        # 调用 LLM 获取回复
        if chat_model == MOONSHOT_MODEL:
            logging.info("使用 Moonshot 模型生成回复")
            reply = call_moonshot_llm(prompt=content_with_memory, system_prompt=role_prompt)
        elif chat_model == DEEPSEEK_MODEL:
            logging.info("使用 DeepSeek 模型生成回复")
            reply = call_deepseek_llm(prompt=content_with_memory, system_prompt=role_prompt)
        else:
            logging.info("默认使用 Moonshot 模型生成回复")
            reply = call_moonshot_llm(prompt=content_with_memory, system_prompt=role_prompt)

        # 判断返回的内容是否包含错误信息
        if "Error" in reply:
            logging.error(f"LLM API 错误：{reply}")
            return "抱歉，处理您的请求时出现错误。"

        return reply

    except Exception as e:
        logging.error(f"Error in reply_with_memory: {e}")
        return "抱歉，我无法处理这个请求。"

# extract_mem() - 使用 LLM 分析用户输入，判断是否有需要抽取的记忆
def extract_mem(content: str, memory_model: str):
    try:
        logging.info(f"分析用户输入的记忆: {content}")

        # 调用 LLM 获取记忆分析结果
        if memory_model == MOONSHOT_MODEL:
            logging.info("使用 Moonshot 模型抽取记忆")
            memory = call_moonshot_llm(prompt=f"用户:{content}", system_prompt=MEM_EXTRACTION_PROMPT)
        else:
            logging.info("使用 DeepSeek 模型抽取记忆")
            memory = call_deepseek_llm(prompt=f"用户:{content}", system_prompt=MEM_EXTRACTION_PROMPT)

        # 判断返回的内容是否包含错误信息
        if "Error" in memory:
            logging.error(f"LLM API 错误：{memory}")
            return False, ""

        logging.info(f"记忆分析结果: {memory}")

        # 如果 LLM 返回的是多个记忆，用 '&&' 连接它们
        if memory == "无":
            logging.info("没有抽取到有效的记忆")
            return False, ""
        else:
            # 如果返回多个记忆，用 "&&" 隔开
            memory_list = memory.split("&&")
            memory_list = [m.strip() for m in memory_list if m.strip()]
            logging.info(f"抽取到的记忆: {memory_list}")

            # 返回拆分后的记忆条目
            return True, memory_list

    except Exception as e:
        logging.error(f"Error in extract_mem: {e}")
        return False, ""

# search_mem() - 搜索与用户输入相关的记忆
def search_mem(content: str, top_k: int, threshold: float = 0.6):
    logging.info(f"搜索记忆的关键词：{content}")

    # 获取所有存储的记忆
    memories = get_all_db()
    logging.info(f"当前记忆库中的记忆数量: {len(memories)}")

    # 如果记忆库为空，直接返回空列表
    if not memories:
        logging.info("记忆库为空，没有可搜索的记忆")
        return []

    # 计算关键词与所有记忆的相似度
    keyword = [content]
    similarity_matrix = compute_similarity(keyword, memories)
    logging.info(f"相似度矩阵：\n{similarity_matrix}")

    # 转换为 numpy 数组以便处理
    similarity = np.array(similarity_matrix).flatten()

    # 判断相似度数组是否为空
    if similarity.size == 0:
        logging.info("相似度数组为空")
        return []

    # 获取相似度大于阈值的索引
    valid_indices = np.where(similarity >= threshold)[0]

    # 如果没有符合条件的记忆，返回空列表
    if valid_indices.size == 0:
        logging.info("没有找到相似度超过阈值的记忆")
        return []

    # 获取 top_k 个最大相似度的索引
    top_k_indices = valid_indices[similarity[valid_indices].argsort()[-top_k:][::-1]]
    logging.info(f"最相似的 {top_k} 个记忆的索引：{top_k_indices}")

    # 获取对应的记忆
    top_k_memories = [(memories[i], similarity[i]) for i in top_k_indices]
    for idx, (memory, sim) in enumerate(top_k_memories):
        logging.info(f"记忆 {idx}: {memory} (相似度: {sim})")

    return top_k_memories

# 添加一个 API 获取所有的 memory
@app.get("/memories")
async def get_all_memories():
    try:
        memories = get_all_db()
        logging.info(f"获取所有记忆: {memories}")
        return {"memories": memories}
    except Exception as e:
        logging.error(f"获取记忆时出错: {e}")
        raise HTTPException(status_code=500, detail=f"Error occurred: {e}")

# 添加一个 API 清空所有记忆
@app.delete("/memories")
async def clear_all_memories():
    try:
        clear_db()  # 调用清空函数
        logging.info("所有记忆已清空")
        return {"message": "所有记忆已清空"}
    except Exception as e:
        logging.error(f"清空记忆时出错: {e}")
        raise HTTPException(status_code=500, detail=f"Error occurred: {e}")

# 创建聊天接口
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # 记录接收到的请求
        logging.info(f"收到用户请求: content={request.content}, chat_model={request.chat_model}, memory_model={request.memory_model}, role_prompt={request.role_prompt}, memory_threshold={request.memory_threshold}, top_k={request.top_k}")

        reply, has_mem, memory = chat_one(
            request.content,
            request.chat_model,
            request.memory_model,
            request.role_prompt,
            request.memory_threshold,
            request.top_k,
            image_bytes_list=request.image_bytes_list,
            vlm_model=request.vlm_model
        )

        logging.info(f"返回给用户的回复: {reply}")

        return {"reply": reply, "has_mem": has_mem, "memory": memory}

    except Exception as e:
        logging.error(f"处理请求时出错: {e}")
        raise HTTPException(status_code=500, detail=f"Error occurred: {e}")

# 测试代码
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)