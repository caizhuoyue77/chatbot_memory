import time
import os
import logging
import base64
from openai import OpenAI
from config import (
    MOONSHOT_API_KEY, DEEPSEEK_API_KEY,
    ROLEPLAY_PROMPT, UNIVERSAL_ROLEPLAY_PROMPT,
    MOONSHOT_MODEL, DEEPSEEK_MODEL,
    MOONSHOT_VLM_MODEL,
    VLM_SYSTEM_PROMPT, VLM_USER_PROMPT
)

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def call_llm(prompt: str, system_prompt: str, model: str, api_key: str, base_url: str, retries: int = 3, delay: int = 20, temperature: float = 0.95):
    """
    通用的 LLM 调用函数，支持不同的 API 配置。

    参数：
    prompt (str): 提供给 LLM 的提示词。
    system_prompt (str): 系统角色的提示词。
    model (str): 使用的 LLM 模型名称。
    api_key (str): API 密钥。
    base_url (str): API 基本 URL。
    retries (int): 重试次数，默认为 3 次。
    delay (int): 重试间隔时间，默认为 20 秒。
    temperature (float): 生成文本的温度，默认为 0.95。

    返回：
    str: LLM 的回复内容，或错误消息。
    """
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    attempt = 0
    while attempt < retries:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                timeout=30  # 设置超时时间为30秒
            )
            return completion.choices[0].message.content
        except Exception as e:
            attempt += 1
            logging.error(f"请求发生错误: {e}，正在重试... (尝试 {attempt}/{retries})")
            if attempt >= retries:
                logging.error("重试次数已达到上限，无法处理请求。")
                return "Error: 请求失败，已尝试多次。"
            logging.info(f"等待 {delay} 秒后重新尝试...")
            time.sleep(delay)

    return "Error: 请求失败，已尝试多次。"

def call_moonshot_llm(prompt: str, system_prompt: str = ROLEPLAY_PROMPT + UNIVERSAL_ROLEPLAY_PROMPT, model: str = MOONSHOT_MODEL, retries: int = 3, delay: int = 20):
    """
    调用 Moonshot API 的封装函数。
    """
    return call_llm(prompt, system_prompt, model, api_key=MOONSHOT_API_KEY, base_url="https://api.moonshot.cn/v1", retries=retries, delay=delay)

def call_moonshot_vlm(
        images: list[bytes],
        prompt: str = VLM_USER_PROMPT,
        system_prompt:str = VLM_SYSTEM_PROMPT,
        model:str = MOONSHOT_VLM_MODEL,
        retries: int = 3,
        delay: int = 20,
        temperature: float = 0.95
):
    """
    调用 Moonshot API 的封装函数。
    """
    return call_vlm(images=images, model=model, api_key=MOONSHOT_API_KEY, base_url="https://api.moonshot.cn/v1", prompt=prompt, system_prompt=system_prompt, retries=retries, delay=delay)

def call_deepseek_llm(prompt: str, system_prompt: str = ROLEPLAY_PROMPT + UNIVERSAL_ROLEPLAY_PROMPT, model: str = DEEPSEEK_MODEL, retries: int = 3, delay: int = 20):
    """
    调用 DeepSeek API 的封装函数。
    """
    return call_llm(prompt, system_prompt, model, DEEPSEEK_API_KEY, "https://api.deepseek.com", retries, delay)

async def call_llm_async(prompt: str, system_prompt: str, model: str, api_key: str, base_url: str, retries: int = 3, delay: int = 20, temperature: float = 0.95):
    """
    异步调用 LLM 的通用函数。
    """
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    attempt = 0
    while attempt < retries:
        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                timeout=30
            )
            return completion.choices[0].message.content
        except Exception as e:
            attempt += 1
            logging.error(f"请求发生错误: {e}，正在重试... (尝试 {attempt}/{retries})")
            if attempt >= retries:
                logging.error("重试次数已达到上限，无法处理请求。")
                return "Error: 请求失败，已尝试多次。"
            logging.info(f"等待 {delay} 秒后重新尝试...")
            await asyncio.sleep(delay)

    return "Error: 请求失败，已尝试多次。"

async def call_moonshot_llm_async(prompt: str, system_prompt: str = ROLEPLAY_PROMPT + UNIVERSAL_ROLEPLAY_PROMPT, model: str = MOONSHOT_MODEL, retries: int = 3, delay: int = 20):
    """
    异步调用 Moonshot API 的封装函数。
    """
    return await call_llm_async(prompt, system_prompt, model, MOONSHOT_API_KEY, "https://api.moonshot.cn/v1", retries, delay)

async def call_deepseek_llm_async(prompt: str, system_prompt: str = ROLEPLAY_PROMPT + UNIVERSAL_ROLEPLAY_PROMPT, model: str = DEEPSEEK_MODEL, retries: int = 3, delay: int = 20):
    """
    异步调用 DeepSeek API 的封装函数。
    """
    return await call_llm_async(prompt, system_prompt, model, DEEPSEEK_API_KEY, "https://api.deepseek.com", retries, delay)

def call_vlm(
        images: list[str],
        api_key: str,
        base_url: str,
        prompt: str = VLM_USER_PROMPT,
        system_prompt:str = VLM_SYSTEM_PROMPT,
        model:str = MOONSHOT_VLM_MODEL,
        retries: int = 3,
        delay: int = 20,
        temperature: float = 0.95
    ):
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    attempt = 0

    message_content = []

    for image in images:
        message_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image}",
                },
            }
        )
    message_content.append(
        {
            "type": "text",
            "text": prompt
        }
    )
    # logging.info(f"message_content: {message_content}")
    while attempt < retries:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        # 注意这里，content 由原来的 str 类型变更为一个 list，这个 list 中包含多个部分的内容，图片（image_url）是一个部分（part），
                        # 文字（text）是一个部分（part）
                        "content": message_content
                    },
                ],
            )
            return completion.choices[0].message.content
        except Exception as e:
            attempt += 1
            logging.error(f"请求发生错误: {e}，正在重试... (尝试 {attempt}/{retries})")
            if attempt >= retries:
                logging.error("重试次数已达到上限，无法处理请求。")
                return "Error: 请求失败，已尝试多次。"
            logging.info(f"等待 {delay} 秒后重新尝试...")
            time.sleep(delay)

    return "Error: 请求失败，已尝试多次。"



if __name__ == "__main__":
    # 测试调用 Moonshot
    prompt = "醒了？"
    moonshot_response = call_moonshot_llm(prompt)
    print("Moonshot Response:", moonshot_response)


    # 测试调用 DeepSeek
    deepseek_response = call_deepseek_llm(prompt)
    print("DeepSeek Response:", deepseek_response)