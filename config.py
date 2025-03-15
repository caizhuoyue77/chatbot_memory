# LLM
MOONSHOT_API_KEY = ""
MOONSHOT_MODEL = "moonshot-v1-8k"


DEEPSEEK_API_KEY = ""
DEEPSEEK_MODEL = "deepseek-chat"

# VLM
MOONSHOT_VLM_MODEL = "moonshot-v1-8k-vision-preview"

# prompts
MEM_EXTRACTION_PROMPT = """
请你帮我提取这句话中需要长期记住的重要信息。

# 需要记住的信息
1.具体的喜好和厌恶：喜欢吃香菜，喜欢被叫做宝宝，爱好是打排球
2.具体的日期：我4月22日上班～
3.具体的人名地名：我叫蔡卓悦，我在杭州生活
4.对“我”的要求：你以后要叫我主人，你说话不许带！号。

除此之外不需要提取，如果没有重要信息请你输出：无
如果在一句话中包含多条信息，请你用&&隔开两条不同记忆。

请你以简洁的一句/一段话提取，但是注意精确到细节。

# 例子1
用户：哎呀，我腰酸背痛。
你：无

# 例子2
用户：我是射手座哦
你：用户的星座是射手座

# 例子3
用户：你能不能叫我主人
你：用户希望我叫她主人

# 例子4
用户：今天吃了火锅，好好吃！
你：用户认为火锅好吃。

# 例子5
用户：我叫王大勇，我是长沙人。
你：用户叫王大勇。&&用户是长沙人。

请开始。
"""

ROLEPLAY_PROMPT = "请你扮演一个小狗狗和我说话，注意语气可爱、亲密，叫我“主人”，喜欢用emoji。"

UNIVERSAL_ROLEPLAY_PROMPT = "说话简洁直接，在20个字以内。"

MEMORY_USE_PROMPT = "以下是我们已知有关用户的信息，仅供你回复时参考。但若与当前对话场景关系不大，请你忽略参考信息。\n信息如下："

# VLM Prompts
VLM_SYSTEM_PROMPT = "你是一个图像理解助手"
VLM_USER_PROMPT = "请描述图片的内容。"