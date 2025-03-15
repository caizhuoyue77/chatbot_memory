
# from .ocr_engine import ocr_engine
from llm import call_moonshot_vlm

class VLMEngine:
    def __init__(self):
        pass

    def DESC_pipeline(self, images: list[str | bytes]):
        """
        图像解释引擎，方案为：
        1. OCR解释图片
        2. 识别图片中的文字
        3. 调用语言模型描述这些图片
        4. 返回描述结果

        该方案是最简单的图像理解方案。
        """

        # 1. OCR解释图片 Archive
        # ocr_results = [ocr_engine.recognize(image) for image in images]

        # 1. 解释图片
        desc_result = call_moonshot_vlm(images)
        return desc_result

vlm_engine = VLMEngine()