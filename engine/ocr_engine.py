from rapidocr import RapidOCR
import numpy as np

class OcrEngine:
    """OCR文字识别引擎封装类
    
    封装RapidOCR的初始化及识别流程，提供标准化的文字识别接口
    """
    
    def __init__(self) -> None:
        """初始化OCR引擎实例"""
        self.engine = RapidOCR()

    def recognize(self, image: str | bytes | np.ndarray) -> tuple[str, ...]:
        """单轮文字识别接口
        
        Args:
            image (str | bytes | np.ndarray): 输入图像，支持以下类型：
                - str: 图像文件路径，例如："./data/sample.jpg"
                - bytes: 图像字节流，例如：open("image.png", "rb").read()
                - np.ndarray: OpenCV格式的BGR图像数组，例如：cv2.imread("input.jpg")

        Returns:
            tuple[str, ...]: 识别到的文本元组，按从左到右、从上到下的顺序排列。
                            例如：("Hello World", "2023-01-01")
            
        示例:
            >>> ocr_engine = OcrEngine()
            >>> ocr_engine.recognize("document.png")
            ("发票代码", "发票号码", "金额：¥120.00")
        """
        # RapidOCR返回结构为(txts, boxes, scores)
        # 其中txts包含所有识别到的文本
        return self.engine(image).txts
    
ocr_engine = OcrEngine()
if __name__ == "__main__":
    ocr_engine = OcrEngine()
    image = "https://github.com/RapidAI/RapidOCR/blob/main/python/tests/test_files/ch_en_num.jpg?raw=true"
    result = ocr_engine.recognize(image)
    print(result)