from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from typing import List
import torch

class ImageCaptioner:
    """
    Generates captions and tags for images using BLIP.
    """
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def generate_caption(self, image: Image.Image) -> str:
        """
        Generate a caption for an image.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        output = self.model.generate(**inputs, max_new_tokens=50)
        caption = self.processor.decode(output[0], skip_special_tokens=True)

        return caption

    def generate_tags(self, caption: str, max_tags: int = 3) -> List[str]:
        """
        Generate simple tags from caption.
        """
        words = caption.lower().split()

        # Basic filtering
        stopwords = {"a", "an", "the", "is", "on", "in", "with", "and", "of"}
        keywords = [word.strip(".,") for word in words if word not in stopwords]

        # Unique + limit
        unique_keywords = list(dict.fromkeys(keywords))

        return unique_keywords[:max_tags]

    def process_image(self, image_path: str) -> dict:
        """
        Full pipeline: caption + tags.
        """
        image = Image.open(image_path).convert("RGB")

        caption = self.generate_caption(image)
        tags = self.generate_tags(caption)

        return {
            "caption": caption,
            "tags": tags
        }
