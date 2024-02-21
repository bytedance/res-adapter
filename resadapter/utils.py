from PIL import ImageDraw, ImageFont

from torchvision import transforms

def draw_text_on_images(image, text):
    pil_image = transforms.ToPILImage()(image)
    draw = ImageDraw.Draw(pil_image)
    font_size = int(0.08 * image.shape[-1])
    font = ImageFont.truetype("assets/Times-Newer-Roman-Bold-Italic.otf", size=font_size)
    draw.text((10, 10), text, fill=(3, 176, 80), font=font)
    image = transforms.ToTensor()(pil_image)
    return image