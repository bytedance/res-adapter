# Copyright (2024) Bytedance Ltd. and/or its affiliates 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 

from PIL import ImageDraw, ImageFont

from torchvision import transforms

def draw_text_on_images(image, text):
    pil_image = transforms.ToPILImage()(image)
    draw = ImageDraw.Draw(pil_image)
    max_length = max(image.shape[-2], image.shape[-1])

    if max_length >= 512:
        font_scale = 0.1
    else:
        font_scale = 0.2
    font_size = int(font_scale * max_length)
    font = ImageFont.truetype("assets/Times-Newer-Roman-Bold-Italic.otf", size=font_size)

    offset = 10
    x = offset
    y = offset

    draw.text((x, y), text, fill=(56, 136, 239), font=font)
    image = transforms.ToTensor()(pil_image)
    return image