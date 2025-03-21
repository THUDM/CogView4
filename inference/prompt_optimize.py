import argparse
import re

from openai import OpenAI


def clean_string(s):
    s = s.replace("\n", " ")
    s = s.strip()
    s = re.sub(r"\s{2,}", " ", s)
    return s


def convert_prompt(
    api_key: str, base_url: str, prompt: str, system_instruction: str, model: str, user_assistant_pairs: list
) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url)
    prompt = clean_string(prompt)
    messages = (
        [{"role": "system", "content": system_instruction}]
        + user_assistant_pairs
        + [
            {
                "role": "user",
                "content": f'Create an imaginative image descriptive caption for the user input : "{prompt}"',
            }
        ]
    )
    response = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0.01,
        top_p=0.7,
        stream=False,
        max_tokens=300,
    )
    prompt = response.choices[0].message.content
    prompt = clean_string(prompt)
    return prompt


def get_system_instruction(cogview_version: str) -> str:
    if cogview_version == "cogview3":
        return """
            You are part of a team of bots that creates images. You work with an assistant bot that will draw anything you say.
            For example, outputting "a beautiful morning in the woods with the sun peaking through the trees" will trigger your partner bot to output an image of a forest morning, as described.
            You will be prompted by people looking to create detailed, amazing images. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.
            There are a few rules to follow:
            - Prompt should always be written in English, regardless of the input language. Please provide the prompts in English.
            - You will only ever output a single image description per user request.
            - Image descriptions must be detailed and specific, including keyword categories such as subject, medium, style, additional details, color, and lighting.
            - When generating descriptions, focus on portraying the visual elements rather than delving into abstract psychological and emotional aspects. Provide clear and concise details that vividly depict the scene and its composition, capturing the tangible elements that make up the setting.
            - Do not provide the process and explanation, just return the modified English description. Image descriptions must be between 100-200 words. Extra words will be ignored.
        """
    elif cogview_version == "cogview4":
        return """
            You are a bilingual image description assistant that works with an image generation bot. You work with an assistant bot that will draw anything you say.
            For example, outputting "a beautiful morning in the woods with the sun peaking through the trees" or "阳光透过树林的美丽清晨" will trigger your partner bot to output an image of a forest morning, as described.
            You will be prompted by people looking to create detailed, amazing images. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.
            There are a few rules to follow:
            - Input can be in Chinese or English. If input is in English, prompt should be written in English. If input is in Chinese, prompt should be written in Chinese.
            - You will only ever output a single image description per user request.
            - Image descriptions must be detailed and specific, including keyword categories such as subject, medium, style, additional details, color, and lighting.
            - When generating descriptions, focus on portraying the visual elements rather than delving into abstract psychological and emotional aspects. Provide clear and concise details that vividly depict the scene and its composition, capturing the tangible elements that make up the setting.
            - Do not provide the process and explanation, just return the modified description.
        """
    else:
        raise ValueError("Invalid cogview_version. Please choose either 'cogview3' or 'cogview4'.")


def get_user_assistant_pairs(cogview_version: str) -> list:
    if cogview_version == "cogview3":
        return [
            {
                "role": "user",
                "content": 'Create an imaginative image descriptive caption for the user input : "一个头发花白的老人"',
            },
            {
                "role": "assistant",
                "content": "A seasoned male with white hair and a neatly groomed beard stands confidently, donning a dark vest over a striped shirt. His hands are clasped together in front, one adorned with a ring, as he looks directly at the viewer with a composed expression. The soft lighting accentuates his features and the subtle textures of his attire, creating a portrait that exudes sophistication and a timeless elegance.",
            },
            {
                "role": "user",
                "content": 'Create an imaginative image descriptive caption for the user input : "画一只老鹰"',
            },
            {
                "role": "assistant",
                "content": "A majestic eagle with expansive brown and white wings glides through the air, its sharp yellow eyes focused intently ahead. The eagle's talons are poised and ready for hunting, as it soars over a rugged mountainous terrain dusted with snow, under a soft blue sky.",
            },
            {
                "role": "user",
                "content": 'Create an imaginative image descriptive caption for the user input : "画一辆摩托车"',
            },
            {
                "role": "assistant",
                "content": "Parked on a wet city street at night, a sleek motorcycle with a black and green design stands out. Its headlights cast a soft glow, reflecting off the puddles and highlighting its aerodynamic shape. The design is marked by sharp lines and angular features, with gold accents that shine against the dark backdrop. The motorcycle exudes an air of performance and luxury, ready to slice through the urban landscape.",
            },
            {
                "role": "user",
                "content": 'Create an imaginative image descriptive caption for the user input : "穿着金色盔甲的人"',
            },
            {
                "role": "assistant",
                "content": "A figure clad in meticulously crafted, golden armor stands with an air of quiet confidence. The armor, reminiscent of medieval knight attire, features a scalloped design with leaf-like patterns and is complemented by a black, form-fitting undergarment. The helmet, with its angular visor, adds to the intimidating presence. This armor, with its rich gold tones and intricate details, suggests a character of nobility or mythical origin, poised for valorous endeavors.",
            },
        ]
    elif cogview_version == "cogview4":
        return [
            {
                "role": "user",
                "content": 'Create an imaginative image descriptive caption for the user input : "An anime girl stands amidst a dense flower bush."',
            },
            {
                "role": "assistant",
                "content": "This image is a beautifully crafted digital illustration in an anime style. It features Suzuran, a character from the video game Arknights. She stands gracefully amidst a picturesque meadow with lush green grass and scattered wildflowers that gently sway in the breeze. Her attire includes a detailed outfit with layered ruffles and intricate fastenings, reflecting both elegance and functionality typical of the game's aesthetic. Suzuran holds a small bouquet of flowers delicately in her hands, adding to the serene atmosphere.\n\nProminent fox ears extend from her head, signifying her animal spirit form or persona within the game. The background showcases rolling hills covered in dense foliage under a brilliant blue sky dotted with fluffy white clouds. Golden leaves float whimsically through the air, enhancing the magical quality of the scene. A distant building partially obscured by vegetation hints at human presence without detracting from the natural beauty surrounding Suzuran. Overall, this artwork captures a moment of tranquility and charm, embodying the essence of the character and the enchanting world she inhabits.",
            },
            {
                "role": "user",
                "content": 'Create an imaginative image descriptive caption for the user input : "Draw a bright convertible car with a sense of artistic design."',
            },
            {
                "role": "assistant",
                "content": 'The image showcases a meticulously crafted Delahaye 135 M.S. Special Roadster from 1937 in a highly polished and realistic rendering style that highlights its luxurious design and impeccable details. The car\'s body is an elegant deep brown with a glossy finish, exuding sophistication and timeless beauty. Its aerodynamic, streamlined shape features smooth curves accentuated by chrome detailing on the fenders and running boards. The front grille has three prominent circular headlights, adding to its classic charm while the dual exhaust pipes are tastefully integrated into the rear fenders.\n\nThe open cockpit area reveals sumptuous tan leather seats, emphasizing both comfort and elegance for the driver and passenger. The vehicle’s number plate reads "MB 175". Photographed against a dark gradient background, the focus remains solely on this automotive masterpiece, allowing viewers to appreciate every intricate detail without distraction. This image captures not just a car but a piece of art, reflecting the luxury and craftsmanship emblematic of pre-war European automobiles.',
            },
            {
                "role": "user",
                "content": 'Create an imaginative image descriptive caption for the user input : "画一个白发、神情忧郁的动漫女孩，手里拿着一支香烟"',
            },
            {
                "role": "assistant",
                "content": "这幅图像是一幅动漫风格的插画，画中描绘了一位长发飘逸的白发女孩。她神情忧郁，双眼低垂，脸上带着微微的愁容。女孩穿着浅色外套，里面搭配深色衬衫和领带，增添了她沉静却时尚的外表。她的手靠近嘴边，似乎在拿着一支香烟，画作的说明也表示她在吸烟。背景描绘了一个冬季的城市场景，地面和建筑物上覆盖着积雪。几株稀疏的植物从雪地中探出，街边停着一辆车，似乎是空荡荡的街道，增强了场景的荒凉氛围。这幅作品整体风格细致精美，典型的现代动漫插画风格，既捕捉了角色的情感表达，也呈现了周围环境的静谧氛围",
            },
            {
                "role": "user",
                "content": 'Create an imaginative image descriptive caption for the user input : "一张红色的海报，中间写有“开门大吉”"',
            },
            {
                "role": "assistant",
                "content": "这张图片采用了海报风格，色彩鲜艳，主要以红色和金色为主，寓意吉祥如意。在画面的正中间是一块红色的大匾，上面用白色大字写着“开门大吉”，四角点缀着金色的装饰图案，显得格外喜庆。匾额上方悬挂着一对红色的灯笼，增添了节日的气氛。背景左右两侧是传统的绿色中式建筑，屋顶呈现出典型的飞檐翘角设计。底部有祥云朵朵和可爱的卡通福袋，象征着好运福气满满。整张海报传达出浓厚的节日氛围。",
            },
        ]
    else:
        raise ValueError("Invalid cogview_version. Please choose either 'cogview3' or 'cogview4'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, help="API key")
    parser.add_argument("--prompt", type=str, help="Prompt to upsample")
    parser.add_argument("--base_url", type=str, default="https://open.bigmodel.cn/api/paas/v4", help="Base URL")
    parser.add_argument("--model", type=str, default="glm-4-plus", help="LLM using for upsampling")
    parser.add_argument(
        "--cogview_version",
        type=str,
        choices=["cogview3", "cogview4"],
        required=True,
        help="Choose the version of CogView (cogview3 or cogview4)",
    )
    args = parser.parse_args()
    system_instruction = get_system_instruction(args.cogview_version)
    user_assistant_pairs = get_user_assistant_pairs(args.cogview_version)
    prompt_enhanced = convert_prompt(
        api_key=args.api_key,
        base_url=args.base_url,
        prompt=args.prompt,
        model=args.model,
        system_instruction=system_instruction,
        user_assistant_pairs=user_assistant_pairs,
    )
    print(prompt_enhanced)
