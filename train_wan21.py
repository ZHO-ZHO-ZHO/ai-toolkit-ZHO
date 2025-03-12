import os
from huggingface_hub import whoami    
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import sys
import gradio as gr
from PIL import Image
import torch
import uuid
import shutil
import json
import yaml
from slugify import slugify
from transformers import AutoProcessor, AutoModelForCausalLM

# 将 ai-toolkit 添加到路径
sys.path.insert(0, os.path.join(os.getcwd(), "ai-toolkit"))
from toolkit.job import get_job

MAX_IMAGES = 150

# Wan2.1 默认配置文件
WAN21_CONFIG = """
job: extension
config:
  name: "my_first_wan21_14b_lora_v1"
  process:
    - type: 'sd_trainer'
      training_folder: "output"
      device: cuda:0
      trigger_word: "p3r5on"
      network:
        type: "lora"
        linear: 32
        linear_alpha: 32
      save:
        dtype: float16
        save_every: 250
        max_step_saves_to_keep: 4
        push_to_hub: false
      datasets:
        - folder_path: "/path/to/images/folder"
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          shuffle_tokens: false
          cache_latents_to_disk: true
          resolution: [ 632 ]
      train:
        batch_size: 1
        steps: 2000
        gradient_accumulation: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        timestep_type: 'sigmoid'
        optimizer: "adamw8bit"
        lr: 1e-4
        optimizer_params:
          weight_decay: 1e-4
        ema_config:
          use_ema: true
          ema_decay: 0.99
        dtype: bf16
        unload_text_encoder: true
      model:
        name_or_path: "Wan-AI/Wan2.1-T2V-14B-Diffusers"
        arch: 'wan21'
        quantize: true
        quantize_te: true
        low_vram: true 
      sample:
        sampler: "flowmatch"
        sample_every: 250
        width: 832
        height: 480
        num_frames: 40
        fps: 15
        prompts:
          - "woman playing the guitar, on stage, singing a song, laser lights, punk rocker"
        neg: ""  
        seed: 42
        walk_seed: true
        guidance_scale: 5
        sample_steps: 30
meta:
  name: "[name]"
  version: '1.0'
"""

def load_captioning(uploaded_files, concept_sentence):
    uploaded_images = [file for file in uploaded_files if not file.endswith('.txt')]
    txt_files = [file for file in uploaded_files if file.endswith('.txt')]
    txt_files_dict = {os.path.splitext(os.path.basename(txt_file))[0]: txt_file for txt_file in txt_files}
    updates = []
    if len(uploaded_images) <= 1:
        raise gr.Error("请至少上传 2 张图片以训练模型（默认设置下理想数量为 4-30 张）")
    elif len(uploaded_images) > MAX_IMAGES:
        raise gr.Error(f"目前最多允许上传 {MAX_IMAGES} 张图片")
    updates.append(gr.update(visible=True))
    for i in range(1, MAX_IMAGES + 1):
        visible = i <= len(uploaded_images)
        updates.append(gr.update(visible=visible))
        image_value = uploaded_images[i - 1] if visible else None
        updates.append(gr.update(value=image_value, visible=visible))
        corresponding_caption = False
        if image_value:
            base_name = os.path.splitext(os.path.basename(image_value))[0]
            if base_name in txt_files_dict:
                with open(txt_files_dict[base_name], 'r') as file:
                    corresponding_caption = file.read()
        text_value = corresponding_caption if visible and corresponding_caption else "[trigger]" if visible and concept_sentence else None
        updates.append(gr.update(value=text_value, visible=visible))
    updates.append(gr.update(visible=True))
    updates.append(gr.update(placeholder=f'动态场景中的人物 {concept_sentence}', value=f'人物在舞台上演奏吉他 {concept_sentence}'))
    updates.append(gr.update(placeholder=f"激光灯光下的 {concept_sentence}"))
    updates.append(gr.update(placeholder=f"朋克摇滚风格的 {concept_sentence}"))
    updates.append(gr.update(visible=True))
    return updates

def hide_captioning():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

def create_dataset(*inputs):
    print("创建数据集")
    images = inputs[0]
    destination_folder = f"datasets/{uuid.uuid4()}"
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    jsonl_file_path = os.path.join(destination_folder, "metadata.jsonl")
    with open(jsonl_file_path, "a") as jsonl_file:
        for index, image in enumerate(images):
            new_image_path = shutil.copy(image, destination_folder)
            original_caption = inputs[index + 1]
            file_name = os.path.basename(new_image_path)
            data = {"file_name": file_name, "prompt": original_caption}
            jsonl_file.write(json.dumps(data) + "\n")
    return destination_folder

def run_captioning(images, concept_sentence, *captions):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        "multimodalart/Florence-2-large-no-flash-attn", torch_dtype=torch_dtype, trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained("multimodalart/Florence-2-large-no-flash-attn", trust_remote_code=True)
    captions = list(captions)
    for i, image_path in enumerate(images):
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        prompt = "<DETAILED_CAPTION>"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text, task=prompt, image_size=(image.width, image.height)
        )
        caption_text = parsed_answer["<DETAILED_CAPTION>"].replace("The image shows ", "")
        if concept_sentence:
            caption_text = f"{caption_text} [trigger]"
        captions[i] = caption_text
        yield captions
    model.to("cpu")
    del model
    del processor

def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and v:
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def start_training(
    lora_name, concept_sentence, steps, lr, rank, low_vram, dataset_folder, sample_1, sample_2, sample_3,
    use_more_advanced_options, more_advanced_options
):
    push_to_hub = True
    if not lora_name:
        raise gr.Error("请填写 LoRA 名称！名称必须唯一。")
    try:
        if whoami()["auth"]["accessToken"]["role"] == "write" or "repo.write" in whoami()["auth"]["accessToken"]["fineGrained"][0]["permissions"]:
            gr.Info(f"开始本地训练 {whoami()['name']}。训练完成后，LoRA 将保存到本地和 Hugging Face。")
        else:
            push_to_hub = False
            gr.Warning("开始本地训练。由于未使用具有写权限的 Hugging Face 令牌，LoRA 将仅保存到本地。")
    except:
        push_to_hub = False
        gr.Warning("开始本地训练。由于未登录 Hugging Face，LoRA 将仅保存到本地。")
    
    print("开始训练")
    slugged_lora_name = slugify(lora_name)
    config = yaml.safe_load(WAN21_CONFIG)
    config["config"]["name"] = slugged_lora_name
    config["config"]["process"][0]["train"]["steps"] = int(steps)
    config["config"]["process"][0]["train"]["lr"] = float(lr)
    config["config"]["process"][0]["network"]["linear"] = int(rank)
    config["config"]["process"][0]["network"]["linear_alpha"] = int(rank)
    config["config"]["process"][0]["datasets"][0]["folder_path"] = dataset_folder
    config["config"]["process"][0]["save"]["push_to_hub"] = push_to_hub
    config["config"]["process"][0]["model"]["low_vram"] = low_vram
    if push_to_hub:
        try:
            username = whoami()["name"]
        except:
            raise gr.Error("获取用户名失败。请确认已登录 Hugging Face。")
        config["config"]["process"][0]["save"]["hf_repo_id"] = f"{username}/{slugged_lora_name}"
        config["config"]["process"][0]["save"]["hf_private"] = True
    if concept_sentence:
        config["config"]["process"][0]["trigger_word"] = concept_sentence
    if sample_1 or sample_2 or sample_3:
        config["config"]["process"][0]["sample"]["prompts"] = []
        if sample_1:
            config["config"]["process"][0]["sample"]["prompts"].append(sample_1)
        if sample_2:
            config["config"]["process"][0]["sample"]["prompts"].append(sample_2)
        if sample_3:
            config["config"]["process"][0]["sample"]["prompts"].append(sample_3)
    else:
        config["config"]["process"][0]["sample"]["prompts"] = [f"{concept_sentence} in a dynamic scene"]
    if use_more_advanced_options:
        more_advanced_options_dict = yaml.safe_load(more_advanced_options)
        config["config"]["process"][0] = recursive_update(config["config"]["process"][0], more_advanced_options_dict)
    
    random_config_name = str(uuid.uuid4())
    os.makedirs("tmp", exist_ok=True)
    config_path = f"tmp/{random_config_name}-{slugged_lora_name}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    job = get_job(config_path)
    job.run()
    job.cleanup()
    return f"训练完成。模型保存为 {slugged_lora_name}"

theme = gr.themes.Monochrome(
    text_size=gr.themes.Size(lg="18px", md="15px", sm="13px", xl="22px", xs="12px", xxl="24px", xxs="9px"),
    font=[gr.themes.GoogleFont("Source Sans Pro"), "ui-sans-serif", "system-ui", "sans-serif"],
)
css = """
h1{font-size: 2em}
h3{margin-top: 0}
#component-1{text-align:center}
.main_ui_logged_out{opacity: 0.3; pointer-events: none}
.tabitem{border: 0px}
.group_padding{padding: .55em}
"""

with gr.Blocks(theme=theme, css=css) as demo:
    gr.Markdown(
        """# Wan2.1 视频模型训练界面 🧞‍♂️
### 使用 [Ostris' AI Toolkit](https://github.com/ostris/ai-toolkit) 轻松训练 Wan2.1 LoRA"""
    )
    with gr.Column() as main_ui:
        with gr.Row():
            lora_name = gr.Textbox(
                label="LoRA 名称",
                info="必须是唯一的名称",
                placeholder="例如：my_wan21_lora",
            )
            concept_sentence = gr.Textbox(
                label="触发词/句子",
                info="用于激活模型的触发词或句子",
                placeholder="例如：p3r5on 或 'punk rocker style'",
            )
        with gr.Group(visible=True) as image_upload:
            with gr.Row():
                images = gr.File(
                    file_types=["image", ".txt"],
                    label="上传训练图像",
                    file_count="multiple",
                    interactive=True,
                    visible=True,
                    scale=1,
                )
                with gr.Column(scale=3, visible=False) as captioning_area:
                    with gr.Column():
                        gr.Markdown(
                            """# 自定义描述
<p style="margin-top:0">可以为每张图像添加自定义描述，或使用 AI 模型生成。[trigger] 将代表触发词/句子。</p>
""", elem_classes="group_padding")
                        do_captioning = gr.Button("使用 Florence-2 添加 AI 描述")
                        output_components = [captioning_area]
                        caption_list = []
                        for i in range(1, MAX_IMAGES + 1):
                            locals()[f"captioning_row_{i}"] = gr.Row(visible=False)
                            with locals()[f"captioning_row_{i}"]:
                                locals()[f"image_{i}"] = gr.Image(
                                    type="filepath", width=111, height=111, min_width=111,
                                    interactive=False, scale=2, show_label=False,
                                    show_share_button=False, show_download_button=False,
                                )
                                locals()[f"caption_{i}"] = gr.Textbox(label=f"描述 {i}", scale=15, interactive=True)
                            output_components.append(locals()[f"captioning_row_{i}"])
                            output_components.append(locals()[f"image_{i}"])
                            output_components.append(locals()[f"caption_{i}"])
                            caption_list.append(locals()[f"caption_{i}"])

        with gr.Accordion("高级选项", open=False):
            steps = gr.Number(label="训练步数", value=2000, minimum=500, maximum=4000, step=1)
            lr = gr.Number(label="学习率", value=1e-4, minimum=1e-6, maximum=1e-3, step=1e-6)
            rank = gr.Number(label="LoRA 秩", value=32, minimum=4, maximum=128, step=4)
            low_vram = gr.Checkbox(label="低显存模式", value=True)
            with gr.Accordion("更多高级选项", open=False):
                use_more_advanced_options = gr.Checkbox(label="使用更多高级选项", value=False)
                more_advanced_options = gr.Code(WAN21_CONFIG, language="yaml")

        with gr.Accordion("样本提示词（可选）", visible=False) as sample:
            gr.Markdown("输入测试提示词以生成训练过程中的样本动画。建议包含触发词/句子。")
            sample_1 = gr.Textbox(label="测试提示词 1")
            sample_2 = gr.Textbox(label="测试提示词 2")
            sample_3 = gr.Textbox(label="测试提示词 3")
        
        output_components.append(sample)
        output_components.append(sample_1)
        output_components.append(sample_2)
        output_components.append(sample_3)
        start = gr.Button("开始训练", visible=False)
        output_components.append(start)
        progress_area = gr.Markdown("")

    dataset_folder = gr.State()

    images.upload(load_captioning, inputs=[images, concept_sentence], outputs=output_components)
    images.delete(load_captioning, inputs=[images, concept_sentence], outputs=output_components)
    images.clear(hide_captioning, outputs=[captioning_area, sample, start])
    
    start.click(fn=create_dataset, inputs=[images] + caption_list, outputs=dataset_folder).then(
        fn=start_training,
        inputs=[lora_name, concept_sentence, steps, lr, rank, low_vram, dataset_folder, sample_1, sample_2, sample_3,
                use_more_advanced_options, more_advanced_options],
        outputs=progress_area,
    )
    do_captioning.click(fn=run_captioning, inputs=[images, concept_sentence] + caption_list, outputs=caption_list)

if __name__ == "__main__":
    demo.launch(share=True, show_error=True)
