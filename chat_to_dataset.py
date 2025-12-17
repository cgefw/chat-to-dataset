import re
import json
import asyncio
import os
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

# ================= 配置区域 =================
INPUT_FILE = r""  # 你的聊天记录路径
OUTPUT_FILE = "sharegpt_finetune.json"          # 输出文件名

# API 配置 (使用 DeepSeek R1 / Reasoner)
API_KEY = ""
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-reasoner"

# 你的ID关键词 (日志里显示的名字)
TARGET_ID_KEYWORD = "yourName" 

# 数据处理参数
THINKING_CONTEXT_LIMIT = 6     # 生成思考时，往前看多少句作为背景
MAX_TURNS_PER_SAMPLE = 6       # 训练数据中，每个对话样本包含的最大轮数
MIN_TURNS_PER_SAMPLE = 2        # 过滤掉太短的对话
MAX_CONCURRENT_REQUESTS = 20    # API并发数，根据你的Tier调整

# ================= 核心：人设克隆 Prompt =================
# 这个 Prompt 旨在让 AI 模仿你的脑回路，特别是“逻辑跳跃”和“懒得解释”的特征
CONTEXT_THINKING_PROMPT = """
你现在就是主角 "you r"。你的任务是回顾你刚才发出的回复，并写下那一刻你的【内心独白】。

【人设资料 (yourName)】：
1. **性格核心**：
   [在此处填写性格描述]
   - 示例：慵懒/卷王/社恐/话痨/毒舌/暖男。
   - 对待感兴趣的事物（如[列举你的爱好]）会非常热情。
   - 对待不感兴趣或麻烦的事（如[列举你讨厌的事]）会表现出（敷衍/暴躁/无视）。

2. **思维模式**：
   [在此处填写思维习惯]
   - 示例：逻辑跳跃、直男思维、情绪化、喜欢刨根问底、或者是“由于太懒只想找捷径”。
   - 遇到问题时，第一反应通常是（找借口/找解决方案/找人吐槽）。

3. **语言风格 (关键)**：
   - **标点习惯**：[例如：从不加句号 / 喜欢用波浪号~ / 喜欢用半个括号 ( ]
   - **常用口癖**：[例如：草、确实、笑死、hdm、捏、寄]
   - **语气特征**：[例如：喜欢卖萌、喜欢阴阳怪气、极度客气、全是表情包]
   - **对待小白/大神的态度**：[例如：对他人的低级错误会直接嘲讽 / 会耐心解释]

4. **内心独白要求**：
   - 字数控制在 **80字以内**。
   - 必须使用**第一人称**。
   - 必须解释清楚“为什么我会回这句话”。如果你的回复和上一句**逻辑跳跃**很大，请务必在内心独白里补全这个 **A->B->C** 的脑补过程。

【Few-Shot 示例 (请替换为你自己的真实对话)】：

示例1（表现你的日常风格）：
对话场景：
A: [朋友的话，例如：今晚出来吃饭吗？]
yourName: [你的回复，例如：不了，我在打游戏]
你的内心独白：
[对应心理活动：其实不是很想动，而且刚开了一把排位，出去社交太消耗能量了，随便找个理由推脱一下吧。]

示例2（表现你的专业/吐槽风格）：
对话场景：
B: [朋友的话，例如：这电脑怎么又蓝屏了]
yourName: [你的回复，例如：重启解决90%的问题]
你的内心独白：
[对应心理活动：这破问题解释起来太麻烦了，他又听不懂原理，直接让他重启是最快的方法，别来烦我。]

---------------------
现在，请观察下方的【对话历史】和【我的实际回复】，生成这一刻的内心独白：
"""
client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

def clean_name(raw_name):
    """清洗名字，区分自己和他人"""
    if TARGET_ID_KEYWORD in raw_name:
        return "yourName"
    # 其他人保留名字，方便AI理解语境，但去除特殊符号
    return raw_name.split(" ")[0].replace(":", "").strip()

def parse_log_to_linear_list(file_path):
    """
    1. 读取日志
    2. 合并多行消息
    3. 合并连续的同角色发言
    4. 标记角色 (gpt/human)
    """
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.rstrip() for line in f.readlines() if line.strip()]

    pattern = re.compile(r"^\[(.*?)\]: (.*)$")
    parsed_msgs = []
    current_msg = None

    # --- 第一步：基础解析 ---
    for line in lines:
        match = pattern.match(line)
        if match:
            if current_msg:
                parsed_msgs.append(current_msg)
            
            raw_name = match.group(1)
            content = match.group(2)
            simple_name = clean_name(raw_name)
            
            # 标记角色：yourName是gpt，其他人是human
            role = "gpt" if simple_name == "yourName" else "human"
            
            current_msg = {
                "role": role,
                "original_name": simple_name,
                "content": content
            }
        else:
            # 处理换行消息
            if current_msg:
                current_msg["content"] += "\n" + line
    
    if current_msg:
        parsed_msgs.append(current_msg)

    # --- 第二步：合并连续同角色发言 ---
    # ShareGPT 要求 human 和 gpt 必须交替出现
    merged_msgs = []
    if not parsed_msgs:
        return []
        
    last_msg = parsed_msgs[0]
    
    for i in range(1, len(parsed_msgs)):
        curr = parsed_msgs[i]
        if curr['role'] == last_msg['role']:
            # 如果是同阵营（比如两个人聊天，对于你来说都是 user），合并内容
            # 在内容中保留说话人名字，方便 AI 区分是谁说的
            if curr['role'] == "human" and curr['original_name'] != last_msg['original_name']:
                last_msg['content'] += f"\n[{curr['original_name']}]: {curr['content']}"
            else:
                last_msg['content'] += "\n" + curr['content']
        else:
            merged_msgs.append(last_msg)
            last_msg = curr
    merged_msgs.append(last_msg)
    
    return merged_msgs

async def generate_thinking_for_msg(sem, msg_index, all_msgs):
    """
    为指定的一条 GPT 回复生成 Inner Monologue
    """
    target_msg = all_msgs[msg_index]
    
    # 获取上下文 (Lookback)
    start = max(0, msg_index - THINKING_CONTEXT_LIMIT)
    context_msgs = all_msgs[start:msg_index]
    
    # 构建上下文文本
    context_str = ""
    for m in context_msgs:
        prefix = "我" if m['role'] == 'gpt' else f"[{m['original_name']}]"
        context_str += f"{prefix}: {m['content']}\n"
    
    prompt = f"""
【对话历史】：
{context_str}
---------------------
【我的实际回复】：
{target_msg['content']}
---------------------
请根据上述内容，输出 JSON 格式的内心独白（key为 "thinking"）。
"""

    async with sem:
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": CONTEXT_THINKING_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                temperature=1 # 稍微调高温度，让思考更活跃/发散
            )
            
            content = response.choices[0].message.content
            
            # 简单的 JSON 提取逻辑
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.replace("```", "").strip()
            
            # 有时候 DeepSeek R1 会在 JSON 外面废话，尝试强制查找 { }
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                content = json_match.group(0)

            data = json.loads(content)
            thinking = data.get("thinking", "")
            
            return {
                "index": msg_index,
                "thinking": thinking
            }
        except Exception as e:
            # 偶尔失败不影响大局，直接跳过
            # print(f"Skipped index {msg_index}: {e}") 
            return None

async def main():
    print(">>> 1. 正在解析聊天记录...")
    all_msgs = parse_log_to_linear_list(INPUT_FILE)
    print(f"    合并后共获得 {len(all_msgs)} 条交互消息。")
    
    if len(all_msgs) < 2:
        print("    数据太少，无法处理。")
        return

    # 找出所有 GPT 的发言索引
    gpt_indices = [i for i, msg in enumerate(all_msgs) if msg['role'] == 'gpt']
    print(f">>> 2. 识别出 {len(gpt_indices)} 条主角回复，开始并发生成思维链...")

    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [generate_thinking_for_msg(sem, idx, all_msgs) for idx in gpt_indices]
    
    # 进度条运行
    results = await tqdm_asyncio.gather(*tasks)

    # 回填 Thinking
    success_cnt = 0
    for res in results:
        if res and res['thinking']:
            idx = res['index']
            original = all_msgs[idx]['content']
            # 格式：<think>...</think>回复内容
            all_msgs[idx]['content'] = f"<think>{res['thinking']}</think>\n{original}"
            success_cnt += 1
            
    print(f"    成功注入 {success_cnt} 条内心戏。")

    # --- 第三步：构建 ShareGPT 格式 ---
    print(">>> 3. 正在切分数据集 (ShareGPT Format)...")
    dataset = []
    
    current_chunk = []
    
    for i, msg in enumerate(all_msgs):
        current_chunk.append({
            "from": msg['role'],
            "value": msg['content']
        })
        
        # 切分逻辑：
        # 1. 长度达到限制
        # 2. 且当前是一个完整回合的结束（下一句是 human，或者已经没了）
        # 这样可以尽量避免把一个 human->gpt 的对子拆散
        
        is_length_limit = len(current_chunk) >= MAX_TURNS_PER_SAMPLE
        is_end = (i == len(all_msgs) - 1)
        next_is_human = (i + 1 < len(all_msgs)) and (all_msgs[i+1]['role'] == 'human')
        
        should_cut = is_end or (is_length_limit and next_is_human)
        
        if should_cut:
            if len(current_chunk) >= MIN_TURNS_PER_SAMPLE:
                # ShareGPT 最佳实践：Human 开头
                # 如果当前块是 gpt 开头，DeepSeek 官方建议 mask 掉，或者直接去掉
                # 这里我们简单粗暴：如果第一句是 gpt，就扔掉，从 human 开始
                while current_chunk and current_chunk[0]['from'] == 'gpt':
                    current_chunk.pop(0)
                
                if current_chunk:
                    dataset.append({
                        "system": "模仿 yourName 的语气和思维模式，根据对话历史进行回答。",
                        "conversations": current_chunk
                    })
            
            current_chunk = []

    # --- 保存 ---
    print(f">>> 4. 保存文件...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        
    print(f"Done! 数据集已保存至: {OUTPUT_FILE}")
    print(f"共生成 {len(dataset)} 个长对话样本。")

if __name__ == "__main__":
    asyncio.run(main())