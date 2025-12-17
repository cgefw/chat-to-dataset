import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import threading
import sys
import os
import re
import json
import asyncio
from openai import AsyncOpenAI
# 注意：在GUI中重定向tqdm比较复杂，这里我们用简单的文本日志代替进度条，或者将tqdm输出重定向
# 为了界面稳定性，本代码将移除tqdm_asyncio的图形化进度条，改为文本日志输出

# ================= 默认配置常量 =================
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL_NAME = "deepseek-reasoner"
DEFAULT_THINKING_CTX = 6
DEFAULT_MAX_TURNS = 6
DEFAULT_MIN_TURNS = 2
DEFAULT_CONCURRENT = 20

# 默认 Prompt 模板
DEFAULT_PROMPT = """你现在就是主角 "you r"。你的任务是回顾你刚才发出的回复，并写下那一刻你的【内心独白】。

【人设资料 (yourName)】：
1. **性格核心**：
   [在此处填写性格描述]
2. **思维模式**：
   [在此处填写思维习惯]
3. **语言风格**：
   - 标点习惯：...
   - 常用口癖：...
4. **内心独白要求**：
   - 字数控制在 **80字以内**。
   - 必须使用**第一人称**。
   - 必须解释清楚“为什么我会回这句话”。如果你的回复和上一句**逻辑跳跃**很大，请务必在内心独白里补全这个 **A->B->C** 的脑补过程。

---------------------
现在，请观察下方的【对话历史】和【我的实际回复】，生成这一刻的内心独白：
"""

# ================= 核心逻辑类 =================
class ConverterLogic:
    def __init__(self, log_callback):
        self.log = log_callback
        self.stop_flag = False

    def clean_name(self, raw_name, target_keyword):
        """清洗名字，区分自己和他人"""
        # 去掉首尾空格
        raw_name = raw_name.strip()
        # 如果包含关键词，直接标记为自己
        if target_keyword in raw_name:
            return "yourName"
        # 简单处理：如果是 "Name(12345)" 这种格式，只取括号前的名字
        # 同时去掉可能存在的冒号等符号
        raw_name = re.split(r"[\(\<]", raw_name)[0].strip()
        return raw_name.replace(":", "").replace("：", "")

    def is_system_message(self, content):
        """过滤系统消息"""
        black_keywords = [
            "撤回了一条消息", "加入了群聊", "拍了拍", 
            "通话时长", "语音通话", "邀请你", 
            "对方已成功接收", "当前版本不支持",
            "均未接听"
        ]
        content = content.strip()
        if not content: return True
        # 过滤纯图片/表情/视频占位符
        if content in ["[图片]", "[表情]", "[视频]", "[动画表情]"]: return True
        
        for kw in black_keywords:
            if kw in content:
                return True
        return False

    def smart_parse(self, lines):
        """
        智能解析器：兼容 QQ/微信导出格式 和 标准格式
        """
        parsed_msgs = []
        
        # 1. 时间戳特征正则 (QQ: 2023-12-12 12:12:12 | WeChat: 12:12)
        time_pattern = re.compile(r"(\d{4}[-/]\d{2}[-/]\d{2})|(\d{1,2}:\d{2})")
        
        # 2. 标准格式正则: [Name]: Content
        standard_pattern = re.compile(r"^\[(.*?)\]: (.*)$")

        current_name = None
        current_content_lines = []

        def flush_msg():
            if current_name and current_content_lines:
                content = "\n".join(current_content_lines).strip()
                if not self.is_system_message(content):
                    parsed_msgs.append({
                        "original_name": current_name,
                        "content": content
                    })

        for line in lines:
            line = line.strip()
            if not line: continue

            # A. 优先尝试匹配标准格式 [Name]: Msg (兼容旧数据)
            std_match = standard_pattern.match(line)
            if std_match:
                flush_msg() 
                current_name = std_match.group(1)
                content = std_match.group(2)
                current_content_lines = [content]
                continue

            # B. 尝试匹配 Header 行 (QQ/微信导出格式)
            # 判定规则：行比较短 + 包含时间戳 + 不是长文本
            is_header = False
            if len(line) < 60 and time_pattern.search(line):
                is_header = True
            
            if is_header:
                flush_msg() # 上一条消息结束
                # 提取名字：把时间戳和多余符号去掉
                # 微信: "张三 12:30" -> "张三"
                # QQ: "李四 2023/1/1 12:00:00" -> "李四"
                temp_name = time_pattern.sub("", line).strip()
                current_name = temp_name
                current_content_lines = []
            else:
                # C. 既不是标准头，也不是新Header，归为上一条的内容
                if current_name:
                    current_content_lines.append(line)

        flush_msg() # 存最后一条
        return parsed_msgs

    def parse_log(self, file_path, target_keyword):
        if not os.path.exists(file_path):
            self.log(f"[Error] 文件不存在: {file_path}")
            return []

        # 读取所有行
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        self.log(f"    正在智能分析 {len(lines)} 行原始数据...")
        
        # === 使用新的智能解析器 ===
        raw_objs = self.smart_parse(lines)
        
        self.log(f"    初步识别出 {len(raw_objs)} 条有效消息，开始格式化...")

        # 后处理：转换为 role 并合并
        merged_msgs = []
        if not raw_objs: return []

        processed_objs = []
        for obj in raw_objs:
            simple_name = self.clean_name(obj['original_name'], target_keyword)
            role = "gpt" if simple_name == "yourName" else "human"
            processed_objs.append({
                "role": role,
                "original_name": simple_name,
                "content": obj['content']
            })

        # 合并连续发言
        last_msg = processed_objs[0]
        for i in range(1, len(processed_objs)):
            curr = processed_objs[i]
            if curr['role'] == last_msg['role']:
                # 同一阵营合并
                if curr['role'] == "human" and curr['original_name'] != last_msg['original_name']:
                    # 不同的人说话（群聊），带上名字
                    last_msg['content'] += f"\n[{curr['original_name']}]: {curr['content']}"
                else:
                    # 同一个人或自己，直接拼内容
                    last_msg['content'] += "\n" + curr['content']
            else:
                merged_msgs.append(last_msg)
                last_msg = curr
        merged_msgs.append(last_msg)
        
        return merged_msgs

    async def generate_thinking(self, client, sem, msg_index, all_msgs, model_name, ctx_limit, prompt_template):
        target_msg = all_msgs[msg_index]
        start = max(0, msg_index - ctx_limit)
        context_msgs = all_msgs[start:msg_index]
        
        context_str = ""
        for m in context_msgs:
            prefix = "我" if m['role'] == 'gpt' else f"[{m['original_name']}]"
            context_str += f"{prefix}: {m['content']}\n"
        
        user_prompt = f"""
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
                    model=model_name,
                    messages=[
                        {"role": "system", "content": prompt_template},
                        {"role": "user", "content": user_prompt},
                    ],
                    stream=False,
                    temperature=1.0
                )
                content = response.choices[0].message.content
                
                # 清洗 JSON
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.replace("```", "").strip()
                
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if json_match:
                    content = json_match.group(0)

                data = json.loads(content)
                return {"index": msg_index, "thinking": data.get("thinking", "")}
            except Exception as e:
                # self.log(f"[Warn] ID {msg_index} 生成失败: {str(e)}")
                return None

    async def run_async(self, config):
        self.log(">>> 1. 正在解析聊天记录...")
        all_msgs = self.parse_log(config['input_file'], config['target_keyword'])
        self.log(f"    合并后共获得 {len(all_msgs)} 条交互消息。")
        
        if len(all_msgs) < 2:
            self.log("[Error] 数据太少，中止。")
            return

        gpt_indices = [i for i, msg in enumerate(all_msgs) if msg['role'] == 'gpt']
        self.log(f">>> 2. 识别出 {len(gpt_indices)} 条主角回复，开始并发生成思维链...")
        
        client = AsyncOpenAI(api_key=config['api_key'], base_url=config['base_url'])
        sem = asyncio.Semaphore(config['concurrent'])
        
        tasks = [
            self.generate_thinking(client, sem, idx, all_msgs, config['model'], config['ctx_limit'], config['prompt']) 
            for idx in gpt_indices
        ]
        
        # 简单的进度展示
        total = len(tasks)
        completed = 0
        results = []
        
        for f in asyncio.as_completed(tasks):
            res = await f
            results.append(res)
            completed += 1
            if completed % 5 == 0 or completed == total:
                self.log(f"    进度: {completed}/{total}")

        success_cnt = 0
        for res in results:
            if res and res['thinking']:
                idx = res['index']
                original = all_msgs[idx]['content']
                all_msgs[idx]['content'] = f"<think>{res['thinking']}</think>\n{original}"
                success_cnt += 1
        
        self.log(f"    成功注入 {success_cnt} 条内心戏。")

        # 切分数据集
        self.log(">>> 3. 正在生成 ShareGPT 格式...")
        dataset = []
        current_chunk = []
        
        for i, msg in enumerate(all_msgs):
            current_chunk.append({"from": msg['role'], "value": msg['content']})
            
            is_limit = len(current_chunk) >= config['max_turns']
            is_end = (i == len(all_msgs) - 1)
            next_human = (i + 1 < len(all_msgs)) and (all_msgs[i+1]['role'] == 'human')
            
            if is_end or (is_limit and next_human):
                if len(current_chunk) >= config['min_turns']:
                    # 确保 Human 开头
                    while current_chunk and current_chunk[0]['from'] == 'gpt':
                        current_chunk.pop(0)
                    
                    if current_chunk:
                        dataset.append({
                            "system": "模仿 yourName 的语气和思维模式，根据对话历史进行回答。",
                            "conversations": current_chunk
                        })
                current_chunk = []

        with open(config['output_file'], 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
            
        self.log(f"[Success] 处理完成！共生成 {len(dataset)} 条数据。")
        self.log(f"文件保存至: {config['output_file']}")

# ================= GUI 主程序 =================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Chat-to-Dataset Converter (DeepSeek R1)")
        self.root.geometry("800x750")

        # 变量绑定
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar(value="sharegpt_finetune.json")
        self.api_key = tk.StringVar()
        self.target_keyword = tk.StringVar(value="YourNameInLog")
        
        # 默认值配置区域变量
        self.base_url = tk.StringVar(value=DEFAULT_BASE_URL)
        self.model_name = tk.StringVar(value=DEFAULT_MODEL_NAME)
        self.ctx_limit = tk.IntVar(value=DEFAULT_THINKING_CTX)
        self.max_turns = tk.IntVar(value=DEFAULT_MAX_TURNS)
        self.min_turns = tk.IntVar(value=DEFAULT_MIN_TURNS)
        self.concurrent = tk.IntVar(value=DEFAULT_CONCURRENT)

        self.setup_ui()

    def setup_ui(self):
        # 1. 文件选择
        frame_file = tk.LabelFrame(self.root, text="文件设置", padx=10, pady=5)
        frame_file.pack(fill="x", padx=10, pady=5)
        
        self.create_file_input(frame_file, "输入日志路径:", self.input_path, True)
        self.create_file_input(frame_file, "输出文件路径:", self.output_path, False)

        # 2. 核心设置
        frame_core = tk.LabelFrame(self.root, text="核心配置", padx=10, pady=5)
        frame_core.pack(fill="x", padx=10, pady=5)

        tk.Label(frame_core, text="API Key:").grid(row=0, column=0, sticky="e")
        tk.Entry(frame_core, textvariable=self.api_key, show="*", width=30).grid(row=0, column=1, sticky="w", padx=5)
        
        tk.Label(frame_core, text="你的日志ID关键词:").grid(row=0, column=2, sticky="e")
        tk.Entry(frame_core, textvariable=self.target_keyword, width=20).grid(row=0, column=3, sticky="w", padx=5)

        tk.Label(frame_core, text="Base URL:").grid(row=1, column=0, sticky="e", pady=5)
        tk.Entry(frame_core, textvariable=self.base_url, width=30).grid(row=1, column=1, sticky="w", padx=5)

        tk.Label(frame_core, text="Model Name:").grid(row=1, column=2, sticky="e")
        tk.Entry(frame_core, textvariable=self.model_name, width=20).grid(row=1, column=3, sticky="w", padx=5)

        # 3. 高级参数 (一行显示)
        frame_adv = tk.LabelFrame(self.root, text="参数调整", padx=10, pady=5)
        frame_adv.pack(fill="x", padx=10, pady=5)
        
        self.create_label_entry(frame_adv, "Thinking上下文数:", self.ctx_limit, 0)
        self.create_label_entry(frame_adv, "最大对话轮数:", self.max_turns, 1)
        self.create_label_entry(frame_adv, "最小对话轮数:", self.min_turns, 2)
        self.create_label_entry(frame_adv, "API并发数:", self.concurrent, 3)

        # 4. Prompt 编辑区
        frame_prompt = tk.LabelFrame(self.root, text="System Prompt (控制思考风格)", padx=10, pady=5)
        frame_prompt.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.txt_prompt = scrolledtext.ScrolledText(frame_prompt, height=8)
        self.txt_prompt.pack(fill="both", expand=True)
        self.txt_prompt.insert(tk.END, DEFAULT_PROMPT)

        # 5. 操作区
        frame_action = tk.Frame(self.root, pady=5)
        frame_action.pack(fill="x", padx=10)
        
        self.btn_run = tk.Button(frame_action, text="开始处理", command=self.start_processing, 
                                 bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), height=2)
        self.btn_run.pack(fill="x")

        # 6. 日志区
        self.txt_log = scrolledtext.ScrolledText(self.root, height=10, bg="#f0f0f0")
        self.txt_log.pack(fill="both", expand=True, padx=10, pady=10)

    def create_file_input(self, parent, label, var, is_input):
        f = tk.Frame(parent)
        f.pack(fill="x", pady=2)
        tk.Label(f, text=label, width=12, anchor="e").pack(side="left")
        tk.Entry(f, textvariable=var).pack(side="left", fill="x", expand=True, padx=5)
        cmd = self.browse_input if is_input else self.browse_output
        tk.Button(f, text="浏览", command=cmd).pack(side="left")

    def create_label_entry(self, parent, text, var, col):
        f = tk.Frame(parent)
        f.grid(row=0, column=col, padx=10)
        tk.Label(f, text=text).pack()
        tk.Entry(f, textvariable=var, width=10, justify="center").pack()

    def browse_input(self):
        f = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if f: self.input_path.set(f)

    def browse_output(self):
        f = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
        if f: self.output_path.set(f)

    def log(self, msg):
        self.txt_log.insert(tk.END, msg + "\n")
        self.txt_log.see(tk.END)

    def start_processing(self):
        # 获取所有配置
        config = {
            "input_file": self.input_path.get(),
            "output_file": self.output_path.get(),
            "api_key": self.api_key.get(),
            "target_keyword": self.target_keyword.get(),
            "base_url": self.base_url.get(),
            "model": self.model_name.get(),
            "ctx_limit": self.ctx_limit.get(),
            "max_turns": self.max_turns.get(),
            "min_turns": self.min_turns.get(),
            "concurrent": self.concurrent.get(),
            "prompt": self.txt_prompt.get("1.0", tk.END).strip()
        }

        # 验证
        if not config["input_file"] or not config["api_key"]:
            messagebox.showerror("错误", "请至少填写 输入文件 和 API Key")
            return

        self.btn_run.config(state="disabled", text="运行中...")
        self.txt_log.delete(1.0, tk.END)
        
        # 启动线程
        t = threading.Thread(target=self.run_thread, args=(config,))
        t.daemon = True
        t.start()

    def run_thread(self, config):
        logic = ConverterLogic(self.log)
        asyncio.run(logic.run_async(config))
        self.btn_run.config(state="normal", text="开始处理")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
