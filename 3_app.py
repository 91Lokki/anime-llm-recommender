import os
import json
import torch
import pandas as pd
import gradio as gr
from difflib import SequenceMatcher
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ================= 設定區 =================
# ★ 如果你已經在 CMD 用 huggingface-cli login 登入過，設為 True 即可
HF_TOKEN = True 

# 路徑設定
BASE_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
# ★ 請確認這裡指到你剛剛訓練完的資料夾 (V13)
LORA_PATH = "./llama3.2_sft_v13"  
CSV_PATH = "mal_anime.csv"             
# =========================================

# --- 1. 載入資料庫 (事實查核層 - RAG) ---
print("📂 Loading Database for RAG...")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"❌ 找不到 {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
cols = {c.lower(): c for c in df.columns}

# 欄位對照
c_title = cols.get('title') or 'title'
c_desc = cols.get('description') or cols.get('synopsis') or 'description'
c_score = cols.get('score') or 'score'
c_img = cols.get('image') or cols.get('img_url') or 'image' 
c_studios = cols.get('studios') or cols.get('studio') or 'studios'
c_year = cols.get('released_year') or cols.get('released_ye') or 'year'

# 轉成 List 方便搜尋
all_titles = df[c_title].dropna().astype(str).tolist()

def find_anime_in_db(query_title):
    """
    資料庫檢索器：
    不使用作弊字典，完全依賴 AI 輸出正確片名，
    只做基本的「包含搜尋」與「模糊搜尋」來容錯。
    """
    query_title = str(query_title).strip()
    query_lower = query_title.lower()
    
    # 1. 精確比對
    match = df[df[c_title].str.lower() == query_lower]
    if not match.empty:
        return match.iloc[0]
    
    # 2. 包含搜尋 (解決 Ponyo -> Gake no Ue no Ponyo)
    # 這是必要的，因為日文片名通常很長，AI 講簡稱是合理的
    if len(query_lower) >= 3:
        for db_t in all_titles:
            db_t_str = str(db_t).lower()
            if query_lower in db_t_str:
                return df[df[c_title] == db_t].iloc[0]

    # 3. 模糊比對 (容許一點點拼字錯誤)
    best_ratio = 0
    best_title = None
    
    for db_t in all_titles:
        if abs(len(query_lower) - len(db_t)) / len(query_lower) > 0.6:
            continue
        ratio = SequenceMatcher(None, query_lower, db_t.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_title = db_t
            
    if best_ratio > 0.6: 
        return df[df[c_title] == best_title].iloc[0]
    
    return None

# --- 2. 載入 AI 模型 (SFT Logic) ---
print(f"🚀 Loading Trained Model from {LORA_PATH}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, 
        quantization_config=bnb_config, 
        device_map="auto", 
        token=HF_TOKEN
    )
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()
except Exception as e:
    print(f"❌ 模型載入失敗：{e}")
    print("請確認 LORA_PATH 資料夾是否存在，並且裡面有 adapter_model.safetensors")
    exit()

# --- 3. 核心推論邏輯 ---
def recommend_pipeline(user_query):
    if not user_query: return ""
    print(f"User Query: {user_query}")

    # A. Prompt (必須嚴格遵守訓練時的格式)
    messages = [
        {"role": "system", "content": "You are an anime recommender. Return a JSON list of 3 anime titles based on the user request."},
        {"role": "user", "content": user_query}
    ]
    
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")
    
    # B. AI 生成 (SFT 大腦運作中)
    with torch.no_grad():
        outputs = model.generate(
            input_ids, 
            max_new_tokens=120, 
            temperature=0.3, # 低溫保證穩定
            do_sample=True,
            repetition_penalty=1.2, # 防止鬼打牆
            pad_token_id=tokenizer.eos_token_id
        )
    
    # C. 解析 AI 的回答
    raw_output = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"🤖 AI Raw Output: {raw_output}")

    titles = []
    try:
        # 嘗試從字串中抓出 JSON List
        start = raw_output.find('[')
        end = raw_output.rfind(']') + 1
        if start != -1 and end != -1:
            titles = json.loads(raw_output[start:end])
        else:
            # 萬一 AI 沒吐 JSON，嘗試用逗號切分
            titles = [t.strip() for t in raw_output.split(',')]
    except:
        titles = [raw_output]

    # D. 結合資料庫生成卡片 (RAG)
    html_output = "<div style='display: flex; flex-direction: column; gap: 15px;'>"
    found_count = 0
    
    for t in titles[:3]: 
        row = find_anime_in_db(t) # 這步就是 RAG：用 AI 的關鍵字去查 DB
        
        if row is not None:
            found_count += 1
            # 從 CSV 讀取正確資訊
            r_title = row.get(c_title, t)
            r_score = row.get(c_score, "N/A")
            r_studio = row.get(c_studios, "Unknown Studio")
            r_year = row.get(c_year, "Unknown Year")
            r_img = row.get(c_img, "") 
            r_desc = str(row.get(c_desc, "No description."))
            
            if len(r_desc) > 180: r_desc = r_desc[:180] + "..."

            # 圖片處理
            if r_img and str(r_img).startswith('http'):
                img_html = f"<img src='{r_img}' style='width: 100px; height: 140px; object-fit: cover; border-radius: 5px;' />" 
            else:
                img_html = "<div style='width:100px; height:140px; background:#444; border-radius:5px; display:flex; align-items:center; justify-content:center; color:#888;'>No Img</div>"

            # 漂亮的卡片 HTML
            card = f"""
            <div style='display: flex; background: #2b2b2b; color: white; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 8px rgba(0,0,0,0.3); border: 1px solid #444;'>
                <div style='padding: 10px;'>
                    {img_html}
                </div>
                <div style='padding: 10px 15px; flex: 1;'>
                    <h3 style='margin: 0 0 5px 0; color: #ffcc00; font-size: 1.2em;'>{r_title}</h3>
                    <div style='font-size: 0.85em; color: #aaa; margin-bottom: 8px;'>
                        📅 {r_year} | 🏢 {r_studio} | ⭐ {r_score}
                    </div>
                    <p style='margin: 0; font-size: 0.9em; line-height: 1.4; color: #ddd;'>{r_desc}</p>
                </div>
            </div>
            """
            html_output += card
        else:
            # 如果 AI 產生了資料庫裡沒有的片 (幻覺)，就不顯示，保持介面乾淨
            print(f"⚠️ Filtered Hallucination or Mismatch: {t}")

    html_output += "</div>"
    
    if found_count == 0:
        return f"<p style='color: #ff6666;'>No matches found in database. AI suggested: {titles}</p>"
        
    return html_output

# --- 4. 啟動 Gradio (V13.5 UI) ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Anime SFT Recommender (V13.5)")
    gr.Markdown("展示 **SFT (知識注入)** 與 **RAG (資料庫檢索)** 的結合成果。")
    gr.Markdown("👉 模型已學會隱藏標籤：**Basketball, Soccer, Time Travel, Ghibli...**")
    
    with gr.Row():
        with gr.Column(scale=4):
            inp = gr.Textbox(label="Your Request", placeholder="e.g., 'Recommend Basketball anime'")
        with gr.Column(scale=1):
            btn = gr.Button("🔍 Recommend", variant="primary")
    
    # 這些是 V13.5 的強項
    gr.Examples(
        examples=[
            "Recommend Basketball anime.",
            "I want to watch Soccer anime.",
            "Recommend old (pre-2015) Action anime.",
            "Recommend anime by Kyoto Animation.",
            "Recommend Ghibli movies.",
            "Recommend Time Travel anime."
        ],
        inputs=inp
    )
    
    out = gr.HTML(label="Result")
    
    btn.click(fn=recommend_pipeline, inputs=inp, outputs=out)
    inp.submit(fn=recommend_pipeline, inputs=inp, outputs=out)

print("✅ System Ready! Click the link below to open.")
demo.launch(share=True)


