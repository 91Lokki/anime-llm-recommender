import os
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer

# ================= 設定區 =================
# ★ 使用系統登入資訊 (請確認已在 CMD 執行 huggingface-cli login)
HF_TOKEN = True 

# 模型與資料設定
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
DATA_FILE = "train_selector_v13_fav.jsonl"  # ✅ 讀取 V13 人氣挖掘版
OUTPUT_DIR = "./llama3.2_sft_v13"           # 存到新資料夾，區分版本
# =========================================

def plot_loss_curve(log_history):
    """
    繪製訓練曲線
    """
    train_steps = []
    train_loss = []
    eval_steps = []
    eval_loss = []

    for entry in log_history:
        if 'loss' in entry and 'step' in entry:
            train_steps.append(entry['step'])
            train_loss.append(entry['loss'])
        elif 'eval_loss' in entry and 'step' in entry:
            eval_steps.append(entry['step'])
            eval_loss.append(entry['eval_loss'])

    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_loss, label='Training Loss', color='blue', alpha=0.6)
    if eval_loss:
        plt.plot(eval_steps, eval_loss, label='Validation Loss', color='red', linewidth=2)
    
    plt.title(f'Learning Curve (Epochs=3)')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve_v13.png') 
    print("📈 Loss 曲線圖已儲存為 loss_curve_v13.png")

def main():
    print(f"🔥 準備開始 V13 特訓 (Epochs=3)...")
    print(f"📂 讀取教材：{DATA_FILE}")
    
    if not os.path.exists(DATA_FILE):
        print(f"❌ 錯誤：找不到 {DATA_FILE}！請先執行第一步 `1_prepare_data_v13_favorites.py`。")
        return

    # 1. 讀取與切分資料
    full_dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    # 切分 5% 當驗證就好，我們要保留更多資料來訓練
    dataset_split = full_dataset.train_test_split(test_size=0.05)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]
    
    print(f"📊 訓練資料: {len(train_dataset)} 筆 | 驗證資料: {len(eval_dataset)} 筆")

    # 2. 設定 4-bit 量化 (3080 Ti 最佳化)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print("🚀 正在載入 Llama 3.2 模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN,
        use_cache=False
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    # 3. LoRA 設定
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # 4. 訓練參數 (高強度版)
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=3,             
        learning_rate=2e-4,
        bf16=True,
        
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,                # 每 200 步驗證一次
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,   # 雖然跑 3 輪，但如果過擬合太嚴重，還是會幫你選最好的存檔
        metric_for_best_model="eval_loss", 
        report_to="none"
    )

    # 5. 開始訓練
    print("🏋️‍♂️ 開始 SFT 微調 (這會花一點時間)...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=args
    )

    trainer.train()
    
    # 6. 存檔
    print("🎨 正在繪製 Loss 曲線...")
    plot_loss_curve(trainer.state.log_history)

    print("💾 正在儲存最佳模型...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n" + "="*40)
    print(f"🎉 V13 訓練完成！")
    print(f"✅ 模型已儲存於：{OUTPUT_DIR}")
    print("👉 下一步：請修改 3_app.py 裡的 LORA_PATH 指向這個新資料夾")
    print("="*40)

if __name__ == "__main__":
    main()


