import pandas as pd
import json
import random
import os
import re
from tqdm import tqdm

# ================= Config =================
CSV_FILE = "mal_anime.csv"
OUTPUT_FILE = "train_selector_v13_fav.jsonl"
MAX_SAMPLES = 10000 
MIN_FAVORITES = 500 # ★ 關鍵：只有收藏數超過 500 的才算「不冷門」
# =========================================

def parse_int_clean(val):
    """處理數字裡的逗號，例如 '87,916' -> 87916"""
    try:
        return int(str(val).replace(',', '').split('.')[0])
    except:
        return 0

def parse_year(year_str):
    try:
        match = re.search(r'\d{4}', str(year_str))
        if match: return int(match.group(0))
    except: pass
    return 0

def main():
    print(f"📂 正在讀取 {CSV_FILE} (V13 人氣挖掘版)...")
    if not os.path.exists(CSV_FILE):
        print("❌ 找不到 CSV 檔")
        return

    df = pd.read_csv(CSV_FILE)
    cols = {c.lower(): c for c in df.columns}
    
    c_title = cols.get('title') or 'title'
    c_genres = cols.get('genres') or 'genres'
    c_favorites = cols.get('favorites') or cols.get('members') or 'favorites' # 讀取 favorites
    c_year = cols.get('released_year') or cols.get('released_ye') or 'year'
    c_studios = cols.get('studios') or cols.get('studio') or 'studios'
    c_desc = cols.get('description') or cols.get('synopsis') or 'description'

    # --- 1. 定義挖掘規則 (Mining Rules) ---
    # 這些是我們教 AI 的「隱藏知識」
    MINING_RULES = {
        "Basketball": ["basketball", "nba", "basket ball", "slam dunk"],
        "Soccer": ["soccer", "football", "world cup", "blue lock"],
        "Baseball": ["baseball", "pitcher", "homerun"],
        "Volleyball": ["volleyball", "haikyuu"],
        "Tennis": ["tennis", "prince of tennis"],
        "Time Travel": ["time travel", "time loop", "future", "past", "steins;gate"],
        "Cyberpunk": ["cyberpunk", "android", "cyborg", "virtual world"],
        "Ghibli-Style": ["ghibli", "miyazaki", "totoro", "spirited away"] # 教它風格
    }
    
    hidden_tag_map = {k: [] for k in MINING_RULES.keys()}
    genre_map = {}
    studio_map = {} 
    title_to_year = {}

    print("⚙️ 正在掃描簡介 + 過濾人氣...")
    
    for _, row in df.iterrows():
        title = str(row.get(c_title, "")).strip()
        genres_str = str(row.get(c_genres, ""))
        studios_str = str(row.get(c_studios, ""))
        desc_str = str(row.get(c_desc, "")).lower()
        
        # ★ 處理 Favorites (去掉逗號)
        fav_val = parse_int_clean(row.get(c_favorites, 0))
        year_val = parse_year(row.get(c_year, 0))

        # ★ 關鍵過濾：
        # 1. 沒片名跳過
        # 2. 人氣太低 (Favorites < 500) 跳過 -> 解決「過於冷門」的問題
        if not title or fav_val < MIN_FAVORITES: continue 
        
        title_to_year[title] = year_val

        genres = set([g.strip() for g in genres_str.split(',') if g.strip()])
        studios = [s.strip() for s in studios_str.split(',') if s.strip()]
        
        for g in genres:
            if g not in genre_map: genre_map[g] = []
            genre_map[g].append(title)
        for s in studios:
            if s not in studio_map: studio_map[s] = []
            studio_map[s].append(title)

        # ★ 挖掘隱藏標籤 (Mining) ★
        for tag, keywords in MINING_RULES.items():
            for kw in keywords:
                if kw in desc_str: 
                    hidden_tag_map[tag].append(title)
                    break 

    # --- 2. 生成訓練劇本 ---
    training_data = []
    
    modes = ["genre"] * 3 + ["studio"] * 1 + ["hidden_tag"] * 4

    print(f"🚀 正在生成 {MAX_SAMPLES} 筆訓練資料...")
    pbar = tqdm(total=MAX_SAMPLES)

    while len(training_data) < MAX_SAMPLES:
        mode = random.choice(modes)
        q = ""
        ans = []
        
        # --- Mode 1: 隱藏標籤 (SFT 核心) ---
        if mode == "hidden_tag":
            target_tag = random.choice(list(hidden_tag_map.keys()))
            titles = hidden_tag_map[target_tag]
            
            # 確保有足夠的片 (至少 3 部)
            if len(titles) < 3: continue
            
            # 隨機挑 3 部「熱門且符合標籤」的片
            # 因為前面已經過濾過 Favorites，所以這裡挑出來的一定是熱門的
            q = random.choice([
                f"Recommend {target_tag} anime.",
                f"I want to watch an anime about {target_tag}.",
                f"Top {target_tag} anime."
            ])
            ans = random.sample(titles, 3)

        # --- Mode 2: Genre (Old/New) ---
        elif mode == "genre":
            target_genre = random.choice(list(genre_map.keys()))
            titles = genre_map[target_genre]
            if len(titles) < 3: continue
            
            sub_mode = random.choice(["normal", "old", "new"])
            if sub_mode == "normal":
                q = f"Recommend {target_genre} anime."
                ans = random.sample(titles, 3)
            elif sub_mode == "old":
                filtered = [t for t in titles if title_to_year.get(t, 0) < 2015 and title_to_year.get(t, 0) > 0]
                if len(filtered) < 3: continue
                q = f"Recommend old (pre-2015) {target_genre} anime."
                ans = random.sample(filtered, 3)
            elif sub_mode == "new":
                filtered = [t for t in titles if title_to_year.get(t, 0) >= 2015]
                if len(filtered) < 3: continue
                q = f"Recommend new (post-2015) {target_genre} anime."
                ans = random.sample(filtered, 3)

        # --- Mode 3: Studio ---
        elif mode == "studio":
            valid_studios = [k for k,v in studio_map.items() if len(v) >= 5]
            if not valid_studios: continue
            target_studio = random.choice(valid_studios)
            titles = studio_map[target_studio]
            q = f"Recommend anime by {target_studio}."
            ans = random.sample(titles, min(3, len(titles)))

        if q and len(ans) >= 3:
            conversation = [
                {"role": "system", "content": "You are an anime recommender. Return a JSON list of 3 anime titles based on the user request."},
                {"role": "user", "content": q},
                {"role": "assistant", "content": json.dumps(ans, ensure_ascii=False)}
            ]
            training_data.append(conversation)
            pbar.update(1)

    pbar.close()
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in training_data:
            json.dump({"messages": entry}, f)
            f.write('\n')

    print(f"🎉 V13 人氣挖掘版完成！")
    print(f"📊 已過濾掉 Favorites < {MIN_FAVORITES} 的冷門作品")
    print(f"🏀 挖掘出 {len(hidden_tag_map['Basketball'])} 部熱門籃球動畫")
    print(f"⚽ 挖掘出 {len(hidden_tag_map['Soccer'])} 部熱門足球動畫")

if __name__ == "__main__":
    main()


