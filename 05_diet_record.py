import streamlit as st
import pandas as pd
from datetime import date, timedelta
import os, json, math, io, zipfile, pathlib, shutil, time

st.set_page_config(page_title="栄養管理ダイエット記録", page_icon="🍱", layout="wide")

# ============================
# ファイルパス / 文字コード / バックアップ
# ============================
DATA_DIR = pathlib.Path(".").resolve()
BACKUP_DIR = DATA_DIR / "backups"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

def _p(name: str) -> str:
    return str(DATA_DIR / name)

LOG_PATH     = _p("diet_log.csv")
FOOD_DB_PATH = _p("food_db.csv")
LIMITS_PATH  = _p("limits.json")
WEIGHT_PATH  = _p("weight_log.csv")
ADVICE_PATH  = _p("advice_log.csv")
PROFILE_PATH = _p("profile.json")

CSV_ENCODING = "utf-8-sig"
ALT_ENCODING = "cp932"

# ============================
# 初期データ
# ============================
DEFAULT_FOOD_DB = [
    {"food": "白ごはん", "unit": "", "per": 1.0, "kcal": 168, "protein": 2.5, "fat": 0.3, "carbs": 37.1, "fiber": 0.3, "sugar": 0.1, "sodium_mg": 1},
    {"food": "玄米", "unit": "", "per": 1.0, "kcal": 165, "protein": 2.8, "fat": 1.0, "carbs": 35.6, "fiber": 1.4, "sugar": 0.5, "sodium_mg": 5},
    {"food": "食パン", "unit": "", "per": 1.0, "kcal": 264, "protein": 9.3, "fat": 4.2, "carbs": 46.7, "fiber": 2.3, "sugar": 5.0, "sodium_mg": 490},
    {"food": "鶏むね（皮なし・加熱）", "unit": "", "per": 1.0, "kcal": 120, "protein": 26.0, "fat": 1.5, "carbs": 0.0, "fiber": 0.0, "sugar": 0.0, "sodium_mg": 65},
    {"food": "卵（全卵）", "unit": "", "per": 1.0, "kcal": 76, "protein": 6.3, "fat": 5.3, "carbs": 0.2, "fiber": 0.0, "sugar": 0.2, "sodium_mg": 62},
]
NUTRIENTS = ["kcal", "protein", "fat", "carbs", "fiber", "sugar", "sodium_mg"]
MEAL_TYPES = ["朝食", "昼食", "夕食", "間食"]

# ============================
# 共通ユーティリティ（BMI/推定式など）
# ============================
def calc_bmi(height_cm: float, weight_kg: float):
    try:
        h_m = max(0.5, float(height_cm)/100.0)
        if weight_kg is None or (isinstance(weight_kg, float) and math.isnan(weight_kg)):
            return None
        return round(weight_kg / (h_m*h_m), 1)
    except Exception:
        return None

def std_weight(height_cm: float):
    try:
        h_m = max(0.5, float(height_cm)/100.0)
        return round(22.0 * h_m * h_m, 1)
    except Exception:
        return None

def bmi_category(bmi: float) -> str:
    if bmi is None:
        return ""
    if bmi < 18.5: return "低体重"
    if bmi < 25:   return "普通体重"
    if bmi < 30:   return "肥満(1度)"
    if bmi < 35:   return "肥満(2度)"
    if bmi < 40:   return "肥満(3度)"
    return "肥満(4度)"

def activity_factor(level: str) -> float:
    if "低" in level: return 1.2
    if "高" in level: return 1.725
    return 1.55  # ふつう

def mifflin_bmr(sex: str, age: int, height_cm: float, weight_kg: float) -> float:
    # Mifflin-St Jeor
    s = 5 if sex == "男性" else -161 if sex == "女性" else 0
    return 10*weight_kg + 6.25*height_cm - 5*age + s

# ---- 早期減量プランのローカル簡易推定（AI失敗時フォールバック）
def local_estimate_limits_rapid_loss(profile: dict) -> dict:
    sex = profile.get("sex","男性")
    age = int(profile.get("age", 28))
    h   = float(profile.get("height_cm", 170.0))
    w   = float(profile.get("current_weight_kg", 65.0))
    act = profile.get("activity","ふつう(週1-3運動)")
    bmr  = mifflin_bmr(sex, age, h, w)
    tdee = bmr * activity_factor(act)
    deficit_ratio = 0.20  # 早期減量プラン：TDEEから約20%赤字
    kcal = max(round(tdee * (1.0 - deficit_ratio)), round(bmr * 0.9))  # 安全側の下限（BMRの90%は下回らない）
    # PFC
    protein_g = round(max(1.8 * w, 2.0 * w), 1)  # 高め（実質2.0g/kg）
    fat_g = round((0.22 * kcal) / 9.0, 1)       # 脂質22%kcal
    carbs_g = round(max(100.0, (kcal - (protein_g*4 + fat_g*9)) / 4.0), 1)  # 最低100g
    # その他
    fiber_g = 20.0
    sugar_g = 45.0
    sodium_mg = 2300.0
    return {
        "kcal": float(kcal),
        "protein": float(protein_g),
        "fat": float(fat_g),
        "carbs": float(carbs_g),
        "fiber": float(fiber_g),
        "sugar": float(sugar_g),
        "sodium_mg": float(sodium_mg),
        "enabled": True,
        "_source": "local_rapid",
    }

def recommend_activity_text(bmi: float, activity_label: str) -> str:
    base = "- 週 **150〜300分** の中強度有酸素\n- 週 **2日以上** の筋トレ（全身）"
    if bmi is None:
        return base
    if bmi < 18.5:
        return (
            "- 週 **120〜150分** の軽〜中強度有酸素（やり過ぎ注意）\n"
            "- 週 **2日** の筋トレ（大筋群・フォーム重視）\n"
            "- 十分なエネルギーとたんぱく質摂取も意識"
        )
    if bmi < 25:
        extra = "（現在の活動「%s」に応じて上限300分側を目指すと体力向上）" % activity_label
        return base + "\n" + extra
    if bmi < 30:
        return (
            "- 週 **200〜300分** の中強度有酸素 + 日常の歩数UP\n"
            "- 週 **3日** の筋トレ（下半身＋大筋群）\n"
            "- 可能なら **インターバル** を週1回追加"
        )
    else:
        return (
            "- 週 **300分以上** の中強度有酸素を段階的に（分割OK）\n"
            "- 週 **3日** の筋トレ（低〜中負荷で継続）\n"
            "- 関節に優しい有酸素（エアロバイク/水中ウォーク）も推奨"
        )

# ---- OpenAIチャット呼び出しのラッパ（temperature非対応モデル対策）
def _chat_create(client, model, messages, temperature=None):
    kwargs = {"model": model, "messages": messages}
    # temperatureを受け付けないモデルをここに列挙（必要に応じて拡張）
    models_no_temp = {"gpt-5"}
    if (temperature is not None) and (model not in models_no_temp):
        kwargs["temperature"] = float(temperature)
    return client.chat.completions.create(**kwargs)

# ============================
# 補助：CSV読み込み
# ============================
@st.cache_data
def get_default_food_df():
    return pd.DataFrame(DEFAULT_FOOD_DB)

def read_csv_smart(file_or_path, is_path=True):
    enc_list = [CSV_ENCODING, ALT_ENCODING, "utf-8"]
    last_err = None
    for enc in enc_list:
        try:
            if is_path:
                return pd.read_csv(file_or_path, encoding=enc)
            else:
                file_or_path.seek(0)
                return pd.read_csv(file_or_path, encoding=enc)
        except Exception as e:
            last_err = e
            continue
    raise last_err

# ============================
# ロード/セーブ
# ============================
def _ensure_food_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "unit" not in df.columns: df["unit"] = ""
    if "per" not in df.columns:  df["per"] = 1.0
    cols = [c for c in df.columns if c in NUTRIENTS]
    if cols: df[cols] = df[cols].astype(float).round(1)
    df["per"] = pd.to_numeric(df["per"], errors="coerce").fillna(1.0).round(1)
    df["unit"] = df["unit"].astype(str)
    df["food"] = df["food"].astype(str)
    return df

def load_food_db(path: str = FOOD_DB_PATH) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            df = read_csv_smart(path, is_path=True)
            return _ensure_food_df_columns(df)
        except Exception:
            pass
    df = get_default_food_df().copy()
    return _ensure_food_df_columns(df)

def save_food_db(df: pd.DataFrame, path: str = FOOD_DB_PATH):
    df = _ensure_food_df_columns(df)
    df.to_csv(path, index=False, encoding=CSV_ENCODING)

def load_log(path: str = LOG_PATH) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            df = read_csv_smart(path, is_path=True)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            cols = [c for c in df.columns if c in NUTRIENTS]
            if cols:
                df[cols] = pd.to_numeric(df[cols], errors="coerce").astype(float).round(1)
            if "per" in df.columns:
                df["per"] = pd.to_numeric(df["per"], errors="coerce").fillna(1.0).round(1)
            if "unit" not in df.columns: df["unit"] = ""
            df["unit"] = df["unit"].astype(str)
            if "meal" in df.columns: df["meal"] = df["meal"].astype(str)
            if "food" in df.columns: df["food"] = df["food"].astype(str)
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=["date", "meal", "food", "unit", "per", *NUTRIENTS])

def save_log(df: pd.DataFrame, path: str = LOG_PATH):
    for c in NUTRIENTS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(1)
    if "per" in df.columns:
        df["per"] = pd.to_numeric(df["per"], errors="coerce").fillna(1.0).round(1)
    if "unit" in df.columns: df["unit"] = df["unit"].astype(str)
    if "meal" in df.columns: df["meal"] = df["meal"].astype(str)
    if "food" in df.columns: df["food"] = df["food"].astype(str)
    df.to_csv(path, index=False, encoding=CSV_ENCODING)

def load_limits(path: str = LIMITS_PATH) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "kcal": 2000.0, "protein": 150.0, "fat": 60.0, "carbs": 260.0,
        "sugar": 50.0, "sodium_mg": 2300.0, "fiber": 20.0, "enabled": False,
    }

def save_limits(limits: dict, path: str = LIMITS_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(limits, f, ensure_ascii=False, indent=2)

def load_weight(path: str = WEIGHT_PATH) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            df = read_csv_smart(path, is_path=True)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            if "weight_kg" in df.columns:
                df["weight_kg"] = pd.to_numeric(df["weight_kg"], errors="coerce").round(1)
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=["date", "weight_kg"])

def save_weight(df: pd.DataFrame, path: str = WEIGHT_PATH):
    if "weight_kg" in df.columns:
        df["weight_kg"] = pd.to_numeric(df["weight_kg"], errors="coerce").round(1)
    df.to_csv(path, index=False, encoding=CSV_ENCODING)

def load_advice_log(path: str = ADVICE_PATH) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            df = read_csv_smart(path, is_path=True)
            for col in ["start_day", "last_day", "created_at"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=[
        "created_at", "model", "window", "include_foods", "simple_mode",
        "start_day", "last_day", "ai_advice"
    ])

def save_advice_log(df: pd.DataFrame, path: str = ADVICE_PATH):
    df.to_csv(path, index=False, encoding=CSV_ENCODING)

def load_profile(path: str = PROFILE_PATH) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"sex": "男性","age": 28,"height_cm": 173.0,"current_weight_kg": 73.0,"activity": "ふつう(週1-3運動)"}

def save_profile(prof: dict, path: str = PROFILE_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prof, f, ensure_ascii=False, indent=2)

# ============================
# バックアップ関連（ZIP）
# ============================
BACKUP_FILES = {
    "diet_log.csv": LOG_PATH, "weight_log.csv": WEIGHT_PATH, "advice_log.csv": ADVICE_PATH,
    "food_db.csv": FOOD_DB_PATH, "limits.json": LIMITS_PATH, "profile.json": PROFILE_PATH,
}

def create_backup_bytes() -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arcname, realpath in BACKUP_FILES.items():
            p = pathlib.Path(realpath)
            if p.exists(): zf.write(p, arcname=arcname)
            else: zf.writestr(arcname, "")
    mem.seek(0)
    return mem.read()

def save_backup_zip_to_disk() -> pathlib.Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = BACKUP_DIR / f"diet_backup_{ts}.zip"
    data = create_backup_bytes()
    with open(out, "wb") as f: f.write(data)
    return out

def list_backups():
    backs = sorted(BACKUP_DIR.glob("diet_backup_*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    return backs

def restore_from_zip_bytes(zip_bytes: bytes) -> dict:
    results = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        if "diet_log.csv" in names:
            try:
                with zf.open("diet_log.csv") as f:
                    df = pd.read_csv(f, encoding="utf-8")
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                st.session_state.log = df; save_log(st.session_state.log)
                results["diet_log.csv"] = "OK"
            except Exception as e: results["diet_log.csv"] = f"NG: {e}"
        if "weight_log.csv" in names:
            try:
                with zf.open("weight_log.csv") as f: df = pd.read_csv(f, encoding="utf-8")
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                st.session_state.weight = df; save_weight(st.session_state.weight)
                results["weight_log.csv"] = "OK"
            except Exception as e: results["weight_log.csv"] = f"NG: {e}"
        if "advice_log.csv" in names:
            try:
                with zf.open("advice_log.csv") as f: df = pd.read_csv(f, encoding="utf-8")
                for col in ["start_day","last_day","created_at"]:
                    if col in df.columns: df[col] = pd.to_datetime(df[col], errors="coerce")
                st.session_state.advice = df; save_advice_log(st.session_state.advice)
                results["advice_log.csv"] = "OK"
            except Exception as e: results["advice_log.csv"] = f"NG: {e}"
        if "food_db.csv" in names:
            try:
                with zf.open("food_db.csv") as f: df = pd.read_csv(f, encoding="utf-8")
                df = _ensure_food_df_columns(df)
                st.session_state.food_db = df; save_food_db(st.session_state.food_db)
                results["food_db.csv"] = "OK"
            except Exception as e: results["food_db.csv"] = f"NG: {e}"
        if "limits.json" in names:
            try:
                with zf.open("limits.json") as f: limits = json.loads(f.read().decode("utf-8"))
                st.session_state.limits = limits; save_limits(st.session_state.limits)
                results["limits.json"] = "OK"
            except Exception as e: results["limits.json"] = f"NG: {e}"
        if "profile.json" in names:
            try:
                with zf.open("profile.json") as f: prof = json.loads(f.read().decode("utf-8"))
                st.session_state.profile = prof; save_profile(st.session_state.profile)
                results["profile.json"] = "OK"
            except Exception as e: results["profile.json"] = f"NG: {e}"
    return results

# ============================
# セッション初期化
# ============================
if "food_db" not in st.session_state: st.session_state.food_db = load_food_db()
if "log" not in st.session_state:     st.session_state.log = load_log()
if "date" not in st.session_state:    st.session_state.date = date.today()
if "limits" not in st.session_state:  st.session_state.limits = load_limits()
if "weight" not in st.session_state:  st.session_state.weight = load_weight()
if "advice" not in st.session_state:  st.session_state.advice = load_advice_log()
if "profile" not in st.session_state: st.session_state.profile = load_profile()
# 上限のAI提案（保存前の一時データ）を確保
if "_limits_proposal" not in st.session_state: st.session_state._limits_proposal = None
if "_limits_proposal_src" not in st.session_state: st.session_state._limits_proposal_src = None

# ============================
# サイドバー（設定とデータ）
# ============================
with st.sidebar:
    st.header("⚙️ 設定とデータ")

    # 食品DB 読み込み
    uploaded_food = st.file_uploader("食品DBをCSVで読み込む（任意）", type=["csv"], accept_multiple_files=False)
    if uploaded_food is not None:
        try:
            df_up = read_csv_smart(uploaded_food, is_path=False)
            required = {"food", *NUTRIENTS}
            if not required.issubset(df_up.columns):
                st.error("CSVに必要な列: food, kcal, protein, fat, carbs, fiber, sugar, sodium_mg（unit, per は任意）")
            else:
                df_up = _ensure_food_df_columns(df_up)
                st.session_state.food_db = df_up; save_food_db(st.session_state.food_db)
                st.success("食品DBを読み込み保存しました")
        except Exception as e:
            st.error(f"読み込みエラー: {e}")

    # 食品の手動追加
    with st.expander("🧾 食品を手動で追加"):
        with st.form("add_food_form", clear_on_submit=True):
            food_name = st.text_input("食品名")
            c = st.columns(3)
            kcal = c[0].number_input("kcal", min_value=0.0, value=100.0)
            protein = c[1].number_input("たんぱく質(g)", min_value=0.0, value=5.0)
            fat = c[2].number_input("脂質(g)", min_value=0.0, value=3.0)
            c2b = st.columns(4)
            carbs = c2b[0].number_input("炭水化物(g)", min_value=0.0, value=15.0)
            fiber = c2b[1].number_input("食物繊維(g)", min_value=0.0, value=1.0)
            sugar = c2b[2].number_input("糖質(g)", min_value=0.0, value=10.0)
            sodium_mg = c2b[3].number_input("ナトリウム(mg)", min_value=0.0, value=100.0)
            submit_food = st.form_submit_button("食品を追加")
        if submit_food:
            if food_name:
                new_row = {"food": str(food_name), "unit": "", "per": 1.0,
                           "kcal": round(float(kcal),1), "protein": round(float(protein),1), "fat": round(float(fat),1),
                           "carbs": round(float(carbs),1), "fiber": round(float(fiber),1), "sugar": round(float(sugar),1),
                           "sodium_mg": round(float(sodium_mg),1)}
                st.session_state.food_db = pd.concat([st.session_state.food_db, pd.DataFrame([new_row])], ignore_index=True)
                save_food_db(st.session_state.food_db); st.success(f"{food_name} をDBに追加しました（保存済み）")
            else:
                st.error("食品名を入力してください")

    # 🧠 AIで食品栄養推定（既存）
    with st.expander("🧠 AIで食品の栄養を推定して追加", expanded=False):
        st.caption("食品名を1つ入力すると、AIが一般的な1食相当の栄養成分を推定します。")
        ai_food_name = st.text_input("食品名（例：照り焼きチキン丼、プロテインバー、コブサラダ）", key="ai_food_name")
        if st.button("AIで推定", key="btn_ai_estimate_food"):
            ai_key_local = st.session_state.get('ai_api_key')
            ai_model_local = st.session_state.get('ai_model', 'gpt-4o-mini')
            if not ai_key_local:
                st.error("OpenAI API Key を下の『🤖 AIアドバイス 設定』で入力してください。")
            elif not ai_food_name.strip():
                st.error("食品名を入力してください。")
            else:
                prompt = (
                    "あなたは栄養士です。与えられた日本語の食品名について、"
                    "次の栄養成分を1つのJSONで返してください。小数1位に丸める。"
                    "キー: kcal, protein_g, fat_g, carbs_g, fiber_g, sugar_g, sodium_mg。"
                    "返答はJSONのみ。\n"
                    f"食品名: {ai_food_name}"
                )
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=ai_key_local)
                    res = _chat_create(
                        client,
                        ai_model_local,
                        messages=[
                            {"role": "system", "content": "常に妥当な1食相当の値を返す。"},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.2,  # gpt-5など非対応モデルでは送信されません
                    )
                    raw = res.choices[0].message.content.strip()
                    try:
                        data = json.loads(raw)
                    except Exception:
                        import re
                        m = re.search(r"\{[\s\S]*\}", raw)
                        if not m: raise ValueError("JSONを抽出できませんでした")
                        data = json.loads(m.group(0))
                    def _num(x):
                        try: return round(float(x),1)
                        except Exception: return 0.0
                    est = {"food": ai_food_name.strip(), "unit":"", "per":1.0,
                           "kcal":_num(data.get("kcal",0)), "protein":_num(data.get("protein_g",0)),
                           "fat":_num(data.get("fat_g",0)), "carbs":_num(data.get("carbs_g",0)),
                           "fiber":_num(data.get("fiber_g",0)), "sugar":_num(data.get("sugar_g",0)),
                           "sodium_mg":_num(data.get("sodium_mg",0))}
                    st.session_state["_ai_food_estimate_result"] = est
                    st.success("推定に成功。下で保存できます。")
                except ModuleNotFoundError:
                    st.error("`openai` がありません。`pip install openai` を実行してください。")
                except Exception as e:
                    st.error(f"AI推定エラー: {e}")

        est = st.session_state.get("_ai_food_estimate_result")
        if est:
            show = pd.DataFrame([{
                "食品名": est["food"], "kcal": est["kcal"], "たんぱく質(g)": est["protein"],
                "脂質(g)": est["fat"], "炭水化物(g)": est["carbs"], "食物繊維(g)": est["fiber"],
                "糖質(g)": est["sugar"], "ナトリウム(mg)": est["sodium_mg"],
            }])
            st.dataframe(show, use_container_width=True)
            if st.button("↑ この推定でDBに追加", key="btn_ai_add_food"):
                new_row = {"food": est["food"], "unit":"", "per":1.0,
                           "kcal": est["kcal"], "protein": est["protein"], "fat": est["fat"],
                           "carbs": est["carbs"], "fiber": est["fiber"], "sugar": est["sugar"],
                           "sodium_mg": est["sodium_mg"]}
                st.session_state.food_db = pd.concat([st.session_state.food_db, pd.DataFrame([new_row])], ignore_index=True)
                save_food_db(st.session_state.food_db); st.success(f"{est['food']} を追加しました（保存済み）")

    with st.expander("🗑️ 食品を削除"):
        foods = sorted(st.session_state.food_db["food"].astype(str).unique().tolist())
        del_select = st.multiselect("削除する食品を選択", foods)
        if st.button("選択した食品を削除"):
            if del_select:
                before = len(st.session_state.food_db)
                st.session_state.food_db = st.session_state.food_db[~st.session_state.food_db["food"].isin(del_select)].reset_index(drop=True)
                save_food_db(st.session_state.food_db)
                st.success(f"{len(del_select)} 件削除（{before} → {len(st.session_state.food_db)}）")
            else:
                st.info("削除対象が選択されていません")

    with st.expander("⚖️ 体重データを削除"):
        if st.session_state.weight.empty:
            st.caption("体重データはまだありません")
        else:
            wtmp = st.session_state.weight.copy()
            wtmp["date"] = pd.to_datetime(wtmp["date"], errors="coerce")
            w_dates = sorted(wtmp["date"].dt.date.unique().tolist())
            del_w = st.multiselect("削除する日付を選択", w_dates, format_func=lambda d: d.strftime("%Y-%m-%d"))
            if st.button("選択した体重データを削除"):
                if del_w:
                    keep_mask = ~wtmp["date"].dt.date.isin(del_w)
                    st.session_state.weight = wtmp.loc[keep_mask].reset_index(drop=True)
                    save_weight(st.session_state.weight)
                    st.success(f"{len(del_w)} 件の体重データを削除しました（保存済み）")
                else:
                    st.info("削除対象が選択されていません")

    # プロフィール（この直下に日次上限設定）
    with st.expander("👤 プロフィール（AIが参照）", expanded=True):
        p = st.session_state.profile
        sex_options = ["男性","女性","その他"]
        act_options = ["低い(座位中心)","ふつう(週1-3運動)","高い(週4+運動)"]
        def safe_index(options, value, fallback=0):
            try: return options.index(value)
            except Exception: return fallback
        colp1, colp2 = st.columns(2)
        with colp1:
            p["sex"] = st.selectbox("性別", sex_options, index=safe_index(sex_options, p.get("sex","男性")))
            p["age"] = int(st.number_input("年齢", min_value=10, max_value=100, value=int(p.get("age",28))))
            p["height_cm"] = float(st.number_input("身長(cm)", min_value=120.0, max_value=230.0, value=float(p.get("height_cm",173.0)), step=0.1, format="%.1f"))
        with colp2:
            latest_w = None
            if not st.session_state.weight.empty:
                wtmp = st.session_state.weight.copy()
                wtmp["date"] = pd.to_datetime(wtmp["date"], errors="coerce")
                wtmp = wtmp.sort_values("date")
                if not wtmp.empty and pd.notnull(wtmp["weight_kg"].iloc[-1]):
                    latest_w = float(wtmp["weight_kg"].iloc[-1])
            default_w = float(p.get("current_weight_kg", 73.0))
            if latest_w is not None: default_w = latest_w
            p["current_weight_kg"] = float(st.number_input("現在体重(kg)", min_value=30.0, max_value=200.0, value=round(default_w,1), step=0.1, format="%.1f"))
            p["activity"] = st.selectbox("活動レベル", act_options, index=safe_index(act_options, p.get("activity","ふつう(週1-3運動)"), fallback=1))
        save_profile(p)

    # === 📏 日次上限設定（プロフィールの直下） ===
    st.markdown("---")
    st.subheader("📏 日次上限設定")
    st.session_state.limits["enabled"] = st.toggle("上限チェックを有効化", value=st.session_state.limits.get("enabled", False))
    cols = st.columns(3)
    st.session_state.limits["kcal"] = cols[0].number_input("kcal 上限", value=float(st.session_state.limits["kcal"]))
    st.session_state.limits["protein"] = cols[1].number_input("たんぱく質(g) 上限", value=float(st.session_state.limits["protein"]))
    st.session_state.limits["fat"] = cols[2].number_input("脂質(g) 上限", value=float(st.session_state.limits["fat"]))
    cols2 = st.columns(4)
    st.session_state.limits["carbs"] = cols2[0].number_input("炭水化物(g) 上限", value=float(st.session_state.limits["carbs"]))
    st.session_state.limits["fiber"] = cols2[1].number_input("食物繊維(g) 上限", value=float(st.session_state.limits["fiber"]))
    st.session_state.limits["sugar"] = cols2[2].number_input("糖質(g) 上限", value=float(st.session_state.limits["sugar"]))
    st.session_state.limits["sodium_mg"] = cols2[3].number_input("ナトリウム(mg) 上限", value=float(st.session_state.limits["sodium_mg"]))

    # === AIで上限を推定（早期減量プラン）→ まず表示、保存ボタンで反映 ===
    st.caption("プロフィールから早期減量プランを前提に日次上限の『提案値』を生成します。まず表で確認し、良ければ保存してください。")
    gen = st.button("🧠 上限を推定（早期減量プラン）")
    if gen:
        prof = st.session_state.profile
        ai_key_local = st.session_state.get('ai_api_key')
        ai_model_local = st.session_state.get('ai_model', 'gpt-4o-mini')
        proposal = None
        src = "ai_rapid"
        if ai_key_local:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=ai_key_local)
                prompt = (
                    "あなたは管理栄養士です。与えられたプロフィールに基づき、"
                    "『早期減量プラン（TDEEから約20%の赤字、P=2.0g/kg、F=20〜25%kcal、Cは残差で最低100g）』を前提に、"
                    "日次の上限（目安）を **JSONのみ** で返してください。数値は小数1位に丸めます。\n"
                    "キー: kcal, protein, fat, carbs, fiber, sugar, sodium_mg\n"
                    "制約:\n"
                    "- kcal は TDEE*(0.8〜0.85) 目安。ただし BMR*0.9 未満にしない\n"
                    "- protein は 体重×2.0g を基準\n"
                    "- fat は 総エネルギーの約22%（±3%許容）\n"
                    "- carbs は 残差で最低100g\n"
                    "- fiber=20, sugar≤45, sodium_mg≈2300\n"
                    f"プロフィール: {json.dumps(prof, ensure_ascii=False)}\n"
                    "出力はJSONのみ。説明や文章は不要。"
                )
                res = _chat_create(
                    client,
                    ai_model_local,
                    messages=[
                        {"role": "system", "content": "安全で現実的な早期減量プランの上限を返す。"},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,  # gpt-5など非対応モデルでは送信されません
                )
                raw = res.choices[0].message.content.strip()
                try:
                    proposal = json.loads(raw)
                except Exception:
                    import re
                    m = re.search(r"\{[\s\S]*\}", raw)
                    if m: proposal = json.loads(m.group(0))
            except ModuleNotFoundError:
                src = "local_rapid"
            except Exception as e:
                st.warning(f"AI推定でエラー: {e}")
                src = "local_rapid"
        else:
            src = "local_rapid"

        if proposal is None:
            proposal = local_estimate_limits_rapid_loss(prof)
            src = "local_rapid"

        # 整形 & 一時保存
        clean = {}
        for k in ["kcal","protein","fat","carbs","fiber","sugar","sodium_mg"]:
            try:
                clean[k] = round(float(proposal.get(k, 0.0)), 1)
            except Exception:
                clean[k] = 0.0
        st.session_state._limits_proposal = clean
        st.session_state._limits_proposal_src = src

    # 提案の表示と保存/破棄ボタン
    if st.session_state._limits_proposal:
        st.markdown("**AI提案（早期減量プラン）**")
        dfp = pd.DataFrame([st.session_state._limits_proposal])
        st.dataframe(dfp, use_container_width=True)
        colp1, colp2 = st.columns(2)
        with colp1:
            if st.button("✅ この上限を保存して有効化"):
                for k, v in st.session_state._limits_proposal.items():
                    st.session_state.limits[k] = float(v)
                st.session_state.limits["enabled"] = True
                save_limits(st.session_state.limits)
                st.success("提案値を保存しました（上限チェックを有効化）。")
                st.session_state._limits_proposal = None
                st.session_state._limits_proposal_src = None
        with colp2:
            if st.button("🗑️ 提案を破棄"):
                st.session_state._limits_proposal = None
                st.session_state._limits_proposal_src = None
                st.info("提案を破棄しました。")

    save_limits(st.session_state.limits)

    # ---- AIアドバイス設定（モデルに gpt-5 追加）
    st.markdown("---")
    st.subheader("🤖 AIアドバイス 設定")
    env_key = os.environ.get("OPENAI_API_KEY", "")
    secret_key = None
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        pass
    if env_key and not secret_key:
        st.caption("🔐 環境変数 OPENAI_API_KEY を検出。必要なら下で上書きできます。"
        )
    if secret_key:
        st.caption("🔐 secrets.toml の OPENAI_API_KEY を検出。必要なら下で上書きできます。")
    api_key_input = st.text_input("OpenAI API Key", type="password", value="")
    st.session_state.ai_api_key = (api_key_input.strip() or secret_key or env_key or None)
    # モデル選択肢に gpt-5 を追加
    st.session_state.ai_model = st.selectbox("モデル", ["gpt-5", "gpt-4o-mini", "gpt-4.1-mini"], index=1)

    # 対象期間（全期間あり）
    ai_window_options = ["全期間", 5, 10, 15, 20]
    st.session_state.ai_window = st.radio("対象期間", ai_window_options, index=2, horizontal=True)
    st.session_state.ai_include_foods = st.checkbox("食事ログもAIに渡す（詳細参照）", value=True)
    st.session_state.ai_debug = st.checkbox("🛠 デバッグ：送信プロンプトを表示", value=False)

    # === 💾 データの保存・バックアップ・復元（統合） ===
    st.markdown("---")
    st.subheader("💾 データの保存・バックアップ・復元")

    log_all2 = st.session_state.log.dropna(subset=["date"]).copy()
    if not log_all2.empty:
        daily_all = log_all2.groupby(log_all2["date"].dt.date)[NUTRIENTS].sum().round(1)
        meals_all = log_all2.groupby(log_all2["date"].dt.date).size().rename("meals")
        w2 = st.session_state.weight.copy()
        w2["date"] = pd.to_datetime(w2["date"], errors="coerce")
        weight_all = w2.set_index(w2["date"].dt.date)[["weight_kg"]] if not w2.empty else pd.DataFrame(columns=["weight_kg"])
        combined = daily_all.join(meals_all, how="outer").join(weight_all, how="outer").sort_index()
        if st.session_state.limits.get("enabled", False):
            for n in NUTRIENTS:
                L = float(st.session_state.limits.get(n, 0) or 0)
                if L > 0 and n in combined.columns:
                    combined[n + "_remaining"] = (L - combined[n]).apply(lambda x: round(x, 1) if pd.notnull(x) and x > 0 else 0.0)
        adv2 = st.session_state.advice.copy()
        if not adv2.empty:
            adv2["date"] = pd.to_datetime(adv2["last_day"], errors="coerce").dt.date
            latest_adv = (adv2.sort_values("created_at").groupby("date", as_index=False).tail(1)[["date","ai_advice"]].set_index("date"))
            combined = combined.join(latest_adv, how="left")
        combined = combined.round(1); combined.index.name = "date"
        csv_combined = combined.reset_index().to_csv(index=False).encode(CSV_ENCODING)
        st.download_button("結合データ（全期間）CSVダウンロード", data=csv_combined, file_name="combined_data.csv", mime="text/csv", use_container_width=True)
    else:
        st.caption("結合できるデータがまだありません")

    csv_all = st.session_state.log.round(1).to_csv(index=False).encode(CSV_ENCODING)
    st.download_button("食事ログCSV", data=csv_all, file_name="diet_log.csv", mime="text/csv", use_container_width=True)
    csv_w = st.session_state.weight.round(1).to_csv(index=False).encode(CSV_ENCODING) if not st.session_state.weight.empty else ("date,weight_kg\n".encode(CSV_ENCODING))
    st.download_button("体重ログCSV", data=csv_w, file_name="weight_log.csv", mime="text/csv", use_container_width=True)
    csv_adv = st.session_state.advice.to_csv(index=False).encode(CSV_ENCODING)
    st.download_button("AIアドバイス履歴CSV", data=csv_adv, file_name="advice_log.csv", mime="text/csv", use_container_width=True)
    st.download_button("現在の食品DBをCSVでダウンロード", data=st.session_state.food_db.to_csv(index=False).encode(CSV_ENCODING), file_name="food_db.csv", mime="text/csv", use_container_width=True)

    # バックアップ作成/ダウンロード
    st.markdown("**バックアップ（ZIP）**")
    colb1, colb2 = st.columns(2)
    with colb1:
        if st.button("今すぐバックアップを作成（ZIP保存）", use_container_width=True):
            try:
                out = save_backup_zip_to_disk()
                st.success(f"保存: backups/{out.name}")
                st.session_state["_last_backup_path"] = str(out)
            except Exception as e:
                st.error(f"バックアップに失敗: {e}")
        last_zip_path = st.session_state.get("_last_backup_path")
        if last_zip_path and pathlib.Path(last_zip_path).exists():
            with open(last_zip_path, "rb") as f:
                st.download_button("↑ 直近バックアップをダウンロード", data=f.read(), file_name=pathlib.Path(last_zip_path).name, mime="application/zip", use_container_width=True)
    with colb2:
        mem_zip = create_backup_bytes()
        st.download_button("現在状態をそのままZIPダウンロード（保存しない）", data=mem_zip, file_name=f"diet_backup_{time.strftime('%Y%m%d-%H%M%S')}.zip", mime="application/zip", use_container_width=True)

    # 復元
    st.markdown("**バックアップから復元**")
    backups = list_backups()
    if backups:
        sel = st.selectbox("backups/ 内のZIPを選択", backups, format_func=lambda p: f"{p.name}  ({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p.stat().st_mtime))})")
        confirm = st.checkbox("選択ZIPで現在データを上書き復元することに同意する")
        if st.button("このバックアップから復元", disabled=not confirm, use_container_width=True):
            try:
                with open(sel, "rb") as f:
                    res = restore_from_zip_bytes(f.read())
                ok = [k for k, v in res.items() if str(v).startswith("OK")]
                ng = {k: v for k, v in res.items() if not str(v).startswith("OK")}
                if ok: st.success("復元成功: " + ", ".join(ok))
                if ng: st.warning("復元できなかった項目: " + ", ".join([f"{k}({v})" for k, v in ng.items()]))
                st.toast("復元完了。メイン画面をご確認ください。")
            except Exception as e:
                st.error(f"復元エラー: {e}")
    else:
        st.caption("backups/ にバックアップがありません。上で作成できます。")

    up_zip = st.file_uploader("ZIPをアップロードして復元", type=["zip"])
    if up_zip is not None:
        try:
            res = restore_from_zip_bytes(up_zip.read())
            ok = [k for k, v in res.items() if str(v).startswith("OK")]
            ng = {k: v for k, v in res.items() if not str(v).startswith("OK")}
            if ok: st.success("復元成功: " + ", ".join(ok))
            if ng: st.warning("復元できなかった項目: " + ", ".join([f"{k}({v})" for k, v in ng.items()]))
            st.toast("アップロードから復元が完了しました。")
        except Exception as e:
            st.error(f"アップロード復元エラー: {e}")

    # （自動保存）
    loaded_ok = (
        isinstance(st.session_state.get("log"), pd.DataFrame) and
        isinstance(st.session_state.get("food_db"), pd.DataFrame) and
        isinstance(st.session_state.get("weight"), pd.DataFrame) and
        isinstance(st.session_state.get("advice"), pd.DataFrame) and
        isinstance(st.session_state.get("profile"), dict)
    )
    if loaded_ok:
        save_log(st.session_state.log); save_weight(st.session_state.weight)
        save_advice_log(st.session_state.advice); save_profile(st.session_state.profile)
    else:
        st.warning("⚠️ データ読込に失敗したため自動保存をスキップしました。")

# ============================
# メインUI
# ============================
st.title("🍱 栄養管理ダイエット記録")
st.caption("食品を選ぶと 1食分として記録。すべて小数点1位で保存・表示。")

# ------- 入力 / 体重（当日） -------
st.markdown("### 入力 / 体重（当日）")
selected_date = st.date_input("表示する日付（入力・体重共通）", value=st.session_state.date, format="YYYY-MM-DD", key="display_date_main")
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### 🍽️ 入力フォーム")
    with st.form("input_form"):
        meal = st.selectbox("食事区分", MEAL_TYPES, index=0)
        db = st.session_state.food_db
        options = db["food"].astype(str).tolist() if not db.empty else []
        if options:
            food = st.selectbox("食品を選択", options, index=0)
            submitted = st.form_submit_button("➕ 1食分を追加", use_container_width=True)
        else:
            st.warning("食品DBが空です。サイドバーから食品を追加してください。")
            food = None; submitted = False
    if submitted and food:
        row = st.session_state.food_db[st.session_state.food_db["food"] == food].iloc[0]
        entry = {"date": pd.to_datetime(selected_date), "meal": meal, "food": row.get("food", food),
                 "unit": str(row.get("unit","")), "per": round(float(row.get("per",1.0)),1)}
        for n in NUTRIENTS: entry[n] = round(float(row[n]),1)
        st.session_state.log = pd.concat([st.session_state.log, pd.DataFrame([entry])], ignore_index=True)
        save_log(st.session_state.log); st.success(f"{food} を1食分追加しました（保存済み）")

with col_right:
    st.markdown("#### ⚖️ 体重の記録（当日）")
    wdf = st.session_state.weight.copy()
    wdf["date"] = pd.to_datetime(wdf["date"], errors="coerce")
    cur = wdf[wdf["date"].dt.date == pd.to_datetime(selected_date).date()]
    def_weight = float(cur["weight_kg"].iloc[0]) if not cur.empty else float(st.session_state.profile.get("current_weight_kg", 73.0))
    input_weight = st.number_input("体重(kg)", min_value=30.0, max_value=200.0, value=round(def_weight,1), step=0.1, format="%.1f")
    if st.button("体重を保存", use_container_width=True):
        st.session_state.weight = wdf[wdf["date"].dt.date != pd.to_datetime(selected_date).date()].copy()
        new_row = pd.DataFrame({"date": [pd.to_datetime(selected_date)], "weight_kg": [round(input_weight,1)]})
        st.session_state.weight = pd.concat([st.session_state.weight, new_row], ignore_index=True)
        save_weight(st.session_state.weight)
        st.session_state.profile["current_weight_kg"] = round(input_weight,1); save_profile(st.session_state.profile)
        st.success("体重を保存しました")

    # === BMI と おすすめ運動（横並び） ===
    h = float(st.session_state.profile.get("height_cm", 173.0))
    bmi_val = calc_bmi(h, float(input_weight))
    std_w = std_weight(h)
    col_bmi, col_move = st.columns([1,2])
    with col_bmi:
        if bmi_val is not None:
            st.metric(label="BMI（当日）", value=f"{bmi_val}", delta=f"基準体重 {std_w}kg")
            st.caption(f"区分: {bmi_category(bmi_val)}")
    with col_move:
        rec = recommend_activity_text(bmi_val, st.session_state.profile.get("activity","ふつう(週1-3運動)"))
        st.markdown("**おすすめの運動量**")
        st.markdown(rec)

# ============================
# 当日の一覧と合計
# ============================
st.markdown("---")
st.subheader(f"📒 {selected_date} の記録")

st.session_state.log["date"] = pd.to_datetime(st.session_state.log["date"], errors="coerce")
mask = st.session_state.log["date"].dt.date == pd.to_datetime(selected_date).date()
day_df = st.session_state.log.loc[mask].copy()

if day_df.empty:
    st.info("この日の記録はまだありません。フォームから追加してください。")
else:
    day_df = day_df.reset_index(drop=False).rename(columns={"index": "_idx"})
    day_df["削除"] = False
    display_cols = ["_idx", "meal", "food", *NUTRIENTS, "削除"]

    show_df = day_df.copy()
    for c in NUTRIENTS:
        if c in show_df.columns:
            show_df[c] = pd.to_numeric(show_df[c], errors="coerce").round(1)

    edited = st.data_editor(
        show_df[display_cols], num_rows="fixed", use_container_width=True,
        key=f"editor_{str(selected_date)}", hide_index=True,
    )
    to_delete = edited[edited["削除"] == True]["_idx"].tolist()
    if to_delete:
        st.session_state.log = st.session_state.log.drop(index=to_delete).reset_index(drop=True)
        save_log(st.session_state.log); st.warning(f"{len(to_delete)} 件を削除しました（保存済み）")

    upd_rows = edited[~edited["_idx"].isin(to_delete)].copy()
    if not upd_rows.empty:
        for _, r in upd_rows.iterrows():
            ridx = int(r["_idx"])
            if ridx in st.session_state.log.index:
                st.session_state.log.at[ridx, "meal"] = str(r.get("meal", ""))
                st.session_state.log.at[ridx, "food"] = str(r.get("food", ""))
                for n in NUTRIENTS:
                    if n in st.session_state.log.columns:
                        val = pd.to_numeric(r.get(n, None), errors="coerce")
                        if pd.notnull(val):
                            st.session_state.log.at[ridx, n] = round(float(val), 1)
        save_log(st.session_state.log)

    latest_day = st.session_state.log.loc[st.session_state.log["date"].dt.date == pd.to_datetime(selected_date).date(), ["meal","food",*NUTRIENTS]].copy()
    for n in NUTRIENTS:
        latest_day[n] = pd.to_numeric(latest_day[n], errors="coerce")
    totals = latest_day[NUTRIENTS].sum(numeric_only=True).round(1)

    colA, colB = st.columns([1,1])
    with colA:
        st.markdown("### 🔢 栄養合計（当日）")
        st.table(totals.to_frame(name="合計"))
    with colB:
        st.markdown("### ⏳ 上限までの残り（不足分）")
        if st.session_state.limits.get("enabled", False):
            rem, over_list = {}, []
            for n in NUTRIENTS:
                limit = float(st.session_state.limits.get(n, 0) or 0)
                val = float(totals.get(n, 0) or 0)
                if limit > 0:
                    diff = round(limit - val, 1)
                    rem[n] = diff if diff > 0 else 0.0
                    if diff < 0: over_list.append((n, round(-diff,1)))
            st.table(pd.Series(rem).to_frame("残り").round(1))
            if over_list:
                msg = "\n".join([f"- {k}: 上限超過 {v:.1f}" for k, v in over_list])
                st.error("⚠️ 上限を超えています\n" + msg)
        else:
            st.info("サイドバーの『日次上限設定』で上限チェックを有効化してください")

    csv_day = latest_day.round(1).to_csv(index=False).encode(CSV_ENCODING)
    st.download_button("この日の記録をCSVでダウンロード", data=csv_day, file_name=f"diet_{selected_date}.csv", mime="text/csv", use_container_width=True)

# ============================
# 直近の集計と可視化（すべて日単位）＋全期間
# ============================
st.markdown("---")
st.subheader("📈 直近期間の集計と可視化（すべて日単位）")

log2 = st.session_state.log.dropna(subset=["date"]).copy()

def style_exceed(df: pd.DataFrame, limits: dict):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    if not limits.get("enabled", False): return styles
    for col in [c for c in df.columns if c in NUTRIENTS]:
        lim = float(limits.get(col, 0) or 0)
        if lim > 0:
            mask = pd.to_numeric(df[col], errors="coerce") > lim
            styles.loc[mask, col] = "color: red; font-weight: 700;"
    return styles

def _meal_presence_daily(rdf: pd.DataFrame) -> pd.DataFrame:
    if rdf.empty: return pd.DataFrame()
    temp = rdf.copy(); temp["date_only"] = temp["date"].dt.date
    presence = (
        temp.groupby(["date_only", "meal"]).size().reset_index(name="cnt")
        .pivot_table(index="date_only", columns="meal", values="cnt", fill_value=0).astype(int)
    )
    presence[presence > 0] = 1
    for m in MEAL_TYPES:
        if m not in presence.columns: presence[m] = 0
    return presence[MEAL_TYPES].sort_index()

if log2.empty:
    st.info("データがありません")
else:
    def render_window(window, is_all=False):
        if is_all:
            last_day = max(log2["date"].dt.date.max(), date.today())
            start_day = min(log2["date"].dt.date.min(), last_day)
        else:
            last_day = max(log2["date"].dt.date.max(), date.today())
            start_day = last_day - timedelta(days=int(window)-1)

        rmask = (log2["date"].dt.date >= start_day) & (log2["date"].dt.date <= last_day)
        rdf = log2.loc[rmask].copy()
        daily = rdf.groupby(rdf["date"].dt.date)[NUTRIENTS].sum().round(1).sort_index()
        meal_presence = _meal_presence_daily(rdf)
        daily = daily.join(meal_presence, how="left").fillna(0)
        for col in MEAL_TYPES: daily[col] = daily[col].astype(int)

        table_df = daily.reset_index().rename(columns={"index":"日付", "date":"日付"})
        styled = table_df.style.apply(style_exceed, limits=st.session_state.limits, axis=None)
        st.caption(f"対象期間: {start_day} 〜 {last_day}。当日の上限を超えた数値は赤字で表示")
        st.dataframe(styled, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### kcal 推移（日単位）")
            if "kcal" in daily.columns and not daily.empty: st.line_chart(daily[["kcal"]])
            else: st.caption("kcal データなし")
        with c2:
            st.markdown("#### 体重推移（日単位）")
            w = st.session_state.weight.copy()
            if not w.empty:
                w["date"] = pd.to_datetime(w["date"], errors="coerce")
                wv = w[(w["date"].dt.date >= start_day) & (w["date"].dt.date <= last_day)].copy().sort_values("date")
                if not wv.empty and "weight_kg" in wv.columns:
                    wt = wv.set_index(wv["date"].dt.date)[["weight_kg"]]
                    st.line_chart(wt)
                else:
                    st.caption("この期間の体重データがありません")
            else:
                st.caption("体重データがありません")

        st.markdown("#### たんぱく質 / 脂質 / 炭水化物 推移（日単位）")
        pfc_cols = [col for col in ["protein","fat","carbs"] if col in daily.columns]
        if pfc_cols: st.line_chart(daily[pfc_cols])
        return rdf, start_day, last_day, daily

    windows = [5, 10, 15, 20, 30, 60, 90]
    tabs = st.tabs([f"{w}日" for w in windows] + ["全期間"])
    for i, window in enumerate(windows):
        with tabs[i]: render_window(window, is_all=False)
    with tabs[-1]:   render_window(window=None, is_all=True)

# ============================
# 🤖 AI ダイエットアドバイス
# ============================
st.markdown("---")
st.subheader("🤖 AI ダイエットアドバイス（OpenAI API）")

ai_key = st.session_state.get('ai_api_key')
ai_model = st.session_state.get('ai_model', 'gpt-4o-mini')
ai_window_choice = st.session_state.get('ai_window', 10)
ai_include_foods = bool(st.session_state.get('ai_include_foods', True))
ai_debug = bool(st.session_state.get('ai_debug', False))
profile = st.session_state.profile

col_ai1, col_ai2 = st.columns([1,1])
with col_ai1: run_ai = st.button("AIで要約とアドバイスを生成")
with col_ai2: simple_mode = st.checkbox("短めに要約（要点のみ）", value=True)

def calc_bmi_safe(height_cm: float, weight_kg: float):
    try: return calc_bmi(height_cm, weight_kg)
    except Exception: return None

def std_weight_safe(height_cm: float):
    try: return std_weight(height_cm)
    except Exception: return None

if run_ai:
    if not ai_key:
        st.error("OpenAI API Key を入力してください（サイドバー）")
    else:
        base = st.session_state.log.dropna(subset=["date"]).copy()
        if base.empty:
            st.info("食事データがありません")
        else:
            last_day = max(base["date"].dt.date.max(), date.today())
            if ai_window_choice == "全期間":
                start_day = min(base["date"].dt.date.min(), last_day); window_label = "全期間"
            else:
                ai_window_int = int(ai_window_choice)
                start_day = last_day - timedelta(days=ai_window_int-1); window_label = f"{ai_window_int}日"

            rmask = (base["date"].dt.date >= start_day) & (base["date"].dt.date <= last_day)
            rdf = base.loc[rmask].copy()
            daily = rdf.groupby(rdf["date"].dt.date)[NUTRIENTS].sum().round(1).sort_index()
            meals = rdf.groupby(rdf["date"].dt.date).size().rename("meals")
            daily = daily.join(meals, how="left").fillna(0); daily["meals"] = daily["meals"].astype(int)

            w = st.session_state.weight.copy()
            if not w.empty:
                w["date"] = pd.to_datetime(w["date"], errors="coerce")
                wmask = (w["date"].dt.date >= start_day) & (w["date"].dt.date <= last_day)
                wv = w.loc[wmask].copy()
                weight_series = wv.set_index(wv["date"].dt.date)["weight_kg"] if not wv.empty else pd.Series(dtype=float)
            else:
                weight_series = pd.Series(dtype=float)

            today_mask = st.session_state.log["date"].dt.date == last_day
            today_tot = st.session_state.log.loc[today_mask, NUTRIENTS].sum().round(1)
            limits = st.session_state.limits
            remaining = {}
            if limits.get("enabled", False):
                for n in NUTRIENTS:
                    L = float(limits.get(n, 0) or 0)
                    v = float(today_tot.get(n, 0) or 0)
                    if L>0:
                        diff = round(L - v, 1)
                        remaining[n] = diff if diff>0 else 0.0

            food_detail_json = None
            if ai_include_foods and not rdf.empty:
                freq = rdf["food"].value_counts().head(30).reset_index()
                freq.columns = ["food", "count"]
                food_sum = rdf.groupby("food")[NUTRIENTS].sum().round(1)
                def top_by(col, n=12):
                    if col not in food_sum.columns: return []
                    return (food_sum[col].sort_values(ascending=False).head(n)
                            .reset_index().rename(columns={col: f"total_{col}"}).to_dict(orient="records"))
                top_dict = {col: top_by(col) for col in NUTRIENTS}
                recent = rdf.sort_values("date").tail(80)[["date", "meal", "food"]].copy()
                recent["date"] = pd.to_datetime(recent["date"]).dt.strftime("%Y-%m-%d")
                recent_records = recent.to_dict(orient="records")
                food_detail = {"食品頻度TOP": freq.to_dict(orient="records"),
                               "栄養素別上位食品": top_dict,
                               "最近の食事明細": recent_records}
                food_detail_json = json.dumps(food_detail, ensure_ascii=False)

            p_sex = profile.get("sex","男性")
            p_age = int(profile.get("age", 28))
            p_h = float(profile.get("height_cm", 173.0))
            p_w = float(profile.get("current_weight_kg", 73.0))
            p_act = profile.get("activity","ふつう(週1-3運動)")
            p_bmi = calc_bmi_safe(p_h, p_w)
            p_std = std_weight_safe(p_h)

            df_for_prompt = daily.reset_index().rename(columns={"date":"日付"}); df_for_prompt["日付"] = df_for_prompt["日付"].astype(str)
            weight_dict = {str(k): float(v) for k, v in weight_series.to_dict().items()}

            system_msg = (
                "あなたは管理栄養士の視点をもつ日本語のダイエットコーチです。"
                "安全で現実的・実行可能な提案を行い、極端な減量や医学的判断は避けます。"
                "具体的な食事のアドバイスを提供し、必要に応じて上限(過剰)と不足の両面に触れてください。"
                "アドバイスは食事ログ・体重推移・上限設定・プロフィールを踏まえてください。"
            )
            base_block = f"""
【プロフィール】{json.dumps({"性別": p_sex, "年齢": p_age, "身長_cm": p_h, "現在体重_kg": p_w, "BMI": p_bmi, "標準体重_kg(BMI22)": p_std, "活動レベル": p_act}, ensure_ascii=False)}
【対象期間】{start_day}〜{last_day}（{window_label}）
【日別合計（kcal/P/F/C/食物繊維/糖質/ナトリウム, meals）】
{df_for_prompt.to_json(orient='records', force_ascii=False)}
【体重(kg) 推移】{json.dumps(weight_dict, ensure_ascii=False)}
【上限設定】{json.dumps({k: float(limits.get(k, 0) or 0) for k in NUTRIENTS}, ensure_ascii=False)}
【今日の不足分（上限到達までの残り, 無い場合は0）】{json.dumps(remaining, ensure_ascii=False)}
"""
            if food_detail_json:
                base_block += f"\n【食品名の参考（頻度/上位/直近明細）】{food_detail_json}\n上の食品名を具体的に引用し、代替食品・調理法・外食やコンビニの選び方まで実行可能な提案を出してください。\n"

            user_msg = base_block + "まず期間の傾向を要約し、次に『改善アクション』を日本語で提案。『体重に関する助言』も含め、最後に注意事項を1行添えてください。"

            try:
                if ai_debug:
                    with st.expander("🛠 デバッグ：送信プロンプト（system / user）", expanded=False):
                        st.code(system_msg, language="markdown")
                        st.code(user_msg, language="markdown")
                from openai import OpenAI
                client = OpenAI(api_key=ai_key)
                resp = _chat_create(
                    client,
                    ai_model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.6,  # gpt-5など非対応モデルでは送信されません
                )
                advice = resp.choices[0].message.content
                st.success("AIアドバイスを生成しました")
                st.markdown(advice)
                st.caption("※ 一般的な情報です。持病・服薬がある場合は医療専門家にご相談ください。")

                new_adv = pd.DataFrame([{
                    "created_at": pd.Timestamp.now(tz="Asia/Tokyo"), "model": ai_model, "window": window_label,
                    "include_foods": bool(ai_include_foods), "simple_mode": bool(simple_mode),
                    "start_day": pd.to_datetime(start_day), "last_day": pd.to_datetime(last_day), "ai_advice": advice,
                }])
                st.session_state.advice = pd.concat([st.session_state.advice, new_adv], ignore_index=True)
                save_advice_log(st.session_state.advice); st.success("AIアドバイスを保存しました（advice_log.csv）")
            except ModuleNotFoundError:
                st.error("`openai` がありません。`pip install openai` を実行し、requirements.txt に `openai>=1.42.0` を追加してください。")
            except Exception as e:
                st.error(f"OpenAI API呼び出しでエラー: {e}")

# ============================
# 📝 直近のAIアドバイス（最新1回）
# ============================
st.markdown("---")
st.subheader("📝 直近のAIアドバイス（最新1回）")

adv_hist = st.session_state.advice.copy()
if adv_hist.empty:
    st.caption("まだAIアドバイスの履歴はありません。上のボタンから生成してください。")
else:
    adv_hist["created_at"] = pd.to_datetime(adv_hist["created_at"], errors="coerce")
    latest = adv_hist.sort_values("created_at").iloc[-1]

    def _fmt_ts(x):
        try:
            ts = pd.to_datetime(x, errors="coerce")
            return ts.strftime("%Y-%m-%d %H:%M") if pd.notnull(ts) else ""
        except Exception:
            return ""

    def _fmt_window(val):
        if pd.isna(val): return ""
        try:
            iv = int(val); return f"{iv}日"
        except Exception:
            return str(val)

    created_s = _fmt_ts(latest.get("created_at"))
    model_s = str(latest.get("model", "") or "")
    window_label = _fmt_window(latest.get("window", ""))

    sd = pd.to_datetime(latest.get("start_day"), errors="coerce")
    ld = pd.to_datetime(latest.get("last_day"), errors="coerce")
    period_s = f"{sd.date()} 〜 {ld.date()}" if (pd.notnull(sd) and pd.notnull(ld)) else ""

    meta_parts = []
    if created_s: meta_parts.append(f"生成日: {created_s}")
    if model_s: meta_parts.append(f"モデル: {model_s}")
    if window_label: meta_parts.append(f"期間: {window_label}")
    if period_s: meta_parts.append(f"（{period_s}）")

    st.caption(" / ".join(meta_parts))
    st.info(str(latest.get("ai_advice", "")))
