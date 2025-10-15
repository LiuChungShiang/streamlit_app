# filename: 05_diet_record.py
import streamlit as st
import pandas as pd
from datetime import date, timedelta
import os, json, math, io, zipfile, base64
import plotly.express as px  # Plotlyでの描画

st.set_page_config(page_title="栄養管理アプリ(AIアドバイザー付き)", page_icon="🍱", layout="wide")

# ============================
# ファイルパス / 文字コード
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

def data_path(filename: str) -> str:
    return os.path.join(DATA_DIR, filename)

def ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

LOG_PATH = data_path("diet_log.csv")
FOOD_DB_PATH = data_path("food_db.csv")
LIMITS_PATH = data_path("limits.json")
WEIGHT_PATH = data_path("weight_log.csv")
ADVICE_PATH = data_path("advice_log.csv")
PROFILE_PATH = data_path("profile.json")

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
# ロード/セーブ関数
# ============================
def _ensure_food_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "unit" not in df.columns:
        df["unit"] = ""
    if "per" not in df.columns:
        df["per"] = 1.0
    cols = [c for c in df.columns if c in NUTRIENTS]
    if cols:
        df[cols] = df[cols].astype(float).round(1)
    df["per"] = pd.to_numeric(df["per"], errors="coerce").fillna(1.0).round(1)
    df["unit"] = df["unit"].astype(str)
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
    ensure_data_dir()
    df = _ensure_food_df_columns(df)
    df.to_csv(path, index=False, encoding=CSV_ENCODING)

def load_log(path: str = LOG_PATH) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            df = read_csv_smart(path, is_path=True)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            cols = [c for c in df.columns if c in NUTRIENTS]
            if cols:
                df[cols] = df[cols].astype(float).round(1)
            if "per" in df.columns:
                df["per"] = pd.to_numeric(df["per"], errors="coerce").round(1)
            if "unit" not in df.columns:
                df["unit"] = ""
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=["date", "meal", "food", "unit", "per", *NUTRIENTS])

def save_log(df: pd.DataFrame, path: str = LOG_PATH):
    ensure_data_dir()
    cols = [c for c in df.columns if c in NUTRIENTS]
    if cols:
        df[cols] = df[cols].astype(float).round(1)
    if "per" in df.columns:
        df["per"] = pd.to_numeric(df["per"], errors="coerce").fillna(1.0).round(1)
    if "unit" in df.columns:
        df["unit"] = df["unit"].astype(str)
    df.to_csv(path, index=False, encoding=CSV_ENCODING)

def load_limits(path: str = LIMITS_PATH) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "kcal": 2000.0,
        "protein": 150.0,
        "fat": 60.0,
        "carbs": 260.0,
        "sugar": 50.0,
        "sodium_mg": 2300.0,
        "fiber": 20.0,
        "enabled": False,
    }

def save_limits(limits: dict, path: str = LIMITS_PATH):
    ensure_data_dir()
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
    ensure_data_dir()
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
    ensure_data_dir()
    df.to_csv(path, index=False, encoding=CSV_ENCODING)

def load_profile(path: str = PROFILE_PATH) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "sex": "男性",
        "age": 28,
        "height_cm": 173.0,
        "current_weight_kg": 73.0,
        "activity": "ふつう(週1-3運動)",
    }

def save_profile(prof: dict, path: str = PROFILE_PATH):
    ensure_data_dir()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prof, f, ensure_ascii=False, indent=2)

# ============================
# 共通：BMI/標準体重
# ============================
def calc_bmi(height_cm: float, weight_kg: float):
    h_m = max(0.5, float(height_cm)/100.0)
    if weight_kg is None or (isinstance(weight_kg, float) and math.isnan(weight_kg)):
        return None
    return round(weight_kg / (h_m*h_m), 1)

def std_weight(height_cm: float):
    h_m = max(0.5, float(height_cm)/100.0)
    return round(22.0 * h_m * h_m, 1)

def calc_bmr(height_cm: float, weight_kg: float, age: int, sex: str):
    """Mifflin-St Jeor方程式で推定静的代謝量を計算"""
    try:
        h = float(height_cm)
        w = float(weight_kg)
        a = int(age)
    except (TypeError, ValueError):
        return None
    if h <= 0 or w <= 0 or a <= 0:
        return None
    sex_factor = 5
    if str(sex) == "女性":
        sex_factor = -161
    elif str(sex) == "その他":
        sex_factor = -78  # 男女の中央値を採用
    bmr = 10 * w + 6.25 * h - 5 * a + sex_factor
    return round(bmr, 0)

def calc_tdee(bmr: float, activity_label: str):
    if bmr is None:
        return None
    multipliers = {
        "低い(座位中心)": 1.2,
        "ふつう(週1-3運動)": 1.55,
        "高い(週4+運動)": 1.725,
    }
    mult = multipliers.get(activity_label, 1.4)
    return round(bmr * mult, 0)

# ============================
# バックアップ（ZIP）ユーティリティ
# ============================
BACKUP_FILES = [FOOD_DB_PATH, LOG_PATH, WEIGHT_PATH, ADVICE_PATH, LIMITS_PATH, PROFILE_PATH]

def make_backup_zip_bytes() -> bytes:
    """現在の保存ファイル群からZIPバイナリを作成"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # food_db
        if os.path.exists(FOOD_DB_PATH):
            zf.write(FOOD_DB_PATH, arcname=FOOD_DB_PATH)
        else:
            df = _ensure_food_df_columns(get_default_food_df().copy())
            zf.writestr(FOOD_DB_PATH, df.to_csv(index=False, encoding=CSV_ENCODING))
        # diet_log
        if os.path.exists(LOG_PATH):
            zf.write(LOG_PATH, arcname=LOG_PATH)
        else:
            zf.writestr(LOG_PATH, "date,meal,food,unit,per," + ",".join(NUTRIENTS) + "\n")
        # weight_log
        if os.path.exists(WEIGHT_PATH):
            zf.write(WEIGHT_PATH, arcname=WEIGHT_PATH)
        else:
            zf.writestr(WEIGHT_PATH, "date,weight_kg\n")
        # advice_log
        if os.path.exists(ADVICE_PATH):
            zf.write(ADVICE_PATH, arcname=ADVICE_PATH)
        else:
            zf.writestr(ADVICE_PATH, "created_at,model,window,include_foods,simple_mode,start_day,last_day,ai_advice\n")
        # limits.json
        if os.path.exists(LIMITS_PATH):
            zf.write(LIMITS_PATH, arcname=LIMITS_PATH)
        else:
            zf.writestr(LIMITS_PATH, json.dumps(load_limits(), ensure_ascii=False, indent=2))
        # profile.json
        if os.path.exists(PROFILE_PATH):
            zf.write(PROFILE_PATH, arcname=PROFILE_PATH)
        else:
            zf.writestr(PROFILE_PATH, json.dumps(load_profile(), ensure_ascii=False, indent=2))
    buf.seek(0)
    return buf.read()

def restore_from_zip(uploaded_zip_bytes: bytes) -> dict:
    """ZIPから各データを復元し、保存＆セッション反映。戻り値は処理結果メッセージ集"""
    results = {}
    with zipfile.ZipFile(io.BytesIO(uploaded_zip_bytes), "r") as zf:
        names = set(zf.namelist())

        # 食品DB
        if FOOD_DB_PATH in names:
            with zf.open(FOOD_DB_PATH) as f:
                b = io.BytesIO(f.read()); b.seek(0)
                df = read_csv_smart(b, is_path=False)
                df = _ensure_food_df_columns(df)
                st.session_state.food_db = df
                save_food_db(df)
                results["food_db"] = f"{FOOD_DB_PATH} を復元しました"
        # 食事ログ
        if LOG_PATH in names:
            with zf.open(LOG_PATH) as f:
                b = io.BytesIO(f.read()); b.seek(0)
                df = read_csv_smart(b, is_path=False)
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                st.session_state.log = df
                save_log(df)
                results["diet_log"] = f"{LOG_PATH} を復元しました"
        # 体重
        if WEIGHT_PATH in names:
            with zf.open(WEIGHT_PATH) as f:
                b = io.BytesIO(f.read()); b.seek(0)
                df = read_csv_smart(b, is_path=False)
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                if "weight_kg" in df.columns:
                    df["weight_kg"] = pd.to_numeric(df["weight_kg"], errors="coerce").round(1)
                st.session_state.weight = df
                save_weight(df)
                results["weight_log"] = f"{WEIGHT_PATH} を復元しました"
        # アドバイス
        if ADVICE_PATH in names:
            with zf.open(ADVICE_PATH) as f:
                b = io.BytesIO(f.read()); b.seek(0)
                df = read_csv_smart(b, is_path=False)
                for col in ["start_day", "last_day", "created_at"]:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                st.session_state.advice = df
                save_advice_log(df)
                results["advice_log"] = f"{ADVICE_PATH} を復元しました"
        # 上限
        if LIMITS_PATH in names:
            with zf.open(LIMITS_PATH) as f:
                limits = json.loads(f.read().decode("utf-8"))
                for k in NUTRIENTS:
                    limits[k] = float(limits.get(k, 0) or 0)
                limits["enabled"] = bool(limits.get("enabled", False))
                st.session_state.limits = limits
                save_limits(limits)
                results["limits"] = f"{LIMITS_PATH} を復元しました"
        # プロフィール
        if PROFILE_PATH in names:
            with zf.open(PROFILE_PATH) as f:
                prof = json.loads(f.read().decode("utf-8"))
                st.session_state.profile = prof
                save_profile(prof)
                results["profile"] = f"{PROFILE_PATH} を復元しました"
    return results

# ============================
# セッション初期化
# ============================
if "food_db" not in st.session_state:
    st.session_state.food_db = load_food_db()
if "log" not in st.session_state:
    st.session_state.log = load_log()
if "date" not in st.session_state:
    st.session_state.date = date.today()
if "limits" not in st.session_state:
    st.session_state.limits = load_limits()
if "weight" not in st.session_state:
    st.session_state.weight = load_weight()
if "advice" not in st.session_state:
    st.session_state.advice = load_advice_log()
if "profile" not in st.session_state:
    st.session_state.profile = load_profile()
if "ai_food_pending" not in st.session_state:
    st.session_state.ai_food_pending = None
if "ai_food_desc_text" not in st.session_state:
    st.session_state.ai_food_desc_text = ""
if "ai_food_desc_widget" not in st.session_state:
    st.session_state.ai_food_desc_widget = st.session_state.ai_food_desc_text

# ============================
# サイドバー（設定とデータ）
# ============================
with st.sidebar:
    st.header("⚙️ 設定とデータ")

    # 手動追加
    with st.expander("食品を手動で追加"):
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
                new_row = {
                    "food": food_name, "unit": "", "per": 1.0,
                    "kcal": round(kcal, 1), "protein": round(protein, 1), "fat": round(fat, 1),
                    "carbs": round(carbs, 1), "fiber": round(fiber, 1), "sugar": round(sugar, 1),
                    "sodium_mg": round(sodium_mg, 1),
                }
                st.session_state.food_db = pd.concat([st.session_state.food_db, pd.DataFrame([new_row])], ignore_index=True)
                save_food_db(st.session_state.food_db)
                st.success(f"{food_name} をDBに追加しました（保存済み）")
            else:
                st.error("食品名を入力してください")

    # 食塩→ナトリウム換算
    with st.expander("食塩(g) → ナトリウム(mg) 換算", expanded=False):
        st.caption("目安換算：NaCl中のNaは約39.3%。食塩1g ≒ ナトリウム約394mg")
        salt_g = st.number_input("食塩相当量 (g)", min_value=0.0, value=0.0, step=0.1)
        sodium_est = round(salt_g * 1000.0 / 2.54, 1)  # ≒393.7mg/g
        st.metric(label="換算結果（ナトリウム）", value=f"{sodium_est} mg/日")

    # AIで食品の栄養推定（プレビュー→保存）
    with st.expander("🤖栄養成分を推定（AI）", expanded=False):
        ai_food_name = st.text_input("食品名（例：照り焼きチキン丼）", value="")
        # Textエリアの状態を最新化（前回入力内容を保持）
        st.session_state.ai_food_desc_text = st.session_state.get(
            "ai_food_desc_widget",
            st.session_state.ai_food_desc_text,
        )

        desc_image = st.file_uploader(
            "説明用に文字起こししたい画像",
            type=["png", "jpg", "jpeg", "webp"],
            key="ai_food_desc_image"
        )
        desc_cols = st.columns(2)
        transcribe_desc = desc_cols[0].button("画像から説明に追加", use_container_width=True)
        clear_desc = desc_cols[1].button("説明をクリア", use_container_width=True)

        if clear_desc:
            st.session_state.ai_food_desc_text = ""
            st.session_state.ai_food_desc_widget = ""
            st.success("説明欄をクリアしました。")

        if transcribe_desc:
            if desc_image is None:
                st.error("説明に追加する画像をアップロードしてください。")
            else:
                env_key_desc = os.environ.get("OPENAI_API_KEY", "")
                secret_key_desc = None
                try:
                    secret_key_desc = st.secrets.get("OPENAI_API_KEY")
                except Exception:
                    pass
                ai_key_desc = (st.session_state.get("ai_api_key") or secret_key_desc or env_key_desc or None)
                if not ai_key_desc:
                    st.error("OpenAI API Key が未設定です。『🤖 AIアドバイス 設定』で入力してください。")
                else:
                    try:
                        from openai import OpenAI
                        desc_image.seek(0)
                        image_bytes = desc_image.read()
                        if not image_bytes:
                            st.error("画像の読み込みに失敗しました。別の画像をお試しください。")
                        else:
                            ext = (os.path.splitext(desc_image.name or "")[1].lower().replace(".", "") or "jpeg")
                            if ext == "jpg":
                                ext = "jpeg"
                            if ext not in ["png", "jpeg", "webp"]:
                                ext = "jpeg"
                            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                            data_url = f"data:image/{ext};base64,{image_b64}"
                            sys_desc = (
                                "あなたはOCRアシスタントです。"
                                "画像内に含まれる日本語の文字列を正確に読み取り、カロリー、タンパク質、脂質、炭水化物、糖質、食物繊維、食塩かナトリウムだけ出力してください。"
                                "出力は抽出した栄養成分に関するテキストのみで、説明や装飾は不要です。"
                            )
                            user_desc = "以下の画像から読み取れるテキストを抽出し、日本語でそのまま出力してください。"
                            client_desc = OpenAI(api_key=ai_key_desc)
                            resp_desc = client_desc.chat.completions.create(
                                model=st.session_state.get("ai_model", "gpt-4o-mini"),
                                messages=[
                                    {"role": "system", "content": sys_desc},
                                    {
                                        "role": "user",
                                        "content": [
                                            {"type": "text", "text": user_desc},
                                            {"type": "image_url", "image_url": {"url": data_url}},
                                        ],
                                    },
                                ],
                                temperature=0.0,
                            )
                            desc_text = resp_desc.choices[0].message.content.strip()
                            if desc_text:
                                current_desc = st.session_state.get("ai_food_desc_text", "")
                                if current_desc:
                                    if not current_desc.endswith("\n"):
                                        current_desc += "\n"
                                    st.session_state.ai_food_desc_text = current_desc + desc_text
                                else:
                                    st.session_state.ai_food_desc_text = desc_text
                                st.session_state.ai_food_desc_widget = st.session_state.ai_food_desc_text
                                st.success("文字起こし結果を説明欄に追加しました。")
                            else:
                                st.info("抽出できるテキストが見つかりませんでした。")
                    except ModuleNotFoundError:
                        st.error("`openai` パッケージが見つかりません。`pip install openai` を実行してください。")
                    except Exception as e:
                        st.error(f"OpenAI呼び出しエラー: {e}")

        # テキストエリアの表示値を最新の説明に同期
        st.session_state.ai_food_desc_widget = st.session_state.ai_food_desc_text
        st.text_area(
            "食品の説明（栄養成分があれば尚良）",
            key="ai_food_desc_widget",
        )
        st.session_state.ai_food_desc_text = st.session_state.ai_food_desc_widget

        colf = st.columns(2)
        run_est = colf[0].button("🤖 推定する", use_container_width=True)
        clear_pending = colf[1].button("クリア", use_container_width=True)

        if clear_pending:
            st.session_state.ai_food_pending = None

        if run_est:
            env_key2 = os.environ.get("OPENAI_API_KEY", "")
            secret_key2 = None
            try:
                secret_key2 = st.secrets.get("OPENAI_API_KEY")
            except Exception:
                pass
            ai_key2 = (st.session_state.get("ai_api_key") or secret_key2 or env_key2 or None)
            if not ai_key2:
                st.error("OpenAI API Key が未設定です。『🤖 AIアドバイス 設定』で入力してください。")
            elif not ai_food_name.strip():
                st.error("食品名を入力してください。")
            else:
                sys2 = (
                    "あなたは日本語の管理栄養士です。与えられた食品名と説明から、"
                    "1食分の概算栄養成分をJSONで推定してください。"
                    "キー: kcal, protein, fat, carbs, fiber, sugar, sodium_mg（すべて数値, 小数1桁）。"
                    "根拠は出力しないでください。"
                )
                desc_text_for_prompt = st.session_state.get("ai_food_desc_text", "")
                user2 = (
                    f"食品名: {ai_food_name}\n"
                    f"説明: {desc_text_for_prompt}\n"
                    "出力はJSONのみ。例: {\"kcal\": 520.0, \"protein\": 32.0, \"fat\": 15.0, "
                    "\"carbs\": 65.0, \"fiber\": 4.0, \"sugar\": 8.0, \"sodium_mg\": 1200.0}"
                )
                try:
                    from openai import OpenAI
                    client2 = OpenAI(api_key=ai_key2)
                    resp2 = client2.chat.completions.create(
                        model=st.session_state.get("ai_model","gpt-4o-mini"),
                        messages=[{"role":"system","content":sys2},{"role":"user","content":user2}],
                        temperature=0.5,
                    )
                    txt2 = resp2.choices[0].message.content.strip()
                    if txt2.startswith("```"):
                        txt2 = txt2.strip("`")
                        txt2 = txt2.split("\n",1)[-1]
                        if txt2.lower().startswith("json"):
                            txt2 = txt2.split("\n",1)[-1]
                        if txt2.endswith("```"):
                            txt2 = txt2[:-3]
                    try:
                        js2 = json.loads(txt2)
                        pending = {"food": ai_food_name.strip(), "unit": "", "per": 1.0}
                        for k in NUTRIENTS:
                            v = js2.get(k, 0.0)
                            pending[k] = round(float(v), 1)
                        st.session_state.ai_food_pending = pending
                        st.success("推定が完了しました。下で内容を確認し、保存してください。")
                    except Exception:
                        st.warning("AI出力のJSON解析に失敗しました。内容を表示します：")
                        st.code(txt2, language="json")
                except ModuleNotFoundError:
                    st.error("`openai` パッケージが見つかりません。`pip install openai` を実行してください。")
                except Exception as e:
                    st.error(f"OpenAI呼び出しエラー: {e}")

        if st.session_state.ai_food_pending:
            st.markdown("**推定結果（1食分の概算）**")
            prev = {k: st.session_state.ai_food_pending[k] for k in ["food", *NUTRIENTS]}
            st.table(pd.Series(prev).to_frame("value"))
            if st.button("この内容でDBに追加（保存）", use_container_width=True):
                st.session_state.food_db = pd.concat(
                    [st.session_state.food_db, pd.DataFrame([st.session_state.ai_food_pending])],
                    ignore_index=True
                )
                save_food_db(st.session_state.food_db)
                st.success(f"『{st.session_state.ai_food_pending['food']}』をDBに保存しました。")
                st.session_state.ai_food_pending = None

    with st.expander("食品を削除/食品名を編集", expanded=False):
        foods = sorted(st.session_state.food_db["food"].astype(str).unique().tolist())
        if not foods:
            st.caption("登録されている食品がありません")
        else:
            st.markdown("**食品名を編集**")
            target_food = st.selectbox("変更する食品を選択", foods, key="edit_food_select")
            new_name = st.text_input("新しい食品名", key="edit_food_new_name")
            if st.button("名前を更新", use_container_width=True):
                new_name_clean = new_name.strip()
                if not new_name_clean:
                    st.error("新しい食品名を入力してください")
                elif new_name_clean == target_food:
                    st.info("同じ名前が入力されています")
                elif new_name_clean in foods:
                    st.error("同じ名前の食品が既に存在します")
                else:
                    mask = st.session_state.food_db["food"].astype(str) == target_food
                    if not mask.any():
                        st.error("選択した食品が見つかりませんでした。")
                    else:
                        st.session_state.food_db.loc[mask, "food"] = new_name_clean
                        st.session_state.food_db = st.session_state.food_db.reset_index(drop=True)
                        save_food_db(st.session_state.food_db)
                        st.success(f"『{target_food}』を『{new_name_clean}』に変更しました（保存済み）")
                        st.session_state.edit_food_select = new_name_clean
                        st.session_state.edit_food_new_name = ""

            st.markdown("---")
            st.markdown("**食品を削除**")
            del_select = st.multiselect("削除する食品を選択", foods)
            if st.button("選択した食品を削除"):
                if del_select:
                    before = len(st.session_state.food_db)
                    st.session_state.food_db = st.session_state.food_db[~st.session_state.food_db["food"].isin(del_select)].reset_index(drop=True)
                    save_food_db(st.session_state.food_db)
                    after = len(st.session_state.food_db)
                    st.success(f"{len(del_select)} 件削除しました（{before} → {after}）")
                else:
                    st.info("削除対象が選択されていません")

    with st.expander("体重データを削除"):
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

    # 🤖 AIアドバイス 設定（※対象期間に30日を追加）
    st.markdown("---")
    with st.expander("🤖AIアドバイス設定（OpenAI API）", expanded=False):
        env_key = os.environ.get("OPENAI_API_KEY", "")
        secret_key = None
        try:
            secret_key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            pass
        if env_key and not secret_key:
            st.caption("🔐 環境変数 OPENAI_API_KEY を検出。必要なら下で上書きできます。")
        if secret_key:
            st.caption("🔐 secrets.toml の OPENAI_API_KEY を検出。必要なら下で上書きできます。")
        api_key_input = st.text_input("OpenAI API Key", type="password", value="")
        st.session_state.ai_api_key = (api_key_input.strip() or secret_key or env_key or None)
        st.session_state.ai_model = st.selectbox("モデル", ["gpt-4o-mini", "gpt-4.1-mini"], index=0)

        # ★ ここに30日を追加
        ai_window_options = [5, 10, 15, 20, 30]
        sel = st.session_state.get('ai_window', 10)
        preselect_idx = 0 if sel == "全期間" else ai_window_options.index(sel) if sel in ai_window_options else ai_window_options.index(10)
        st.session_state.ai_window = st.radio("対象期間", ai_window_options, index=preselect_idx, horizontal=True)
        st.session_state.ai_include_foods = st.checkbox("食事ログもAIに渡す", value=st.session_state.get('ai_include_foods', True))
        st.session_state.ai_debug = st.checkbox("🛠 デバッグ：送信プロンプトを表示", value=st.session_state.get('ai_debug', False))

    # ============================
    # 📦 エクスポート / 自動保存（折りたたみ対応）
    # ============================
    st.markdown("---")
    with st.expander("📦 エクスポート / 自動保存", expanded=False):

        # 既存の各CSVダウンロード
        st.download_button(
            "現在の食品DBをCSVでダウンロード",
            data=st.session_state.food_db.to_csv(index=False).encode(CSV_ENCODING),
            file_name="food_db.csv",
            mime="text/csv",
            use_container_width=True,
        )

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
                latest_adv = (
                    adv2.sort_values("created_at")
                        .groupby("date", as_index=False)
                        .tail(1)[["date", "ai_advice"]]
                        .set_index("date")
                )
                combined = combined.join(latest_adv, how="left")

            combined = combined.round(1)
            combined.index.name = "date"
            csv_combined = combined.reset_index().to_csv(index=False).encode(CSV_ENCODING)
            st.download_button(
                "結合データ（全期間）CSVをダウンロード",
                data=csv_combined,
                file_name="combined_data.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.caption("結合できるデータがまだありません")

        csv_all = st.session_state.log.round(1).to_csv(index=False).encode(CSV_ENCODING)
        st.download_button("食事ログCSV", data=csv_all, file_name=LOG_PATH, mime="text/csv", use_container_width=True)

        csv_w = st.session_state.weight.round(1).to_csv(index=False).encode(CSV_ENCODING) if not st.session_state.weight.empty else ("date,weight_kg\n".encode(CSV_ENCODING))
        st.download_button("体重ログCSV", data=csv_w, file_name=WEIGHT_PATH, mime="text/csv", use_container_width=True)

        csv_adv = st.session_state.advice.to_csv(index=False).encode(CSV_ENCODING)
        st.download_button("AIアドバイス履歴CSV", data=csv_adv, file_name=ADVICE_PATH, mime="text/csv", use_container_width=True)

        st.caption("💾 このアプリは入力や操作のたびに各CSV/JSONへ自動保存しています。")

        # 🗜 一括バックアップ（ZIP）
        st.markdown("#### 🗜 一括バックアップ（ZIP）")
        backup_bytes = make_backup_zip_bytes()
        st.download_button(
            "すべてのデータをZIPでダウンロード（バックアップ）",
            data=backup_bytes,
            file_name="diet_backup.zip",
            mime="application/zip",
            use_container_width=True
        )

        # ♻️ 復元：ZIPを読み込んで一括復元
        st.markdown("#### ♻️ ZIPからの復元")
        up_zip = st.file_uploader("バックアップZIPを選択", type=["zip"], key="zip_restore_uploader")
        if up_zip is not None:
            try:
                results = restore_from_zip(up_zip.read())
                if results:
                    for k, msg in results.items():
                        st.success(msg)
                    st.success("一括復元が完了しました（すべて保存済み）")
                else:
                    st.info("ZIP内に対応するファイルが見つかりませんでした")
            except zipfile.BadZipFile:
                st.error("ZIPファイルが壊れている可能性があります")
            except Exception as e:
                st.error(f"ZIP復元中にエラー: {e}")

        # 🧩 復元：各ファイル単位
        st.markdown("#### 🧩 個別ファイルから復元")
        colu1, colu2 = st.columns(2)
        with colu1:
            up_log = st.file_uploader("食事ログ（diet_log.csv）", type=["csv"], key="log_restore")
            if up_log is not None:
                try:
                    df = read_csv_smart(up_log, is_path=False)
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    st.session_state.log = df
                    save_log(df)
                    st.success("diet_log.csv を復元＆保存しました")
                except Exception as e:
                    st.error(f"食事ログ復元エラー: {e}")

            up_weight = st.file_uploader("体重ログ（weight_log.csv）", type=["csv"], key="weight_restore")
            if up_weight is not None:
                try:
                    df = read_csv_smart(up_weight, is_path=False)
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    if "weight_kg" in df.columns:
                        df["weight_kg"] = pd.to_numeric(df["weight_kg"], errors="coerce").round(1)
                    st.session_state.weight = df
                    save_weight(df)
                    st.success("weight_log.csv を復元＆保存しました")
                except Exception as e:
                    st.error(f"体重ログ復元エラー: {e}")

            up_food = st.file_uploader("食品DB（food_db.csv）", type=["csv"], key="food_restore")
            if up_food is not None:
                try:
                    df = read_csv_smart(up_food, is_path=False)
                    required = {"food", *NUTRIENTS}
                    if not required.issubset(df.columns):
                        st.error("CSVに必要な列: food, kcal, protein, fat, carbs, fiber, sugar, sodium_mg（unit, per は任意）")
                    else:
                        df = _ensure_food_df_columns(df)
                        st.session_state.food_db = df
                        save_food_db(df)
                        st.success("food_db.csv を復元＆保存しました")
                except Exception as e:
                    st.error(f"食品DB復元エラー: {e}")

        with colu2:
            up_adv = st.file_uploader("AIアドバイス履歴（advice_log.csv）", type=["csv"], key="adv_restore")
            if up_adv is not None:
                try:
                    df = read_csv_smart(up_adv, is_path=False)
                    for col in ["start_day", "last_day", "created_at"]:
                        if col in df.columns:
                            df[col] = pd.to_datetime(df[col], errors="coerce")
                    st.session_state.advice = df
                    save_advice_log(df)
                    st.success("advice_log.csv を復元＆保存しました")
                except Exception as e:
                    st.error(f"アドバイス履歴復元エラー: {e}")

            up_limits = st.file_uploader("日次上限設定（limits.json）", type=["json"], key="limits_restore")
            if up_limits is not None:
                try:
                    txt = up_limits.read().decode("utf-8")
                    limits = json.loads(txt)
                    for k in NUTRIENTS:
                        limits[k] = float(limits.get(k, 0) or 0)
                    limits["enabled"] = bool(limits.get("enabled", False))
                    st.session_state.limits = limits
                    save_limits(limits)
                    st.success("limits.json を復元＆保存しました")
                except Exception as e:
                    st.error(f"上限設定復元エラー: {e}")

            up_profile = st.file_uploader("プロフィール（profile.json）", type=["json"], key="profile_restore")
            if up_profile is not None:
                try:
                    txt = up_profile.read().decode("utf-8")
                    prof = json.loads(txt)
                    st.session_state.profile = prof
                    save_profile(prof)
                    st.success("profile.json を復元＆保存しました")
                except Exception as e:
                    st.error(f"プロフィール復元エラー: {e}")

        # 末尾で自動保存（冪等）
        save_log(st.session_state.log)
        save_weight(st.session_state.weight)
        save_advice_log(st.session_state.advice)
        save_profile(st.session_state.profile)

# ============================
# メインUI
# ============================
st.title("🍱 栄養管理ダイエット記録")
st.caption("食品を選ぶと 1食分として記録。すべて小数点1位で保存・表示。")

# ------- 1×2：入力フォーム & 体重の記録（当日） -------
selected_date = st.date_input("表示する日付（入力・体重共通）", value=st.session_state.date, format="YYYY-MM-DD", key="display_date_main")
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### 🍽️ 入力フォーム")
    food = None
    with st.form("input_form"):
        meal = st.selectbox("食事区分", MEAL_TYPES, index=0)
        db = st.session_state.food_db
        options = db["food"].tolist()
        food = st.selectbox("食品を選択", options, index=0 if options else None)
        submitted = st.form_submit_button("➕ 1食分を追加", use_container_width=True)
    if submitted and food:
        row = st.session_state.food_db[st.session_state.food_db["food"] == food].iloc[0]
        entry = {"date": pd.to_datetime(selected_date), "meal": meal, "food": row.get("food", food),
                 "unit": str(row.get("unit", "")), "per": round(float(row.get("per", 1.0)), 1)}
        for n in NUTRIENTS:
            entry[n] = round(float(row[n]), 1)
        st.session_state.log = pd.concat([st.session_state.log, pd.DataFrame([entry])], ignore_index=True)
        save_log(st.session_state.log)
        st.success(f"{food} を1食分追加しました（保存済み）")

with col_right:
    st.markdown("#### ⚖️ 体重の記録")
    wdf = st.session_state.weight.copy()
    wdf["date"] = pd.to_datetime(wdf["date"], errors="coerce")
    cur = wdf[wdf["date"].dt.date == pd.to_datetime(selected_date).date()]
    def_weight = float(cur["weight_kg"].iloc[0]) if not cur.empty else float(st.session_state.profile.get("current_weight_kg", 73.0))
    input_weight = st.number_input("体重(kg)", min_value=0.0, value=round(def_weight,1), step=0.1, format="%.1f")
    prof_for_bmi = st.session_state.profile
    bmi_val = calc_bmi(prof_for_bmi.get("height_cm",173.0), input_weight)
    st.caption(f"BMI: {bmi_val if bmi_val is not None else '—'} / 標準体重(BMI22): {std_weight(prof_for_bmi.get('height_cm',173.0))} kg")
    if st.button("体重を保存", use_container_width=True):
        st.session_state.weight = wdf[wdf["date"].dt.date != pd.to_datetime(selected_date).date()].copy()
        new_row = pd.DataFrame({"date": [pd.to_datetime(selected_date)], "weight_kg": [round(input_weight,1)]})
        st.session_state.weight = pd.concat([st.session_state.weight, new_row], ignore_index=True)
        save_weight(st.session_state.weight)
        st.session_state.profile["current_weight_kg"] = round(input_weight,1)
        save_profile(st.session_state.profile)
        st.success("体重を保存しました")

# ===== プロフィール（AIが参照）
with st.expander("👤プロフィール", expanded=False):
    p = st.session_state.profile
    colp1, colp2 = st.columns(2)
    with colp1:
        p["sex"] = st.selectbox("性別", ["男性","女性","その他"], index=0 if p.get("sex","男性")=="男性" else (1 if p.get("sex")=="女性" else 2))
        p["age"] = int(st.number_input("年齢", min_value=10, max_value=100, value=int(p.get("age",28))))
        p["height_cm"] = float(st.number_input("身長(cm)", min_value=120.0, max_value=230.0, value=float(p.get("height_cm",173.0)), step=0.1, format="%.1f"))
    with colp2:
        latest_w = None
        if not st.session_state.weight.empty:
            wtmp = st.session_state.weight.copy().sort_values("date")
            if not wtmp.empty and pd.notnull(wtmp["weight_kg"].iloc[-1]):
                latest_w = float(wtmp["weight_kg"].iloc[-1])
        default_w = float(p.get("current_weight_kg", 73.0))
        if latest_w:
            default_w = latest_w
        p["current_weight_kg"] = float(st.number_input("現在体重(kg)", min_value=30.0, max_value=200.0, value=default_w, step=0.1, format="%.1f"))
        p["activity"] = st.selectbox("活動レベル", ["低い(座位中心)","ふつう(週1-3運動)","高い(週4+運動)"], index=["低い(座位中心)","ふつう(週1-3運動)","高い(週4+運動)"].index(p.get("activity","ふつう(週1-3運動)")))

    bmi_prof = calc_bmi(p.get("height_cm", 173.0), p.get("current_weight_kg", 73.0))
    std_w_prof = std_weight(p.get("height_cm", 173.0))
    bmr_prof = calc_bmr(p.get("height_cm", 173.0), p.get("current_weight_kg", 73.0), p.get("age", 28), p.get("sex", "男性"))
    tdee_prof = calc_tdee(bmr_prof, p.get("activity", "ふつう(週1-3運動)"))

    col_metrics = st.columns(4)
    col_metrics[0].metric("BMI", f"{bmi_prof:.1f}" if bmi_prof is not None else "—")
    col_metrics[1].metric("標準体重(BMI22)", f"{std_w_prof} kg")
    col_metrics[2].metric("基礎代謝量(BMR)", f"{bmr_prof:.0f} kcal/日" if bmr_prof is not None else "—")
    col_metrics[3].metric("推定消費カロリー(TDEE)", f"{tdee_prof:.0f} kcal/日" if tdee_prof is not None else "—")
    st.caption("※BMRはMifflin-St Jeor、TDEEは活動レベル別の推定係数で計算しています。")

    save_profile(p)

# ======= （変更1）📏 日次上限設定：プロフィールの直下に移動＋折り畳み =======
with st.expander("📏 日次上限設定", expanded=False):
    st.session_state.limits["enabled"] = st.toggle("上限チェックを有効化", value=st.session_state.limits.get("enabled", False))
    cols_lim = st.columns(3)
    st.session_state.limits["kcal"] = cols_lim[0].number_input("kcal 上限", value=float(st.session_state.limits["kcal"]))
    st.session_state.limits["protein"] = cols_lim[1].number_input("たんぱく質(g) 上限", value=float(st.session_state.limits["protein"]))
    st.session_state.limits["fat"] = cols_lim[2].number_input("脂質(g) 上限", value=float(st.session_state.limits["fat"]))
    cols_lim2 = st.columns(4)
    st.session_state.limits["carbs"] = cols_lim2[0].number_input("炭水化物(g) 上限", value=float(st.session_state.limits["carbs"]))
    st.session_state.limits["fiber"] = cols_lim2[1].number_input("食物繊維(g) 上限", value=float(st.session_state.limits["fiber"]))
    st.session_state.limits["sugar"] = cols_lim2[2].number_input("糖質(g) 上限", value=float(st.session_state.limits["sugar"]))
    st.session_state.limits["sodium_mg"] = cols_lim2[3].number_input("ナトリウム(mg) 上限", value=float(st.session_state.limits["sodium_mg"]))

    # AIで上限推定（減量プラン）
    st.markdown("---")
    if st.button("🤖減量プランの上限を推定（AI）", use_container_width=True):
        env_key = os.environ.get("OPENAI_API_KEY", "")
        secret_key = None
        try:
            secret_key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            pass
        ai_key = (st.session_state.get("ai_api_key") or secret_key or env_key or None)
        if not ai_key:
            st.error("OpenAI API Key が未設定です。サイドバー『🤖 AIアドバイス 設定』で入力してください。")
        else:
            prof = st.session_state.profile
            prompt = {
                "sex": prof.get("sex", "男性"),
                "age": int(prof.get("age", 28)),
                "height_cm": float(prof.get("height_cm", 173.0)),
                "current_weight_kg": float(prof.get("current_weight_kg", 73.0)),
                "activity": prof.get("activity", "ふつう(週1-3運動)"),
                "goal": "減量(週あたり0.25〜0.5kg目安)",
            }
            sys = (
                "あなたは日本語の管理栄養士です。"
                "プロフィールから現実的な減量プランの1日あたり目標値をJSONで返してください。"
                "高たんぱく・適正脂質・適正炭水化物の範囲を踏まえ、値は小数1桁で。"
                "必ず以下のキーを全て含めてください: kcal, protein, fat, carbs, fiber, sugar, sodium_mg。"
            )
            user = f"プロフィール: {json.dumps(prompt, ensure_ascii=False)}\n出力はJSONのみ。例: {{\"kcal\": 1800.0, \"protein\": 130.0, ...}}"
            try:
                from openai import OpenAI
                client = OpenAI(api_key=ai_key)
                resp = client.chat.completions.create(
                    model=st.session_state.get("ai_model","gpt-4o-mini"),
                    messages=[{"role":"system","content":sys},{"role":"user","content":user}],
                    temperature=0.4,
                )
                txt = resp.choices[0].message.content.strip()
                if txt.startswith("```"):
                    txt = txt.strip("`")
                    txt = txt.split("\n",1)[-1]
                    if txt.lower().startswith("json"):
                        txt = txt.split("\n",1)[-1]
                    if txt.endswith("```"):
                        txt = txt[:-3]
                try:
                    js = json.loads(txt)
                    for k in NUTRIENTS:
                        if k in js and isinstance(js[k], (int,float)):
                            st.session_state.limits[k] = round(float(js[k]),1)
                    st.session_state.limits["enabled"] = True
                    save_limits(st.session_state.limits)
                    st.success("AI推定の上限を反映しました（保存済み）")
                except Exception:
                    st.warning("AI出力のJSON解析に失敗しました。内容を表示します：")
                    st.code(txt, language="json")
            except ModuleNotFoundError:
                st.error("`openai` パッケージが見つかりません。`pip install openai` を実行してください。")
            except Exception as e:
                st.error(f"OpenAI呼び出しエラー: {e}")

    # 入力変更を保存
    save_limits(st.session_state.limits)

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
        show_df[display_cols],
        num_rows="dynamic",
        use_container_width=True,
        key=f"editor_{selected_date}",
        hide_index=True,
    )
    to_delete = edited[edited["削除"] == True]["_idx"].tolist()
    if to_delete:
        st.session_state.log = st.session_state.log.drop(index=to_delete).reset_index(drop=True)
        save_log(st.session_state.log)
        st.warning(f"{len(to_delete)} 件を削除しました（保存済み）")

    totals = edited[NUTRIENTS].sum().round(1)

    colA, colB = st.columns([1,1])
    with colA:
        st.markdown("### 🔢 栄養合計（当日）")
        st.table(totals.to_frame(name="合計"))
    with colB:
        st.markdown("### ⏳ 上限までの残り（不足分）")
        remaining_for_ui = {}
        over_list = []
        if st.session_state.limits.get("enabled", False):
            rem = {}
            for n in NUTRIENTS:
                limit = float(st.session_state.limits.get(n, 0) or 0)
                val = float(totals.get(n, 0) or 0)
                if limit > 0:
                    diff = round(limit - val, 1)
                    rem[n] = diff if diff > 0 else 0.0
                    if diff < 0:
                        over_list.append((n, round(-diff,1)))
            remaining_for_ui = rem
            st.table(pd.Series(rem).to_frame("残り").round(1))
            if over_list:
                msg = "\n".join([f"- {k}: 上限超過 {v:.1f}" for k, v in over_list])
                st.error("⚠️ 上限を超えています\n" + msg)
        else:
            st.info("上限チェックを有効化すると不足分が表示されます（『📏 日次上限設定』を展開）")

        # ======= （変更2）既存食品からのAIアドバイス =======
        if st.button(" 🤖 既存食品から埋め合わせ提案（AI）", use_container_width=True, key="btn_ai_suggest_existing_foods"):
            if not st.session_state.limits.get("enabled", False):
                st.error("上限チェックが無効です。『📏 日次上限設定』で有効化してください。")
            else:
                env_key = os.environ.get("OPENAI_API_KEY", "")
                secret_key = None
                try:
                    secret_key = st.secrets.get("OPENAI_API_KEY")
                except Exception:
                    pass
                ai_key = (st.session_state.get("ai_api_key") or secret_key or env_key or None)
                if not ai_key:
                    st.error("OpenAI API Key をサイドバーで設定してください。")
                else:
                    # 食品DBをJSON化（大きすぎる場合は上位N件に絞る）
                    db = _ensure_food_df_columns(st.session_state.food_db.copy())
                    db = db.dropna(subset=["food"]).drop_duplicates(subset=["food"])
                    # 軽量化：最大200品まで
                    db_small = db.head(200)[["food","per","unit",*NUTRIENTS]].round(1)
                    foods_json = db_small.to_dict(orient="records")
                    # プロンプト
                    sys_sugg = (
                        "あなたは日本語の管理栄養士です。"
                        "与えられた『不足分（上限までの残り）』を、提供された食品リスト（既存DB）だけから埋める提案をしてください。"
                        "各食品は per を1食とする。"
                        "『不足分（上限までの残り）』を超えないでください。"
                        "出力はJSONのみ。以下の形式で返してください。"
                        " {\"suggestions\": [{\"food\": \"食品名\", \"servings\": 1.0, \"note\": \"栄養成分を表示\"}, ...], "
                        "\"expected_total\": {\"kcal\": ..., \"protein\": ..., \"fat\": ..., \"carbs\": ..., \"fiber\": ..., \"sugar\": ..., \"sodium_mg\": ...} }"
                    )
                    user_sugg = {
                        "date": str(selected_date),
                        "remaining_targets": remaining_for_ui,     # 不足（0なら無理に満たさない）
                        "food_db": foods_json
                    }
                    try:
                        from openai import OpenAI
                        client = OpenAI(api_key=ai_key)
                        resp = client.chat.completions.create(
                            model=st.session_state.get("ai_model","gpt-4o-mini"),
                            messages=[
                                {"role":"system","content":sys_sugg},
                                {"role":"user","content":json.dumps(user_sugg, ensure_ascii=False)}
                            ],
                            temperature=0.4,
                        )
                        txt = resp.choices[0].message.content.strip()
                        if txt.startswith("```"):
                            txt = txt.strip("`")
                            txt = txt.split("\n",1)[-1]
                            if txt.lower().startswith("json"):
                                txt = txt.split("\n",1)[-1]
                            if txt.endswith("```"):
                                txt = txt[:-3]
                        try:
                            js = json.loads(txt)
                            sugg = js.get("suggestions", [])
                            exp_total = js.get("expected_total", {})
                            if not isinstance(sugg, list):
                                sugg = []
                            # 表示
                            if sugg:
                                df_s = pd.DataFrame(sugg)
                                if "servings" in df_s.columns:
                                    df_s["servings"] = pd.to_numeric(df_s["servings"], errors="coerce").fillna(1.0).round(2)
                                st.success("提案が生成されました（既存食品のみ）")
                                st.table(df_s)
                            else:
                                st.info("提案結果が空でした。食品DBを増やすか、上限値を調整して再実行してください。")
                            #if isinstance(exp_total, dict) and exp_total:
                            #    st.caption("想定合計摂取（この提案を採用した場合）")
                            #    st.table(pd.Series(exp_total).round(1).to_frame("想定量"))
                        except Exception:
                            st.warning("AI出力のJSON解析に失敗しました。内容を表示します：")
                            st.code(txt, language="json")
                    except ModuleNotFoundError:
                        st.error("`openai` パッケージが見つかりません。`pip install openai` を実行してください。")
                    except Exception as e:
                        st.error(f"OpenAI呼び出しエラー: {e}")

    csv_day = edited.drop(columns=["_idx", "削除"]).round(1).to_csv(index=False).encode(CSV_ENCODING)

# ============================
# 直近の集計と可視化（すべて日単位）
# ============================
st.markdown("---")
st.subheader("📈 直近期間の集計と可視化")

log2 = st.session_state.log.dropna(subset=["date"]).copy()

def style_exceed(df: pd.DataFrame, limits: dict):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    if not limits.get("enabled", False):
        return styles
    for col in df.columns:
        if col in NUTRIENTS:
            lim = float(limits.get(col, 0) or 0)
            if lim > 0:
                mask = df[col] > lim
                styles.loc[mask, col] = "color: red; font-weight: 700;"
    return styles

def daily_meal_presence(rdf: pd.DataFrame) -> pd.DataFrame:
    if rdf.empty:
        return pd.DataFrame(columns=MEAL_TYPES)
    rdf["date_only"] = rdf["date"].dt.date
    pres = (
        rdf.groupby(["date_only", "meal"])
           .size()
           .unstack(fill_value=0)
           .reindex(columns=MEAL_TYPES, fill_value=0)
    )
    pres = (pres > 0).astype(int)
    pres.index.name = None
    pres = pres.rename_axis(None, axis=1)
    return pres

if log2.empty:
    st.info("データがありません")
else:
    def render_window(window):
        if window == "all":
            last_day = log2["date"].dt.date.max()
            start_day = log2["date"].dt.date.min()
        else:
            last_day = max(log2["date"].dt.date.max(), date.today())
            start_day = last_day - timedelta(days=int(window)-1)

        rmask = (log2["date"].dt.date >= start_day) & (log2["date"].dt.date <= last_day)
        rdf = log2.loc[rmask].copy()

        daily_raw = rdf.groupby(rdf["date"].dt.date)[NUTRIENTS].sum()
        daily = daily_raw.round(0).astype("Int64").sort_index()

        presence = daily_meal_presence(rdf)
        daily = daily.join(presence, how="left").fillna(0)
        for m in MEAL_TYPES:
            if m in daily.columns:
                daily[m] = daily[m].astype(int)

        table_df = daily.reset_index().rename(columns={"index": "日付"})
        table_df = table_df.rename(columns={"date": "日付"})
        if "日付" not in table_df.columns:
            table_df = table_df.rename(columns={table_df.columns[0]: "日付"})

        styled = table_df.style.apply(style_exceed, limits=st.session_state.limits, axis=None)
        st.caption(f"対象期間: {start_day} 〜 {last_day}。当日の上限を超えた数値は赤字で表示")
        st.dataframe(styled, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### kcal 推移")
            if "kcal" in daily.columns and not daily.empty:
                kdf = daily[["kcal"]].copy()
                kdf["日付"] = kdf.index
                kdf = kdf.reset_index(drop=True)
                fig_kcal = px.line(kdf, x="日付", y="kcal", markers=True)
                fig_kcal.update_layout(margin=dict(l=0,r=0,t=10,b=0), yaxis_title="kcal", xaxis_title="")
                st.plotly_chart(fig_kcal, use_container_width=True, key=f"fig_kcal_{window}")
            else:
                st.caption("kcal データなし")
        with c2:
            st.markdown("#### 体重推移")
            w = st.session_state.weight.copy()
            if not w.empty:
                w["date"] = pd.to_datetime(w["date"], errors="coerce")
                wv = w[(w["date"].dt.date >= start_day) & (w["date"].dt.date <= last_day)].copy().sort_values("date")
                if not wv.empty and "weight_kg" in wv.columns:
                    wt = wv.set_index(wv["date"].dt.date)[["weight_kg"]]
                    wdf_plot = wt.copy()
                    wdf_plot["日付"] = wdf_plot.index
                    wdf_plot = wdf_plot.reset_index(drop=True)
                    fig_w = px.line(wdf_plot, x="日付", y="weight_kg", markers=True)
                    fig_w.update_layout(margin=dict(l=0,r=0,t=10,b=0), yaxis_title="kg", xaxis_title="")
                    st.plotly_chart(fig_w, use_container_width=True, key=f"fig_weight_{window}")
                else:
                    st.caption("この期間の体重データがありません")
            else:
                st.caption("体重データがありません")

        st.markdown("#### たんぱく質 / 脂質 / 炭水化物 推移")
        pfc_cols = [col for col in ["protein","fat","carbs"] if col in daily.columns]
        if pfc_cols:
            pfc_df = daily[pfc_cols].copy()
            pfc_df["日付"] = pfc_df.index
            pfc_melt = pfc_df.reset_index(drop=True).melt(id_vars="日付", value_vars=pfc_cols, var_name="栄養素", value_name="量")
            fig_pfc = px.line(pfc_melt, x="日付", y="量", color="栄養素", markers=True)
            fig_pfc.update_layout(margin=dict(l=0,r=0,t=10,b=0), yaxis_title="g", xaxis_title="")
            st.plotly_chart(fig_pfc, use_container_width=True, key=f"fig_pfc_{window}")
        else:
            st.caption("P/F/C データなし")

        col_rank, col_meal_avg = st.columns(2)
        with col_rank:
            st.markdown("#### よく食べる食品ランキング")
            foods_df = rdf.dropna(subset=["food"]).copy()
            if foods_df.empty:
                st.caption("対象期間の食事データがありません")
            else:
                label_map = {
                    "kcal": "カロリー(kcal)",
                    "protein": "たんぱく質(g)",
                    "fat": "脂質(g)",
                    "carbs": "炭水化物(g)",
                    "sodium_mg": "ナトリウム(mg)",
                }
                target_nutrients = [c for c in ["kcal", "protein", "fat", "carbs", "sodium_mg"] if c in foods_df.columns]
                counts = foods_df.groupby("food").size().rename("回数")
                frames = [counts]
                if target_nutrients:
                    means = foods_df.groupby("food")[target_nutrients].mean()
                    means = means.rename(columns={c: label_map.get(c, c) for c in target_nutrients})
                    frames.append(means)

                stats = pd.concat(frames, axis=1)
                if not stats.empty:
                    for col in stats.columns:
                        if col != "回数":
                            stats[col] = stats[col].fillna(0.0)
                    stats["回数"] = stats["回数"].fillna(0).astype(int)
                    sort_cols = ["回数"]
                    kcal_avg_col = "カロリー(kcal)"
                    if kcal_avg_col in stats.columns:
                        sort_cols.append(kcal_avg_col)
                    stats = stats.sort_values(sort_cols, ascending=[False] * len(sort_cols))
                    stats = stats.head(4)
                    numeric_cols = [c for c in stats.columns if c not in ["回数"]]
                    for col in numeric_cols:
                        stats[col] = stats[col].astype(float).round(1)
                    stats = stats.reset_index().rename(columns={"food": "食品"})
                    st.dataframe(stats, use_container_width=True, hide_index=True)
                    st.caption("※対象期間において利用回数が多い順に最大4件表示")
                else:
                    st.caption("対象期間の食事データがありません")

        with col_meal_avg:
            st.markdown("#### 食事区分ごとの平均値")
            meal_df = rdf.dropna(subset=["meal"]).copy()
            if meal_df.empty:
                st.caption("対象期間の食事データがありません")
            else:
                meal_df["date"] = pd.to_datetime(meal_df["date"], errors="coerce")
                meal_df = meal_df.dropna(subset=["date"])
                if meal_df.empty:
                    st.caption("対象期間の食事データがありません")
                else:
                    meal_df["day"] = meal_df["date"].dt.date
                    daily_counts = meal_df.groupby(["day", "meal"]).size().unstack(fill_value=0)
                    daily_kcal = None
                    if "kcal" in meal_df.columns:
                        daily_kcal = meal_df.groupby(["day", "meal"])["kcal"].sum().unstack(fill_value=0)
                    if daily_counts.empty:
                        st.caption("対象期間の食事データがありません")
                    else:
                        avg_counts = daily_counts.mean(axis=0).rename("1日平均件数")
                        avg_df = avg_counts.to_frame()
                        if daily_kcal is not None and not daily_kcal.empty:
                            avg_kcal = daily_kcal.mean(axis=0).rename("1日平均カロリー(kcal)")
                            avg_df = avg_df.join(avg_kcal, how="left")
                        avg_df = avg_df.reset_index().rename(columns={"meal": "食事区分"})
                        avg_df["1日平均件数"] = avg_df["1日平均件数"].round(2)
                        if "1日平均カロリー(kcal)" in avg_df.columns:
                            avg_df["1日平均カロリー(kcal)"] = avg_df["1日平均カロリー(kcal)"].round(1)
                        avg_df = avg_df.sort_values("1日平均件数", ascending=False)
                        st.dataframe(avg_df, use_container_width=True, hide_index=True)
                        st.caption("※対象期間における1日あたりの平均件数と平均カロリー（食事区分別）")

        return rdf, start_day, last_day, daily

    windows = ["全期間", 5, 10, 15, 20, 30, 60, 90]
    tabs = st.tabs([str(w) + ("日" if isinstance(w, int) else "") for w in windows])
    for t, window in zip(tabs, windows):
        with t:
            win_key = "all" if window == "全期間" else int(window)
            render_window(win_key)

# ============================
# 🤖 AI ダイエットアドバイス（OpenAI API）
# ============================
st.markdown("---")
st.subheader("🤖ダイエットアドバイス（AI）")

ai_key = st.session_state.get('ai_api_key')
ai_model = st.session_state.get('ai_model', 'gpt-4o-mini')
ai_window_sel = st.session_state.get('ai_window', 10)
ai_include_foods = bool(st.session_state.get('ai_include_foods', True))
ai_debug = bool(st.session_state.get('ai_debug', False))
profile = st.session_state.profile

col_ai1, col_ai2 = st.columns([1,1])
with col_ai1:
    run_ai = st.button("AIで要約とアドバイスを生成")
with col_ai2:
    simple_mode = st.checkbox("短めに要約（要点のみ）", value=True)

if run_ai:
    if not ai_key:
        st.error("OpenAI API Key を入力してください（サイドバー）")
    else:
        base = st.session_state.log.dropna(subset=["date"]).copy()
        if base.empty:
            st.info("食事データがありません")
        else:
            if ai_window_sel == "全期間":
                start_day = base["date"].dt.date.min()
                last_day = base["date"].dt.date.max()
            else:
                last_day = max(base["date"].dt.date.max(), date.today())
                start_day = last_day - timedelta(days=int(ai_window_sel)-1)

            rmask = (base["date"].dt.date >= start_day) & (base["date"].dt.date <= last_day)
            rdf = base.loc[rmask].copy()
            daily = rdf.groupby(rdf["date"].dt.date)[NUTRIENTS].sum().round(1).sort_index()

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

            p = profile
            p_sex = p.get("sex","男性")
            p_age = int(p.get("age", 28))
            p_h = float(p.get("height_cm", 173.0))
            p_w = float(p.get("current_weight_kg", 73.0))
            p_bmi = calc_bmi(p_h, p_w)
            p_std = std_weight(p_h)
            p_act = p.get("activity","ふつう(週1-3運動)")

            df_for_prompt = daily.reset_index().rename(columns={"date":"日付"})
            df_for_prompt["日付"] = df_for_prompt["日付"].astype(str)
            weight_dict = {str(k): float(v) for k, v in weight_series.to_dict().items()}

            system_msg = (
                "あなたは管理栄養士の視点をもつ日本語のダイエットコーチです。"
                "安全で現実的・実行可能な提案を行い、極端な減量や医学的判断は避けます。"
                "具体的な食事のアドバイスを提供し、必要に応じて上限(過剰)と不足の両面に触れてください。"
                "アドバイスは食事ログ・体重推移・上限設定・プロフィールを踏まえてください。"
            )
            style = "簡潔に3〜5個の箇条書き" if simple_mode else "見出し付きで要約→提案の順に詳しく"

            prof_block = {
                "性別": p_sex, "年齢": p_age, "身長_cm": p_h, "現在体重_kg": p_w,
                "BMI": p_bmi, "標準体重_kg(BMI22)": p_std, "活動レベル": p_act
            }

            base_block = f"""
【プロフィール】{json.dumps(prof_block, ensure_ascii=False)}
【対象期間】{start_day}〜{last_day}（{(last_day-start_day).days+1}日）
【日別合計（kcal/たんぱく質/脂質/炭水化物/食物繊維/糖質/ナトリウム）】
{df_for_prompt.to_json(orient='records', force_ascii=False)}
【体重(kg) 推移】{json.dumps(weight_dict, ensure_ascii=False)}
【上限設定】{json.dumps({k: float(limits.get(k, 0) or 0) for k in NUTRIENTS}, ensure_ascii=False)}
【今日の不足分（上限到達までの残り, 無い場合は0）】{json.dumps(remaining, ensure_ascii=False)}
"""
            if bool(st.session_state.get('ai_include_foods', True)) and not rdf.empty:
                freq = rdf["food"].value_counts().head(30).reset_index()
                freq.columns = ["food", "count"]
                food_sum = rdf.groupby("food")[NUTRIENTS].sum().round(1)
                def top_by(col, n=12):
                    if col not in food_sum.columns:
                        return []
                    return (
                        food_sum[col]
                        .sort_values(ascending=False)
                        .head(n)
                        .reset_index()
                        .rename(columns={col: f"total_{col}"})
                        .to_dict(orient="records")
                    )
                top_dict = {col: top_by(col) for col in NUTRIENTS}
                recent = rdf.sort_values("date").tail(80)[["date", "meal", "food"]].copy()
                recent["date"] = pd.to_datetime(recent["date"]).dt.strftime("%Y-%m-%d")
                recent_records = recent.to_dict(orient="records")
                food_detail = {
                    "食品頻度TOP": freq.to_dict(orient="records"),
                    "栄養素別上位食品": top_dict,
                    "最近の食事明細": recent_records,
                }
                base_block += f"\n【食品名に基づく参考情報（頻度/上位/直近明細）】{json.dumps(food_detail, ensure_ascii=False)}\n"

            user_msg = (
                base_block +
                f"まず{style}で『期間の傾向』を要約し、"
                "次に『改善アクション』を日本語で提案してください。"
                "必ず『体重に関する助言』を含めてください。"
                "最後に注意事項を1行添えてください。"
            )

            if bool(st.session_state.get('ai_debug', False)):
                with st.expander("🛠 デバッグ：送信プロンプト（system / user）", expanded=False):
                    st.code(system_msg, language="markdown")
                    st.code(user_msg, language="markdown")

            try:
                from openai import OpenAI
                client = OpenAI(api_key=ai_key)
                resp = client.chat.completions.create(
                    model=ai_model,
                    messages=[{"role": "system", "content": system_msg},
                              {"role": "user", "content": user_msg}],
                    temperature=0.6,
                )
                advice = resp.choices[0].message.content
                st.success("AIアドバイスを生成しました")
                st.markdown(advice)

                new_adv = pd.DataFrame([{
                    "created_at": pd.Timestamp.now(tz="Asia/Tokyo"),
                    "model": ai_model,
                    "window": "all" if ai_window_sel == "全期間" else int(ai_window_sel),
                    "include_foods": bool(st.session_state.get('ai_include_foods', True)),
                    "simple_mode": bool(simple_mode),
                    "start_day": pd.to_datetime(start_day),
                    "last_day": pd.to_datetime(last_day),
                    "ai_advice": advice,
                }])
                st.session_state.advice = pd.concat([st.session_state.advice, new_adv], ignore_index=True)
                save_advice_log(st.session_state.advice)
                st.success("AIアドバイスを保存しました（advice_log.csv）")
            except ModuleNotFoundError:
                st.error("`openai` パッケージが見つかりません。`pip install openai` を実行してください。")
            except Exception as e:
                st.error(f"OpenAI API呼び出しでエラーが発生しました: {e}")

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
    created_s = pd.to_datetime(latest["created_at"]).strftime("%Y-%m-%d %H:%M") if pd.notnull(latest.get("created_at")) else ""
    model_s = str(latest.get("model", ""))
    window_s = latest.get("window", 0)
    window_disp = "全期間" if str(window_s) == "all" else f"{int(window_s)}日"
    period_s = ""
    if pd.notnull(latest.get("start_day")) and pd.notnull(latest.get("last_day")):
        sd = pd.to_datetime(latest["start_day"], errors="coerce")
        ld = pd.to_datetime(latest["last_day"], errors="coerce")
        if pd.notnull(sd) and pd.notnull(ld):
            period_s = f"{sd.date()} 〜 {ld.date()}"
    st.caption(f"生成日: {created_s} / モデル: {model_s} / 期間: {window_disp}" + (f"（{period_s}）" if period_s else ""))
    st.info(str(latest.get("ai_advice", "")))
