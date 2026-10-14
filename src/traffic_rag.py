import re
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

from src.config import (
    MODEL_FILE,
    MODELS_DIR,
    CONGESTION_LABELS,
)
from src.vector_store import TrafficRetriever, translate_query

FEATURE_STATS_FILE = MODELS_DIR / "segment_feature_stats.joblib"

# Arabic Day of week mapping
ARABIC_DAYS = {
    "الاثنين": 0, "الإثنين": 0, "monday": 0,
    "الثلاثاء": 1, "tuesday": 1,
    "الاربعاء": 2, "الأربعاء": 2, "wednesday": 2,
    "الخميس": 3, "thursday": 3,
    "الجمعة": 4, "friday": 4,
    "السبت": 5, "saturday": 5,
    "الاحد": 6, "الأحد": 6, "sunday": 6,
}

class TrafficPredictor:
    def __init__(self):
        print(f"[*] Loading trained XGBoost models from: {MODEL_FILE}")
        self.bundle = joblib.load(MODEL_FILE)
        self.clf_risk = self.bundle.get("clf_risk", self.bundle.get("classifier"))
        self.clf_multi = self.bundle.get("clf_multi", self.bundle.get("classifier"))
        self.reg = self.bundle["regressor"]
        self.feature_cols = self.bundle["feature_cols"]
        self.labels = self.bundle["labels"]
        
        self.stats_dict = joblib.load(FEATURE_STATS_FILE) if FEATURE_STATS_FILE.exists() else None
        
    def predict(self, segment_info: Dict[str, Any], yr: int = 2026, m: int = 10, d: int = 13, hh: int = 17, mm: int = 0) -> Dict[str, Any]:
        """
        Runs upgraded XGBoost prediction with 89.5% risk accuracy and R2=0.80 volume.
        """
        dt = datetime(yr, m, d, hh, mm)
        day_of_week = dt.weekday()
        is_weekend = int(day_of_week >= 5)
        is_rush_hour = int(hh in [7, 8, 9, 16, 17, 18, 19])
        
        hour_sin = np.sin(2 * np.pi * hh / 24.0)
        hour_cos = np.cos(2 * np.pi * hh / 24.0)
        month_sin = np.sin(2 * np.pi * m / 12.0)
        month_cos = np.cos(2 * np.pi * m / 12.0)
        
        seg_id = int(segment_info.get("segment_id", 0))
        boro_str = str(segment_info.get("boro", "Manhattan"))
        
        # Historical baseline statistics lookup
        global_mean = self.stats_dict["global_mean"] if self.stats_dict else 85.0
        seg_mean = global_mean
        seg_std = 20.0
        seg_max = global_mean * 2.0
        seg_hour_mean = global_mean
        seg_day_mean = global_mean
        boro_hour_mean = global_mean
        
        if self.stats_dict:
            seg_info_stat = self.stats_dict["seg_stats"].get(seg_id)
            if seg_info_stat:
                seg_mean = seg_info_stat.get("seg_mean", seg_mean)
                seg_std = seg_info_stat.get("seg_std", seg_std)
                seg_max = seg_info_stat.get("seg_max", seg_max)
            seg_hour_mean = self.stats_dict["seg_hour_stats"].get((seg_id, hh), seg_mean)
            seg_day_mean = self.stats_dict["seg_day_stats"].get((seg_id, day_of_week), seg_mean)
            boro_hour_mean = self.stats_dict["boro_hour_stats"].get((boro_str, hh), seg_mean)
            
        row = {
            "Yr": yr,
            "M": m,
            "D": d,
            "HH": hh,
            "MM": mm,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "is_rush_hour": is_rush_hour,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "month_sin": month_sin,
            "month_cos": month_cos,
            "seg_mean": seg_mean,
            "seg_hour_mean": seg_hour_mean,
            "seg_day_mean": seg_day_mean,
            "seg_std": seg_std,
            "seg_max": seg_max,
            "boro_hour_mean": boro_hour_mean,
        }
        X_df = pd.DataFrame([row])[self.feature_cols]
        
        # 1. Congestion Classification (73.4% accuracy)
        pred_level = int(self.clf_multi.predict(X_df)[0])
        
        # 2. Severe Congestion / Incident Bottleneck Risk (89.5% accuracy)
        risk_probas = self.clf_risk.predict_proba(X_df)[0]
        severe_risk_prob = float(risk_probas[1]) if len(risk_probas) > 1 else float(risk_probas[0])
        is_severe_risk = int(self.clf_risk.predict(X_df)[0])
        
        # 3. Volume Regression (R2 = 0.80)
        pred_vol = max(0, int(round(float(self.reg.predict(X_df)[0]))))
        
        # Severity evaluation
        if is_severe_risk == 1 or severe_risk_prob > 0.45 or pred_level == 3:
            risk_tag = "HIGH (خطر تكدس واختناق مرتفع)"
            alert_icon = "🔴"
        elif pred_level == 2 or severe_risk_prob > 0.25:
            risk_tag = "MEDIUM (كثافة مرورية ملحوظة)"
            alert_icon = "🟡"
        else:
            risk_tag = "LOW (انسيابي / طبيعي)"
            alert_icon = "🟢"
            
        return {
            "predicted_volume": pred_vol,
            "congestion_level": pred_level,
            "congestion_label": CONGESTION_LABELS[pred_level],
            "severe_risk_prob": round(severe_risk_prob * 100, 1),
            "is_severe_risk": bool(is_severe_risk),
            "risk_tag": risk_tag,
            "alert_icon": alert_icon,
            "is_rush_hour": bool(is_rush_hour),
            "hour": hh,
            "day_of_week": day_of_week,
        }


def classify_intent(query: str) -> str:
    """Classifies user query intent: GREETING, HELP, THANKS, OFF_TOPIC, or TRAFFIC_QUERY."""
    norm = query.strip().lower()
    
    # 1. Greetings & Chitchat
    greeting_patterns = [
        r"^(?:مرحبا|مرحباً|أهلا|اهلا|السلام عليكم|صباح الخير|مساء الخير|كيفك|كيف حالك|عامل ايه|ازيك|شخبارك|اخبارك)",
        r"^(?:hello|hi|hey|good morning|good evening|how are you|how do you do|sup)\b"
    ]
    for p in greeting_patterns:
        if re.search(p, norm):
            # If query is short or just a greeting without a street query
            if len(norm.split()) <= 4:
                return "GREETING"
                
    # 2. Identity / Capabilities / Help
    identity_patterns = [
        r"(?:من انت|من أنت|مين انت|مين حضرتك|عرف نفسك|who are you|what are you)",
        r"(?:تقدر تعمل ايه|ماذا تفعل|كيف تساعدني|ما هي وظيفتك|ما وظيفتك|what can you do|help|مساعدة)"
    ]
    for p in identity_patterns:
        if re.search(p, norm):
            return "HELP"
            
    # 3. Gratitude
    if re.search(r"^(?:شكرا|شكراً|تسلم|يعطيك العافية|مشكور|thanks|thank you)\b", norm):
        return "THANKS"
        
    # 4. Check for Traffic Indicators or Boroughs / Streets
    boros = ["manhattan", "brooklyn", "queens", "bronx", "staten island", "مانهاتن", "بروكلين", "كوينز", "برونكس", "ستاتن"]
    traffic_keywords = [
        "شارع", "طريق", "جسر", "كوبري", "محور", "زحمة", "ازدحام", "مرور", "حوادث", "سير", "تكدس", "اختناق",
        "street", "st", "avenue", "ave", "boulevard", "blvd", "bridge", "br", "expressway", "pkwy", "parkway",
        "traffic", "jam", "congestion", "accident", "bottleneck"
    ]
    
    has_traffic_intent = any(k in norm for k in traffic_keywords) or any(b in norm for b in boros)
    if has_traffic_intent:
        return "TRAFFIC_QUERY"
        
    # Check if query contains numbers or known street landmarks
    if re.search(r"\b\d+(?:st|nd|rd|th)?\b", norm) or any(w in norm for w in ["flatbush", "broadway", "washington", "fulton", "atlantic", "hunts", "richmond"]):
        return "TRAFFIC_QUERY"
        
    return "OFF_TOPIC"


class TrafficRAG:
    """
    Retrieval-Augmented Generation (RAG) System:
    Retrieves real street segments and history, feeds them into the upgraded XGBoost model (89.5% accuracy),
    and synthesizes actionable AI traffic insights.
    """
    def __init__(self):
        self.retriever = TrafficRetriever()
        self.predictor = TrafficPredictor()
        
    def extract_time_from_query(self, query: str):
        """Extracts specified hour or day of week from query text."""
        norm = query.lower()
        extracted_hour = 17 # default 5 PM rush hour
        extracted_day = None
        
        # Search for hour (e.g., "الساعة 6", "6 pm", "18:00")
        hour_match = re.search(r"(?:الساعة\s*|at\s*)(\d{1,2})", norm)
        if hour_match:
            h = int(hour_match.group(1))
            if "مساء" in norm or "pm" in norm:
                extracted_hour = h if h >= 12 else h + 12
            elif "صباح" in norm or "am" in norm:
                extracted_hour = h if h < 12 else 0
            else:
                extracted_hour = h
                
        # Search for day of week
        for d_name, d_code in ARABIC_DAYS.items():
            if d_name in norm:
                extracted_day = d_code
                break
                
        return extracted_hour, extracted_day

    def query(self, user_query: str, boro: str = None, top_k: int = 2) -> Dict[str, Any]:
        """
        Executes full RAG workflow with Intent Routing:
        1. Classify Query Intent (Greeting, Help, Thanks, Off-topic, or Traffic)
        2. If Traffic: Retrieve segments -> Predict XGBoost -> Synthesize Report
        """
        intent = classify_intent(user_query)
        
        # Handle Conversational Intents directly
        if intent == "GREETING":
            return {
                "query": user_query,
                "intent": intent,
                "segments": [],
                "answer": "أهلاً بك! أنا بخير وفي أتم الجاهزية. 🚦\n\nأنا **مساعدك الذكي لحركة المرور في مدينة نيويورك**، ويمكنني التنبؤ بمستوى الازدحام ومخاطر الحوادث وتدفق المركبات بدقة تصل إلى **89.5%**.\n\n💡 **جرّب سؤالي مثلاً:**\n* *\"إيه حالة المرور في شارع Flatbush Avenue الساعة 6 مساءً؟\"*\n* *\"هل فيه زحمة أو خطر حوادث عند Washington Bridge الصبح الساعة 8؟\"*\n* *\"ما هي حالة السير في شارع برودواي؟\"*"
            }
            
        elif intent == "HELP":
            return {
                "query": user_query,
                "intent": intent,
                "segments": [],
                "answer": "### 🛠️ كيف يمكنني مساعدتك؟\nأنا نظام ذكاء اصطناعي (RAG) مدعوم بنموذج XGBoost متقدم للبيانات الضخمة في نيويورك. أستطيع تقديم:\n1. **التنبؤ بحجم السيارات (Traffic Volume):** بدقة $R^2 = 80\\%$ وبخطأ منخفض.\n2. **رصد مناطق الخطر والاختناق (Severe Incident Risk):** بدقة **89.5%**.\n3. **توصيات القيادة:** اقتراح مسارات بديلة في أوقات الذروة.\n\nفقط اكتب اسم الشارع والوقت الذي ترغب بالاستفسار عنه!"
            }
            
        elif intent == "THANKS":
            return {
                "query": user_query,
                "intent": intent,
                "segments": [],
                "answer": "على الرحب والسعة دائماً! أتمنى لك رحلة آمنة وطريقاً انسيابياً خ الياً من الزحام. 🚗✨"
            }
            
        elif intent == "OFF_TOPIC":
            return {
                "query": user_query,
                "intent": intent,
                "segments": [],
                "answer": "أنا مساعد متخصص حصرياً في **حركة المرور وتوقع الاختناقات ومناطق الحوادث لمدينة نيويورك**. 🏙️\n\nيرجى كتابة اسم شارع أو محور مروري محدد في نيويورك (مثل *Broadway* أو *Flatbush Ave* أو *5th Ave*) مع التوقيت، وسأقوم بتحليله لك فوراً!"
            }
            
        # Intent is TRAFFIC_QUERY: execute RAG pipeline
        hour, day_code = self.extract_time_from_query(user_query)
        
        # 1. Retrieve segments
        retrieved_segments = self.retriever.search(user_query, boro=boro, top_k=top_k)
        
        if not retrieved_segments:
            return {
                "query": user_query,
                "intent": intent,
                "segments": [],
                "answer": f"عذراً، لم أتمكن من العثور على شارع يطابق استفسارك: \"{user_query}\" في سجلات مدينة نيويورك.\n\nيرجى التأكد من اسم الشارع وكتابته بالإنجليزية أو العربية (مثل *Flatbush Avenue*, *Washington Bridge*, *Broadway*, *Hunts Point Avenue*)."
            }
            
        # 2. Predict for each segment
        results = []
        for seg in retrieved_segments:
            pred = self.predictor.predict(seg, hh=hour)
            results.append({
                "metadata": seg,
                "prediction": pred
            })
            
        # 3. Build Augmented Context and AI Response
        answer_text = self._generate_response(user_query, results, hour)
        
        return {
            "query": user_query,
            "intent": intent,
            "target_hour": f"{hour:02d}:00",
            "segments": results,
            "answer": answer_text
        }

    def _generate_response(self, user_query: str, results: List[Dict[str, Any]], hour: int) -> str:
        """
        Synthesizes structured, data-grounded AI traffic report in Arabic.
        """
        lines = []
        lines.append(f"### 🚦 تقرير المساعد الذكي لحركة المرور (RAG Traffic Analysis)")
        lines.append(f"**الاستفسار:** \"{user_query}\"")
        lines.append(f"**التوقيت المحلل:** الساعة `{hour:02d}:00`\n")
        lines.append("---")
        
        for idx, res in enumerate(results, 1):
            meta = res["metadata"]
            pred = res["prediction"]
            
            street = meta.get("street", "Unknown")
            from_st = meta.get("from_st", "")
            to_st = meta.get("to_st", "")
            boro = meta.get("boro", "")
            direction = meta.get("direction", "")
            avg_vol = meta.get("avg_vol", 0)
            peak_h = meta.get("peak_hour", 17)
            
            pred_vol = pred["predicted_volume"]
            status = pred["congestion_label"]
            risk_rate = pred["severe_risk_prob"]
            icon = pred["alert_icon"]
            risk_tag = pred["risk_tag"]
            
            lines.append(f"#### {idx}. شارع **{street}** ({boro} - الاتجاه: {direction})")
            lines.append(f"* **المقطع:** من `{from_st}` إلى `{to_st}`")
            lines.append(f"* **حالة السير المتوقعة:** {icon} **{status}**")
            lines.append(f"* **حجم التدفق المتوقع (Volume):** `{pred_vol}` مركبة / 15 دقيقة (المعدل التاريخي: `{avg_vol}`)")
            lines.append(f"* **مؤشر خطر الاختناق والحوادث (دقة 89.5%):** **{risk_rate}%** ({risk_tag})")
            lines.append(f"* **ساعة الذروة التاريخية للشارع:** `{peak_h:02d}:00`")
            
            # Recommendation
            if pred["is_severe_risk"] or risk_rate > 35:
                lines.append(f"> ⚠️ **توصية ذكية:** يُتوقع تكدس مروري ملحوظ في هذا التوقيت. يُفضل استخدام مسار بديل أو تجنب السفر في هذا النطاق إن أمكن.")
            else:
                lines.append(f"> ✅ **توصية ذكية:** حركة السير متوقعة بشكل انسيابي وطبيعي، ولا توجد مؤشرات على اختناقات مرورية حادة.")
            lines.append("")
            
        return "\n".join(lines)

if __name__ == "__main__":
    rag = TrafficRAG()
    test_q = "إيه حالة المرور في شارع Flatbush Avenue الساعة 6 مساءً؟"
    res = rag.query(test_q)
    print(res["answer"])
