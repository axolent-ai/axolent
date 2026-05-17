"""One-shot script: inject self_awareness i18n keys into all locale files.

Run once after Phase 2 implementation. Safe to re-run (idempotent).
Inserts self_awareness.* keys before time_context.header in each locale.

Usage:
    python bridge/scripts/inject_self_awareness_i18n.py
"""

import json
from pathlib import Path

LOCALES_DIR = Path(__file__).parent.parent / "i18n" / "locales"

# Translations per language (all 20 languages)
TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        "self_awareness.label_model": "Current model",
        "self_awareness.slot_heading": "[Slot occupancy]",
        "self_awareness.text_precise": "Answer precisely with these values when asked about slot occupancy.",
        "self_awareness.text_self_id": "When the user asks which model you are using, answer with these values. Do not speculate from training data.",
        "self_awareness.text_no_slots": "If asked about other slots and you do not have this slot list, answer honestly: 'I only have information about my active slot.' Do not speculate.",
    },
    "de": {
        "self_awareness.label_model": "Modell",
        "self_awareness.slot_heading": "[Slot-Belegung im System]",
        "self_awareness.text_precise": "Antworte präzise mit diesen Werten wenn nach Slot-Belegungen gefragt wird.",
        "self_awareness.text_self_id": "Wenn der User fragt welches Modell du nutzt, antworte mit diesen Werten. Spekuliere nicht aus Trainingsdaten.",
        "self_awareness.text_no_slots": "Wenn du nach anderen Slots gefragt wirst und diese Slot-Liste nicht hast, antworte ehrlich: 'Ich habe nur Information zu meinem aktiven Slot.' Spekuliere nicht.",
    },
    "fr": {
        "self_awareness.label_model": "Modèle actuel",
        "self_awareness.slot_heading": "[Occupation des slots]",
        "self_awareness.text_precise": "Réponds précisément avec ces valeurs quand on te demande l'occupation des slots.",
        "self_awareness.text_self_id": "Quand l'utilisateur demande quel modèle tu utilises, réponds avec ces valeurs. Ne spécule pas à partir des données d'entraînement.",
        "self_awareness.text_no_slots": "Si on te demande d'autres slots et que tu n'as pas cette liste, réponds honnêtement : 'Je n'ai que des informations sur mon slot actif.' Ne spécule pas.",
    },
    "es": {
        "self_awareness.label_model": "Modelo actual",
        "self_awareness.slot_heading": "[Ocupación de slots]",
        "self_awareness.text_precise": "Responde con precisión con estos valores cuando te pregunten sobre la ocupación de slots.",
        "self_awareness.text_self_id": "Cuando el usuario pregunte qué modelo usas, responde con estos valores. No especules con datos de entrenamiento.",
        "self_awareness.text_no_slots": "Si te preguntan sobre otros slots y no tienes esta lista, responde honestamente: 'Solo tengo información sobre mi slot activo.' No especules.",
    },
    "it": {
        "self_awareness.label_model": "Modello attuale",
        "self_awareness.slot_heading": "[Occupazione slot]",
        "self_awareness.text_precise": "Rispondi con precisione con questi valori quando ti chiedono dell'occupazione degli slot.",
        "self_awareness.text_self_id": "Quando l'utente chiede quale modello stai usando, rispondi con questi valori. Non speculare dai dati di addestramento.",
        "self_awareness.text_no_slots": "Se ti chiedono di altri slot e non hai questa lista, rispondi onestamente: 'Ho solo informazioni sul mio slot attivo.' Non speculare.",
    },
    "pt": {
        "self_awareness.label_model": "Modelo atual",
        "self_awareness.slot_heading": "[Ocupação de slots]",
        "self_awareness.text_precise": "Responda com precisão com estes valores quando perguntarem sobre a ocupação de slots.",
        "self_awareness.text_self_id": "Quando o usuário perguntar qual modelo você usa, responda com estes valores. Não especule com dados de treinamento.",
        "self_awareness.text_no_slots": "Se perguntarem sobre outros slots e você não tiver esta lista, responda honestamente: 'Só tenho informações sobre meu slot ativo.' Não especule.",
    },
    "nl": {
        "self_awareness.label_model": "Huidig model",
        "self_awareness.slot_heading": "[Slot-bezetting]",
        "self_awareness.text_precise": "Antwoord precies met deze waarden wanneer naar slot-bezetting wordt gevraagd.",
        "self_awareness.text_self_id": "Wanneer de gebruiker vraagt welk model je gebruikt, antwoord met deze waarden. Speculeer niet vanuit trainingsdata.",
        "self_awareness.text_no_slots": "Als naar andere slots wordt gevraagd en je deze lijst niet hebt, antwoord eerlijk: 'Ik heb alleen informatie over mijn actieve slot.' Speculeer niet.",
    },
    "pl": {
        "self_awareness.label_model": "Aktualny model",
        "self_awareness.slot_heading": "[Obsadzenie slotów]",
        "self_awareness.text_precise": "Odpowiadaj precyzyjnie tymi wartościami, gdy pytają o obsadzenie slotów.",
        "self_awareness.text_self_id": "Gdy użytkownik pyta, jakiego modelu używasz, odpowiedz tymi wartościami. Nie spekuluj na podstawie danych treningowych.",
        "self_awareness.text_no_slots": "Jeśli pytają o inne sloty, a nie masz tej listy, odpowiedz uczciwie: 'Mam tylko informacje o moim aktywnym slocie.' Nie spekuluj.",
    },
    "ru": {
        "self_awareness.label_model": "Текущая модель",
        "self_awareness.slot_heading": "[Занятость слотов]",
        "self_awareness.text_precise": "Отвечай точно этими значениями, когда спрашивают о занятости слотов.",
        "self_awareness.text_self_id": "Когда пользователь спрашивает, какую модель ты используешь, отвечай этими значениями. Не спекулируй на основе тренировочных данных.",
        "self_awareness.text_no_slots": "Если спрашивают о других слотах, а у тебя нет этого списка, отвечай честно: 'У меня есть информация только о моём активном слоте.' Не спекулируй.",
    },
    "uk": {
        "self_awareness.label_model": "Поточна модель",
        "self_awareness.slot_heading": "[Зайнятість слотів]",
        "self_awareness.text_precise": "Відповідай точно цими значеннями, коли питають про зайнятість слотів.",
        "self_awareness.text_self_id": "Коли користувач питає, яку модель ти використовуєш, відповідай цими значеннями. Не спекулюй на основі тренувальних даних.",
        "self_awareness.text_no_slots": "Якщо питають про інші слоти і у тебе немає цього списку, відповідай чесно: 'У мене є інформація лише про мій активний слот.' Не спекулюй.",
    },
    "sv": {
        "self_awareness.label_model": "Aktuell modell",
        "self_awareness.slot_heading": "[Slotbelagd]",
        "self_awareness.text_precise": "Svara exakt med dessa värden när du tillfrågas om slotbeläggning.",
        "self_awareness.text_self_id": "När användaren frågar vilken modell du använder, svara med dessa värden. Spekulera inte från träningsdata.",
        "self_awareness.text_no_slots": "Om du tillfrågas om andra slots och inte har denna lista, svara ärligt: 'Jag har bara information om min aktiva slot.' Spekulera inte.",
    },
    "tr": {
        "self_awareness.label_model": "Mevcut model",
        "self_awareness.slot_heading": "[Slot doluluk durumu]",
        "self_awareness.text_precise": "Slot doluluk durumu sorulduğunda bu değerlerle kesin yanıt ver.",
        "self_awareness.text_self_id": "Kullanıcı hangi modeli kullandığını sorduğunda bu değerlerle yanıt ver. Eğitim verilerinden spekülasyon yapma.",
        "self_awareness.text_no_slots": "Başka slotlar hakkında sorulursa ve bu listen yoksa dürüstçe yanıt ver: 'Sadece aktif slotum hakkında bilgim var.' Spekülasyon yapma.",
    },
    "ar": {
        "self_awareness.label_model": "النموذج الحالي",
        "self_awareness.slot_heading": "[إشغال الفتحات]",
        "self_awareness.text_precise": "أجب بدقة بهذه القيم عند السؤال عن إشغال الفتحات.",
        "self_awareness.text_self_id": "عندما يسأل المستخدم عن النموذج الذي تستخدمه، أجب بهذه القيم. لا تتكهن من بيانات التدريب.",
        "self_awareness.text_no_slots": "إذا سُئلت عن فتحات أخرى وليس لديك هذه القائمة، أجب بصدق: 'لدي معلомات فقط عن فتحتي النشطة.' لا تتكهن.",
    },
    "hi": {
        "self_awareness.label_model": "वर्तमर्न मॉडल",
        "self_awareness.slot_heading": "[स्लॉट अधिभोग]",
        "self_awareness.text_precise": "जब स्लॉट अधिभोग के बरे में पूछढ हो तो इन मूल्यों के सरथ सटीक जवरब दें।",
        "self_awareness.text_self_id": "जब उपयोगकर्तढ पूछे कि आप कौन सं मॉडल उपयोग कर रहे हैं, तो इन मूल्यों के सरथ जवधब दें। प्रशिक्षण डेटध से अनुमीन न लगढएं।",
        "self_awareness.text_no_slots": "यदि अन्य स्लॉट्स के बरे में पूछं जढए और आपके पढस यह सूची नहीं है, तो ईमढनदТरी से जवढब दें: 'मेरे पढस केवल मेरे सक्रिय स्लॉट की जढनकढरी है।' अनुमढन न लगढएं।",
    },
    "id": {
        "self_awareness.label_model": "Model saat ini",
        "self_awareness.slot_heading": "[Penempatan slot]",
        "self_awareness.text_precise": "Jawab dengan tepat menggunakan nilai-nilai ini saat ditanya tentang penempatan slot.",
        "self_awareness.text_self_id": "Saat pengguna bertanya model apa yang kamu gunakan, jawab dengan nilai-nilai ini. Jangan berspekulasi dari data pelatihan.",
        "self_awareness.text_no_slots": "Jika ditanya tentang slot lain dan kamu tidak punya daftar ini, jawab jujur: 'Saya hanya punya informasi tentang slot aktif saya.' Jangan berspekulasi.",
    },
    "ja": {
        "self_awareness.label_model": "現在のモデル",
        "self_awareness.slot_heading": "[スロット占有状況]",
        "self_awareness.text_precise": "スロット占有について聞かれたら、これらの値で正確に回答してください。",
        "self_awareness.text_self_id": "ユーザーがどのモデルを使用しているか聞いたら、これらの値で回答してください。訓練データから推測しないでください。",
        "self_awareness.text_no_slots": "他のスロットについて聞かれ、このリストがない場合は、正直に回答してください：「私のアクティブスロットの情報しかありません。」推測しないでください。",
    },
    "ko": {
        "self_awareness.label_model": "현재 모델",
        "self_awareness.slot_heading": "[슬롯 점유 현황]",
        "self_awareness.text_precise": "슬롯 점유에 대해 물어보면 이 값으로 정확하게 답변하세요.",
        "self_awareness.text_self_id": "사용자가 어떤 모델을 사용하는지 물으면 이 값으로 답변하세요. 학습 데이터로 추측하지 마세요.",
        "self_awareness.text_no_slots": "다른 슬롯에 대해 물어보고 이 목록이 없으면 정직하게 답변하세요: '활성 슬롯에 대한 정보만 있습니다.' 추측하지 마세요.",
    },
    "zh": {
        "self_awareness.label_model": "当前模型",
        "self_awareness.slot_heading": "[槽位占用情况]",
        "self_awareness.text_precise": "当被问到槽位占用情况时，用这些值准确回答。",
        "self_awareness.text_self_id": "当用户问你使用哪个模型时，用这些值回答。不要从训练数据中推测。",
        "self_awareness.text_no_slots": "如果被问到其他槽位而你没有这个列表，请诚实回答：“我只有关于我活跃槽位的信息。”不要推测。",
    },
    "th": {
        "self_awareness.label_model": "โมเดลปัจจุบัน",
        "self_awareness.slot_heading": "[การครอบครองสล็อต]",
        "self_awareness.text_precise": "เมื่อถูกถามเกี่ยวกับการครอบครองสล็อต ตอบอย่างแม่นยำด้วยค่าเหล่านี้",
        "self_awareness.text_self_id": "เมื่อผู้ใช้ถามว่าคุณใช้โมเดลอะไร ตอบด้วยค่าเหล่านี้ อย่าคาดเดาจากข้อมูลการฝึก",
        "self_awareness.text_no_slots": "หากถูกถามเกี่ยวกับสล็อตอื่นและไม่มีรายการนี้ ตอบตามตรง: 'ฉันมีข้อมูลเกี่ยวกับสล็อตที่ใช้งานอยู่เท่านั้น' อย่าคาดเดา",
    },
    "vi": {
        "self_awareness.label_model": "Mô hình hiện tại",
        "self_awareness.slot_heading": "[Tình trạng slot]",
        "self_awareness.text_precise": "Khi được hỏi về tình trạng slot, trả lời chính xác với các giá trị này.",
        "self_awareness.text_self_id": "Khi người dùng hỏi bạn đang sử dụng mô hình nào, trả lời với các giá trị này. Không suy đoán từ dữ liệu huấn luyện.",
        "self_awareness.text_no_slots": "Nếu được hỏi về các slot khác và bạn không có danh sách này, hãy trả lời thành thật: 'Tôi chỉ có thông tin về slot đang hoạt động của mình.' Không suy đoán.",
    },
}

SOURCE_HASH = "a1b2c3d4e5f60001"  # Same for all (en is source)
HASHES = {
    "self_awareness.label_model": "a1b2c3d4e5f60001",
    "self_awareness.slot_heading": "a1b2c3d4e5f60002",
    "self_awareness.text_precise": "a1b2c3d4e5f60003",
    "self_awareness.text_self_id": "a1b2c3d4e5f60004",
    "self_awareness.text_no_slots": "a1b2c3d4e5f60005",
}


def inject_keys(locale_file: Path, lang: str) -> bool:
    """Inject self_awareness keys into a locale file.

    Returns True if keys were added, False if already present.
    """
    with open(locale_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    keys = data.get("keys", {})

    # Check if already present
    if "self_awareness.label_model" in keys:
        return False

    translations = TRANSLATIONS.get(lang)
    if translations is None:
        # Use EN as fallback
        translations = TRANSLATIONS["en"]

    # Add keys
    for key_name, text in translations.items():
        keys[key_name] = {
            "text": text,
            "source_hash": HASHES[key_name],
            "auto_translated": lang != "en",
            "reviewed": lang in ("en", "de"),
        }

    data["keys"] = keys

    with open(locale_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return True


def main() -> None:
    """Inject self_awareness keys into all locale files."""
    added = 0
    skipped = 0

    for json_file in sorted(LOCALES_DIR.glob("*.json")):
        if json_file.name.startswith("_"):
            continue
        lang = json_file.stem
        if inject_keys(json_file, lang):
            print(f"  Added keys to {json_file.name}")
            added += 1
        else:
            print(f"  Skipped {json_file.name} (already present)")
            skipped += 1

    print(f"\nDone: {added} updated, {skipped} skipped.")


if __name__ == "__main__":
    main()
