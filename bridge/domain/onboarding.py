"""Onboarding domain: wizard configuration and language mapping.

Defines the available languages for the setup wizard,
wizard texts in multiple languages, and the wizard step structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Supported languages: code -> (display_name, native_name)
# Used for the Wizard language picker AND for /lang command extension.
# ---------------------------------------------------------------------------

WIZARD_LANGUAGES: dict[str, str] = {
    "de": "Deutsch",
    "en": "English",
    "fr": "Français",
    "es": "Español",
    "it": "Italiano",
    "pt": "Português",
    "nl": "Nederlands",
    "pl": "Polski",
    "sv": "Svenska",
    "tr": "Türkçe",
    "ru": "Русский",
    "uk": "Українська",
    "zh": "中文",
    "ja": "日本語",
    "ko": "한국어",
    "ar": "العربية",
    "hi": "हिन्दी",
    "id": "Bahasa Indo.",
    "th": "ภาษาไทย",
    "vi": "Tiếng Việt",
}

# All valid language codes (20 named + "auto")
VALID_LANGUAGE_CODES: set[str] = set(WIZARD_LANGUAGES.keys()) | {"auto"}

# Keyboard layout: 5 rows of 4, plus auto row
LANGUAGE_KEYBOARD_ROWS: list[list[str]] = [
    ["de", "en", "fr", "es"],
    ["it", "pt", "nl", "pl"],
    ["sv", "tr", "ru", "uk"],
    ["zh", "ja", "ko", "ar"],
    ["hi", "id", "th", "vi"],
]

# ---------------------------------------------------------------------------
# Wizard UI texts (i18n)
# ---------------------------------------------------------------------------

# Step 1: Language selection prompt
_STEP1_TEXTS: dict[str, str] = {
    "de": "Willkommen! Wähle deine bevorzugte Sprache:",
    "en": "Welcome! Choose your preferred language:",
}

# Step 2: Completion message (template with {lang_name} placeholder)
# We provide translations for all 20 languages.
_STEP2_TEXTS: dict[str, str] = {
    "de": (
        "Du bist startklar! 🚀\n\n"
        "Sprache: {lang_name}\n\n"
        "Wichtig zu wissen:\n"
        "• Deine Sprache ist nicht fix, du kannst sie jederzeit über /settings ändern\n"
        "• Mit /help siehst du alle Befehle\n"
        "• Stell mir einfach eine Frage und los geht's"
    ),
    "en": (
        "You're all set! 🚀\n\n"
        "Language: {lang_name}\n\n"
        "Good to know:\n"
        "• Your language isn't locked, you can change it anytime via /settings\n"
        "• Use /help to see all commands\n"
        "• Just ask me a question and let's go"
    ),
    "fr": (
        "Tu es prêt(e) ! 🚀\n\n"
        "Langue : {lang_name}\n\n"
        "Bon à savoir :\n"
        "• Ta langue n'est pas figée, tu peux la changer à tout moment via /settings\n"
        "• Utilise /help pour voir toutes les commandes\n"
        "• Pose-moi simplement une question et c'est parti"
    ),
    "es": (
        "¡Todo listo! 🚀\n\n"
        "Idioma: {lang_name}\n\n"
        "Importante:\n"
        "• Tu idioma no es fijo, puedes cambiarlo en cualquier momento via /settings\n"
        "• Usa /help para ver todos los comandos\n"
        "• Hazme una pregunta y empezamos"
    ),
    "it": (
        "Tutto pronto! 🚀\n\n"
        "Lingua: {lang_name}\n\n"
        "Da sapere:\n"
        "• La tua lingua non è fissa, puoi cambiarla in qualsiasi momento via /settings\n"
        "• Usa /help per vedere tutti i comandi\n"
        "• Fammi una domanda e partiamo"
    ),
    "pt": (
        "Tudo pronto! 🚀\n\n"
        "Idioma: {lang_name}\n\n"
        "Importante:\n"
        "• Seu idioma não é fixo, você pode mudá-lo a qualquer momento via /settings\n"
        "• Use /help para ver todos os comandos\n"
        "• Me faça uma pergunta e vamos começar"
    ),
    "nl": (
        "Je bent klaar! 🚀\n\n"
        "Taal: {lang_name}\n\n"
        "Goed om te weten:\n"
        "• Je taal staat niet vast, je kunt het altijd wijzigen via /settings\n"
        "• Gebruik /help om alle commando's te zien\n"
        "• Stel me gewoon een vraag en we beginnen"
    ),
    "pl": (
        "Gotowe! 🚀\n\n"
        "Język: {lang_name}\n\n"
        "Warto wiedzieć:\n"
        "• Twój język nie jest na stałe, możesz go zmienić w każdej chwili przez /settings\n"
        "• Użyj /help aby zobaczyć wszystkie komendy\n"
        "• Po prostu zadaj mi pytanie i zaczynamy"
    ),
    "sv": (
        "Du är redo! 🚀\n\n"
        "Språk: {lang_name}\n\n"
        "Bra att veta:\n"
        "• Ditt språk är inte låst, du kan ändra det när som helst via /settings\n"
        "• Använd /help för att se alla kommandon\n"
        "• Ställ en fråga så kör vi"
    ),
    "tr": (
        "Hazırsın! 🚀\n\n"
        "Dil: {lang_name}\n\n"
        "Bilmen gerekenler:\n"
        "• Dilin sabit değil, istediğin zaman /settings üzerinden değiştirebilirsin\n"
        "• Tüm komutları görmek için /help kullan\n"
        "• Bana bir soru sor ve başlayalım"
    ),
    "ru": (
        "Всё готово! 🚀\n\n"
        "Язык: {lang_name}\n\n"
        "Полезно знать:\n"
        "• Язык не фиксирован, его можно изменить в любое время через /settings\n"
        "• Используй /help чтобы увидеть все команды\n"
        "• Просто задай вопрос и поехали"
    ),
    "uk": (
        "Все готово! 🚀\n\n"
        "Мова: {lang_name}\n\n"
        "Корисно знати:\n"
        "• Мова не фіксована, її можна змінити будь-коли через /settings\n"
        "• Використовуй /help щоб побачити всі команди\n"
        "• Просто постав питання і починаємо"
    ),
    "zh": (
        "准备就绪！🚀\n\n"
        "语言：{lang_name}\n\n"
        "须知：\n"
        "• 语言不是固定的，你可以随时通过 /settings 更改\n"
        "• 使用 /help 查看所有命令\n"
        "• 直接问我问题就可以开始了"
    ),
    "ja": (
        "準備完了！🚀\n\n"
        "言語：{lang_name}\n\n"
        "知っておくべきこと：\n"
        "• 言語は固定ではありません。/settings からいつでも変更できます\n"
        "• /help で全コマンドを確認できます\n"
        "• 質問をどうぞ、始めましょう"
    ),
    "ko": (
        "준비 완료! 🚀\n\n"
        "언어: {lang_name}\n\n"
        "알아두면 좋은 점:\n"
        "• 언어는 고정되지 않으며 /settings에서 언제든 변경할 수 있습니다\n"
        "• /help로 모든 명령어를 확인하세요\n"
        "• 질문하면 바로 시작합니다"
    ),
    "ar": (
        "🚀 !أنت جاهز\n\n"
        "اللغة: {lang_name}\n\n"
        ":من المهم معرفته\n"
        "• لغتك ليست ثابتة، يمكنك تغييرها في أي وقت عبر /settings\n"
        "• استخدم /help لرؤية جميع الأوامر\n"
        "• اسألني سؤالاً وهيا نبدأ"
    ),
    "hi": (
        "तैयार हो! 🚀\n\n"
        "भाषा: {lang_name}\n\n"
        "जानने योग्य:\n"
        "• आपकी भाषा स्थायी नहीं है, आप इसे कभी भी /settings से बदल सकते हैं\n"
        "• सभी कमांड देखने के लिए /help का उपयोग करें\n"
        "• बस मुझसे कोई सवाल पूछें और शुरू करते हैं"
    ),
    "id": (
        "Siap! 🚀\n\n"
        "Bahasa: {lang_name}\n\n"
        "Penting untuk diketahui:\n"
        "• Bahasa kamu tidak tetap, bisa diubah kapan saja lewat /settings\n"
        "• Gunakan /help untuk melihat semua perintah\n"
        "• Tanyakan saja dan kita mulai"
    ),
    "th": (
        "พร้อมแล้ว! 🚀\n\n"
        "ภาษา: {lang_name}\n\n"
        "สิ่งที่ควรรู้:\n"
        "• ภาษาไม่ได้ถูกล็อค คุณสามารถเปลี่ยนได้ตลอดเวลาผ่าน /settings\n"
        "• ใช้ /help เพื่อดูคำสั่งทั้งหมด\n"
        "• ถามคำถามได้เลย แล้วเราจะเริ่มกัน"
    ),
    "vi": (
        "Sẵn sàng! 🚀\n\n"
        "Ngôn ngữ: {lang_name}\n\n"
        "Lưu ý:\n"
        "• Ngôn ngữ không cố định, bạn có thể thay đổi bất cứ lúc nào qua /settings\n"
        "• Dùng /help để xem tất cả lệnh\n"
        "• Hãy hỏi tôi một câu hỏi và bắt đầu thôi"
    ),
}

# Auto-detect option texts
_AUTO_DETECT_TEXTS: dict[str, str] = {
    "de": "Automatisch erkennen (Empfohlen)",
    "en": "Auto-detect (Recommended)",
}

# Button labels
_SKIP_WIZARD_TEXTS: dict[str, str] = {
    "de": "Setup überspringen",
    "en": "Skip Setup",
}

_LETS_GO_TEXTS: dict[str, str] = {
    "de": "Los geht's",
    "en": "Let's go",
}

# Onboarding hint after 3 messages (for users who skipped)
_ONBOARDING_HINT_TEXTS: dict[str, str] = {
    "de": "Tipp: Über /onboarding kannst du das Setup nachholen.",
    "en": "Tip: Use /onboarding to complete the setup.",
}

# Wizard done (completion) texts
_WIZARD_DONE_TEXTS: dict[str, str] = {
    "de": "Viel Spaß!",
    "en": "Enjoy!",
    "fr": "Amusez-vous bien !",
    "es": "¡A disfrutar!",
    "it": "Buon divertimento!",
    "pt": "Aproveite!",
    "nl": "Veel plezier!",
    "pl": "Miłej zabawy!",
    "sv": "Ha det kul!",
    "tr": "İyi eğlenceler!",
    "ru": "Приятного использования!",
    "uk": "Приємного користування!",
    "zh": "祝你使用愉快！",
    "ja": "楽しんでください！",
    "ko": "즐겁게 사용하세요!",
    "ar": "استمتع!",
    "hi": "आनंद लें!",
    "id": "Selamat menikmati!",
    "th": "ขอให้สนุก!",
    "vi": "Chúc bạn vui vẻ!",
}

# Wizard skip texts
_WIZARD_SKIP_STEP1_TEXTS: dict[str, str] = {
    "de": "Setup übersprungen.",
    "en": "Setup skipped.",
}

_WIZARD_SKIP_STEP2_TEXTS: dict[str, str] = {
    "de": "Setup abgeschlossen.",
    "en": "Setup completed.",
}

# /start welcome texts for onboarded users
_START_WELCOME_TEXTS: dict[str, str] = {
    "de": (
        "Axolent ist bereit.\n\n"
        "Schick mir eine Frage und ich beantworte sie.\n\n"
        "Tipp: Du kannst Bot-Nachrichten als Bookmark speichern. "
        "Antworte einfach mit /save."
    ),
    "en": (
        "Axolent is ready.\n\n"
        "Send me a question and I'll answer it.\n\n"
        "Tip: You can bookmark bot messages. "
        "Just reply with /save."
    ),
    "fr": (
        "Axolent est prêt.\n\n"
        "Pose-moi une question et je te réponds.\n\n"
        "Astuce : tu peux enregistrer les messages du bot. "
        "Réponds simplement avec /save."
    ),
    "es": (
        "Axolent está listo.\n\n"
        "Hazme una pregunta y te respondo.\n\n"
        "Consejo: puedes guardar los mensajes del bot. "
        "Simplemente responde con /save."
    ),
    "it": (
        "Axolent è pronto.\n\n"
        "Fammi una domanda e ti rispondo.\n\n"
        "Suggerimento: puoi salvare i messaggi del bot. "
        "Rispondi semplicemente con /save."
    ),
    "pt": (
        "Axolent está pronto.\n\n"
        "Me faça uma pergunta e eu respondo.\n\n"
        "Dica: você pode salvar as mensagens do bot. "
        "Basta responder com /save."
    ),
    "nl": (
        "Axolent is klaar.\n\n"
        "Stel me een vraag en ik beantwoord het.\n\n"
        "Tip: je kunt bot-berichten opslaan als bladwijzer. "
        "Antwoord gewoon met /save."
    ),
    "pl": (
        "Axolent jest gotowy.\n\n"
        "Zadaj mi pytanie, a odpowiem.\n\n"
        "Wskazówka: możesz zapisywać wiadomości bota. "
        "Po prostu odpowiedz /save."
    ),
    "sv": (
        "Axolent är redo.\n\n"
        "Ställ en fråga så svarar jag.\n\n"
        "Tips: du kan spara bot-meddelanden som bokmärken. "
        "Svara bara med /save."
    ),
    "tr": (
        "Axolent hazır.\n\n"
        "Bana bir soru sor, cevaplayayım.\n\n"
        "İpucu: Bot mesajlarını yer imi olarak kaydedebilirsin. "
        "Sadece /save ile yanıtla."
    ),
    "ru": (
        "Axolent готов.\n\n"
        "Задай мне вопрос, и я отвечу.\n\n"
        "Подсказка: ты можешь сохранять сообщения бота. "
        "Просто ответь /save."
    ),
    "uk": (
        "Axolent готовий.\n\n"
        "Постав мені питання, і я відповім.\n\n"
        "Підказка: ти можеш зберігати повідомлення бота. "
        "Просто відповідай /save."
    ),
    "zh": (
        "Axolent 已就绪。\n\n"
        "问我一个问题，我来回答。\n\n"
        "提示：你可以将机器人消息加入书签。"
        "只需回复 /save。"
    ),
    "ja": (
        "Axolent の準備ができました。\n\n"
        "質問をどうぞ、お答えします。\n\n"
        "ヒント：ボットのメッセージをブックマークできます。"
        "/save と返信するだけです。"
    ),
    "ko": (
        "Axolent 준비 완료.\n\n"
        "질문하면 답변해 드립니다.\n\n"
        "팁: 봇 메시지를 북마크할 수 있습니다. "
        "/save로 답장하세요."
    ),
    "ar": (
        "Axolent جاهز.\n\n"
        "اسألني سؤالاً وسأجيبك.\n\n"
        "نصيحة: يمكنك حفظ رسائل البوت كإشارة مرجعية. "
        "فقط أجب بـ /save."
    ),
    "hi": (
        "Axolent तैयार है।\n\n"
        "मुझसे कोई सवाल पूछें, मैं जवाब दूँगा।\n\n"
        "सुझाव: आप बॉट संदेशों को बुकमार्क कर सकते हैं। "
        "बस /save के साथ जवाब दें।"
    ),
    "id": (
        "Axolent siap.\n\n"
        "Tanyakan sesuatu dan saya akan menjawab.\n\n"
        "Tips: kamu bisa menyimpan pesan bot sebagai bookmark. "
        "Cukup balas dengan /save."
    ),
    "th": (
        "Axolent พร้อมแล้ว\n\n"
        "ถามคำถามได้เลย แล้วจะตอบให้\n\n"
        "เคล็ดลับ: คุณสามารถบันทึกข้อความบอทเป็นบุ๊กมาร์กได้ "
        "เพียงตอบกลับด้วย /save"
    ),
    "vi": (
        "Axolent đã sẵn sàng.\n\n"
        "Hãy hỏi tôi một câu hỏi, tôi sẽ trả lời.\n\n"
        "Mẹo: bạn có thể lưu tin nhắn bot làm dấu trang. "
        "Chỉ cần trả lời bằng /save."
    ),
}


def get_step1_text(lang: str = "de") -> str:
    """Returns the Step 1 prompt text in the given language."""
    return _STEP1_TEXTS.get(lang, _STEP1_TEXTS["en"])


def get_step2_text(lang: str, lang_name: str) -> str:
    """Returns the Step 2 completion text in the given language.

    Args:
        lang: ISO language code.
        lang_name: Display name of the chosen language.
    """
    template = _STEP2_TEXTS.get(lang, _STEP2_TEXTS["en"])
    return template.format(lang_name=lang_name)


def get_auto_detect_text(lang: str = "de") -> str:
    """Returns the auto-detect button text."""
    return _AUTO_DETECT_TEXTS.get(lang, _AUTO_DETECT_TEXTS["en"])


def get_skip_wizard_text(lang: str = "de") -> str:
    """Returns the skip button text."""
    return _SKIP_WIZARD_TEXTS.get(lang, _SKIP_WIZARD_TEXTS["en"])


def get_lets_go_text(lang: str = "de") -> str:
    """Returns the 'Let's go' button text."""
    return _LETS_GO_TEXTS.get(lang, _LETS_GO_TEXTS["en"])


def get_onboarding_hint_text(lang: str = "de") -> str:
    """Returns the onboarding hint for skipped users."""
    return _ONBOARDING_HINT_TEXTS.get(lang, _ONBOARDING_HINT_TEXTS["en"])


def get_wizard_done_text(lang: str = "de") -> str:
    """Returns the wizard completion text in the given language."""
    return _WIZARD_DONE_TEXTS.get(lang, _WIZARD_DONE_TEXTS["en"])


def get_wizard_skip_step1_text(lang: str = "de") -> str:
    """Returns the wizard skip (step 1) text."""
    return _WIZARD_SKIP_STEP1_TEXTS.get(lang, _WIZARD_SKIP_STEP1_TEXTS["en"])


def get_wizard_skip_step2_text(lang: str = "de") -> str:
    """Returns the wizard skip (step 2) text."""
    return _WIZARD_SKIP_STEP2_TEXTS.get(lang, _WIZARD_SKIP_STEP2_TEXTS["en"])


def get_start_welcome_text(lang: str = "de") -> str:
    """Returns the /start welcome text for onboarded users."""
    return _START_WELCOME_TEXTS.get(lang, _START_WELCOME_TEXTS["en"])


def get_language_name(code: str) -> str:
    """Returns the display name for a language code.

    Args:
        code: ISO language code.

    Returns:
        Native language name, or the code itself if unknown.
    """
    if code == "auto":
        return "Auto-detect"
    return WIZARD_LANGUAGES.get(code, code)


@dataclass(frozen=True, slots=True)
class OnboardingState:
    """Onboarding state for a user.

    Attributes:
        user_id: Telegram User-ID.
        onboarded: Whether the user completed or skipped-past-step-1 the wizard.
        wizard_lang: Language chosen in wizard step 1 (None if not yet chosen).
        skip_count: Number of times user has sent messages without onboarding.
        hint_shown: Whether the 3-message onboarding hint has been shown.
    """

    user_id: int
    onboarded: bool = False
    wizard_lang: Optional[str] = None
    skip_count: int = 0
    hint_shown: bool = False
