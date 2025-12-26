import subprocess
import sys
from pathlib import Path
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

project_root = Path(__file__).parent.parent.parent
token_file = Path(__file__).parent / ".token"


def load_token():
    if not token_file.exists():
        raise FileNotFoundError(f"Файл с токеном не найден: {token_file}")
    with open(token_file, "r", encoding="utf-8") as f:
        token = f.read().strip()
    if not token:
        raise ValueError("Токен пустой")
    return token


def run_predict(text: str) -> str:
    predict_script = project_root / "MLProj" / "ml" / "scripts" / "predict.py"
    if not predict_script.exists():
        return f"Ошибка: скрипт predict.py не найден: {predict_script}"
    try:
        result = subprocess.run(
            [sys.executable, str(predict_script), text],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(project_root),
            timeout=30,
        )
        if result.returncode != 0:
            return f"Ошибка при выполнении predict.py:\n{result.stderr}"
        return result.stdout
    except subprocess.TimeoutExpired:
        return "Ошибка: время выполнения превышено"
    except Exception as e:
        return f"Ошибка: {str(e)}"


def start(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Привет! Отправь мне текст, и я определю его намерение.\n"
        "Например: 'I need to activate my card'"
    )


def handle_message(update: Update, context: CallbackContext):
    text = update.message.text
    if not text or not text.strip():
        update.message.reply_text("Пожалуйста, отправь текст для классификации.")
        return
    update.message.reply_text("Обрабатываю...")
    result = run_predict(text)
    update.message.reply_text(result)


def main():
    try:
        token = load_token()
    except Exception as e:
        print(f"Ошибка загрузки токена: {e}")
        sys.exit(1)
    updater = Updater(token=token, use_context=True)
    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    print("Бот запущен...")
    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
