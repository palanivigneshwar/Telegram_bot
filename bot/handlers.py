import os
from telegram import Update
from telegram.ext import ContextTypes

from rag.pipeline import RAGPipeline
from vision.captioner import ImageCaptioner

# Initialize systems ONCE

rag_pipeline = RAGPipeline(data_path="data/docs")
captioner = ImageCaptioner()

# Folder to store images

IMAGE_DIR = "data/images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# /start command

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
    "👋 Welcome! I'm your GenAI bot.\n\n"
    "I can:\n"
    "• Answer questions from documents (/ask)\n"
    "• Describe images (just send one)\n\n"
    "Use /help to learn more."
    )

# /help command

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
    "🤖 Available Commands:\n\n"
    "/ask <question> - Ask questions from knowledge base\n"
    "/image - Send an image directly (no command needed)\n"
    "/help - Show this message\n\n"
    "📌 Examples:\n"
    "/ask What is AI?\n"
    "(or just upload an image)"
    )
    await update.message.reply_text(help_text)

# /ask command

async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
        "Please provide a question.\nExample: /ask What is AI?"
        )
        return

    query = " ".join(context.args)

    await update.message.reply_text("🔍 Searching for answer...")

    try:
        answer = rag_pipeline.query(query)
        await update.message.reply_text(f"🧠 Answer:\n\n{answer}")

    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

# Image handler (FULL IMPLEMENTATION)

async def image_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await update.message.reply_text("🖼️ Processing image...")

        # Get highest resolution photo
        photo = update.message.photo[-1]

        # Download file
        file = await context.bot.get_file(photo.file_id)

        file_path = os.path.join(IMAGE_DIR, f"{photo.file_id}.jpg")
        await file.download_to_drive(file_path)

        # Process image
        result = captioner.process_image(file_path)

        caption = result["caption"]
        tags = result["tags"]

        response = (
            f"📝 Caption:\n{caption}\n\n"
            f"🏷️ Tags:\n{', '.join(tags)}"
        )

        await update.message.reply_text(response)

    except Exception as e:
        await update.message.reply_text(f"❌ Error processing image: {str(e)}")
