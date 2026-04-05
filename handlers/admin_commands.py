"""
Admin command handlers - restricted to admin only
"""
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes, CommandHandler, MessageHandler, filters
from utils.logger import logger
from config.settings import YOUR_ADMIN_ID
from services.ads import AdService
from schedulers.news_scheduler import NewsScheduler
from schedulers.analysis_scheduler import AnalysisScheduler

async def check_admin(update: Update) -> bool:
    """Verify user is admin"""
    user_id = update.effective_user.id
    if user_id != YOUR_ADMIN_ID:
        await update.message.reply_text("❌ You don't have permission to use this command.")
        logger.warning(f"Unauthorized admin command attempt by user {user_id}")
        return False
    return True

async def cmd_sendnews(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin: Broadcast news to all subscribed chats"""
    if not await check_admin(update):
        return
    
    try:
        await update.message.reply_text("📰 Broadcasting news to all subscribed chats...")
        
        bot_instance = context.application.bot_data.get('bot_instance')
        await NewsScheduler.broadcast_news(bot_instance)
        
        await update.message.reply_text("✅ News broadcast complete!")
        logger.info("Admin news broadcast executed")
        
    except Exception as e:
        logger.error(f"Error in /sendnews: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)[:100]}")

async def cmd_sendanalysis(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin: Broadcast analysis to all subscribed chats"""
    if not await check_admin(update):
        return
    
    try:
        await update.message.reply_text("📊 Broadcasting analysis to all subscribed chats...")
        
        bot_instance = context.application.bot_data.get('bot_instance')
        await AnalysisScheduler.broadcast_analysis(bot_instance)
        
        await update.message.reply_text("✅ Analysis broadcast complete!")
        logger.info("Admin analysis broadcast executed")
        
    except Exception as e:
        logger.error(f"Error in /sendanalysis: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)[:100]}")

async def cmd_subscriptions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin: List target channels from config"""
    if not await check_admin(update):
        return

    try:
        bot_instance = context.application.bot_data.get('bot_instance')
        chats = bot_instance.get_subscribed_chats()

        if not chats:
            await update.message.reply_text(
                "📭 No channels configured.\n\n"
                "Add channels in <code>.env</code>:\n"
                "<code>TARGET_CHANNELS=-1001234567890</code>\n\n"
                "Then restart the bot."
            )
            return

        message = "<b>📢 Target Channels:</b>\n\n"
        for chat_id, chat_type in chats:
            message += f"• <code>{chat_id}</code> ({chat_type})\n"
        
        await update.message.reply_text(message, parse_mode='HTML')
        logger.info(f"Admin viewed subscriptions ({len(chats)} chats)")
        
    except Exception as e:
        logger.error(f"Error in /subscriptions: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)[:100]}")

async def cmd_setwelcome(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin: Set custom welcome message"""
    if not await check_admin(update):
        return
    
    try:
        if not context.args:
            await update.message.reply_text(
                "Usage: /setwelcome <message>\n"
                "Use {username} as placeholder for username\n"
                "Example: /setwelcome Welcome {username} to our trading group!"
            )
            return
        
        welcome_msg = ' '.join(context.args)
        
        bot_instance = context.application.bot_data.get('bot_instance')
        bot_instance.set_setting('welcome_message', welcome_msg)
        
        await update.message.reply_text(f"✅ Welcome message updated!")
        logger.info(f"Admin updated welcome message")
        
    except Exception as e:
        logger.error(f"Error in /setwelcome: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)[:100]}")

async def cmd_postmessage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin: Post text advertisement"""
    if not await check_admin(update):
        return
    
    try:
        # Parse: /postmessage <chat_id> <message> [--button=Label:URL]
        if len(context.args) < 2:
            await update.message.reply_text(
                "Usage: /postmessage <chat_id> <message> [--button=Label:URL]\n"
                "Example: /postmessage 123456789 Check out our service! --button=Click:https://example.com"
            )
            return
        
        chat_id_str = context.args[0]
        chat_id = AdService.validate_chat_id(chat_id_str)
        if not chat_id:
            await update.message.reply_text("❌ Invalid chat ID")
            return
        
        # Extract message and buttons
        message_parts = context.args[1:]
        buttons_str = None
        
        # Check for button argument
        message_text = []
        for part in message_parts:
            if part.startswith('--button='):
                buttons_str = part[9:]
            else:
                message_text.append(part)
        
        content = ' '.join(message_text)
        
        bot_instance = context.application.bot_data.get('bot_instance')
        ad_id = bot_instance.add_ad(
            update.effective_user.id, chat_id, content, 'text', buttons_str
        )
        
        if ad_id:
            # Send the ad
            try:
                await context.bot.send_message(chat_id=chat_id, text=content, parse_mode='HTML')
                bot_instance.update_ad_status(ad_id, 'posted')
                await update.message.reply_text(f"✅ Ad #{ad_id} posted!")
            except Exception as send_err:
                logger.error(f"Failed to send ad to {chat_id}: {send_err}")
                await update.message.reply_text(f"⚠️ Ad created (#{ad_id}) but failed to send: {str(send_err)[:100]}")
        else:
            await update.message.reply_text("❌ Failed to create ad")
        
    except Exception as e:
        logger.error(f"Error in /postmessage: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)[:100]}")

async def cmd_postimage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin: Post image advertisement"""
    if not await check_admin(update):
        return
    
    try:
        # Parse: /postimage <chat_id> <image_url> <caption>
        if len(context.args) < 3:
            await update.message.reply_text(
                "Usage: /postimage <chat_id> <image_url> <caption>\n"
                "Example: /postimage 123456789 https://example.com/image.jpg Check this out!"
            )
            return
        
        chat_id_str = context.args[0]
        chat_id = AdService.validate_chat_id(chat_id_str)
        if not chat_id:
            await update.message.reply_text("❌ Invalid chat ID")
            return
        
        image_url = context.args[1]
        caption = ' '.join(context.args[2:])
        
        bot_instance = context.application.bot_data.get('bot_instance')
        ad_id = bot_instance.add_ad(
            update.effective_user.id, chat_id, caption, 'image', 
            image_url=image_url
        )
        
        if ad_id:
            # Send the image
            try:
                await context.bot.send_photo(
                    chat_id=chat_id, 
                    photo=image_url, 
                    caption=caption, 
                    parse_mode='HTML'
                )
                bot_instance.update_ad_status(ad_id, 'posted')
                await update.message.reply_text(f"✅ Image ad #{ad_id} posted!")
            except Exception as send_err:
                logger.error(f"Failed to send image ad to {chat_id}: {send_err}")
                await update.message.reply_text(f"⚠️ Ad created (#{ad_id}) but failed to send: {str(send_err)[:100]}")
        else:
            await update.message.reply_text("❌ Failed to create ad")
        
    except Exception as e:
        logger.error(f"Error in /postimage: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)[:100]}")

async def cmd_schedulepost(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin: Schedule post for later"""
    if not await check_admin(update):
        return
    
    try:
        # Parse: /schedulepost <chat_id> '<YYYY-MM-DD HH:MM'> <message>
        if len(context.args) < 3:
            await update.message.reply_text(
                "Usage: /schedulepost <chat_id> '<YYYY-MM-DD HH:MM'> <message>\n"
                "Example: /schedulepost 123456789 '2024-12-25 14:30' Holiday promo!"
            )
            return
        
        chat_id_str = context.args[0]
        chat_id = AdService.validate_chat_id(chat_id_str)
        if not chat_id:
            await update.message.reply_text("❌ Invalid chat ID")
            return
        
        schedule_time = context.args[1].strip("'\"")
        if not AdService.validate_schedule_time(schedule_time):
            await update.message.reply_text("❌ Invalid time format. Use YYYY-MM-DD HH:MM")
            return
        
        content = ' '.join(context.args[2:])
        
        bot_instance = context.application.bot_data.get('bot_instance')
        ad_id = bot_instance.add_ad(
            update.effective_user.id, chat_id, content, 'text',
            scheduled_time=schedule_time
        )
        
        if ad_id:
            await update.message.reply_text(
                f"✅ Ad #{ad_id} scheduled for {schedule_time}\n"
                "(Note: Auto-execution coming soon)"
            )
        else:
            await update.message.reply_text("❌ Failed to schedule ad")
        
    except Exception as e:
        logger.error(f"Error in /schedulepost: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)[:100]}")

async def cmd_viewads(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin: View recent ads"""
    if not await check_admin(update):
        return
    
    try:
        bot_instance = context.application.bot_data.get('bot_instance')
        ads = bot_instance.get_all_ads(20)
        
        if not ads:
            await update.message.reply_text("📭 No ads found.")
            return
        
        message = "<b>📋 Recent Advertisements:</b>\n\n"
        for ad in ads:
            message += AdService.format_ad_preview(ad)
        
        await update.message.reply_text(message, parse_mode='HTML')
        logger.info(f"Admin viewed ads")
        
    except Exception as e:
        logger.error(f"Error in /viewads: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)[:100]}")

def setup_admin_handlers(application):
    """Register all admin command handlers"""
    application.add_handler(CommandHandler('sendnews', cmd_sendnews))
    application.add_handler(CommandHandler('sendanalysis', cmd_sendanalysis))
    application.add_handler(CommandHandler('subscriptions', cmd_subscriptions))
    application.add_handler(CommandHandler('setwelcome', cmd_setwelcome))
    application.add_handler(CommandHandler('postmessage', cmd_postmessage))
    application.add_handler(CommandHandler('postimage', cmd_postimage))
    application.add_handler(CommandHandler('schedulepost', cmd_schedulepost))
    application.add_handler(CommandHandler('viewads', cmd_viewads))
