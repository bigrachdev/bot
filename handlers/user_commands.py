"""
User command handlers for market data queries
"""
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes, CommandHandler, CallbackQueryHandler
from utils.logger import logger
from config.settings import TOP_STOCKS, TOP_FOREX
from schedulers.news_scheduler import NewsScheduler
from schedulers.analysis_scheduler import AnalysisScheduler

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command - show market menu"""
    try:
        user_id = update.effective_user.id
        username = update.effective_user.username or update.effective_user.first_name
        chat = update.effective_chat
        chat_id = chat.id

        bot_instance = context.application.bot_data.get('bot_instance')
        bot_instance.subscribe_chat(chat_id, chat.type, chat.title or username)

        # Get custom welcome message
        welcome_msg = bot_instance.get_setting('welcome_message',
                                                f'Welcome, {username}! Use the buttons below to explore markets. 📈')
        welcome_msg = welcome_msg.format(username=username)
        
        # Create inline keyboard for market selection
        keyboard = [
            [InlineKeyboardButton("📈 Stocks", callback_data='market_stocks')],
            [InlineKeyboardButton("💱 Forex", callback_data='market_forex')],
            [InlineKeyboardButton("📰 Get News Now", callback_data='action_news')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            f"{welcome_msg}\n\n"
            "Choose a market to explore or get latest news:",
            reply_markup=reply_markup
        )
        
        logger.info(f"User {user_id} started bot in chat {chat_id}")
        
    except Exception as e:
        logger.error(f"Error in /start: {e}")
        await update.message.reply_text("❌ An error occurred. Please try again.")

async def handle_market_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle market selection buttons"""
    try:
        query = update.callback_query
        chat_id = query.message.chat_id
        
        if query.data == 'market_stocks':
            await show_stocks_menu(query, context)
        elif query.data == 'market_forex':
            await show_forex_menu(query, context)
        elif query.data == 'action_news':
            await handle_news_request(query, context, chat_id)
        
    except Exception as e:
        logger.error(f"Error handling market button: {e}")

async def show_stocks_menu(query, context):
    """Show stocks pagination menu"""
    try:
        parts = query.data.split('_')
        page = int(parts[-1]) if len(parts) > 2 and parts[-1].isdigit() else 0
        page_size = 5
        start = page * page_size
        end = start + page_size
        
        stocks = TOP_STOCKS[start:end]
        
        keyboard = [
            [InlineKeyboardButton(stock['name'], callback_data=f"stock_{stock['symbol']}")]
            for stock in stocks
        ]
        
        # Navigation buttons
        nav_buttons = []
        if page > 0:
            nav_buttons.append(InlineKeyboardButton("⬅️ Previous", callback_data=f'market_stocks_{page-1}'))
        if end < len(TOP_STOCKS):
            nav_buttons.append(InlineKeyboardButton("Next ➡️", callback_data=f'market_stocks_{page+1}'))
        
        if nav_buttons:
            keyboard.append(nav_buttons)
        
        keyboard.append([InlineKeyboardButton("↩️ Back", callback_data='back_to_main')])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            text=f"📈 <b>Top Stocks</b> (Page {page + 1})\n\nSelect a stock to analyze:",
            reply_markup=reply_markup,
            parse_mode='HTML'
        )
        
    except Exception as e:
        logger.error(f"Error showing stocks menu: {e}")

async def show_forex_menu(query, context):
    """Show forex pagination menu"""
    try:
        parts = query.data.split('_')
        page = int(parts[-1]) if len(parts) > 2 and parts[-1].isdigit() else 0
        page_size = 5
        start = page * page_size
        end = start + page_size
        
        forex = TOP_FOREX[start:end]
        
        keyboard = [
            [InlineKeyboardButton(pair['name'], callback_data=f"forex_{pair['symbol']}")]
            for pair in forex
        ]
        
        # Navigation buttons
        nav_buttons = []
        if page > 0:
            nav_buttons.append(InlineKeyboardButton("⬅️ Previous", callback_data=f'market_forex_{page-1}'))
        if end < len(TOP_FOREX):
            nav_buttons.append(InlineKeyboardButton("Next ➡️", callback_data=f'market_forex_{page+1}'))
        
        if nav_buttons:
            keyboard.append(nav_buttons)
        
        keyboard.append([InlineKeyboardButton("↩️ Back", callback_data='back_to_main')])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            text=f"💱 <b>Forex Pairs</b> (Page {page + 1})\n\nSelect a pair to analyze:",
            reply_markup=reply_markup,
            parse_mode='HTML'
        )
        
    except Exception as e:
        logger.error(f"Error showing forex menu: {e}")

async def handle_news_request(query, context, chat_id):
    """Handle on-demand news request"""
    try:
        await query.edit_message_text("📰 Fetching latest news...")
        
        await NewsScheduler.send_news_to_chat(context.application.bot_data.get('bot_instance'), chat_id)
        
        logger.info(f"News sent to chat {chat_id}")
        
    except Exception as e:
        logger.error(f"Error fetching news: {e}")

async def handle_symbol_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle individual symbol selection"""
    try:
        query = update.callback_query
        chat_id = query.message.chat_id
        
        # Extract symbol and market type from callback data
        data_parts = query.data.split('_', 1)
        if len(data_parts) < 2:
            return
        
        market_type = data_parts[0]
        symbol = data_parts[1]
        
        # Send analysis
        if market_type == 'stock':
            await AnalysisScheduler.send_analysis_to_chat(
                context.application.bot_data.get('bot_instance'), chat_id, 'stocks'
            )
        elif market_type == 'forex':
            await AnalysisScheduler.send_analysis_to_chat(
                context.application.bot_data.get('bot_instance'), chat_id, 'forex'
            )
        
        await query.answer()
        
    except Exception as e:
        logger.error(f"Error handling symbol selection: {e}")

async def handle_back_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle back button"""
    try:
        query = update.callback_query
        
        keyboard = [
            [InlineKeyboardButton("📈 Stocks", callback_data='market_stocks')],
            [InlineKeyboardButton("💱 Forex", callback_data='market_forex')],
            [InlineKeyboardButton("📰 Get News Now", callback_data='action_news')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            text="Choose a market to explore:",
            reply_markup=reply_markup
        )
        
    except Exception as e:
        logger.error(f"Error handling back button: {e}")

def setup_user_handlers(application):
    """Register all user command handlers"""
    application.add_handler(CommandHandler('start', cmd_start))
    application.add_handler(CallbackQueryHandler(handle_market_buttons, pattern='^market_'))
    application.add_handler(CallbackQueryHandler(handle_symbol_selection, pattern='^(stock|forex)_'))
    application.add_handler(CallbackQueryHandler(handle_back_button, pattern='^back_to_main$'))
    application.add_handler(CallbackQueryHandler(handle_market_buttons, pattern='^action_'))
