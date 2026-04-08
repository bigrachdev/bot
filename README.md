 Market Data Bot - News & Signal Edition

A Telegram bot that provides real-time market news, technical analysis, and trading signals for stocks, crypto, and forex.

 Features

 📰 News Broadcasting
- News aggregation every 25 minutes from multiple sources (CNBC RSS, MarketWatch, Yahoo Finance, Investing.com, NewsAPI, Alpha Vantage)
- One major headline from each source per broadcast (avoids flooding, ensures source diversity)
- Each post includes full article details + live stock prices from Finnhub
- Automatic broadcasting to all subscribed channels/groups
- De-duplication - won't post the same news twice within 24 hours
- On-demand news - Users can request news anytime with the "Get News Now" button
- Admin broadcast - Force send news immediately with `/sendnews` command

 📈 Technical Analysis & Signals
- Hourly analysis broadcast for top 10 stocks
- Strong signals only - Posts only STRONG BUY 🔥 and STRONG SELL 🔴 recommendations
- Medium detail - Shows signal + key oscillators + moving averages
- Daily timeframe - Based on daily candle analysis
- Admin broadcast - Force send analysis immediately with `/sendanalysis` command

 📊 Real-time Market Analysis
- On-demand analysis - Users can analyze any stock/crypto/forex manually
- Multiple market categories: Stocks, Crypto, Forex
- Top 50 markets for each category with pagination
- Detailed signals: Buy/Sell/Neutral counts, Oscillators, Moving Averages

 👥 User Management
- Auto-subscription - Chats automatically subscribe when /start is used
- Welcome messages - Customizable welcome for new members
- Admin control - Admin-only commands for configuration

 📢 Advertisement Management (Admin Panel)
- Text ads - Post promotional messages with optional buttons/links  
- Image ads - Share promotional images with captions
- Scheduled posts - Plan ads for specific date/time (UTC)
- Ad tracking - All ads stored in database with status (pending/posted/scheduled)
- Ad history - View and audit all posted advertisements
- Target specific chats - Send ads to selected channels (by chat ID)
- Button support - Add clickable links to text ads

 Setup

 1. Install Dependencies
```bash
pip install -r requirements.txt
```

 2. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Required:
- `BOT_TOKEN` - Get from [@BotFather](https://t.me/botfather) on Telegram
- `YOUR_ADMIN_ID` - Your Telegram user ID (get it from [@userinfobot](https://t.me/userinfobot))

For News Features (at least one required):
- `NEWSAPI_KEY` - Free from [NewsAPI.org](https://newsapi.org/) (generous free tier)
- `ALPHAVANTAGE_KEY` - Free from [Alpha Vantage](https://www.alphavantage.co/) (25 calls/day on free tier)

For Enhanced Technical Analysis (required for stock prices):
- `FINNHUB_KEY` - Free from [Finnhub](https://finnhub.io/) (60 calls/minute, includes RSI, MACD, SMA indicators AND real-time stock prices)

 3. Run the Bot
```bash
python bot.py
```

The bot will:
- Start the Telegram bot polling
- Initialize the SQLite database
- Set up hourly news scheduler
- Ready to accept `/start` commands

 Commands

 User Commands
- `/start` - Show market category menu
- Button: "📰 Get News Now" - Fetch and display latest news

 Admin Commands (requires YOUR_ADMIN_ID config)
- `/sendnews` - Broadcast news immediately to all target channels
- `/sendanalysis` - Broadcast strong trading signals immediately (top 10 stocks)
- `/subscriptions` - View target channels configured in `.env`
- `/setwelcome <message>` - Set custom welcome message for new members
  - Use `{username}` placeholder for member's name

 Admin Advertisement System
- `/postmessage <chat_id> <message>` - Post text ad to specific chat
  - Optional buttons: `--button=Label:URL --button=Label2:URL2`
  - Example: `/postmessage -1001234567890 Check out signals! --button=Learn:https://example.com`
  
- `/postimage <chat_id> <image_url> <caption>` - Post image ad 
  - Example: `/postimage -1001234567890 https://example.com/pic.jpg Special offer ends today!`
  
- `/schedulepost <chat_id> <datetime> <message>` - Schedule ad for later
  - Format datetime as `YYYY-MM-DD HH:MM` (UTC)
  - Example: `/schedulepost -1001234567890 '2026-04-05 14:30' Don't miss this opportunity!`
  
- `/viewads` - View ad history (last 20 ads with status)

 How It Works

 News Aggregation Flow
1. Bot starts and initializes scheduler
2. Every hour at minute 0 (e.g., 3:00, 4:00, etc.), the scheduler triggers `broadcast_news()`
3. Bot fetches news from configured APIs
4. News is de-duplicated against 24-hour cache
5. Top 2 news items sent to all subscribed chats
6. News items cached to prevent re-posting

 Analysis Broadcast Flow
1. Every hour at minute 30 (e.g., 3:30, 4:30, etc.), the scheduler triggers `broadcast_analysis()`
2. Bot fetches technical analysis for top 10 stocks (daily timeframe)
3. Filters for STRONG signals only (STRONG BUY or STRONG SELL)
4. Formats analysis with signal + key indicators (oscillators, moving averages)
5. Sends formatted message to all subscribed chats
6. Staggered from news broadcasts (news at :00, analysis at :30) to avoid message spam

 Advertisement Flow
1. Admin uses `/postmessage`, `/postimage`, or `/schedulepost` command
2. Ad details stored in database with status `pending` or `scheduled`
3. For immediate posts (postmessage, postimage):
   - Message/image sent to specified chat immediately
   - Status updated to `posted` in database
4. For scheduled posts (schedulepost):
   - Ad stored with scheduled time
   - Future: Scheduler can post at scheduled time
   - Status: `scheduled` → `posted`
5. Admin can view all ads with `/viewads` command

Finding Chat ID:
- Run `/subscriptions` to see all subscribed chat IDs
- Or use group chat tools to get the chat ID
- Chat IDs are negative numbers for groups/channels

 Subscription System
- Chats automatically subscribe when users click `/start`
- Chat ID, type (private/group/supergroup), and name are stored in database

## 📢 Channel Broadcast Setup

Channels are configured directly in `.env` — no subscription commands needed.

1. **Add the bot as an admin in your channel:**
   - Go to your channel → Administrators → Add Administrator
   - Search for your bot and add it
   - Grant **"Post Messages"** permission (required)

2. **Get the channel ID:**
   - Forward a message from the channel to [@userinfobot](https://t.me/userinfobot)
   - Or check channel info — IDs look like `-1001234567890`

3. **Add the channel ID to `.env`:**
   ```env
   # Single channel:
   TARGET_CHANNELS=-1001234567890

   # Multiple channels (comma-separated):
   TARGET_CHANNELS=-1001234567890,-1009876543210
   ```

4. **Restart the bot** — broadcasts will start automatically

**Important Notes:**
- Channel IDs typically start with `-100`
- Multiple channels: separate with commas, no spaces
- Bot must be admin in each channel with "Post Messages" permission

 Technical Analysis
- Uses [tradingview-ta](https://github.com/AnalysisIndicators/tradingview-ta) library
- Analyzes daily candles for each selected symbol
- Returns recommendation + detailed oscillator/moving average data
- Supports 50 pre-configured top stocks, 45+ forex pairs
- Crypto list dynamically fetched from CoinGecko
- Broadcast analysis: Filters and posts only STRONG signals hourly

 Database Schema

The bot uses SQLite with four main tables:

```sql
-- Settings (customizable messages, configuration)
CREATE TABLE settings (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- Subscriptions (subscribed chats for news and analysis)
CREATE TABLE subscriptions (
    chat_id INTEGER PRIMARY KEY,
    chat_type TEXT,           -- 'private', 'group', 'supergroup'
    chat_name TEXT,
    subscribed_at TIMESTAMP
);

-- News Cache (prevent duplicate posts)
CREATE TABLE news_cache (
    news_id TEXT PRIMARY KEY,
    title TEXT,
    source TEXT,
    posted_at TIMESTAMP,
    url TEXT
);

-- Advertisements (admin-posted ads tracking)
CREATE TABLE advertisements (
    ad_id INTEGER PRIMARY KEY AUTOINCREMENT,
    admin_id INTEGER,
    chat_id INTEGER,
    content TEXT,
    content_type TEXT,        -- 'text', 'image'
    buttons TEXT,             -- JSON string of button data
    image_url TEXT,
    scheduled_time TIMESTAMP, -- For scheduled posts
    posted_at TIMESTAMP,
    status TEXT,              -- 'pending', 'posted', 'scheduled'
    created_at TIMESTAMP
);
```

 Troubleshooting

 No news or analysis being posted
1. Check `NEWSAPI_KEY` and/or `ALPHAVANTAGE_KEY` are set in `.env`
2. For enhanced analysis, ensure `FINNHUB_KEY` is set in `.env`
3. Verify API keys are valid and have remaining quota
3. Check bot logs: `tail -f market_bot.log`
4. Use `/sendnews` or `/sendanalysis` admin commands to test immediately
5. Ensure at least one chat has subscribed (sent `/start` command)

 Analysis shows "No strong signals found"
- This is normal if no stocks have STRONG BUY or STRONG SELL signals
- Check individual stocks manually with `/start` → Click stock
- Only STRONG signals are broadcast (not regular BUY/SELL)

 Ad not posting to chat
1. Verify chat ID is correct - use `/subscriptions` to confirm
2. Check if bot is member of target chat
3. Ensure chat exists and is accessible
4. Check logs for error details
5. Make sure to quote scheduled time: `/schedulepost <id> '2026-04-05 14:30' message`

 Bot crashes
1. Ensure all dependencies installed: `pip install -r requirements.txt`
2. Verify `BOT_TOKEN` is correct
3. Check logs in `market_bot.log`
4. Ensure Python 3.7+ is being used

 Scheduler not running
- Bot will log "✅ News and Analysis schedulers started" on successful startup
- Check that `apscheduler` is installed
- Verify bot has permission to write to logs

 Ad Usage Examples

Text ad with buttons:
```
/postmessage -1001234567890 Check out our trading signals! 🚀 --button=Visit:https://example.com --button=Learn:https://docs.example.com
```

Image ad:
```
/postimage -1001234567890 https://example.com/banner.jpg Limited time offer - Sign up now!
```

Schedule an ad:
```
/schedulepost -1001234567890 '2026-04-05 18:00' Market analysis coming at 6 PM UTC!
```

View ad history:
```
/viewads
```

 Architecture Notes

- SQLite for persistent storage (subscriptions, cache, settings)
- APScheduler for hourly news scheduling (with cron trigger)
- asyncio for concurrent operations
- python-telegram-bot v20+ for Telegram API
- requests for HTTP calls to news APIs
- tradingview-ta for technical analysis

 Future Enhancements

- [ ] User-specific symbol watchlists
- [ ] Sentiment analysis integration
- [ ] Price alerts (buy/sell levels)
- [ ] Discord/Slack integration
- [ ] Web dashboard for analytics
- [ ] More news sources (Reddit, Twitter, financial blogs)
- [ ] Custom timezone support for scheduled posts

 License

MIT
