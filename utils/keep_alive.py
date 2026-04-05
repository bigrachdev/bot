"""
Keep-alive module for Render/Heroku free tier
Starts a lightweight HTTP server and pings it periodically to prevent sleep
"""
import asyncio
import aiohttp
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
from utils.logger import logger
from config.settings import KEEP_ALIVE_INTERVAL

class KeepAliveHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for keep-alive pings"""

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'Bot is alive!')

    def log_message(self, format, *args):
        """Suppress default logging"""
        pass

def start_server(port=8080):
    """Start the keep-alive HTTP server"""
    server = HTTPServer(('0.0.0.0', port), KeepAliveHandler)
    logger.info(f"✅ Keep-alive server started on port {port}")
    server.serve_forever()

async def ping_server(port=8080):
    """Ping the server periodically to keep it alive"""
    url = f"http://localhost:{port}/"

    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        logger.debug("Keep-alive ping successful")
        except Exception as e:
            logger.warning(f"Keep-alive ping failed: {e}")

        await asyncio.sleep(KEEP_ALIVE_INTERVAL)

def start_keep_alive():
    """Start the keep-alive server in a background thread"""
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    logger.info("✅ Keep-alive thread started")

    return server_thread
