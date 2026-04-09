"""Keep-alive module for Render/Heroku free tier
Starts a lightweight HTTP server and pings it periodically to prevent sleep

"""
import asyncio
import aiohttp
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
from utils.logger import logger

class KeepAliveHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for keep-alive pings"""

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'OK')

    def log_message(self, format, *args):
        """Suppress default logging"""
        pass

def start_server(port=8080):
    """Start the keep-alive HTTP server"""
    try:
        server = HTTPServer(('0.0.0.0', port), KeepAliveHandler)
        logger.info(f"✅ Keep-alive server started on port {port}")
        server.serve_forever()
    except Exception as e:
        logger.error(f"❌ Keep-alive server failed to start: {e}")

async def ping_server(port=8080, interval_seconds=240):
    """Ping the server periodically to keep it alive
    
    Args:
        port: Port to ping
        interval_seconds: Seconds between pings (default 240 = 4 min for Render)
    """
    # Try to get external URL from Render environment
    render_url = os.getenv('RENDER_EXTERNAL_URL')
    localhost_url = f"http://localhost:{port}/"
    url_to_ping = render_url if render_url else localhost_url
    
    ping_count = 0
    session = None
    
    logger.info(f"Keep-alive pinging: {url_to_ping} every {interval_seconds}s")

    while True:
        try:
            if session is None or session.closed:
                session = aiohttp.ClientSession()
            
            async with session.get(url_to_ping, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    ping_count += 1
                    logger.debug(f"Keep-alive ping #{ping_count} succeeded")
        except asyncio.CancelledError:
            logger.info("Keep-alive ping task cancelled")
            if session and not session.closed:
                await session.close()
            break
        except Exception as e:
            logger.warning(f"Keep-alive ping failed: {e}")
        
        try:
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            logger.info("Keep-alive sleep interrupted")
            if session and not session.closed:
                await session.close()
            break
    
    if session and not session.closed:
        await session.close()

def start_keep_alive(port=8080):
    """Start the keep-alive server in a background thread"""
    try:
        server_thread = threading.Thread(target=start_server, args=(port,), daemon=True)
        server_thread.start()
        logger.info(f"✅ Keep-alive thread started on port {port}")
        return server_thread
    except Exception as e:
        logger.error(f"❌ Failed to start keep-alive thread: {e}")
        return None
