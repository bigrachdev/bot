"""
Advertisement management service
"""
from utils.logger import logger

class AdService:
    """Handle advertisement operations"""
    
    @staticmethod
    def parse_buttons(button_string: str) -> list:
        """Parse button string into key_markup format
        Format: "Label1:URL1|Label2:URL2"
        """
        if not button_string:
            return []
        
        try:
            buttons = []
            pairs = button_string.split('|')
            for pair in pairs:
                label, url = pair.split(':')
                buttons.append({'text': label.strip(), 'url': url.strip()})
            return buttons
        except Exception as e:
            logger.error(f"Failed to parse buttons: {e}")
            return []

    @staticmethod
    def format_ad_preview(ad: dict) -> str:
        """Format ad for preview in /viewads command"""
        try:
            ad_id = ad.get('ad_id', 'N/A')
            status = ad.get('status', 'N/A')
            content_type = ad.get('content_type', 'text')
            chat_id = ad.get('chat_id', 'N/A')
            created_at = ad.get('created_at', 'N/A')
            
            status_emoji = {
                'pending': '⏳',
                'posted': '✅',
                'scheduled': '⏰'
            }.get(status, '❓')
            
            preview = f"{status_emoji} <b>Ad #{ad_id}</b>\n"
            preview += f"   Status: {status}\n"
            preview += f"   Type: {content_type}\n"
            preview += f"   Chat: {chat_id}\n"
            preview += f"   Created: {created_at}\n\n"
            
            return preview
        except Exception as e:
            logger.error(f"Failed to format ad preview: {e}")
            return "Error formatting ad"

    @staticmethod
    def validate_chat_id(chat_id_str: str) -> int:
        """Validate and parse chat ID"""
        try:
            chat_id = int(chat_id_str)
            return chat_id
        except ValueError:
            logger.warning(f"Invalid chat ID: {chat_id_str}")
            return None

    @staticmethod
    def validate_schedule_time(time_str: str) -> bool:
        """Validate schedule time format YYYY-MM-DD HH:MM"""
        try:
            from datetime import datetime
            datetime.strptime(time_str, '%Y-%m-%d %H:%M')
            return True
        except ValueError:
            logger.warning(f"Invalid time format: {time_str}")
            return False
