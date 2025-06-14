import logging
from src.database.connection import clear_database

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Starting database cleanup...")
        clear_database()
        logger.info("Database cleanup completed successfully!")
    except Exception as e:
        logger.error(f"Error during database cleanup: {str(e)}")
        raise

if __name__ == "__main__":
    main() 