import sqlite3
import os
from datetime import datetime
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path='fake_news_detection.db'):
        """
        Initialize the database manager with the specified database path.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self.connection = None
        
    def connect(self):
        """
        Establish connection to the SQLite database.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row  # Enable dict-like access to rows
            logger.info(f"Successfully connected to database: {self.db_path}")
            return True
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database: {e}")
            return False
    
    def disconnect(self):
        """
        Close the database connection.
        """
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
    
    def create_tables(self):
        """
        Create the required tables: history and users.
        
        Returns:
            bool: True if tables created successfully, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            
            # Create history table
            history_table_sql = """
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
            
            # Create users table
            users_table_sql = """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                role TEXT NOT NULL DEFAULT 'user'
            )
            """
            
            cursor.execute(history_table_sql)
            cursor.execute(users_table_sql)
            
            self.connection.commit()
            logger.info("Tables created successfully")
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Error creating tables: {e}")
            return False
    
    def insert_history_record(self, content, prediction, confidence):
        """
        Insert a new record into the history table.
        
        Args:
            content (str): The news content that was analyzed
            prediction (str): The prediction result (e.g., 'fake', 'real')
            confidence (float): The confidence score of the prediction
            
        Returns:
            int: The ID of the inserted record, or None if failed
        """
        try:
            cursor = self.connection.cursor()
            sql = """
            INSERT INTO history (content, prediction, confidence, timestamp)
            VALUES (?, ?, ?, ?)
            """
            
            timestamp = datetime.now().isoformat()
            cursor.execute(sql, (content, prediction, confidence, timestamp))
            self.connection.commit()
            
            record_id = cursor.lastrowid
            logger.info(f"History record inserted with ID: {record_id}")
            return record_id
            
        except sqlite3.Error as e:
            logger.error(f"Error inserting history record: {e}")
            return None
    
    def insert_user(self, username, role='user'):
        """
        Insert a new user into the users table.
        
        Args:
            username (str): The username
            role (str): The user role (default: 'user')
            
        Returns:
            int: The ID of the inserted user, or None if failed
        """
        try:
            cursor = self.connection.cursor()
            sql = "INSERT INTO users (username, role) VALUES (?, ?)"
            
            cursor.execute(sql, (username, role))
            self.connection.commit()
            
            user_id = cursor.lastrowid
            logger.info(f"User inserted with ID: {user_id}")
            return user_id
            
        except sqlite3.IntegrityError:
            logger.warning(f"User '{username}' already exists")
            return None
        except sqlite3.Error as e:
            logger.error(f"Error inserting user: {e}")
            return None
    
    def get_history_records(self, limit=100):
        """
        Retrieve history records from the database.
        
        Args:
            limit (int): Maximum number of records to retrieve
            
        Returns:
            list: List of history records as dictionaries
        """
        try:
            cursor = self.connection.cursor()
            sql = "SELECT * FROM history ORDER BY timestamp DESC LIMIT ?"
            
            cursor.execute(sql, (limit,))
            records = [dict(row) for row in cursor.fetchall()]
            
            logger.info(f"Retrieved {len(records)} history records")
            return records
            
        except sqlite3.Error as e:
            logger.error(f"Error retrieving history records: {e}")
            return []
    
    def clear_history_records(self):
        """
        Delete all records from the history table.
        
        Returns:
            bool: True if deletion successful, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            sql = "DELETE FROM history"
            
            cursor.execute(sql)
            self.connection.commit()
            
            deleted_count = cursor.rowcount
            logger.info(f"Cleared {deleted_count} history records")
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Error clearing history records: {e}")
            return False
    
    def get_user_by_username(self, username):
        """
        Retrieve a user by username.
        
        Args:
            username (str): The username to search for
            
        Returns:
            dict: User record as dictionary, or None if not found
        """
        try:
            cursor = self.connection.cursor()
            sql = "SELECT * FROM users WHERE username = ?"
            
            cursor.execute(sql, (username,))
            row = cursor.fetchone()
            
            if row:
                user = dict(row)
                logger.info(f"User found: {username}")
                return user
            else:
                logger.info(f"User not found: {username}")
                return None
                
        except sqlite3.Error as e:
            logger.error(f"Error retrieving user: {e}")
            return None
    
    def initialize_database(self):
        """
        Initialize the database by connecting and creating tables.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if self.connect():
            return self.create_tables()
        return False

# Utility functions for easy database operations
def get_database_manager():
    """
    Get a database manager instance.
    
    Returns:
        DatabaseManager: Initialized database manager
    """
    return DatabaseManager()

def test_database_connection():
    """
    Test the database connection and table creation.
    
    Returns:
        bool: True if test successful, False otherwise
    """
    try:
        db_manager = get_database_manager()
        
        # Test connection and table creation
        if not db_manager.initialize_database():
            logger.error("Failed to initialize database")
            return False
        
        # Test inserting a sample record
        sample_content = "This is a test news article for database validation."
        sample_prediction = "real"
        sample_confidence = 0.85
        
        record_id = db_manager.insert_history_record(
            sample_content, sample_prediction, sample_confidence
        )
        
        if record_id is None:
            logger.error("Failed to insert test record")
            return False
        
        # Test retrieving records
        records = db_manager.get_history_records(limit=1)
        if not records:
            logger.error("Failed to retrieve test records")
            return False
        
        # Test user insertion
        user_id = db_manager.insert_user("test_user", "admin")
        if user_id is None:
            logger.error("Failed to insert test user")
            return False
        
        # Test user retrieval
        user = db_manager.get_user_by_username("test_user")
        if not user:
            logger.error("Failed to retrieve test user")
            return False
        
        db_manager.disconnect()
        logger.info("Database test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database test failed: {e}")
        return False

if __name__ == "__main__":
    # Run database test when script is executed directly
    print("Testing database connection and operations...")
    success = test_database_connection()
    
    if success:
        print("✅ Database test passed successfully!")
    else:
        print("❌ Database test failed!")
        exit(1)