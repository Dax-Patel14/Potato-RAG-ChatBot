import sqlite3
from datetime import datetime
import os

# Use an absolute path or a path relative to the project root
# ensuring the database file is created in a reliable location
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'chat_history.db')

# Initialize database and tables if not exist
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chats (
        id TEXT PRIMARY KEY,
        name TEXT,
        created_at TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id TEXT,
        sender TEXT,
        content TEXT,
        timestamp TEXT,
        FOREIGN KEY(chat_id) REFERENCES chats(id)
    )''')
    conn.commit()
    conn.close()

# Add a new chat
def add_chat(chat_id, name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Check if chat already exists to prevent unique constraint errors
    c.execute('SELECT id FROM chats WHERE id = ?', (chat_id,))
    if c.fetchone() is None:
        c.execute('INSERT INTO chats (id, name, created_at) VALUES (?, ?, ?)',
                  (chat_id, name, datetime.now().isoformat()))
        conn.commit()
    conn.close()

# Add a message to a chat
def add_message(chat_id, sender, content):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO messages (chat_id, sender, content, timestamp) VALUES (?, ?, ?, ?)',
              (chat_id, sender, content, datetime.now().isoformat()))
    conn.commit()
    conn.close()

# Get all chats
def get_chats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, name, created_at FROM chats ORDER BY created_at DESC')
    chats = c.fetchall()
    conn.close()
    return chats

# Get messages for a chat
def get_messages(chat_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT sender, content, timestamp FROM messages WHERE chat_id=? ORDER BY timestamp', (chat_id,))
    messages = c.fetchall()
    conn.close()
    return messages

# Rename a chat
def rename_chat(chat_id, new_name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('UPDATE chats SET name=? WHERE id=?', (new_name, chat_id))
    conn.commit()
    conn.close()

# Delete a chat and its messages
def delete_chat(chat_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM messages WHERE chat_id=?', (chat_id,))
    c.execute('DELETE FROM chats WHERE id=?', (chat_id,))
    conn.commit()
    conn.close()

# Call init_db() at module load
init_db()