import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.database import init_db

if __name__ == "__main__":
    init_db()
    print("Database tables created.")