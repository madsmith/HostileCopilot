import logging
import sqlite3
import time
from datetime import datetime

from hostile_copilot.config import OmegaConfig

from .types import ScanData

logger = logging.getLogger(__name__)

class MiningLogger:
    def __init__(self, config: OmegaConfig):
        self._config = config
        self._conn: sqlite3.Connection | None = None

    def initialize(self):
        self._initialize_database()

    def _initialize_database(self):
        db_file = self._config.get("mining_logger.database_file", "mining_logger.db")
        assert db_file is not None, "Missing config 'mining_logger.database_file'"

        conn = sqlite3.connect(db_file)

        cursor = conn.cursor()

        # Check if the info table exists to determine current schema version
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='info'")
        table_exists = cursor.fetchone() is not None
        
        current_version = 0
        if table_exists:
            # Get current schema version from the info table
            cursor.execute("SELECT value FROM info WHERE key='schema_version'")
            result = cursor.fetchone()
            if result:
                current_version = int(result[0])
        
        # Run all necessary migrations in sequence
        self._run_migrations(conn, current_version)
        
        conn.commit()

        self._conn = conn
    
    def _run_migrations(self, conn, current_version):
        """
        Run migrations sequentially to update the database schema.
        
        Args:
            conn: SQLite connection
            current_version: Current schema version of the database
        """
        # Dictionary of migration functions keyed by target version
        migrations = {
            1: self._migrate_to_v1,
        }
        
        # Apply migrations in version order
        for version in sorted([v for v in migrations.keys() if v > current_version]):
            logger.info(f"Migrating database schema to version {version}")
            migrations[version](conn)
            
            # Update schema version
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO info (key, value, updated_at) VALUES (?, ?, ?)",
                ("schema_version", str(version), datetime.now().isoformat())
            )
            conn.commit()

    def _migrate_to_v1(self, conn):
        cursor = conn.cursor()

        # Create info table for key-value pairs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS info (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create scan table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scan (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                object_type TEXT,
                object_subtype TEXT,
                mass INTEGER,
                difficulty TEXT,
                resistance INTEGER,
                instability REAL,
                size REAL,
                location TEXT,
                time INTEGER
            )
        ''')

        # Create scan_compositions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scan_compositions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_id INTEGER,
                material_name TEXT,
                percentage REAL,
                FOREIGN KEY (scan_id) REFERENCES scan (id)
            )
        ''')

    def shutdown(self):
        self._conn.close()

    def log(self, scan_data: ScanData, location: str | None = None):
        assert self._conn is not None, "Database connection is not initialized"
        assert isinstance(scan_data, ScanData), "Scan data must be of type ScanData"

        cursor = self._conn.cursor()
        try:
            # Begin transaction
            cursor.execute('BEGIN')

            # Insert scan data into the scan table
            cursor.execute('''
                INSERT INTO scan (object_type, object_subtype, mass, difficulty, resistance, instability, size, location, time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                scan_data.object_type,
                scan_data.object_subtype,
                scan_data.mass,
                scan_data.difficulty,
                scan_data.resistance,
                scan_data.instability,
                scan_data.size,
                location,
                int(time.time())
            ))

            # Get the last inserted scan ID
            scan_id = cursor.lastrowid

            # Insert composition data into the scan_compositions table
            for scan_item in scan_data.composition:
                cursor.execute('''
                    INSERT INTO scan_compositions (scan_id, material_name, percentage)
                    VALUES (?, ?, ?)
                ''', (scan_id, scan_item.material, scan_item.percentage))

            self._conn.commit()
        except Exception:
            self._conn.rollback()
            logger.exception("Error logging scan data")
        