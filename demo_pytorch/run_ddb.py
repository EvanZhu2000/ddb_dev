#!/usr/bin/env python3
"""
Simple DolphinDB script runner
"""

import dolphindb as ddb
import sys
import os

def run_dolphindb_script(script_path):
    print("=" * 50)
    print("DOLPHINDB SCRIPT RUNNER")
    print("=" * 50)
    
    # Try to connect to DolphinDB
    session = ddb.session()
    
    connection_attempts = [
        {'host': 'localhost', 'port': 8848}
    ]
    
    connected = False
    for attempt in connection_attempts:
        try:
            print(f"Trying {attempt['host']}:{attempt['port']}...")
            session.connect(attempt['host'], attempt['port'])
            print(f"✓ Connected to DolphinDB at {attempt['host']}:{attempt['port']}")
            connected = True
            break
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            continue
    
    if not connected:
        print("\n" + "="*50)
        print("DOLPHINDB CONNECTION FAILED")
        print("="*50)
        print("Please ensure DolphinDB server is running.")
        return False
    
    # Try to login
    try:
        session.login("admin", "123456")
        print("✓ Logged in to DolphinDB")
    except Exception as e:
        print(f"Login failed: {e}")
        print("Continuing without login...")
    
    # Test basic functionality
    try:
        version = session.run("version()")
        print(f"✓ DolphinDB version: {version}")
    except Exception as e:
        print(f"Version check failed: {e}")
    
    # Read and execute script
    print(f"\nReading script: {script_path}")
    if not os.path.exists(script_path):
        print(f"Error: Script file {script_path} not found")
        return False
    
    with open(script_path, 'r') as f:
        script_content = f.read()
    
    print("Executing DolphinDB script...")
    print("=" * 50)
    
    try:
        result = session.run(script_content)
        print("Script executed successfully!")
        if result is not None:
            print(f"Result: {result}")
        return True
    except Exception as e:
        print(f"Script execution failed: {e}")
        return False
    finally:
        session.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_ddb.py <script.dos>")
        sys.exit(1)
    
    script_path = sys.argv[1]
    success = run_dolphindb_script(script_path)
    
    if success:
        print("\n✓ Script completed successfully!")
    else:
        print("\n✗ Script failed!")
        sys.exit(1)
