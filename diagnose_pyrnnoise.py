import sys
import os

print("--- Python Import Diagnostic ---")

# 1. Check for shadow folders in your project
search_paths = [os.getcwd(), os.path.join(os.getcwd(), 'features')]

for path in search_paths:
    rogue_dir = os.path.join(path, 'pyrnnoise')
    rogue_file = os.path.join(path, 'pyrnnoise.py')
    
    if os.path.exists(rogue_dir) or os.path.exists(rogue_file):
        print("\n🚨 NAME COLLISION DETECTED 🚨")
        print("Python is trying to load a local file/folder instead of the real library:")
        if os.path.exists(rogue_dir): print(f" -> {rogue_dir}")
        if os.path.exists(rogue_file): print(f" -> {rogue_file}")
        print("\nPlease delete or rename this file/folder. It is blocking RNNoise from loading!")
        sys.exit(1)

# 2. Check for a corrupted site-packages folder
for path in sys.path:
    if 'site-packages' in path and '.venv' in path:
        lib_dir = os.path.join(path, 'pyrnnoise')
        init_file = os.path.join(lib_dir, '__init__.py')
        
        if os.path.isdir(lib_dir) and not os.path.exists(init_file):
            print("\n🚨 CORRUPTED SITE-PACKAGES DETECTED 🚨")
            print(f"Pip left a broken, empty folder at: {lib_dir}")
            print("\nTo fix this, run these exact commands in PowerShell:")
            print("1. Remove-Item -Recurse -Force .\\.venv\\Lib\\site-packages\\pyrnnoise")
            print("2. pip install pyrnnoise --no-cache-dir")
            sys.exit(1)

print("\nAll folders look clean. Testing import directly...")
try:
    import pyrnnoise
    print(f"Success! Library loaded from: {getattr(pyrnnoise, '__path__', pyrnnoise.__file__)}")
except ImportError as e:
    print(f"ImportError: {e}")


# Save this file in your main `audioTranscription` folder and run `python diagnose_pyrnnoise.py` in your terminal. It will print out the exact path to the rogue file/folder that needs to be deleted or fixed!