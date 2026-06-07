"""Make the project root importable so `from algorithms... import ...`,
`from pipeline import ...`, `from tokenizer import ...` work under pytest."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
