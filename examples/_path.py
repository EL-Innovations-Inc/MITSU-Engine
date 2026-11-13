"""Lets the examples run from a fresh clone without installing the package."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
