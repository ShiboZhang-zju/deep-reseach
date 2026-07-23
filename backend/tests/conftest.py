"""Pytest configuration — set AUTO_CREATE_SCHEMA for test DB init."""
import os
os.environ["AUTO_CREATE_SCHEMA"] = "true"
