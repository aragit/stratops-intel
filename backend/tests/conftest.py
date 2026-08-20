"""Shared pytest fixtures for StratOps Intel backend tests."""

import asyncio
import os
from typing import AsyncGenerator
from uuid import UUID, uuid4

import pytest
import structlog