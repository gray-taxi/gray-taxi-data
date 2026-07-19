"""
conftest.py -- pytest 공통 설정

기능: 프로젝트 루트를 sys.path에 추가하여 테스트에서 `from src.xxx import ...` 형태로
소스 모듈을 임포트할 수 있게 한다.

변경내역
  2026-07-19  최초 작성
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
