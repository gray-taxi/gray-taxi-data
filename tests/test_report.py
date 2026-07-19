"""
test_report.py -- src/report.py 간단 단위 테스트

기능: 파일 읽기 헬퍼(_read_text/_read_json)와 generate_final_report()의
Jinja2 렌더링이 정상 동작하는지 확인한다.

변경내역
  2026-07-19  최초 작성
"""

import json

from src.report import _read_text, _read_json, generate_final_report


# _read_text가 대상 파일이 없을 때 지정한 안내 문구를 *(...)*  형태로 반환하는지 확인한다.
def test_read_text_returns_placeholder_when_missing(tmp_path):
    missing = tmp_path / "missing.md"
    assert _read_text(str(missing), "없어요") == "*(없어요)*"


# _read_text가 파일이 존재할 때 그 내용을 그대로 반환하는지 확인한다.
def test_read_text_returns_file_content(tmp_path):
    path = tmp_path / "existing.md"
    path.write_text("hello", encoding="utf-8")
    assert _read_text(str(path), "없어요") == "hello"


# _read_json이 대상 파일이 없을 때 지정한 기본값을 그대로 반환하는지 확인한다.
def test_read_json_returns_default_when_missing(tmp_path):
    missing = tmp_path / "missing.json"
    assert _read_json(str(missing), {"x": 1}) == {"x": 1}


# _read_json이 파일이 존재할 때 JSON 내용을 올바르게 파싱해 반환하는지 확인한다.
def test_read_json_parses_existing_file(tmp_path):
    path = tmp_path / "existing.json"
    path.write_text(json.dumps({"r2": 0.9}), encoding="utf-8")
    assert _read_json(str(path), {}) == {"r2": 0.9}


# generate_final_report가 최소 템플릿(metrics.r2, ttests 길이 출력)을 실제로 렌더링해
# 파일로 저장하고, 없는 산출물 경로들은 예외 없이 기본값으로 대체되는지 확인한다.
def test_generate_final_report_renders_with_minimal_template(tmp_path):
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    (template_dir / "minimal.md.j2").write_text(
        "R2={{ metrics.r2 }} / TTESTS={{ ttests | length }}",
        encoding="utf-8",
    )

    metrics_path = tmp_path / "model_metrics.json"
    metrics_path.write_text(json.dumps({"r2": 0.5}), encoding="utf-8")

    output_path = tmp_path / "report.md"

    generate_final_report(
        template_dir=str(template_dir),
        template_name="minimal.md.j2",
        output_path=str(output_path),
        stat_summary_path=str(tmp_path / "no_stat.md"),
        metrics_path=str(metrics_path),
        loading_comparison_path=str(tmp_path / "no_loading.md"),
        preprocess_summary_path=str(tmp_path / "no_preprocess.md"),
        ttest_path=str(tmp_path / "no_ttest.json"),
    )

    content = output_path.read_text(encoding="utf-8")
    assert "R2=0.5" in content
    assert "TTESTS=0" in content
