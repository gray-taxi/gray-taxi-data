"""
NYC Yellow Taxi 2026-05 — 자동화 통합 보고서 생성 모듈
=====================================================
이 모듈은 Jinja2를 사용하여 분석 결과(통계 요약 마크다운, ML 통합 평가 지표)를 템플릿에 연동하고,
최종 종합 보고서인 report.md를 루트 디렉토리에 자동으로 렌더링합니다.
"""

import os
import json
import shutil
from jinja2 import Environment, FileSystemLoader

def generate_final_report(
    template_dir: str,
    template_name: str,
    output_path: str,
    stat_summary_path: str,
    metrics_path: str,
    fallback_stat_summary_path: str
) -> None:
    # 1. 기술통계/상관분석 요약 마크다운 읽기 (혹은 복사 후 읽기)
    if not os.path.exists(stat_summary_path):
        if os.path.exists(fallback_stat_summary_path):
            print(f"[Report] outputs/stat_summary.md가 없으므로 {fallback_stat_summary_path}에서 복사합니다.")
            os.makedirs(os.path.dirname(stat_summary_path), exist_ok=True)
            shutil.copy(fallback_stat_summary_path, stat_summary_path)
        else:
            print("[Warning] 기술통계 요약 마크다운 파일을 찾을 수 없습니다. 빈 내용으로 대체합니다.")
            
    if os.path.exists(stat_summary_path):
        with open(stat_summary_path, "r", encoding="utf-8") as f:
            stat_summary_content = f.read()
    else:
        stat_summary_content = "*(기술통계 요약 파일이 존재하지 않습니다)*"

    # 2. ML 평가 메트릭 JSON 읽기
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_content = json.load(f)
        except Exception as e:
            print(f"[Warning] 모델 메트릭 파일 파싱 실패: {e}")
            metrics_content = {}
    else:
        print("[Warning] 모델 메트릭 파일이 존재하지 않습니다. 빈 지표로 대체합니다.")
        metrics_content = {}

    # 3. 시각화 차트 메타데이터 정의 (상대 경로 적용)
    visualizations = [
        {
            "title": "시간대별 택시 이용량 (정적 차트 예시)",
            "image_path": "outputs/taxi_demand_by_hour.png",
            "description": "퇴근 시간대인 오후 18시~19시에 택시 수요가 가장 집중되며, 출근 시간대인 오전 7~8시에도 피크가 형성되는 양상을 보입니다."
        },
        {
            "title": "승객 수에 따른 평균 팁 금액",
            "image_path": "outputs/tip_by_passenger_count.png",
            "description": "승객 수에 따른 평균 팁은 $2.8~$3.0 구간에서 안정적인 패턴을 보이며, 단체 승객(5~6명)의 경우 평균 팁이 미세하게 높은 경향이 있습니다."
        },
        {
            "title": "주요 시간대별 평균 팁 금액",
            "image_path": "outputs/avg_tip_by_period.png",
            "description": "심야(night) 시간대의 평균 팁 금액이 가장 높은 수치로 나타나며, 주간(daytime) 시간대는 상대적으로 다소 낮은 값을 갖습니다."
        }
    ]

    # 4. Jinja2 환경 로드 및 렌더링
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_name)

    rendered_report = template.render(
        stat_summary=stat_summary_content,
        visualizations=visualizations,
        metrics=metrics_content,
    )

    # 5. 최종 report.md 저장
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rendered_report)
    print(f"[Report] 보고서가 성공적으로 빌드되었습니다: {output_path}")

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    generate_final_report(
        template_dir=os.path.join(PROJECT_ROOT, "templates"),
        template_name="report_template.md.j2",
        output_path=os.path.join(PROJECT_ROOT, "report.md"),
        stat_summary_path=os.path.join(PROJECT_ROOT, "outputs", "stat_summary.md"),
        metrics_path=os.path.join(PROJECT_ROOT, "outputs", "model_metrics.json"),
        fallback_stat_summary_path=os.path.join(PROJECT_ROOT, "visualize", "heewon", "stat_summary.md")
    )
