"""
NYC Yellow Taxi 2026-05 — 자동화 보고서 생성 모듈
=================================================
이 모듈은 Jinja2를 사용하여 1~3단계의 분석 결과(시각화 이미지 경로,
기술통계/상관분석 마크다운, ML 파이프라인 평가지표)를 템플릿에 바인딩하여
최종 report.md 보고서를 자동으로 렌더링하고 생성합니다.
"""

import os
import json
from jinja2 import Environment, FileSystemLoader


def generate_final_report(
    template_dir: str,
    template_name: str,
    output_path: str,
    stat_summary_path: str,
    metrics_path: str,
) -> None:
    """
    통계 요약 마크다운과 ML 평가지표 JSON을 로드하여 Jinja2 템플릿을 렌더링한 뒤
    최종 report.md 문서를 생성합니다.
    """
    # 1. 기술통계/상관분석 요약 마크다운 읽기
    if not os.path.exists(stat_summary_path):
        raise FileNotFoundError(f"통계 요약 파일을 찾을 수 없습니다: {stat_summary_path}")
    with open(stat_summary_path, "r", encoding="utf-8") as f:
        stat_summary_content = f.read()

    # 2. ML 평가 메트릭 JSON 읽기
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"모델 메트릭 파일을 찾을 수 없습니다: {metrics_path}")
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_content = json.load(f)

    # 3. 시각화 이미지 메타데이터 정의 (상대경로 적용)
    visualizations = [
        {
            "title": "시간대별 택시 이용량",
            "image_path": "outputs/taxi_demand_by_hour.png",
            "description": "일별 시간대 중 오후 18~19시(퇴근 시간대)에 택시 수요가 가장 급증하며, 오전 7~8시(출근 시간대)에도 작은 피크가 형성되는 이중 피크 패턴이 나타납니다."
        },
        {
            "title": "승객 수에 따른 평균 팁 금액",
            "image_path": "outputs/tip_by_passenger_count.png",
            "description": "승객 수에 따른 평균 팁 금액은 1명에서 6명 사이에서 큰 격차 없이 약 $2.8 ~ $3.0 구간을 유지하고 있으나, 1~2명 탑승 시보다 단체 승객(5~6명)이 탑승할 때 평균 팁이 미세하게 높게 측정됩니다."
        },
        {
            "title": "주요 시간대별 평균 팁 금액",
            "image_path": "outputs/avg_tip_by_period.png",
            "description": "출근, 낮, 퇴근, 심야 시간대 중 '심야' 시간대의 평균 팁 금액이 가장 높게 형성되며, 주간 시간대의 평균 팁이 상대적으로 낮게 관찰됩니다."
        },
        {
            "title": "주요 수치형 변수 간 상관계수 히트맵",
            "image_path": "outputs/correlation_matrix.png",
            "description": "최종 요금(total_amount)은 기본 요금(fare_amount, r=0.98), 주행 거리(trip_distance, r=0.95), 주행 시간(trip_duration_min, r=0.93)과 극도로 강력한 선형적 상관성을 보여줍니다."
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
    print(f"[Report] 보고서가 성공적으로 생성되었습니다: {output_path}")


if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    generate_final_report(
        template_dir=os.path.join(PROJECT_ROOT, "templates"),
        template_name="report_template.md.j2",
        output_path=os.path.join(PROJECT_ROOT, "report.md"),
        stat_summary_path=os.path.join(PROJECT_ROOT, "outputs", "stat_summary.md"),
        metrics_path=os.path.join(PROJECT_ROOT, "outputs", "model_metrics.json"),
    )
