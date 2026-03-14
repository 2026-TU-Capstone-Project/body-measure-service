#!/usr/bin/env python3
"""
신체분석 서비스 현황 및 개선 계획 보고서 PDF 생성
"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── 한글 폰트 등록 ────────────────────────────────────────────────────────
_FONT_PATHS = [
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/Library/Fonts/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
]
_FONT_BOLD_PATHS = [
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/Library/Fonts/NanumGothicBold.ttf",
]

FONT_NAME      = "Korean"
FONT_NAME_BOLD = "KoreanBold"

for path in _FONT_PATHS:
    if os.path.exists(path):
        pdfmetrics.registerFont(TTFont(FONT_NAME, path))
        break

for path in _FONT_BOLD_PATHS:
    if os.path.exists(path):
        pdfmetrics.registerFont(TTFont(FONT_NAME_BOLD, path))
        break

# ── 색상 정의 ─────────────────────────────────────────────────────────────
COLOR_PRIMARY   = colors.HexColor("#1B3A6B")
COLOR_ACCENT    = colors.HexColor("#2E6FD4")
COLOR_GREEN     = colors.HexColor("#2D7D46")
COLOR_RED       = colors.HexColor("#C0392B")
COLOR_ORANGE    = colors.HexColor("#E67E22")
COLOR_LIGHT_BG  = colors.HexColor("#F4F6FA")
COLOR_TABLE_HDR = colors.HexColor("#2E6FD4")
COLOR_ROW_ALT   = colors.HexColor("#EBF2FF")

# ── 스타일 정의 ───────────────────────────────────────────────────────────
def make_styles():
    styles = getSampleStyleSheet()

    title = ParagraphStyle(
        "Title_KR", fontName=FONT_NAME_BOLD, fontSize=20,
        textColor=COLOR_PRIMARY, spaceAfter=4, leading=26, alignment=1,
    )
    subtitle = ParagraphStyle(
        "Subtitle_KR", fontName=FONT_NAME, fontSize=11,
        textColor=colors.HexColor("#555555"), spaceAfter=2, alignment=1,
    )
    h1 = ParagraphStyle(
        "H1_KR", fontName=FONT_NAME_BOLD, fontSize=14,
        textColor=COLOR_PRIMARY, spaceBefore=14, spaceAfter=5, leading=20,
        borderPad=4,
    )
    h2 = ParagraphStyle(
        "H2_KR", fontName=FONT_NAME_BOLD, fontSize=11,
        textColor=COLOR_ACCENT, spaceBefore=10, spaceAfter=4, leading=16,
    )
    body = ParagraphStyle(
        "Body_KR", fontName=FONT_NAME, fontSize=9.5,
        leading=16, spaceAfter=4, textColor=colors.HexColor("#333333"),
    )
    bullet = ParagraphStyle(
        "Bullet_KR", fontName=FONT_NAME, fontSize=9.5,
        leading=16, leftIndent=14, spaceAfter=2,
        textColor=colors.HexColor("#333333"), bulletIndent=4,
    )
    note = ParagraphStyle(
        "Note_KR", fontName=FONT_NAME, fontSize=8.5,
        leading=14, textColor=colors.HexColor("#666666"),
        leftIndent=10, spaceAfter=2,
    )
    summary = ParagraphStyle(
        "Summary_KR", fontName=FONT_NAME, fontSize=9.5,
        leading=16, leftIndent=10, rightIndent=10,
        textColor=COLOR_PRIMARY, backColor=COLOR_LIGHT_BG,
        borderPad=8, spaceBefore=6, spaceAfter=6,
    )
    return dict(
        title=title, subtitle=subtitle, h1=h1, h2=h2,
        body=body, bullet=bullet, note=note, summary=summary,
    )


def build_table(data, col_widths, header_row=True):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    style = [
        ("FONTNAME",    (0, 0), (-1, -1), FONT_NAME),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_ROW_ALT]),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    if header_row:
        style += [
            ("BACKGROUND",  (0, 0), (-1, 0), COLOR_TABLE_HDR),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), FONT_NAME_BOLD),
            ("FONTSIZE",    (0, 0), (-1, 0), 9.5),
            ("ALIGN",       (0, 0), (-1, 0), "CENTER"),
        ]
    t.setStyle(TableStyle(style))
    return t


def generate_pdf(output_path):
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        topMargin=20*mm, bottomMargin=20*mm,
        leftMargin=22*mm, rightMargin=22*mm,
    )
    S = make_styles()
    W = A4[0] - 44*mm   # 본문 너비

    story = []

    # ── 표지 영역 ────────────────────────────────────────────────────────
    story.append(Spacer(1, 12*mm))
    story.append(Paragraph("신체 분석 서비스 현황 및 개선 계획", S["title"]))
    story.append(Paragraph("Body Analysis Service — Limitation Report & Roadmap", S["subtitle"]))
    story.append(Spacer(1, 2*mm))
    story.append(HRFlowable(width="100%", thickness=1.5, color=COLOR_PRIMARY))
    story.append(Spacer(1, 6*mm))

    # ── 1. 현재 정확도 현황 ──────────────────────────────────────────────
    story.append(Paragraph("1. 현재 정확도 현황", S["h1"]))
    story.append(Paragraph(
        "두 인물(F009, F010)을 실측 데이터(GT)와 비교한 결과이며, "
        "온라인 의류 구매 서비스의 일반적인 허용 오차 기준은 ±2~3cm (약 3~5%) 입니다.",
        S["body"],
    ))
    story.append(Spacer(1, 3*mm))

    acc_data = [
        ["항목", "평균 오차율", "정확도", "실용 기준"],
        ["하의 총장",  "1.3%",  "98.7%", "✅  실용 가능"],
        ["가슴 둘레",  "4.6%",  "95.4%", "✅  실용 가능"],
        ["허벅지 둘레","6.0%",  "94.0%", "△  허용 범위"],
        ["상의 총장",  "18.8%", "81.2%", "❌  실용 불가"],
        ["어깨 너비",  "20.0%", "80.0%", "❌  실용 불가"],
        ["전체 평균",  "10.2%", "89.8%", "❌  미흡"],
    ]
    t = build_table(acc_data, [W*0.28, W*0.20, W*0.20, W*0.32])
    # 실용 불가 행 강조
    t.setStyle(TableStyle([
        ("TEXTCOLOR", (3, 4), (3, 4), COLOR_RED),
        ("TEXTCOLOR", (3, 5), (3, 5), COLOR_RED),
        ("TEXTCOLOR", (3, 6), (3, 6), COLOR_RED),
        ("FONTNAME",  (0, 6), (-1, 6), FONT_NAME_BOLD),
        ("BACKGROUND",(0, 6), (-1, 6), colors.HexColor("#FFF3CD")),
    ]))
    story.append(t)
    story.append(Spacer(1, 6*mm))

    # ── 2. 측면 사진 추가 재학습 불가 이유 ──────────────────────────────
    story.append(Paragraph("2. 측면 사진 추가 재학습이 당장 불가능한 이유", S["h1"]))

    story.append(Paragraph("① 기존 학습 데이터(NIA21)에 측면 사진이 없음", S["h2"]))
    bullets1 = [
        "NIA21 데이터셋은 정면 촬영 이미지만 포함",
        "측면 사진은 현재 테스트용으로 촬영한 2쌍만 존재",
        "딥러닝 학습을 위해서는 정면+측면 쌍으로 구성된 데이터가 필요하며, 이 형태의 데이터가 현재 없음",
    ]
    for b in bullets1:
        story.append(Paragraph(f"• {b}", S["bullet"]))

    story.append(Spacer(1, 3*mm))
    story.append(Paragraph("② 아키텍처는 완성됐으나 학습 데이터 구성이 병목", S["h2"]))
    bullets2 = [
        "DualBlazePose(정면+측면 이중 스트림 모델) 설계 및 구현은 완료된 상태",
        "side_encoder와 확장된 attention MLP는 정면+측면 쌍 데이터로 학습되어야만 의미 있는 결과를 냄",
        "현재는 해당 부분이 랜덤 초기화 상태이므로 단일뷰보다 오히려 성능이 낮음",
    ]
    for b in bullets2:
        story.append(Paragraph(f"• {b}", S["bullet"]))
    story.append(Spacer(1, 6*mm))

    # ── 3. 서비스가 완전하지 않은 이유 ──────────────────────────────────
    story.append(Paragraph("3. 현재 서비스가 완전하지 않은 이유", S["h1"]))

    story.append(Paragraph("① 어깨 너비의 구조적 오차 (평균 20%)", S["h2"]))
    bullets3 = [
        "모델이 학습한 '어깨너비'와 GT의 '어깨사이너비(견봉~견봉)' 측정 기준이 상이할 가능성",
        "정면 이미지만으로는 어깨 끝점의 3D 위치를 정확히 특정하기 어려움",
        "두 인물 모두 동일한 방향(+7cm 과대)으로 오차 발생 → 모델의 구조적 편향 확인",
    ]
    for b in bullets3:
        story.append(Paragraph(f"• {b}", S["bullet"]))

    story.append(Spacer(1, 2*mm))
    story.append(Paragraph("② 상의 총장의 누적 오차 (평균 18.8%)", S["h2"]))
    bullets4 = [
        "상의 총장 = 목뒤높이 − 엉덩이높이로 계산",
        "정면 사진만으로는 엉덩이 아랫선의 깊이(depth) 파악이 어려워 일관되게 약 7cm 과소 예측",
        "두 인물 모두 동일한 방향의 오차 → 구조적 편향임을 실험적으로 확인",
    ]
    for b in bullets4:
        story.append(Paragraph(f"• {b}", S["bullet"]))

    story.append(Spacer(1, 2*mm))
    story.append(Paragraph("③ 학습-서비스 환경 간 도메인 갭(Domain Gap)", S["h2"]))

    domain_data = [
        ["구분",         "학습 데이터 (NIA21)",  "실제 서비스 환경"],
        ["촬영 환경",    "스튜디오 통제 환경",    "일상 환경 (배경 무작위)"],
        ["의복 상태",    "맨몸 또는 속옷 착용",   "일반 의류 착용"],
        ["조명",         "전문 스튜디오 조명",    "자연광 / 실내 조명"],
    ]
    story.append(build_table(domain_data, [W*0.22, W*0.39, W*0.39]))
    story.append(Paragraph(
        "※ 위 환경 차이만으로도 실제 서비스 시 정확도가 유의미하게 저하될 수 있음",
        S["note"],
    ))
    story.append(Spacer(1, 6*mm))

    # ── 4. 향후 정밀도 향상 계획 ─────────────────────────────────────────
    story.append(Paragraph("4. 향후 정밀도 향상 계획", S["h1"]))

    story.append(Paragraph("단기 — 즉시 적용 가능", S["h2"]))
    short_data = [
        ["방법", "기대 효과"],
        ["편향 보정(Bias Correction) 적용",
         "어깨·상의총장의 일관된 과대 예측을 통계 보정으로 완화"],
        ["다양한 체형 인원으로 추가 검증",
         "보정 계수의 신뢰도 향상"],
    ]
    story.append(build_table(short_data, [W*0.42, W*0.58]))
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph("중기 — 측면 데이터 구성 후", S["h2"]))
    mid_data = [
        ["방법", "기대 효과"],
        ["기존 NIA21 인물 대상 측면 사진 추가 촬영",
         "정면+측면 쌍 데이터셋 구성 가능"],
        ["DualBlazePose 재학습",
         "side_encoder가 측면의 엉덩이 위치·체형 깊이 학습\n→ 상의 총장·어깨 오차 감소 기대"],
        ["Attention 입력에 허리둘레 반영",
         "체형 다양성 대응력 향상"],
    ]
    story.append(build_table(mid_data, [W*0.42, W*0.58]))
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph("장기 — 고도화 단계", S["h2"]))
    long_data = [
        ["방법", "기대 효과"],
        ["일상 환경 이미지로 도메인 적응(Domain Adaptation)",
         "스튜디오↔실환경 간 정확도 격차 해소"],
        ["다양한 체형·연령·성별 데이터 확장",
         "특정 체형 편향 제거"],
        ["3D 추정 모델 도입 검토",
         "깊이 정보를 직접 확보 → 엉덩이높이·둘레 계산의 근본적 개선"],
    ]
    story.append(build_table(long_data, [W*0.42, W*0.58]))
    story.append(Spacer(1, 6*mm))

    # ── 요약 ────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=COLOR_ACCENT))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph("종합 요약", S["h1"]))
    story.append(Paragraph(
        "현재 시스템은 하의 관련 치수(총장, 가슴)에서는 실용적인 수준의 정확도를 보이나, "
        "상의 총장과 어깨너비에서 구조적 오차가 발생하고 있습니다. "
        "근본 원인은 기존 학습 데이터(NIA21)가 정면 사진만으로 구성되어 있어 "
        "측면 정보를 활용할 수 없다는 점입니다. "
        "DualBlazePose 아키텍처는 이미 구현 완료된 상태이며, "
        "NIA21 인물들에 대한 측면 사진을 추가 구성하여 재학습하는 것이 "
        "정밀도 향상의 핵심 과제입니다.",
        S["summary"],
    ))

    doc.build(story)
    print(f"PDF 생성 완료: {output_path}")


if __name__ == "__main__":
    out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "신체분석_서비스_현황_및_개선계획.pdf",
    )
    generate_pdf(out)
