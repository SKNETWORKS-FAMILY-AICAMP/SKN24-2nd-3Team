import platform
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


_FONT_MAP = {
    "Darwin": ["AppleGothic", "Apple SD Gothic Neo"],
    "Windows": ["Malgun Gothic", "맑은 고딕"],
    "Linux": ["NanumGothic", "NanumBarunGothic", "DejaVu Sans"],
}


def _find_available_font(candidates):
    """설치된 폰트 중 후보 목록에서 첫 번째로 찾은 폰트명을 반환."""
    installed = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in installed:
            return font
    return None


def set_korean_font():
    """
    OS에 맞는 한글 matplotlib 폰트를 자동으로 설정

    - macOS  : AppleGothic → Apple SD Gothic Neo
    - Windows: Malgun Gothic → 맑은 고딕
    - Linux  : NanumGothic → NanumBarunGothic → DejaVu Sans

    Returns:
        설정된 폰트 이름 (설정 실패 시 'sans-serif')
    """
    os_name = platform.system()
    candidates = _FONT_MAP.get(os_name, [])

    font = _find_available_font(candidates)
    if font is None:
        font = "sans-serif"
        print(f"[PLOT_CONFIG] Cannot find Korean font. Using '{font}' (Korean may be broken)")
    else:
        print(f"[PLOT_CONFIG] OS={os_name!r} → '{font}' font will be used for Korean text.")

    plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False
    return font
