# ui.py
from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# 현재 파일 기준 절대 경로
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

router = APIRouter(tags=["UI"])
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

@router.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/ui")

@router.get("/ui", response_class=HTMLResponse)
def serve_ui(request: Request):
    # 템플릿에 주입할 값들(필요시 확장)
    context = {
        "request": request,
        "app_title": "PRISM Prediction",
        # 서버에서 베이스 경로를 바꾸고 싶다면 이런 값도 넘길 수 있어요
        "api_base": "",  # 같은 도메인/포트 사용 시 빈 문자열
    }
    return templates.TemplateResponse("ui.html", context)

def mount_static(app):
    # /static 경로로 정적 자원 서빙
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
