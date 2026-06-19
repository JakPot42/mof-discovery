from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import func, select

from config import DEMO_MODE, TOP_K_DEFAULT
from models import MOFStructure, init_db, get_db
from seed_data import load_seed_data
from ranker import UseCase, USE_CASE_LABELS, USE_CASE_DESCRIPTIONS, USE_CASE_KEY_METRIC, rank
from explainer import explain


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    db = next(get_db())
    load_seed_data(db)
    db.close()
    yield


app = FastAPI(title="MOF Discovery Query Tool", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


def _get_all_mofs(db: Session) -> list[dict]:
    rows = db.execute(select(MOFStructure)).scalars().all()
    return [r.to_dict() for r in rows]


@app.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    use_case: str | None = Query(default=None),
    context: str | None = Query(default=None),
    top_k: int = Query(default=TOP_K_DEFAULT, ge=3, le=50),
    db: Session = Depends(get_db),
):
    results = None
    explanation = None
    selected_use_case = None
    error = None

    if use_case:
        try:
            selected_use_case = UseCase(use_case)
        except ValueError:
            error = f"Unknown use case: {use_case}"

        if selected_use_case:
            mofs = _get_all_mofs(db)
            results = rank(mofs, selected_use_case, top_k=top_k)
            explanation = explain(selected_use_case, context or "", results)

    total_mofs = db.execute(select(func.count()).select_from(MOFStructure)).scalar() or 0

    return templates.TemplateResponse(request, "index.html", {
        "use_cases": {uc.value: USE_CASE_LABELS[uc] for uc in UseCase},
        "use_case_descriptions": {uc.value: USE_CASE_DESCRIPTIONS[uc] for uc in UseCase},
        "key_metrics": {uc.value: USE_CASE_KEY_METRIC[uc] for uc in UseCase},
        "selected_use_case": selected_use_case.value if selected_use_case else None,
        "context": context or "",
        "top_k": top_k,
        "results": results,
        "explanation": explanation,
        "error": error,
        "total_mofs": total_mofs,
        "demo_mode": DEMO_MODE,
    })


@app.get("/mof/{name}", response_class=HTMLResponse)
async def mof_detail(request: Request, name: str, db: Session = Depends(get_db)):
    row = db.execute(
        select(MOFStructure).where(MOFStructure.name == name)
    ).scalar_one_or_none()
    if row is None:
        return HTMLResponse("<h2>MOF not found</h2>", status_code=404)
    return templates.TemplateResponse(request, "detail.html", {"mof": row.to_dict()})


@app.get("/api/stats")
async def stats(db: Session = Depends(get_db)):
    total = db.execute(select(func.count()).select_from(MOFStructure)).scalar() or 0
    with_water = db.execute(
        select(func.count()).select_from(MOFStructure)
        .where(MOFStructure.water_uptake_g_g_20rh.isnot(None))
    ).scalar() or 0
    with_co2 = db.execute(
        select(func.count()).select_from(MOFStructure)
        .where(MOFStructure.co2_uptake_mmol_g_015bar.isnot(None))
    ).scalar() or 0
    with_h2 = db.execute(
        select(func.count()).select_from(MOFStructure)
        .where(MOFStructure.h2_uptake_wt_1bar_77k.isnot(None))
    ).scalar() or 0
    with_ch4 = db.execute(
        select(func.count()).select_from(MOFStructure)
        .where(MOFStructure.ch4_uptake_cm3_cm3_35bar.isnot(None))
    ).scalar() or 0
    return JSONResponse({
        "total_mofs": total,
        "with_water_data_20rh": with_water,
        "with_co2_data_015bar": with_co2,
        "with_h2_data_1bar_77k": with_h2,
        "with_ch4_data_35bar": with_ch4,
        "demo_mode": DEMO_MODE,
    })
