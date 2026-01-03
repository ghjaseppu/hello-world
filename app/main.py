from datetime import date, time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator

from .compute import AstroEngine, ComputeParams, DemManager, compute_alignment

app = FastAPI(title="Lunar Alignment Explorer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

astro_engine = AstroEngine()
dem_manager = DemManager()


class ComputeRequest(BaseModel):
    latitude: float = Field(..., description="Observer latitude in degrees")
    longitude: float = Field(..., description="Observer longitude in degrees")
    altitude_m: float = Field(0, description="Ground altitude of observer in meters")
    date: date = Field(..., description="Local date for computation")
    start_time: time = Field(..., description="Local start time")
    end_time: time = Field(..., description="Local end time (can be earlier than start to pass midnight)")
    timezone_offset_minutes: int = Field(0, description="Local timezone offset versus UTC, in minutes")
    step_seconds: int = Field(600, description="Time step in seconds")
    max_distance_km: float = Field(100.0, description="Maximum search distance from observer in km")
    distance_step_m: float = Field(500.0, description="Step between candidates along anti-azimuth line")
    tolerance_deg: float = Field(0.5, description="Maximum angular separation in degrees")
    observer_height_m: float = Field(1.7, description="Height of observer above ground in meters")
    target_extra_height_m: float = Field(0.0, description="Additional height applied at candidate points")
    use_dem_los: bool = Field(False, description="Whether to apply DEM-based line-of-sight checks")
    require_moon_above_horizon: bool = Field(
        False, description="If true, keep only candidates where the Moon is above the horizon"
    )

    @validator("step_seconds", "max_distance_km", "distance_step_m")
    def ensure_positive(cls, v):  # type: ignore[override]
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @validator("tolerance_deg")
    def ensure_tolerance(cls, v):  # type: ignore[override]
        if v <= 0:
            raise ValueError("Tolerance must be positive")
        return v


@app.get("/")
def index() -> FileResponse:
    return FileResponse("frontend/index.html")


@app.post("/api/compute")
def compute(request: ComputeRequest):
    params = ComputeParams(
        latitude=request.latitude,
        longitude=request.longitude,
        altitude_m=request.altitude_m,
        date_value=request.date,
        start_time=request.start_time,
        end_time=request.end_time,
        timezone_offset_minutes=request.timezone_offset_minutes,
        step_seconds=request.step_seconds,
        max_distance_km=request.max_distance_km,
        distance_step_m=request.distance_step_m,
        tolerance_deg=request.tolerance_deg,
        observer_height_m=request.observer_height_m,
        target_extra_height_m=request.target_extra_height_m,
        use_dem_los=request.use_dem_los,
        require_moon_above_horizon=request.require_moon_above_horizon,
    )

    if params.use_dem_los and not dem_manager.has_dem():
        raise HTTPException(status_code=400, detail="DEM requested but no GeoTIFF found in data/dem")

    feature_collection = compute_alignment(astro_engine, dem_manager, params)
    return feature_collection
