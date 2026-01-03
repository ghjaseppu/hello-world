import math
import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from pyproj import Geod, Transformer
from rasterio.errors import RasterioIOError
from skyfield.api import Loader, wgs84

# Global geodetic helpers
GEOD = Geod(ellps="WGS84")
TRANSFORMER_GEO_TO_ECEF = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
TRANSFORMER_ECEF_TO_ENU_CACHE: Dict[Tuple[float, float], Transformer] = {}


@dataclass
class DemResource:
    dataset: rasterio.io.DatasetReader

    def sample_elevation(self, lon: float, lat: float) -> Optional[float]:
        try:
            for val in self.dataset.sample([(lon, lat)]):
                return float(val[0])
        except Exception:
            return None
        return None


@dataclass
class AlignmentResult:
    position: Tuple[float, float, float]
    separation_deg: float
    moon_alt_deg: float
    note: str


class AstroEngine:
    def __init__(self) -> None:
        # Skyfield loader caches ephemerides under ~/.skyfield by default
        self._loader = Loader(os.path.expanduser("~/.skyfield"))
        self.ts = self._loader.timescale()
        self.ephemeris = self._loader("de421.bsp")
        self.moon = self.ephemeris["moon"]

    def moon_alt_az(
        self, lat: float, lon: float, elevation_m: float, timestamp: datetime
    ) -> Tuple[float, float]:
        sf_time = self.ts.from_datetime(timestamp)
        location = wgs84.latlon(latitude_degrees=lat, longitude_degrees=lon, elevation_m=elevation_m)
        difference = location.at(sf_time).observe(self.moon).apparent()
        alt, az, _ = difference.altaz()
        return alt.degrees, az.degrees


class DemManager:
    def __init__(self, dem_dir: str = "data/dem") -> None:
        self.dem_dir = dem_dir
        self.dem: Optional[DemResource] = None
        self._load_dem()

    def _load_dem(self) -> None:
        if not os.path.isdir(self.dem_dir):
            return
        for fname in os.listdir(self.dem_dir):
            if fname.lower().endswith((".tif", ".tiff")):
                path = os.path.join(self.dem_dir, fname)
                try:
                    dataset = rasterio.open(path)
                    self.dem = DemResource(dataset=dataset)
                    return
                except RasterioIOError:
                    continue

    def has_dem(self) -> bool:
        return self.dem is not None

    def elevation(self, lon: float, lat: float) -> Optional[float]:
        if not self.dem:
            return None
        return self.dem.sample_elevation(lon, lat)


@dataclass
class ComputeParams:
    latitude: float
    longitude: float
    altitude_m: float
    date_value: date
    start_time: time
    end_time: time
    timezone_offset_minutes: int
    step_seconds: int
    max_distance_km: float
    distance_step_m: float
    tolerance_deg: float
    observer_height_m: float
    target_extra_height_m: float
    use_dem_los: bool
    require_moon_above_horizon: bool


@dataclass
class TimeEntry:
    timestamp_utc: datetime
    timestamp_local: datetime


def build_time_range(params: ComputeParams) -> List[TimeEntry]:
    tzinfo = timezone(timedelta(minutes=params.timezone_offset_minutes))
    start_dt_local = datetime.combine(params.date_value, params.start_time, tzinfo=tzinfo)
    end_dt_local = datetime.combine(params.date_value, params.end_time, tzinfo=tzinfo)
    if end_dt_local <= start_dt_local:
        end_dt_local += timedelta(days=1)

    entries: List[TimeEntry] = []
    current = start_dt_local
    step = timedelta(seconds=params.step_seconds)
    while current <= end_dt_local:
        entries.append(
            TimeEntry(timestamp_utc=current.astimezone(timezone.utc), timestamp_local=current)
        )
        current += step
    return entries


def az_alt_from_vector(lat: float, lon: float, vector_ecef: Tuple[float, float, float]) -> Tuple[float, float]:
    # Convert vector expressed in ECEF (target - origin) into ENU and derive azimuth/altitude.
    x, y, z = vector_ecef
    # Calculate rotation matrix components for ENU
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    sin_lat, cos_lat = math.sin(lat_rad), math.cos(lat_rad)
    sin_lon, cos_lon = math.sin(lon_rad), math.cos(lon_rad)

    east = -sin_lon * x + cos_lon * y
    north = -sin_lat * cos_lon * x - sin_lat * sin_lon * y + cos_lat * z
    up = cos_lat * cos_lon * x + cos_lat * sin_lon * y + sin_lat * z

    horizontal_dist = math.hypot(east, north)
    azimuth = (math.degrees(math.atan2(east, north)) + 360.0) % 360.0
    altitude = math.degrees(math.atan2(up, horizontal_dist))
    return azimuth, altitude


def spherical_separation(az1: float, alt1: float, az2: float, alt2: float) -> float:
    v1 = np.array([
        math.cos(math.radians(alt1)) * math.sin(math.radians(az1)),
        math.cos(math.radians(alt1)) * math.cos(math.radians(az1)),
        math.sin(math.radians(alt1)),
    ])
    v2 = np.array([
        math.cos(math.radians(alt2)) * math.sin(math.radians(az2)),
        math.cos(math.radians(alt2)) * math.cos(math.radians(az2)),
        math.sin(math.radians(alt2)),
    ])
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def ecef_from_llh(lon: float, lat: float, h: float) -> Tuple[float, float, float]:
    x, y, z = TRANSFORMER_GEO_TO_ECEF.transform(lon, lat, h)
    return float(x), float(y), float(z)


def line_of_sight_clear(
    dem: DemManager,
    origin: Tuple[float, float, float],
    target: Tuple[float, float, float],
    sample_spacing_m: float = 90.0,
) -> bool:
    if not dem.has_dem():
        return True

    lon_o, lat_o, alt_o = origin
    lon_t, lat_t, alt_t = target
    _, _, dist_m = GEOD.inv(lon_o, lat_o, lon_t, lat_t)
    if dist_m <= 0:
        return True

    steps = max(1, int(dist_m / sample_spacing_m))
    samples = GEOD.npts(lon_o, lat_o, lon_t, lat_t, steps - 1)
    coords = [(lon_o, lat_o)] + samples + [(lon_t, lat_t)]

    for idx, (lon, lat) in enumerate(coords):
        fraction = idx / max(len(coords) - 1, 1)
        line_alt = alt_o + fraction * (alt_t - alt_o)
        dem_alt = dem.elevation(lon, lat)
        if dem_alt is None:
            continue
        if dem_alt > line_alt:
            return False
    return True


def choose_candidate_for_time(
    engine: AstroEngine,
    dem: DemManager,
    params: ComputeParams,
    time_entry: TimeEntry,
) -> Optional[AlignmentResult]:
    # Observer details
    obs_lat = params.latitude
    obs_lon = params.longitude
    obs_ground = params.altitude_m
    if params.use_dem_los and dem.has_dem():
        dem_elev = dem.elevation(obs_lon, obs_lat)
        if dem_elev is not None:
            obs_ground = dem_elev
    observer_alt = obs_ground + params.observer_height_m

    moon_alt_at_obs, moon_az_at_obs = engine.moon_alt_az(
        obs_lat, obs_lon, observer_alt, time_entry.timestamp_utc
    )
    anti_azimuth = (moon_az_at_obs + 180.0) % 360.0

    best: Optional[AlignmentResult] = None
    distance_m = params.distance_step_m
    max_distance_m = params.max_distance_km * 1000.0

    while distance_m <= max_distance_m:
        lon_p, lat_p, _ = GEOD.fwd(obs_lon, obs_lat, anti_azimuth, distance_m)
        ground_alt_p = dem.elevation(lon_p, lat_p) if dem.has_dem() else None
        ground_alt_p = ground_alt_p if ground_alt_p is not None else params.altitude_m
        alt_p = ground_alt_p + params.target_extra_height_m

        moon_alt_p, moon_az_p = engine.moon_alt_az(lat_p, lon_p, alt_p, time_entry.timestamp_utc)
        if params.require_moon_above_horizon and moon_alt_p <= 0:
            distance_m += params.distance_step_m
            continue

        # Direction from P to O
        ecef_p = ecef_from_llh(lon_p, lat_p, alt_p)
        ecef_o = ecef_from_llh(obs_lon, obs_lat, observer_alt)
        vector_po = (ecef_o[0] - ecef_p[0], ecef_o[1] - ecef_p[1], ecef_o[2] - ecef_p[2])
        dir_az, dir_alt = az_alt_from_vector(lat_p, lon_p, vector_po)

        separation = spherical_separation(dir_az, dir_alt, moon_az_p, moon_alt_p)

        if separation <= params.tolerance_deg:
            los_ok = True
            if params.use_dem_los and dem.has_dem():
                los_ok = line_of_sight_clear(
                    dem,
                    origin=(lon_p, lat_p, alt_p),
                    target=(obs_lon, obs_lat, observer_alt),
                    sample_spacing_m=max(params.distance_step_m, 30.0),
                )
            if los_ok:
                if best is None or separation < best.separation_deg:
                    best = AlignmentResult(
                        position=(lon_p, lat_p, alt_p),
                        separation_deg=separation,
                        moon_alt_deg=moon_alt_p,
                        note="aligned",
                    )
        distance_m += params.distance_step_m

    return best


def compute_alignment(engine: AstroEngine, dem: DemManager, params: ComputeParams) -> Dict:
    time_entries = build_time_range(params)
    features: List[Dict] = []
    line_coords: List[Tuple[float, float, float]] = []

    for entry in time_entries:
        candidate = choose_candidate_for_time(engine, dem, params, entry)
        if not candidate:
            continue
        lon, lat, alt = candidate.position
        line_coords.append((lon, lat, alt))
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat, alt]},
                "properties": {
                    "t_local": entry.timestamp_local.isoformat(),
                    "sep_deg": candidate.separation_deg,
                    "moon_alt_deg": candidate.moon_alt_deg,
                    "note": candidate.note,
                },
            }
        )

    if line_coords:
        features.insert(
            0,
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": line_coords},
                "properties": {"description": "Candidate trajectory"},
            },
        )

    return {"type": "FeatureCollection", "features": features}
