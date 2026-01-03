# Lunar Alignment Explorer

Application FastAPI + MapLibre pour explorer des trajectoires où la Lune est alignée avec un point d'observation O.

## Prérequis
- Python 3.10+
- Accès réseau (pour le premier téléchargement de l'éphéméride `de421.bsp` par skyfield)
- (Optionnel) Un GeoTIFF DEM placé dans `data/dem/` pour activer les tests de ligne de vue.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Démarrage
```bash
uvicorn app.main:app --reload
```

Ouvrez ensuite `http://127.0.0.1:8000` dans votre navigateur. L'interface MapLibre est servie directement par FastAPI.

## Données DEM (optionnel)
Déposez un fichier GeoTIFF (par ex. Copernicus DEM GLO-30, SRTM/NASADEM) dans `data/dem/`. Le premier fichier `.tif`/`.tiff` trouvé sera chargé. Cochez ensuite « Utiliser le GeoTIFF pour la ligne de vue » dans l'interface pour activer les masquages terrain.

## API
`POST /api/compute`

Payload JSON :
- `latitude`, `longitude` : position du point O (°)
- `altitude_m` : altitude du sol au point O (m)
- `observer_height_m` : hauteur de l'observateur au-dessus du sol (m)
- `target_extra_height_m` : hauteur ajoutée aux points P (m)
- `date` (YYYY-MM-DD), `start_time`, `end_time` (HH:MM) et `timezone_offset_minutes` (minutes est d'UTC) pour définir la fenêtre temporelle (si l'heure de fin est avant celle de début, on passe au lendemain)
- `step_seconds` : pas temporel (s)
- `max_distance_km` : distance maximale des points P par rapport à O (km)
- `distance_step_m` : pas de balayage le long de l'anti-azimut (m)
- `tolerance_deg` : tolérance de séparation angulaire (°)
- `use_dem_los` : activer la ligne de vue basée sur le DEM
- `require_moon_above_horizon` : filtrer les positions où la Lune est sous l'horizon

Réponse : `FeatureCollection` GeoJSON contenant une `LineString` de la trajectoire et un `Point` par timestamp accepté (propriétés `t_local`, `sep_deg`, `moon_alt_deg`, `note`).

## Notes sur le calcul
- Position et azimut/altitude lunaires calculés en topocentrique avec parallaxe via Skyfield (`de421.bsp`).
- Les candidats P sont échantillonnés le long de l'anti-azimut lunaire depuis O jusqu'à `max_distance_km` avec le pas `distance_step_m`, puis filtrés par la tolérance angulaire.
- La ligne de visée P→O utilise le DEM si présent et si `use_dem_los` est actif. Sans DEM, l'application fonctionne en 2D (aucun masquage terrain appliqué).
