from __future__ import annotations

import io
import math
import re
from datetime import date
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st


EA_WQ_BASE = "https://environment.data.gov.uk/water-quality"
EA_SAMPLING_POINT_URL = f"{EA_WQ_BASE}/data/sampling-point"
EA_OBSERVATION_URL = f"{EA_WQ_BASE}/data/observation"
EA_CRS = "http://www.opengis.net/def/crs/EPSG/0/27700"
MISSING_TEXT = {"", "nan", "none", "<na>", "not available", "not recorded", "unknown"}


def _first_existing(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _clean_text(value: object, fallback: str = "Not recorded") -> str:
    if value is None or pd.isna(value):
        return fallback
    text = str(value).strip()
    return fallback if text.casefold() in MISSING_TEXT else text


def _recorded_values(series: pd.Series) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for value in series.tolist():
        text = _clean_text(value, "")
        key = text.casefold()
        if text and key not in MISSING_TEXT and key not in seen:
            values.append(text)
            seen.add(key)
    return sorted(values, key=str.casefold)


def _api_csv(url: str, params: dict[str, object], *, timeout: int = 45) -> pd.DataFrame:
    """Request CSV from the EA Water Quality Explorer API.

    The current WQE API accepts POST requests with query parameters and no JSON body.
    A GET fallback is retained because the service has changed during its migration.
    """
    query = urlencode(params, doseq=True)
    full_url = f"{url}?{query}"
    headers = {
        "Accept": "text/csv",
        "Accept-Crs": EA_CRS,
        "CSV-Header": "present",
        "API-Version": "1",
        "User-Agent": "Storm-Overflow-Risk-Insights/1.0",
    }

    last_error: Exception | None = None
    for method in ("POST", "GET"):
        try:
            request = Request(
                full_url,
                data=b"" if method == "POST" else None,
                headers=headers,
                method=method,
            )
            with urlopen(request, timeout=timeout) as response:  # nosec B310 - fixed government host
                payload = response.read().decode("utf-8-sig", errors="replace")
            if not payload.strip():
                return pd.DataFrame()
            return pd.read_csv(io.StringIO(payload), dtype_backend="numpy_nullable")
        except (HTTPError, URLError, TimeoutError, ValueError, pd.errors.ParserError) as exc:
            last_error = exc
            if isinstance(exc, HTTPError) and exc.code not in {400, 404, 405, 406, 415, 422, 500, 502, 503, 504}:
                break

    raise RuntimeError(f"Environment Agency Water Quality Explorer request failed: {last_error}")


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def _sampling_points_near(latitude: float, longitude: float, radius_km: float) -> pd.DataFrame:
    params = {
        "latitude": round(float(latitude), 6),
        "longitude": round(float(longitude), 6),
        "radius": float(radius_km),
        "limit": 2500,
    }
    return _api_csv(EA_SAMPLING_POINT_URL, params)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def _historical_observations(
    point_notation: str,
    date_from: str,
    date_to: str,
    maximum_records: int,
) -> tuple[pd.DataFrame, bool]:
    """Paginate one official EA sampling point's observations."""
    page_size = 250
    skip = 0
    pages: list[pd.DataFrame] = []
    capped = False

    while skip < maximum_records:
        limit = min(page_size, maximum_records - skip)
        page = _api_csv(
            EA_OBSERVATION_URL,
            {
                "pointNotation": point_notation,
                "dateFrom": date_from,
                "dateTo": date_to,
                "limit": limit,
                "skip": skip,
                "complianceOnly": "false",
            },
        )
        if page.empty:
            break
        pages.append(page)
        received = len(page)
        skip += received
        if received < limit:
            break
        if skip >= maximum_records:
            capped = True
            break

    if not pages:
        return pd.DataFrame(), False
    return pd.concat(pages, ignore_index=True, sort=False), capped


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1 - a)))


def _parameter_category(name: object) -> str:
    text = _clean_text(name, "").casefold()
    if not text:
        return "Other / not classified"

    pathogen_terms = (
        "e. coli", "escherichia", "enterococc", "coliform", "salmonella",
        "clostrid", "campylobacter", "cryptospor", "giardia", "bacteria",
        "bacteri", "faecal", "fecal", "virus", "norovirus", "pathogen",
    )
    nutrient_terms = (
        "ammon", "nitrate", "nitrite", "nitrogen", "phosphate", "phosphorus",
        "orthophosphate", "bod", "biochemical oxygen demand", "chemical oxygen demand",
        "suspended solid", "total organic carbon", "chlorophyll",
    )
    metal_terms = (
        "lead", "mercury", "cadmium", "arsenic", "chromium", "nickel", "copper",
        "zinc", "iron", "manganese", "aluminium", "aluminum", "cobalt", "selenium",
        "silver", "tin", "vanadium", "antimony", "beryllium", "molybdenum",
    )
    organic_terms = (
        "pfas", "pfos", "pfoa", "pesticide", "herbicide", "fungicide", "insecticide",
        "hydrocarbon", "benzene", "toluene", "xylene", "phenol", "pcb", "dioxin",
        "voc", "solvent", "glyphosate", "atrazine", "microplastic", "pharmaceutical",
    )
    physical_terms = (
        "temperature", "ph", "conductivity", "conductance", "oxygen", "turbidity",
        "salinity", "alkalinity", "hardness", "colour", "color", "chloride", "sulphate",
        "sulfate", "calcium", "magnesium", "sodium", "potassium",
    )

    if any(term in text for term in pathogen_terms):
        return "Pathogen / microbiological"
    if any(term in text for term in metal_terms):
        return "Metals / inorganic contaminants"
    if any(term in text for term in organic_terms):
        return "Organic / emerging contaminants"
    if any(term in text for term in nutrient_terms):
        return "Nutrients / sewage indicators"
    if any(term in text for term in physical_terms):
        return "Physical / general chemistry"
    return "Other measured determinands"


def _normalise_observations(raw: pd.DataFrame, selected_receiving_water: str) -> pd.DataFrame:
    if raw.empty:
        return raw

    frame = raw.copy()
    date_col = _first_existing(frame, ["phenomenonTime", "measurement_datetime", "measurement_date", "sample_date"])
    station_id_col = _first_existing(frame, ["samplingPoint.notation", "sampling_point_id", "pointNotation"])
    station_name_col = _first_existing(frame, ["samplingPoint.prefLabel", "sampling_point_name", "samplingPoint.label"])
    det_code_col = _first_existing(frame, ["determinand.notation", "determinand_code", "det_id"])
    det_name_col = _first_existing(frame, ["determinand.prefLabel", "project_parameter_name", "determinand_name", "parameter"])
    result_col = _first_existing(frame, ["result", "result_as_reported", "exact_numeric_result", "result_numeric"])
    unit_col = _first_existing(frame, ["unit", "reported_unit"])
    purpose_col = _first_existing(frame, ["samplingPurpose", "sampling_purpose"])
    material_col = _first_existing(frame, ["sampleMaterialType", "sample_material_type", "material_type"])

    normalised = pd.DataFrame(index=frame.index)
    normalised["Measurement date"] = pd.to_datetime(frame[date_col], errors="coerce") if date_col else pd.NaT
    normalised["Year"] = normalised["Measurement date"].dt.year.astype("Int64")
    normalised["Monitoring station ID"] = frame[station_id_col].astype("string") if station_id_col else ""
    normalised["Monitoring station"] = frame[station_name_col].astype("string") if station_name_col else normalised["Monitoring station ID"]
    normalised["Determinand code"] = frame[det_code_col].astype("string") if det_code_col else ""
    normalised["Determinand / parameter"] = frame[det_name_col].astype("string") if det_name_col else "Not recorded"
    normalised["Reported result"] = frame[result_col].astype("string") if result_col else ""
    normalised["Unit"] = frame[unit_col].astype("string") if unit_col else ""
    normalised["Sampling purpose"] = frame[purpose_col].astype("string") if purpose_col else ""
    normalised["Sample material"] = frame[material_col].astype("string") if material_col else ""
    normalised["Category"] = normalised["Determinand / parameter"].map(_parameter_category)

    numeric_text = normalised["Reported result"].astype("string").str.extract(
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False
    )
    normalised["Numeric result"] = pd.to_numeric(numeric_text, errors="coerce")
    normalised["Qualifier"] = normalised["Reported result"].astype("string").str.extract(r"^\s*([<>≤≥])", expand=False)

    coded = normalised["Reported result"].fillna("").astype(str).str.casefold()
    normalised["Coded detection"] = coded.map(
        lambda value: "Present / detected"
        if any(term in value for term in ("present", "detected", "positive"))
        else ("Not found / negative" if any(term in value for term in ("not found", "absent", "negative")) else "")
    )

    water_key = re.sub(r"[^a-z0-9]+", " ", selected_receiving_water.casefold()).strip()
    water_tokens = {token for token in water_key.split() if len(token) >= 4 and token not in {"river", "brook", "stream", "water", "canal"}}

    def relationship(station: object) -> str:
        station_key = re.sub(r"[^a-z0-9]+", " ", _clean_text(station, "").casefold()).strip()
        station_tokens = set(station_key.split())
        if water_tokens and water_tokens.intersection(station_tokens):
            return "Named-water match at an EA monitoring station"
        return "Nearby EA monitoring station — proximity evidence"

    normalised["Evidence relationship"] = normalised["Monitoring station"].map(relationship)
    normalised["Source"] = "Environment Agency Water Quality Explorer"
    return normalised


def _summary_by_determinand(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    keys = ["Determinand / parameter", "Determinand code", "Category", "Unit"]
    working = frame.copy()

    def year_list(series: pd.Series) -> str:
        years = sorted({int(v) for v in series.dropna().tolist()})
        return ", ".join(map(str, years))

    def stations(series: pd.Series) -> str:
        vals = _recorded_values(series)
        return "; ".join(vals[:6]) + (f"; +{len(vals)-6} more" if len(vals) > 6 else "")

    summary = (
        working.groupby(keys, dropna=False)
        .agg(
            **{
                "First recorded year": ("Year", "min"),
                "Latest recorded year": ("Year", "max"),
                "Years recorded": ("Year", year_list),
                "Observations": ("Year", "size"),
                "Minimum numeric result": ("Numeric result", "min"),
                "Median numeric result": ("Numeric result", "median"),
                "Maximum numeric result": ("Numeric result", "max"),
                "Coded detections": ("Coded detection", lambda s: int((s == "Present / detected").sum())),
                "Monitoring stations": ("Monitoring station", stations),
            }
        )
        .reset_index()
        .sort_values(["Category", "Determinand / parameter"], kind="stable")
    )
    return summary


def _load_local_quality() -> pd.DataFrame:
    path = Path("data/water_quality_records.csv.gz")
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, compression="gzip", low_memory=False)
    except Exception:
        return pd.DataFrame()


def render_receiving_water_history_explorer(
    high_risk_2025: pd.DataFrame,
    *,
    company_column: str | None = None,
    place_column: str | None = None,
) -> None:
    """Interactive drill-down beneath the 2025 High-risk relationship flow."""
    if high_risk_2025 is None or high_risk_2025.empty:
        return

    frame = high_risk_2025.copy()
    company_col = company_column if company_column in frame.columns else _first_existing(
        frame, ["water_company_name", "water_company", "company"]
    )
    place_col = place_column if place_column in frame.columns else _first_existing(
        frame, ["town_city", "town_or_city", "nearest_town", "place_name", "town"]
    )
    treatment_col = _first_existing(
        frame,
        [
            "_priority_treatment_site",
            "wasc_site_name",
            "source_site_name_wasc_operational",
            "source_site_name_wasc_operational_optional",
        ],
    )
    receiving_col = _first_existing(
        frame,
        [
            "_priority_receiving_water",
            "receiving_water",
            "source_receiving_water",
            "source_receiving_water_environment_common_name_ea_consents_database",
        ],
    )
    outlet_col = _first_existing(
        frame,
        ["_priority_discharge_site", "site_name", "outlet_name", "source_site_name", "asset_name"],
    )
    permit_col = _first_existing(frame, ["permit_reference", "permit_ref", "permit_number"])
    lat_col = _first_existing(frame, ["latitude", "lat"])
    lon_col = _first_existing(frame, ["longitude", "lon", "long"])

    if not treatment_col or not receiving_col:
        st.info("Treatment-site or receiving-water names are not available for these records.")
        return

    st.markdown("### 🔎 Treatment site & receiving-water evidence explorer")
    st.caption(
        "Choose a 2025 High-risk relationship to see the named operational/treatment site, linked outlets and receiving water. "
        "You can then load official Environment Agency historical water-quality observations from 2000 onwards."
    )

    filter_cols = st.columns(3)
    working = frame.copy()

    with filter_cols[0]:
        company_options = _recorded_values(working[company_col]) if company_col else []
        selected_company = st.selectbox(
            "Water company",
            ["All companies", *company_options],
            key="priority_history_company",
        )
    if company_col and selected_company != "All companies":
        working = working.loc[working[company_col].astype(str).str.strip().eq(selected_company)].copy()

    treatment_options = _recorded_values(working[treatment_col])
    with filter_cols[1]:
        selected_treatment = st.selectbox(
            "Treatment / operational site",
            ["All treatment sites", *treatment_options],
            key="priority_history_treatment",
        )
    if selected_treatment != "All treatment sites":
        working = working.loc[working[treatment_col].astype(str).str.strip().eq(selected_treatment)].copy()

    receiving_options = _recorded_values(working[receiving_col])
    with filter_cols[2]:
        selected_water = st.selectbox(
            "Receiving water",
            ["Choose a receiving water", *receiving_options],
            key="priority_history_receiving_water",
        )

    if selected_water == "Choose a receiving water":
        st.info("Select a receiving water to open its linked-site and historical evidence view.")
        return

    selected = working.loc[working[receiving_col].astype(str).str.strip().eq(selected_water)].copy()
    if selected.empty:
        st.info("No 2025 High-risk outlet is linked to that receiving-water selection.")
        return

    display_columns: list[str] = []
    rename: dict[str, str] = {}
    for col, label in [
        (company_col, "Water company"),
        (treatment_col, "Treatment / operational site"),
        (outlet_col, "Storm-overflow outlet / site"),
        (receiving_col, "Receiving water"),
        (place_col, "Town / city"),
        (permit_col, "Permit reference"),
    ]:
        if col and col in selected.columns and col not in display_columns:
            display_columns.append(col)
            rename[col] = label

    st.markdown("#### Linked 2025 High-risk sites")
    if display_columns:
        linked_table = selected[display_columns].copy().rename(columns=rename).drop_duplicates()
        st.dataframe(linked_table, use_container_width=True, hide_index=True)
        st.download_button(
            "Download linked sites",
            data=linked_table.to_csv(index=False).encode("utf-8"),
            file_name="selected_high_risk_treatment_receiving_water_links.csv",
            mime="text/csv",
            key="download_selected_priority_history_links",
        )

    st.markdown("#### Historical water-quality evidence")
    st.info(
        "This is documented Environment Agency monitoring evidence, not proof that a storm overflow caused a result. "
        "A nearby monitoring station may be upstream or downstream and may reflect other pollution sources. "
        "The Water Quality Explorer archive covers 2000 onwards, so this should not be described as every test ever carried out."
    )

    if not lat_col or not lon_col:
        st.warning("Coordinates are not available for this selected receiving-water link, so nearby EA monitoring stations cannot be matched safely.")
        return

    coordinates = selected[[lat_col, lon_col]].copy()
    coordinates[lat_col] = pd.to_numeric(coordinates[lat_col], errors="coerce")
    coordinates[lon_col] = pd.to_numeric(coordinates[lon_col], errors="coerce")
    coordinates = coordinates.dropna()
    if coordinates.empty:
        st.warning("No valid mapped coordinates are available for the selected outlet(s).")
        return

    centre_lat = float(coordinates[lat_col].median())
    centre_lon = float(coordinates[lon_col].median())

    control_cols = st.columns([1, 1, 1])
    with control_cols[0]:
        radius_km = st.selectbox(
            "Monitoring-station search radius",
            [0.5, 1.0, 2.0, 5.0],
            index=2,
            format_func=lambda v: f"{v:g} km",
            key="priority_history_radius",
        )
    with control_cols[1]:
        start_year = st.number_input(
            "From year",
            min_value=2000,
            max_value=date.today().year,
            value=2000,
            step=1,
            key="priority_history_start_year",
        )
    with control_cols[2]:
        record_cap = st.selectbox(
            "Maximum observations to load",
            [2500, 5000, 10000, 25000],
            index=2,
            format_func=lambda value: f"{value:,}",
            key="priority_history_record_cap",
            help="Higher limits may take longer for long-running monitoring stations.",
        )

    try:
        with st.spinner("Finding official EA water-quality monitoring stations near this receiving water…"):
            stations = _sampling_points_near(centre_lat, centre_lon, float(radius_km))
    except Exception as exc:
        st.warning(f"EA Water Quality Explorer is temporarily unavailable for this location: {exc}")
        local_quality = _load_local_quality()
        if not local_quality.empty:
            st.caption("The dashboard still contains its packaged 2025 water-quality extract; use the Water quality section while the live archive is unavailable.")
        return

    if stations.empty:
        st.warning(f"No EA water-quality sampling points were returned within {radius_km:g} km of the mapped outlet location.")
        return

    station_id_col = _first_existing(stations, ["notation", "samplingPoint.notation", "sampling_point_id", "id"])
    station_name_col = _first_existing(stations, ["prefLabel", "samplingPoint.prefLabel", "label", "sampling_point_name"])
    station_lat_col = _first_existing(stations, ["latitude", "lat"])
    station_lon_col = _first_existing(stations, ["longitude", "long", "lon"])

    if not station_id_col:
        st.warning("EA sampling-point response did not contain a usable monitoring-station identifier.")
        return

    station_view = stations.copy()
    station_view["_station_id"] = station_view[station_id_col].astype(str).str.strip()
    station_view["_station_name"] = (
        station_view[station_name_col].astype(str).str.strip()
        if station_name_col
        else station_view["_station_id"]
    )

    if station_lat_col and station_lon_col:
        station_view[station_lat_col] = pd.to_numeric(station_view[station_lat_col], errors="coerce")
        station_view[station_lon_col] = pd.to_numeric(station_view[station_lon_col], errors="coerce")
        station_view["_distance_km"] = station_view.apply(
            lambda row: _haversine_km(centre_lat, centre_lon, float(row[station_lat_col]), float(row[station_lon_col]))
            if pd.notna(row[station_lat_col]) and pd.notna(row[station_lon_col])
            else pd.NA,
            axis=1,
        )
    else:
        station_view["_distance_km"] = pd.NA

    station_view = station_view.loc[station_view["_station_id"].ne("")].drop_duplicates("_station_id")
    station_view = station_view.sort_values("_distance_km", na_position="last")
    if station_view.empty:
        st.warning("No usable EA monitoring-station identifiers were returned for this location.")
        return

    station_labels: dict[str, str] = {}
    for _, row in station_view.head(50).iterrows():
        station_id = row["_station_id"]
        station_name = _clean_text(row["_station_name"], station_id)
        distance = row.get("_distance_km")
        label = f"{station_name} — {station_id}"
        if pd.notna(distance):
            label += f" ({float(distance):.2f} km from mapped outlet)"
        station_labels[label] = station_id

    selected_station_label = st.selectbox(
        "EA monitoring station",
        list(station_labels.keys()),
        key="priority_history_station",
        help="Choose the station whose official historical observations you want to inspect.",
    )
    point_notation = station_labels[selected_station_label]

    if not st.button(
        "Load historical observations",
        type="primary",
        key="load_priority_history_observations",
        help="Loads official Environment Agency observations for the selected station from the chosen year to today.",
    ):
        st.caption("Historical observations are loaded only when requested so the presentation dashboard stays fast.")
        return

    date_from = f"{int(start_year):04d}-01-01"
    date_to = date.today().isoformat()

    try:
        with st.spinner(f"Loading official EA observations for {point_notation} from {start_year} to today…"):
            raw_history, capped = _historical_observations(point_notation, date_from, date_to, int(record_cap))
    except Exception as exc:
        st.error(f"Historical observations could not be loaded from the EA archive: {exc}")
        return

    if raw_history.empty:
        st.info("No observations were returned for the selected monitoring station and period.")
        return

    history = _normalise_observations(raw_history, selected_water)
    history = history.loc[history["Measurement date"].notna()].copy()
    if history.empty:
        st.info("The EA response contained no dated observations that could be displayed.")
        return

    if capped:
        st.warning(
            f"The display reached the selected {int(record_cap):,}-observation cap. "
            "Increase the limit if you need the remaining archive records for this station."
        )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Observations", f"{len(history):,}")
    metric_cols[1].metric("Determinand types", f"{history['Determinand / parameter'].nunique():,}")
    metric_cols[2].metric("Years represented", f"{history['Year'].nunique():,}")
    metric_cols[3].metric("First–latest", f"{int(history['Year'].min())}–{int(history['Year'].max())}")

    category_options = sorted(history["Category"].dropna().astype(str).unique().tolist())
    selected_categories = st.multiselect(
        "Water-quality categories",
        category_options,
        default=category_options,
        key="priority_history_categories",
    )
    filtered = history.loc[history["Category"].isin(selected_categories)].copy() if selected_categories else history.iloc[0:0].copy()

    search_text = st.text_input(
        "Search pollutant, pathogen, contaminant or parameter",
        placeholder="e.g. E. coli, ammonia, lead, PFAS, phosphate",
        key="priority_history_search",
    ).strip()
    if search_text:
        mask = filtered["Determinand / parameter"].astype(str).str.contains(search_text, case=False, na=False, regex=False)
        filtered = filtered.loc[mask].copy()

    summary = _summary_by_determinand(filtered)
    st.markdown("##### Parameters, contaminants and pathogens recorded by year")
    st.caption(
        "A recorded numerical measurement is not automatically a breach or harmful concentration. "
        "Use coded Present/Detected results and regulatory context carefully; the table reports what the archive measured, not causal attribution."
    )
    st.dataframe(summary, use_container_width=True, hide_index=True, height=min(620, 80 + 35 * min(len(summary), 15)))

    st.markdown("##### Exact dated observations")
    detail_columns = [
        "Measurement date", "Year", "Monitoring station", "Monitoring station ID",
        "Determinand / parameter", "Determinand code", "Category", "Reported result",
        "Unit", "Coded detection", "Sampling purpose", "Sample material",
        "Evidence relationship", "Source",
    ]
    details = filtered[detail_columns].sort_values("Measurement date", ascending=False)
    st.dataframe(details, use_container_width=True, hide_index=True, height=520)
    st.download_button(
        "Download historical water-quality evidence",
        data=details.to_csv(index=False).encode("utf-8"),
        file_name=f"ea_water_quality_history_{re.sub(r'[^A-Za-z0-9_-]+', '_', point_notation)}.csv",
        mime="text/csv",
        key="download_priority_history_quality",
    )
    st.caption(
        "Source: Environment Agency Water Quality Explorer. Archive coverage is from 2000 onwards and may be updated or corrected through EA quality assurance."
    )
