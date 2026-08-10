from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="England EDM Spill-Risk Dashboard",
    page_icon="💧",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"

RISK_ORDER = ["Low", "Medium", "High"]

RISK_COLOURS = {
    "Low": "#2E7D32",
    "Medium": "#E69F00",
    "High": "#C62828",
}

st.markdown(
    """
    <style>
      .block-container {
          padding-top: 1.25rem;
          padding-bottom: 3rem;
      }

      div[data-testid="stMetric"] {
          background: #F5F8FA;
          border: 1px solid #CAD9E4;
          border-left: 6px solid #6F9FC5;
          padding: 12px;
          border-radius: 7px;
      }

      .edm-note {
          background: #FFF3CD;
          border-left: 7px solid #D28A00;
          padding: 12px 15px;
          border-radius: 4px;
          color: #17324D;
          margin: 8px 0 14px;
      }

      .edm-info {
          background: #E8F1F8;
          border-left: 7px solid #4477AA;
          padding: 12px 15px;
          border-radius: 4px;
          color: #17324D;
          margin: 8px 0 14px;
      }

      .edm-risk {
          background: #FCE8E6;
          border-left: 7px solid #C62828;
          padding: 12px 15px;
          border-radius: 4px;
          color: #17324D;
          margin: 8px 0 14px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_table(name):
    path = DATA_DIR / f"{name}.csv.gz"

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(
        path,
        low_memory=False,
        compression="gzip",
    )


@st.cache_resource(show_spinner=False)
def load_model_bundle():
    path = (
        ARTIFACT_DIR
        / "final_trained_2026_forecast_model.joblib"
    )

    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_input_metadata():
    path = ARTIFACT_DIR / "input_metadata.json"

    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def pretty(value):
    return (
        str(value)
        .replace("_", " ")
        .strip()
        .title()
    )


def value_text(value):
    if pd.isna(value):
        return "Not available"

    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"

    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.1f}"

    return str(value)


def show_kpis(frame):
    if frame.empty:
        st.info("KPI table is unavailable.")
        return

    for start in range(0, len(frame), 4):
        block = frame.iloc[start:start + 4]
        columns = st.columns(len(block))

        for column, (_, row) in zip(
            columns,
            block.iterrows(),
        ):
            with column:
                st.metric(
                    str(row.get("KPI", "Measure")),
                    value_text(row.get("Value")),
                )

                meaning = row.get("Meaning")

                if pd.notna(meaning):
                    st.caption(str(meaning))


def risk_bar(frame, risk_column, title):
    counts = (
        frame[risk_column]
        .astype("string")
        .value_counts()
        .reindex(RISK_ORDER, fill_value=0)
        .rename_axis("Risk")
        .reset_index(name="Records")
    )

    figure = px.bar(
        counts,
        x="Risk",
        y="Records",
        color="Risk",
        category_orders={"Risk": RISK_ORDER},
        color_discrete_map=RISK_COLOURS,
        text_auto=",.0f",
        title=title,
    )

    figure.update_layout(
        showlegend=False,
        template="plotly_white",
        height=430,
    )

    return figure


def available_values(frame, column):
    if column not in frame.columns:
        return []

    return sorted(
        frame[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


def filter_map_table(
    frame,
    risk_column,
    company_column,
    place_column,
    key_prefix,
):
    first, second, third = st.columns(3)

    with first:
        companies = (
            ["All companies"]
            + available_values(
                frame,
                company_column,
            )
        )

        company = st.selectbox(
            "Water company",
            companies,
            key=f"{key_prefix}_company",
        )

    with second:
        available_risks = available_values(
            frame,
            risk_column,
        )

        risks = ["All risks"] + [
            risk
            for risk in RISK_ORDER
            if risk in available_risks
        ]

        risk = st.selectbox(
            "Risk category",
            risks,
            key=f"{key_prefix}_risk",
        )

    with third:
        places = (
            ["All towns/cities"]
            + available_values(
                frame,
                place_column,
            )
        )

        place = st.selectbox(
            "Town/city",
            places,
            key=f"{key_prefix}_place",
        )

    search = st.text_input(
        "Optional site, permit, receiving-water "
        "or grid-reference search",
        key=f"{key_prefix}_search",
    ).strip()

    filtered = frame.copy()

    if (
        company != "All companies"
        and company_column in filtered.columns
    ):
        filtered = filtered.loc[
            filtered[company_column]
            .astype(str)
            .eq(company)
        ]

    if risk != "All risks":
        filtered = filtered.loc[
            filtered[risk_column]
            .astype(str)
            .eq(risk)
        ]

    if (
        place != "All towns/cities"
        and place_column in filtered.columns
    ):
        filtered = filtered.loc[
            filtered[place_column]
            .astype(str)
            .eq(place)
        ]

    if search:
        searchable = [
            name
            for name in [
                "site_name",
                "permit_reference",
                "receiving_water",
                "catchment",
                "catchment_name",
                "parsed_grid_reference",
            ]
            if name in filtered.columns
        ]

        match = pd.Series(
            False,
            index=filtered.index,
        )

        for name in searchable:
            match |= (
                filtered[name]
                .astype("string")
                .str.contains(
                    search,
                    case=False,
                    regex=False,
                    na=False,
                )
            )

        filtered = filtered.loc[match]

    return filtered


def map_figure(frame, risk_column, title):
    plotting = frame.copy()

    plotting["latitude"] = pd.to_numeric(
        plotting["latitude"],
        errors="coerce",
    )

    plotting["longitude"] = pd.to_numeric(
        plotting["longitude"],
        errors="coerce",
    )

    plotting = plotting.dropna(
        subset=["latitude", "longitude"]
    )

    hover_columns = [
        column
        for column in [
            "site_name",
            "water_company_name",
            "official_place_name",
            "town_or_city",
            "receiving_water",
            "catchment",
            "catchment_name",
            "parsed_grid_reference",
            "probability_low",
            "probability_medium",
            "probability_high",
            "prediction_confidence",
            "confidence_flag",
        ]
        if column in plotting.columns
    ]

    figure = px.scatter_mapbox(
        plotting,
        lat="latitude",
        lon="longitude",
        color=risk_column,
        category_orders={
            risk_column: RISK_ORDER
        },
        color_discrete_map=RISK_COLOURS,
        hover_data=hover_columns,
        zoom=5,
        height=720,
        title=title,
    )

    figure.update_layout(
        mapbox_style="open-street-map",
        margin=dict(
            l=0,
            r=0,
            t=45,
            b=0,
        ),
    )

    return figure


def download_table(frame, filename):
    st.download_button(
        "Download the filtered table",
        data=frame.to_csv(
            index=False
        ).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def aligned_probabilities(model, values):
    raw = np.asarray(
        model.predict_proba(values),
        dtype=float,
    )

    aligned = np.zeros(
        (len(values), 3),
        dtype=float,
    )

    for position, class_code in enumerate(
        model.classes_
    ):
        aligned[:, int(class_code)] = (
            raw[:, position]
        )

    return aligned


def probability_logit(probabilities):
    clipped = np.clip(
        np.asarray(
            probabilities,
            dtype=float,
        ),
        1e-6,
        1 - 1e-6,
    )

    return np.log(
        clipped / (1 - clipped)
    ).reshape(-1, 1)


def apply_calibrators(raw, calibrators):
    calibrated = np.zeros_like(
        raw,
        dtype=float,
    )

    for class_code, calibrator in enumerate(
        calibrators
    ):
        if calibrator is None:
            calibrated[:, class_code] = (
                raw[:, class_code]
            )
        else:
            calibrated[:, class_code] = (
                calibrator.predict_proba(
                    probability_logit(
                        raw[:, class_code]
                    )
                )[:, 1]
            )

    totals = calibrated.sum(
        axis=1,
        keepdims=True,
    )

    invalid = (
        ~np.isfinite(totals[:, 0])
        | (totals[:, 0] <= 0)
    )

    calibrated[invalid] = raw[invalid]

    totals = calibrated.sum(
        axis=1,
        keepdims=True,
    )

    return calibrated / totals


st.title(
    "England EDM Spill-Risk Dashboard"
)

st.caption(
    "Observed 2023–2025 evidence, "
    "2026 model predictions and "
    "transparent uncertainty"
)

st.markdown(
    """
    <div class="edm-note">
      <b>Important:</b> predicted risk categories
      are decision-support outputs. They are not
      confirmed future spills and do not establish
      pollution volume, ecological damage or legal
      responsibility.
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Dashboard section",
    [
        "Overview",
        "Observed map",
        "Predicted 2026 map",
        "Companies",
        "Towns and cities",
        "2025 versus 2026",
        "Models and metrics",
        "Individual prediction",
        "Water quality",
        "Data quality",
        "Evidence search",
        "Method and limitations",
    ],
)

st.sidebar.caption(
    "Public dashboard generated from "
    "the verified Colab outputs."
)


if page == "Overview":
    st.header("Overview")

    observed_kpis = load_table(
        "observed_kpis"
    )

    forecast_kpis = load_table(
        "forecast_kpis"
    )

    observed = load_table(
        "observed_locations"
    )

    forecast = load_table(
        "forecast_map_points"
    )

    st.subheader("Observed evidence")
    show_kpis(observed_kpis)

    st.subheader("2026 forecast")
    show_kpis(forecast_kpis)

    left, right = st.columns(2)

    with left:
        if not observed.empty:
            figure = risk_bar(
                observed,
                "period_risk_category",
                "Observed location categories",
            )

            st.plotly_chart(
                figure,
                use_container_width=True,
            )

    with right:
        if not forecast.empty:
            figure = risk_bar(
                forecast,
                "predicted_2026_risk",
                "Predicted 2026 categories",
            )

            st.plotly_chart(
                figure,
                use_container_width=True,
            )


elif page == "Observed map":
    st.header(
        "Observed 2023–2025 risk map"
    )

    observed = load_table(
        "observed_locations"
    )

    if observed.empty:
        st.error(
            "The observed-location export "
            "is missing."
        )

    else:
        place_column = (
            "official_place_name"
            if "official_place_name"
            in observed.columns
            else "town_or_city"
        )

        filtered = filter_map_table(
            observed,
            "period_risk_category",
            "water_company_name",
            place_column,
            "observed",
        )

        st.write(
            "Locations matching the filters: "
            f"**{len(filtered):,}**"
        )

        if filtered.empty:
            st.warning(
                "No observed locations match "
                "the selected filters."
            )

        else:
            figure = map_figure(
                filtered,
                "period_risk_category",
                "Observed EDM risk locations",
            )

            st.plotly_chart(
                figure,
                use_container_width=True,
            )

            download_table(
                filtered,
                "filtered_observed_locations.csv",
            )


elif page == "Predicted 2026 map":
    st.header(
        "Predicted 2026 risk map"
    )

    st.markdown(
        """
        <div class="edm-risk">
          These are calibrated model predictions,
          not observed 2026 events. Review
          low-confidence and close-probability
          cases before using the results.
        </div>
        """,
        unsafe_allow_html=True,
    )

    forecast = load_table(
        "forecast_map_points"
    )

    if forecast.empty:
        st.error(
            "The forecast-map export is missing."
        )

    else:
        filtered = filter_map_table(
            forecast,
            "predicted_2026_risk",
            "water_company_name",
            "town_or_city",
            "forecast",
        )

        confidence_options = (
            ["All confidence levels"]
            + available_values(
                forecast,
                "confidence_flag",
            )
        )

        confidence = st.selectbox(
            "Prediction confidence",
            confidence_options,
        )

        if (
            confidence
            != "All confidence levels"
        ):
            filtered = filtered.loc[
                filtered["confidence_flag"]
                .astype(str)
                .eq(confidence)
            ]

        st.write(
            "Forecast locations matching "
            f"the filters: **{len(filtered):,}**"
        )

        if filtered.empty:
            st.warning(
                "No forecast locations match "
                "the selected filters."
            )

        else:
            figure = map_figure(
                filtered,
                "predicted_2026_risk",
                "Predicted 2026 EDM risk",
            )

            st.plotly_chart(
                figure,
                use_container_width=True,
            )

            download_table(
                filtered,
                "filtered_2026_predictions.csv",
            )


elif page == "Companies":
    st.header(
        "Water-company comparison"
    )

    rankings = load_table(
        "company_rankings"
    )

    annual = load_table(
        "annual_company_trends"
    )

    if rankings.empty:
        st.error(
            "The company-ranking export "
            "is missing."
        )

    else:
        st.dataframe(
            rankings,
            use_container_width=True,
            hide_index=True,
        )

        required_columns = {
            "water_company_name",
            "high_risk_unique_locations",
        }

        if required_columns.issubset(
            rankings.columns
        ):
            plot = rankings.sort_values(
                "high_risk_unique_locations"
            )

            figure = px.bar(
                plot,
                x="high_risk_unique_locations",
                y="water_company_name",
                orientation="h",
                title=(
                    "Unique observed "
                    "High-risk locations"
                ),
                color_discrete_sequence=[
                    RISK_COLOURS["High"]
                ],
            )

            figure.update_layout(
                template="plotly_white",
                height=max(
                    500,
                    42 * len(plot),
                ),
            )

            st.plotly_chart(
                figure,
                use_container_width=True,
            )

    if (
        not annual.empty
        and "water_company_name"
        in annual.columns
    ):
        company_options = available_values(
            annual,
            "water_company_name",
        )

        if company_options:
            company = st.selectbox(
                "Company annual trend",
                company_options,
            )

            company_rows = annual.loc[
                annual["water_company_name"]
                .astype(str)
                .eq(company)
            ].copy()

            risk_columns = [
                column
                for column in [
                    "low_risk_unique_locations",
                    "medium_risk_unique_locations",
                    "high_risk_unique_locations",
                ]
                if column
                in company_rows.columns
            ]

            if risk_columns:
                long = company_rows.melt(
                    id_vars=["reporting_year"],
                    value_vars=risk_columns,
                    var_name="Risk",
                    value_name="Locations",
                )

                long["Risk"] = (
                    long["Risk"]
                    .str.replace(
                        "_risk_unique_locations",
                        "",
                        regex=False,
                    )
                    .str.title()
                )

                figure = px.line(
                    long,
                    x="reporting_year",
                    y="Locations",
                    color="Risk",
                    markers=True,
                    color_discrete_map=
                        RISK_COLOURS,
                    title=(
                        "Annual observed "
                        f"categories: {company}"
                    ),
                )

                st.plotly_chart(
                    figure,
                    use_container_width=True,
                )


elif page == "Towns and cities":
    st.header(
        "Town and city evidence"
    )

    towns = load_table(
        "town_trends"
    )

    if towns.empty:
        st.error(
            "The town-trend export is missing."
        )

    else:
        place_options = available_values(
            towns,
            "official_place_name",
        )

        place = st.selectbox(
            "Town or city",
            place_options,
        )

        row = towns.loc[
            towns["official_place_name"]
            .astype(str)
            .eq(place)
        ].iloc[0]

        st.write(
            "**Water company or companies:** "
            f"{row.get('water_companies', 'Not available')}"
        )

        st.write(
            "**Trend:** "
            f"{row.get('trend_2023_to_2025', 'Not available')}"
        )

        st.write(
            "**Risk history:** "
            f"{row.get('town_risk_transition', 'Not available')}"
        )

        trend = pd.DataFrame(
            {
                "Year": [
                    2023,
                    2024,
                    2025,
                ],
                "Counted spills": [
                    row.get(
                        f"counted_spills_{year}"
                    )
                    for year in [
                        2023,
                        2024,
                        2025,
                    ]
                ],
                "Duration hours": [
                    row.get(
                        f"duration_hours_{year}"
                    )
                    for year in [
                        2023,
                        2024,
                        2025,
                    ]
                ],
            }
        )

        left, right = st.columns(2)

        with left:
            figure = px.line(
                trend,
                x="Year",
                y="Counted spills",
                markers=True,
                title=f"Counted spills: {place}",
            )

            st.plotly_chart(
                figure,
                use_container_width=True,
            )

        with right:
            figure = px.line(
                trend,
                x="Year",
                y="Duration hours",
                markers=True,
                title=f"Recorded duration: {place}",
            )

            st.plotly_chart(
                figure,
                use_container_width=True,
            )

        st.dataframe(
            towns.head(100),
            use_container_width=True,
            hide_index=True,
        )


elif page == "2025 versus 2026":
    st.header(
        "Observed 2025 versus predicted 2026"
    )

    comparison = load_table(
        "observed_vs_predicted"
    )

    companies = load_table(
        "observed_predicted_company_comparison"
    )

    if comparison.empty:
        st.error(
            "The observed-versus-predicted "
            "export is missing."
        )

    else:
        matrix = pd.crosstab(
            comparison["observed_2025_risk"],
            comparison["predicted_2026_risk"],
        ).reindex(
            index=RISK_ORDER,
            columns=RISK_ORDER,
            fill_value=0,
        )

        figure = go.Figure(
            go.Heatmap(
                z=matrix.to_numpy(),
                x=matrix.columns,
                y=matrix.index,
                text=matrix.to_numpy(),
                texttemplate="%{text:,}",
                colorscale="Blues",
            )
        )

        figure.update_layout(
            title=(
                "Observed-to-predicted "
                "transition matrix"
            ),
            xaxis_title=(
                "Predicted 2026 risk"
            ),
            yaxis_title=(
                "Observed 2025 risk"
            ),
            height=520,
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True,
        )

    if not companies.empty:
        st.subheader(
            "Company-level comparison"
        )

        st.dataframe(
            companies,
            use_container_width=True,
            hide_index=True,
        )


elif page == "Models and metrics":
    st.header("Model evaluation")

    st.markdown(
        """
        <div class="edm-info">
          <b>Evaluation wording:</b> the 2025
          cohort is presented as a validation
          cohort because it has also informed
          sensitivity and training-window
          comparisons. New later data is needed
          for a final external test.
        </div>
        """,
        unsafe_allow_html=True,
    )

    models = load_table(
        "model_comparison"
    )

    metrics = load_table(
        "validation_metrics"
    )

    report = load_table(
        "classification_report"
    )

    if not models.empty:
        st.subheader(
            "Training-period cross-validation"
        )

        st.dataframe(
            models,
            use_container_width=True,
            hide_index=True,
        )

    if not metrics.empty:
        st.subheader(
            "2025 validation metrics"
        )

        metrics = metrics.rename(
            columns=lambda value: (
                str(value).replace(
                    "Test",
                    "2025 validation",
                )
            )
        )

        st.dataframe(
            metrics,
            use_container_width=True,
            hide_index=True,
        )

    if not report.empty:
        st.subheader(
            "Class-level precision, recall and F1"
        )

        st.dataframe(
            report,
            use_container_width=True,
            hide_index=True,
        )


elif page == "Individual prediction":
    st.header(
        "Individual 2026 risk prediction"
    )

    st.caption(
        "Enter prior-year information using "
        "the same fields and fitted preprocessing "
        "as the selected model."
    )

    try:
        bundle = load_model_bundle()
        metadata = load_input_metadata()

    except Exception as error:
        st.error(
            "The saved prediction model "
            f"could not be loaded: {error}"
        )
        st.stop()

    st.write(
        "**Selected model:** "
        f"{bundle.get('selected_model_name', 'Not recorded')}"
    )

    values = {}

    with st.form("prediction_form"):
        st.subheader(
            "Previous-year numeric information"
        )

        numeric_columns = metadata.get(
            "numeric_columns",
            [],
        )

        numeric_widgets = st.columns(2)

        for position, column in enumerate(
            numeric_columns
        ):
            default = float(
                metadata.get(
                    "numeric_defaults",
                    {},
                ).get(
                    column,
                    0.0,
                )
                or 0.0
            )

            with numeric_widgets[
                position % 2
            ]:
                values[column] = (
                    st.number_input(
                        pretty(column),
                        value=default,
                        format="%.3f",
                    )
                )

        st.subheader(
            "Site information"
        )

        categorical_columns = metadata.get(
            "categorical_columns",
            [],
        )

        categorical_widgets = st.columns(2)

        for position, column in enumerate(
            categorical_columns
        ):
            options = metadata.get(
                "categorical_options",
                {},
            ).get(
                column,
                ["__MISSING__"],
            )

            if not options:
                options = ["__MISSING__"]

            display_options = [
                (
                    "Not recorded"
                    if option == "__MISSING__"
                    else option
                )
                for option in options
            ]

            with categorical_widgets[
                position % 2
            ]:
                selected = st.selectbox(
                    pretty(column),
                    display_options,
                )

                values[column] = (
                    "__MISSING__"
                    if selected == "Not recorded"
                    else selected
                )

        submitted = st.form_submit_button(
            "Calculate prediction",
            type="primary",
        )

    if submitted:
        raw_input = pd.DataFrame(
            [values]
        )

        for column in categorical_columns:
            raw_input[column] = (
                raw_input[column]
                .astype("string")
                .fillna("__MISSING__")
                .astype(str)
            )

        transformed = np.asarray(
            bundle["preprocessor"].transform(
                raw_input
            ),
            dtype=np.float32,
        )

        transformed[
            ~np.isfinite(transformed)
        ] = np.nan

        raw_probabilities = (
            aligned_probabilities(
                bundle["model"],
                transformed,
            )
        )

        probabilities = apply_calibrators(
            raw_probabilities,
            bundle[
                "probability_calibrators"
            ],
        )[0]

        code = int(
            np.argmax(probabilities)
        )

        labels = bundle.get(
            "risk_code_to_label",
            {
                0: "Low",
                1: "Medium",
                2: "High",
            },
        )

        prediction = labels.get(
            code,
            labels.get(
                str(code),
                str(code),
            ),
        )

        sorted_probabilities = np.sort(
            probabilities
        )

        margin = float(
            sorted_probabilities[-1]
            - sorted_probabilities[-2]
        )

        confidence = float(
            probabilities.max()
        )

        columns = st.columns(4)

        columns[0].metric(
            "Predicted risk",
            prediction,
        )

        columns[1].metric(
            "Low probability",
            f"{probabilities[0]:.1%}",
        )

        columns[2].metric(
            "Medium probability",
            f"{probabilities[1]:.1%}",
        )

        columns[3].metric(
            "High probability",
            f"{probabilities[2]:.1%}",
        )

        if (
            confidence < 0.60
            or margin < 0.15
        ):
            st.warning(
                "Review required: confidence is "
                "low or the two leading "
                "probabilities are close."
            )

        else:
            st.success(
                "This prediction meets the "
                "dashboard's higher-confidence rule."
            )

        st.caption(
            "This is a model prediction and "
            "should not be interpreted as a "
            "confirmed future event."
        )


elif page == "Water quality":
    st.header(
        "Nearby-station water-quality evidence"
    )

    show_kpis(
        load_table("water_quality_kpis")
    )

    quality = load_table(
        "water_quality_records"
    )

    coverage = load_table(
        "water_quality_coverage"
    )

    if not coverage.empty:
        st.subheader(
            "Coverage by company"
        )

        st.dataframe(
            coverage,
            use_container_width=True,
            hide_index=True,
        )

    if quality.empty:
        st.info(
            "No public water-quality "
            "export is available."
        )

    else:
        filtered = quality.copy()

        if "company" in filtered.columns:
            company_options = (
                ["All companies"]
                + available_values(
                    filtered,
                    "company",
                )
            )

            company = st.selectbox(
                "Company",
                company_options,
            )

            if company != "All companies":
                filtered = filtered.loc[
                    filtered["company"]
                    .astype(str)
                    .eq(company)
                ]

        if (
            "project_parameter_name"
            in filtered.columns
        ):
            parameter_options = (
                ["All parameters"]
                + available_values(
                    filtered,
                    "project_parameter_name",
                )
            )

            parameter = st.selectbox(
                "Measured parameter",
                parameter_options,
            )

            if parameter != "All parameters":
                filtered = filtered.loc[
                    filtered[
                        "project_parameter_name"
                    ]
                    .astype(str)
                    .eq(parameter)
                ]

        st.dataframe(
            filtered.head(5000),
            use_container_width=True,
            hide_index=True,
        )

        download_table(
            filtered,
            "filtered_water_quality_evidence.csv",
        )

        st.caption(
            "Nearby monitoring evidence shows "
            "geographic association, not causation."
        )


elif page == "Data quality":
    st.header(
        "Data quality and audit evidence"
    )

    audit_tables = [
        (
            "Target-leakage audit",
            "leakage_audit",
        ),
        (
            "Temporal matching audit",
            "temporal_matching_audit",
        ),
        (
            "Model-data usability",
            "model_data_quality",
        ),
        (
            "Missing measurements",
            "missing_measurement_audit",
        ),
        (
            "Coordinate quality",
            "coordinate_quality",
        ),
    ]

    for title, name in audit_tables:
        st.subheader(title)

        frame = load_table(name)

        if frame.empty:
            st.info(
                f"{title} is unavailable."
            )

        else:
            st.dataframe(
                frame,
                use_container_width=True,
                hide_index=True,
            )


elif page == "Evidence search":
    st.header(
        "Search verified dashboard records"
    )

    question = st.text_input(
        "Enter a company, town/city, site, "
        "receiving water or permit reference"
    )

    if question.strip():
        query = question.strip()

        search_tables = [
            (
                "Observed locations",
                "observed_locations",
            ),
            (
                "Forecast locations",
                "forecast_map_points",
            ),
            (
                "Town/city trends",
                "town_trends",
            ),
            (
                "Company rankings",
                "company_rankings",
            ),
        ]

        results_found = False

        for title, name in search_tables:
            frame = load_table(name)

            if frame.empty:
                continue

            text_columns = (
                frame.select_dtypes(
                    include=[
                        "object",
                        "string",
                    ]
                ).columns
            )

            match = pd.Series(
                False,
                index=frame.index,
            )

            for column in text_columns:
                match |= (
                    frame[column]
                    .astype("string")
                    .str.contains(
                        query,
                        case=False,
                        regex=False,
                        na=False,
                    )
                )

            result = frame.loc[match]

            if not result.empty:
                results_found = True

                st.subheader(
                    f"{title}: "
                    f"{len(result):,} matching records"
                )

                st.dataframe(
                    result.head(500),
                    use_container_width=True,
                    hide_index=True,
                )

        if not results_found:
            st.warning(
                "No dashboard records matched "
                "that search."
            )


else:
    st.header(
        "Method and limitations"
    )

    st.markdown(
        """
        - Observed Low, Medium and High categories
          come from the verified Excel labels.
        - Earlier-year measurements are used to
          predict the following year's category.
        - Macro F1 gives equal importance to Low,
          Medium and High classes.
        - High-risk recall measures how many actual
          High-risk records were detected.
        - High-risk precision measures how many
          High-risk predictions were genuinely High.
        - Predictions support investigation and
          prioritisation; they are not confirmed events.
        - Spill duration is not sewage volume.
        - Nearby water-quality observations do not
          prove that a specific outlet caused a result.
        - Town names are geographic references and
          do not prove an event occurred inside a
          town boundary.
        - The 2025 cohort has informed model
          assessment and is validation data.
        - Final performance should be checked using
          genuinely new later data.
        """
    )
