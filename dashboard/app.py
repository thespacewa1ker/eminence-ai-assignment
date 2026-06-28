"""Streamlit dashboard for BFSI media reputation analytics."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "classified_dataset.csv"
COMPANY_NAME = "Eminence Strategy"

REQUIRED_COLUMNS = [
    "Title",
    "Driver",
    "Sub driver",
    "Sentiment",
    "Reason",
]

SENTIMENT_COLORS = {
    "Positive": "#16a34a",
    "Neutral": "#2563eb",
    "Negative": "#dc2626",
}

SECTION_ICON = {
    "Executive Summary": " ",
    "Reputation Overview": "",
    "Analytics": "",
    "Discussion Themes": "",
    "Insights": "",
    "Content Explorer": "",
}

STOPWORDS = {
    "about",
    "after",
    "also",
    "amid",
    "among",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "its",
    "more",
    "new",
    "of",
    "on",
    "or",
    "over",
    "said",
    "says",
    "than",
    "that",
    "the",
    "their",
    "this",
    "to",
    "with",
}


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    """Load the classified dataset once per Streamlit cache cycle."""
    if not path.exists():
        raise FileNotFoundError(
            f"Classified dataset not found at {path}. "
            "Run the classification pipeline before launching the dashboard."
        )

    df = pd.read_csv(path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required dashboard columns: {missing_columns}")

    return df


def apply_filters(
    df: pd.DataFrame,
    drivers: list[str],
    sub_drivers: list[str],
    sentiments: list[str],
    title_search: str,
) -> pd.DataFrame:
    """Apply sidebar and search filters to the dataset."""
    filtered = df.copy()

    if drivers:
        filtered = filtered[filtered["Driver"].isin(drivers)]

    if sub_drivers:
        filtered = filtered[filtered["Sub driver"].isin(sub_drivers)]

    if sentiments:
        filtered = filtered[filtered["Sentiment"].isin(sentiments)]

    if title_search.strip():
        filtered = filtered[
            filtered["Title"]
            .fillna("")
            .str.contains(title_search.strip(), case=False, na=False)
        ]

    return filtered


def inject_css() -> None:
    """Apply dashboard-level styling."""
    st.markdown(
        """
        <style>
            :root {
                --navy: #0f2747;
                --navy-soft: #17365f;
                --slate: #475569;
                --muted: #64748b;
                --line: #e2e8f0;
                --panel: #ffffff;
                --page: #f5f7fb;
                --positive: #16a34a;
                --neutral: #2563eb;
                --negative: #dc2626;
            }

            .stApp {
                background: var(--page);
                color: #0f172a;
            }

            [data-testid="stSidebar"] {
                background: #ffffff;
                border-right: 1px solid var(--line);
            }

            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3 {
                color: var(--navy);
            }

            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2rem;
                max-width: 1480px;
            }

            .dashboard-header {
                background: linear-gradient(135deg, #0f2747 0%, #17365f 100%);
                border-radius: 14px;
                color: #ffffff;
                padding: 1.55rem 1.8rem;
                margin-bottom: 1.1rem;
                box-shadow: 0 18px 45px rgba(15, 39, 71, 0.16);
            }

            .dashboard-kicker {
                color: #bfdbfe;
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 0.35rem;
            }

            .dashboard-title {
                font-size: 2rem;
                font-weight: 760;
                line-height: 1.15;
                margin: 0;
            }

            .dashboard-subtitle {
                color: #dbeafe;
                font-size: 0.98rem;
                margin-top: 0.55rem;
                max-width: 820px;
            }

            .section-label {
                align-items: center;
                color: var(--navy);
                display: flex;
                font-size: 1.18rem;
                font-weight: 760;
                gap: 0.55rem;
                margin: 1.25rem 0 0.55rem;
            }

            .section-subtitle {
                color: var(--muted);
                font-size: 0.9rem;
                margin: -0.2rem 0 0.8rem;
            }

            .metric-card {
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 12px;
                box-shadow: 0 8px 28px rgba(15, 23, 42, 0.06);
                min-height: 112px;
                padding: 1rem;
            }

            .metric-label {
                color: var(--muted);
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.05em;
                text-transform: uppercase;
            }

            .metric-value {
                color: var(--navy);
                font-size: 1.75rem;
                font-weight: 800;
                margin-top: 0.5rem;
            }

            .metric-accent-positive .metric-value { color: var(--positive); }
            .metric-accent-neutral .metric-value { color: var(--neutral); }
            .metric-accent-negative .metric-value { color: var(--negative); }

            .insight-card {
                background: #ffffff;
                border: 1px solid var(--line);
                border-left: 4px solid var(--navy-soft);
                border-radius: 12px;
                box-shadow: 0 8px 28px rgba(15, 23, 42, 0.05);
                min-height: 118px;
                padding: 1rem 1.05rem;
            }

            .insight-title {
                color: var(--muted);
                font-size: 0.74rem;
                font-weight: 760;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }

            .insight-value {
                color: var(--navy);
                font-size: 1.05rem;
                font-weight: 760;
                margin-top: 0.45rem;
            }

            .insight-note {
                color: var(--slate);
                font-size: 0.82rem;
                margin-top: 0.35rem;
            }

            .chart-card {
                background: #ffffff;
                border: 1px solid var(--line);
                border-radius: 12px;
                box-shadow: 0 8px 28px rgba(15, 23, 42, 0.05);
                padding: 0.75rem 0.85rem 0.35rem;
            }

            .table-card {
                background: #ffffff;
                border: 1px solid var(--line);
                border-radius: 12px;
                box-shadow: 0 8px 28px rgba(15, 23, 42, 0.05);
                padding: 0.95rem;
            }

            .sidebar-note {
                background: #f8fafc;
                border: 1px solid var(--line);
                border-radius: 10px;
                color: var(--slate);
                font-size: 0.82rem;
                line-height: 1.45;
                padding: 0.8rem;
            }

            div[data-testid="stDownloadButton"] button {
                background: var(--navy);
                border: 1px solid var(--navy);
                color: #ffffff;
                font-weight: 700;
            }

            div[data-testid="stDownloadButton"] button:hover {
                background: var(--navy-soft);
                border-color: var(--navy-soft);
                color: #ffffff;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def chart_layout(fig: Figure, height: int = 380) -> Figure:
    """Apply consistent Plotly styling to dashboard charts."""
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=60, b=45),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#334155", family="Arial"),
        title=dict(font=dict(color="#0f2747", size=17), x=0.02),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor="#e2e8f0", zeroline=False)
    fig.update_yaxes(gridcolor="#e2e8f0", zeroline=False)
    return fig


def render_header() -> None:
    """Render the professional dashboard header."""
    st.markdown(
        f"""
        <div class="dashboard-header">
            <div class="dashboard-kicker">{COMPANY_NAME}</div>
            <h1 class="dashboard-title">BFSI Media Reputation Analytics Dashboard</h1>
            <div class="dashboard-subtitle">
                Executive view of classified media mentions across reputation drivers,
                sentiment, discussion themes, and article-level evidence.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section(title: str, subtitle: str | None = None) -> None:
    """Render a consistent section heading."""
    icon = SECTION_ICON.get(title, "Ã¢â€“Â¸")
    st.markdown(
        f'<div class="section-label"><span>{icon}</span><span>{title}</span></div>',
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def render_metric_card(
    label: str,
    value: int,
    accent: str | None = None,
) -> None:
    """Render one styled KPI card."""
    accent_class = f" metric-accent-{accent}" if accent else ""
    st.markdown(
        f"""
        <div class="metric-card{accent_class}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(df: pd.DataFrame) -> None:
    """Render top-level KPI cards."""
    total_articles = len(df)
    positive_articles = int((df["Sentiment"] == "Positive").sum())
    neutral_articles = int((df["Sentiment"] == "Neutral").sum())
    negative_articles = int((df["Sentiment"] == "Negative").sum())
    total_drivers = int(df["Driver"].nunique(dropna=True))
    total_sub_drivers = int(df["Sub driver"].nunique(dropna=True))

    metrics = [
        ("Total Articles", total_articles, None),
        ("Positive", positive_articles, "positive"),
        ("Neutral", neutral_articles, "neutral"),
        ("Negative", negative_articles, "negative"),
        ("Drivers", total_drivers, None),
        ("Sub Drivers", total_sub_drivers, None),
    ]

    cols = st.columns(6, gap="small")
    for col, (label, value, accent) in zip(cols, metrics):
        with col:
            render_metric_card(label, value, accent)


def empty_chart(message: str) -> None:
    """Display a consistent empty-state message for chart areas."""
    st.info(message)


def chart_card(fig: Figure) -> None:
    """Render a Plotly chart inside a styled visual container."""
    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True)


def render_reputation_overview(df: pd.DataFrame) -> None:
    """Render driver and sentiment distribution charts."""
    if df.empty:
        empty_chart("No records match the selected filters.")
        return

    left, right = st.columns((1.25, 1), gap="medium")

    driver_counts = df["Driver"].value_counts().reset_index()
    driver_counts.columns = ["Driver", "Articles"]
    fig_driver = px.bar(
        driver_counts,
        x="Driver",
        y="Articles",
        title="Driver Distribution",
        text="Articles",
        color_discrete_sequence=["#17365f"],
    )
    fig_driver.update_layout(xaxis_title=None, yaxis_title="Articles")
    fig_driver.update_traces(textposition="outside", cliponaxis=False)

    sentiment_counts = df["Sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Articles"]
    fig_sentiment = px.pie(
        sentiment_counts,
        names="Sentiment",
        values="Articles",
        title="Sentiment Distribution",
        hole=0.52,
        color="Sentiment",
        color_discrete_map=SENTIMENT_COLORS,
    )
    fig_sentiment.update_traces(textposition="inside", textinfo="percent+label")

    with left:
        chart_card(chart_layout(fig_driver, height=390))
    with right:
        chart_card(chart_layout(fig_sentiment, height=390))


def render_analytics(df: pd.DataFrame) -> None:
    """Render cross-driver analytics charts."""
    if df.empty:
        empty_chart("No analytics available for the selected filters.")
        return

    left, right = st.columns(2, gap="medium")

    driver_sentiment = (
        df.groupby(["Driver", "Sentiment"])
        .size()
        .reset_index(name="Articles")
    )
    fig_driver_sentiment = px.bar(
        driver_sentiment,
        x="Driver",
        y="Articles",
        color="Sentiment",
        title="Driver vs Sentiment",
        barmode="stack",
        color_discrete_map=SENTIMENT_COLORS,
    )
    fig_driver_sentiment.update_layout(xaxis_title=None, yaxis_title="Articles")

    top_sub_drivers = df["Sub driver"].value_counts().head(10).reset_index()
    top_sub_drivers.columns = ["Sub Driver", "Articles"]
    fig_top_sub_drivers = px.bar(
        top_sub_drivers.sort_values("Articles"),
        x="Articles",
        y="Sub Driver",
        orientation="h",
        title="Top 10 Sub Drivers",
        text="Articles",
        color_discrete_sequence=["#2563eb"],
    )
    fig_top_sub_drivers.update_layout(xaxis_title="Articles", yaxis_title=None)
    fig_top_sub_drivers.update_traces(textposition="outside", cliponaxis=False)

    with left:
        chart_card(chart_layout(fig_driver_sentiment, height=410))
    with right:
        chart_card(chart_layout(fig_top_sub_drivers, height=410))


def get_discussion_terms(df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """Compute frequent discussion terms from titles and classification reasons."""
    text = " ".join(
        df[["Title", "Reason"]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .tolist()
    )
    tokens = re.findall(r"[A-Za-z][A-Za-z]{2,}", text.lower())
    terms = [token for token in tokens if token not in STOPWORDS]
    counts = Counter(terms).most_common(limit)
    return pd.DataFrame(counts, columns=["Theme", "Mentions"])


def render_discussion_themes(df: pd.DataFrame) -> None:
    """Render sub-driver distribution and top discussion themes."""
    if df.empty:
        empty_chart("No discussion themes available for the selected filters.")
        return

    left, right = st.columns(2, gap="medium")

    sub_driver_counts = (
        df["Sub driver"]
        .value_counts()
        .sort_values(ascending=True)
        .reset_index()
    )
    sub_driver_counts.columns = ["Sub Driver", "Articles"]
    fig_sub_driver = px.bar(
        sub_driver_counts,
        x="Articles",
        y="Sub Driver",
        orientation="h",
        title="Sub Driver Distribution",
        text="Articles",
        color_discrete_sequence=["#17365f"],
    )
    fig_sub_driver.update_layout(xaxis_title="Articles", yaxis_title=None)
    fig_sub_driver.update_traces(textposition="outside", cliponaxis=False)

    discussion_terms = get_discussion_terms(df)
    fig_terms = px.bar(
        discussion_terms.sort_values("Mentions"),
        x="Mentions",
        y="Theme",
        orientation="h",
        title="Top Discussion Themes",
        text="Mentions",
        color_discrete_sequence=["#0f766e"],
    )
    fig_terms.update_layout(xaxis_title="Mentions", yaxis_title=None)
    fig_terms.update_traces(textposition="outside", cliponaxis=False)

    with left:
        chart_card(chart_layout(fig_sub_driver, height=430))
    with right:
        chart_card(chart_layout(fig_terms, height=430))


def strongest_driver_by_sentiment(df: pd.DataFrame, sentiment: str) -> tuple[str, int]:
    """Return the driver with the highest count for a selected sentiment."""
    sentiment_df = df[df["Sentiment"] == sentiment]
    if sentiment_df.empty:
        return "Not available", 0

    counts = sentiment_df["Driver"].value_counts()
    return str(counts.index[0]), int(counts.iloc[0])


def build_overall_reputation_summary(df: pd.DataFrame) -> str:
    """Build a dynamic reputation summary from filtered sentiment mix."""
    if df.empty:
        return "No records match the selected filters."

    sentiment_counts = df["Sentiment"].value_counts()
    top_sentiment = str(sentiment_counts.index[0])
    top_count = int(sentiment_counts.iloc[0])
    share = top_count / len(df) * 100

    return (
        f"{top_sentiment} coverage leads the filtered view, representing "
        f"{share:.1f}% of {len(df):,} articles."
    )


def render_insight_card(title: str, value: str, note: str) -> None:
    """Render a single dynamic insight card."""
    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-title">{title}</div>
            <div class="insight-value">{value}</div>
            <div class="insight-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insights(df: pd.DataFrame) -> None:
    """Render dynamically generated key insights."""
    if df.empty:
        empty_chart("No insights available for the selected filters.")
        return

    driver_counts = df["Driver"].value_counts()
    sub_driver_counts = df["Sub driver"].value_counts()
    positive_driver, positive_count = strongest_driver_by_sentiment(df, "Positive")
    negative_driver, negative_count = strongest_driver_by_sentiment(df, "Negative")

    most_discussed_driver = str(driver_counts.index[0]) if not driver_counts.empty else "N/A"
    most_discussed_driver_count = int(driver_counts.iloc[0]) if not driver_counts.empty else 0
    most_discussed_sub_driver = (
        str(sub_driver_counts.index[0]) if not sub_driver_counts.empty else "N/A"
    )
    most_discussed_sub_driver_count = (
        int(sub_driver_counts.iloc[0]) if not sub_driver_counts.empty else 0
    )

    insight_rows = [
        (
            "Most Discussed Driver",
            most_discussed_driver,
            f"{most_discussed_driver_count:,} articles in the filtered view.",
        ),
        (
            "Most Discussed Sub Driver",
            most_discussed_sub_driver,
            f"{most_discussed_sub_driver_count:,} articles mention this theme.",
        ),
        (
            "Most Positive Driver",
            positive_driver,
            f"{positive_count:,} positive articles classified under this driver.",
        ),
        (
            "Most Negative Driver",
            negative_driver,
            f"{negative_count:,} negative articles classified under this driver.",
        ),
        (
            "Overall Reputation Summary",
            "Filtered sentiment mix",
            build_overall_reputation_summary(df),
        ),
    ]

    top_cols = st.columns(3, gap="medium")
    bottom_cols = st.columns(2, gap="medium")
    cols = list(top_cols) + list(bottom_cols)

    for col, (title, value, note) in zip(cols, insight_rows):
        with col:
            render_insight_card(title, value, note)


def render_sidebar_filters(df: pd.DataFrame) -> tuple[list[str], list[str], list[str], str]:
    """Render sidebar controls and return selected filters."""
    st.sidebar.markdown("## Filters")
    st.sidebar.markdown(
        """
        <div class="sidebar-note">
            Refine the dashboard by reputation taxonomy, sentiment, or title keyword.
            All KPIs, charts, insights, and exports update dynamically.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("### Reputation Taxonomy")

    driver_options = sorted(df["Driver"].dropna().unique().tolist())
    selected_drivers = st.sidebar.multiselect("Driver", driver_options)

    driver_filtered_df = (
        df[df["Driver"].isin(selected_drivers)] if selected_drivers else df
    )
    sub_driver_options = sorted(
        driver_filtered_df["Sub driver"].dropna().unique().tolist()
    )
    selected_sub_drivers = st.sidebar.multiselect("Sub Driver", sub_driver_options)

    st.sidebar.markdown("### Sentiment")
    sentiment_options = sorted(df["Sentiment"].dropna().unique().tolist())
    selected_sentiments = st.sidebar.multiselect("Sentiment", sentiment_options)

    st.sidebar.markdown("### Search")
    title_search = st.sidebar.text_input("Search Titles", placeholder="Enter title keyword")

    return selected_drivers, selected_sub_drivers, selected_sentiments, title_search


def render_table(df: pd.DataFrame) -> None:
    """Render the interactive records table and CSV download."""
    display_columns = ["Title", "Driver", "Sub driver", "Sentiment", "Reason"]

    with st.container(border=True):
        st.caption(f"Showing {len(df):,} filtered articles")
        st.dataframe(
            df[display_columns],
            use_container_width=True,
            hide_index=True,
            height=460,
        )

        csv_data = df[display_columns].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Filtered CSV",
            data=csv_data,
            file_name="filtered_reputation_mentions.csv",
            mime="text/csv",
        )


def main() -> None:
    """Run the Streamlit dashboard."""
    st.set_page_config(
        page_title="BFSI Media Reputation Analytics Dashboard",
        page_icon="Ã°Å¸â€œÅ ",
        layout="wide",
    )
    inject_css()
    render_header()

    try:
        df = load_data(DATA_PATH)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    selected_drivers, selected_sub_drivers, selected_sentiments, title_search = (
        render_sidebar_filters(df)
    )

    filtered_df = apply_filters(
        df=df,
        drivers=selected_drivers,
        sub_drivers=selected_sub_drivers,
        sentiments=selected_sentiments,
        title_search=title_search,
    )

    render_section("Executive Summary", "Snapshot of current filtered media coverage.")
    render_kpis(filtered_df)

    render_section(
        "Reputation Overview",
        "Distribution of media coverage across reputation drivers and sentiment.",
    )
    render_reputation_overview(filtered_df)

    render_section(
        "Analytics",
        "Cross-driver sentiment patterns and the most active sub-driver themes.",
    )
    render_analytics(filtered_df)

    render_section(
        "Discussion Themes",
        "Sub-driver spread and recurring terms from titles and classification reasons.",
    )
    render_discussion_themes(filtered_df)

    render_section(
        "Insights",
        "Automatically generated readout based on the filtered dataset.",
    )
    render_insights(filtered_df)

    render_section(
        "Content Explorer",
        "Article-level evidence with the current sidebar filters applied.",
    )
    render_table(filtered_df)


if __name__ == "__main__":
    main()
