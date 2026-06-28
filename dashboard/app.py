"""Streamlit dashboard for BFSI media reputation analytics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "classified_dataset.csv"

REQUIRED_COLUMNS = [
    "Title",
    "Driver",
    "Sub driver",
    "Sentiment",
    "Reason",
]


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


def render_kpis(df: pd.DataFrame) -> None:
    """Render top-level KPI cards."""
    total_articles = len(df)
    positive_articles = int((df["Sentiment"] == "Positive").sum())
    neutral_articles = int((df["Sentiment"] == "Neutral").sum())
    negative_articles = int((df["Sentiment"] == "Negative").sum())
    total_drivers = int(df["Driver"].nunique(dropna=True))
    total_sub_drivers = int(df["Sub driver"].nunique(dropna=True))

    cols = st.columns(6)
    cols[0].metric("Total Articles", f"{total_articles:,}")
    cols[1].metric("Positive Articles", f"{positive_articles:,}")
    cols[2].metric("Neutral Articles", f"{neutral_articles:,}")
    cols[3].metric("Negative Articles", f"{negative_articles:,}")
    cols[4].metric("Total Drivers", f"{total_drivers:,}")
    cols[5].metric("Total Sub Drivers", f"{total_sub_drivers:,}")


def empty_chart(message: str) -> None:
    """Display a consistent empty-state message for chart areas."""
    st.info(message)


def render_charts(df: pd.DataFrame) -> None:
    """Render interactive Plotly charts."""
    if df.empty:
        empty_chart("No records match the selected filters.")
        return

    left, right = st.columns(2)

    driver_counts = df["Driver"].value_counts().reset_index()
    driver_counts.columns = ["Driver", "Articles"]
    fig_driver = px.bar(
        driver_counts,
        x="Driver",
        y="Articles",
        title="Driver Distribution",
        text="Articles",
    )
    fig_driver.update_layout(xaxis_title=None, yaxis_title="Articles")
    left.plotly_chart(fig_driver, use_container_width=True)

    sentiment_counts = df["Sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Articles"]
    fig_sentiment = px.pie(
        sentiment_counts,
        names="Sentiment",
        values="Articles",
        title="Sentiment Distribution",
        hole=0.45,
    )
    right.plotly_chart(fig_sentiment, use_container_width=True)

    left, right = st.columns(2)

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
    )
    fig_sub_driver.update_layout(xaxis_title="Articles", yaxis_title=None)
    left.plotly_chart(fig_sub_driver, use_container_width=True)

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
    )
    fig_driver_sentiment.update_layout(xaxis_title=None, yaxis_title="Articles")
    right.plotly_chart(fig_driver_sentiment, use_container_width=True)

    top_sub_drivers = df["Sub driver"].value_counts().head(10).reset_index()
    top_sub_drivers.columns = ["Sub Driver", "Articles"]
    fig_top_sub_drivers = px.bar(
        top_sub_drivers.sort_values("Articles"),
        x="Articles",
        y="Sub Driver",
        orientation="h",
        title="Top 10 Sub Drivers",
        text="Articles",
    )
    fig_top_sub_drivers.update_layout(xaxis_title="Articles", yaxis_title=None)
    st.plotly_chart(fig_top_sub_drivers, use_container_width=True)


def render_table(df: pd.DataFrame) -> None:
    """Render the interactive records table and CSV download."""
    display_columns = ["Title", "Driver", "Sub driver", "Sentiment", "Reason"]
    st.subheader("Content Explorer")
    st.dataframe(
        df[display_columns],
        use_container_width=True,
        hide_index=True,
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
        layout="wide",
    )

    st.title("BFSI Media Reputation Analytics Dashboard")

    try:
        df = load_data(DATA_PATH)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    st.sidebar.header("Filters")
    driver_options = sorted(df["Driver"].dropna().unique().tolist())
    selected_drivers = st.sidebar.multiselect("Driver", driver_options)

    driver_filtered_df = (
        df[df["Driver"].isin(selected_drivers)] if selected_drivers else df
    )
    sub_driver_options = sorted(driver_filtered_df["Sub driver"].dropna().unique().tolist())
    selected_sub_drivers = st.sidebar.multiselect("Sub Driver", sub_driver_options)

    sentiment_options = sorted(df["Sentiment"].dropna().unique().tolist())
    selected_sentiments = st.sidebar.multiselect("Sentiment", sentiment_options)

    title_search = st.sidebar.text_input("Search Titles")

    filtered_df = apply_filters(
        df=df,
        drivers=selected_drivers,
        sub_drivers=selected_sub_drivers,
        sentiments=selected_sentiments,
        title_search=title_search,
    )

    render_kpis(filtered_df)
    st.divider()
    render_charts(filtered_df)
    st.divider()
    render_table(filtered_df)


if __name__ == "__main__":
    main()
