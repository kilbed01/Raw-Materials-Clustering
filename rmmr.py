"""
To run, open the Command Prompt app and run streamlit run "[Python file path]".

The file will run in the browser.
"""

import csv
import datetime as dt
import os
from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from kneed import KneeLocator
from scipy.spatial import distance
from sklearn.cluster import KMeans


st.set_page_config(
    page_title="RMMR",
    page_icon="❓",
    initial_sidebar_state="expanded",
)


@st.cache_data
def convert_df(df: pd.DataFrame):
    """
    Convert df to csv and encode to be able to download.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    Encoded dataframe for downloading.

    """
    return df.to_csv()


def uniquify(path: str) -> str:
    """
    Check path for existing file. If file exists, increase counter in file
    path by 1 until file does not already exist.

    Parameters
    ----------
    path : str

    Returns
    -------
    path : str
        Updated with new counter.

    """
    filename, extension = os.path.splitext(path)
    counter = 1
    path = f"{filename}.{str(counter)}{extension}"
    while os.path.exists(path):
        counter += 1
        path = f"{filename}.{str(counter)}{extension}"
    return path


def find_headers(df):
    """
    Determine the proper color and PSD headers in Proficient

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    color_labels : list(str)
    psd_labels : list(str)
    color_headers : list(str)
    psd_headers : list(str)

    """

    if ("L_AVERAGE" in df.columns) and ("L_AVERAGE_WET" in df.columns):
        if df["L_AVERAGE"].isna().sum() <= df["L_AVERAGE_WET"].isna().sum():
            color_labels = ["L_AVERAGE", "A_AVERAGE", "B_AVERAGE"]
            psd_labels = ["PSD"]
        else:
            color_labels = ["L_AVERAGE_WET", "A_AVERAGE_WET", "B_AVERAGE_WET"]
            psd_labels = [
                "45_MICRON_CAMBRIA_MICROTRAC",
                "30_MICRON_CAMBRIA_MICROTRAC",
                "10_MICRON_CAMBRIA_MICROTRAC",
                "2_MICRON_CAMBRIA_MICROTRAC",
                "D10_CAMBRIA_MICROTRAC",
                "D50_CAMBRIA_MICROTRAC",
                "D90_CAMBRIA_MICROTRAC",
            ]
    elif "L_AVERAGE" in df.columns:
        color_labels = ["L_AVERAGE", "A_AVERAGE", "B_AVERAGE"]
        psd_labels = ["PSD"]
    elif "L_AVERAGE_WET" in df.columns:
        color_labels = ["L_AVERAGE_WET", "A_AVERAGE_WET", "B_AVERAGE_WET"]
        psd_labels = [
            "45_MICRON_CAMBRIA_MICROTRAC",
            "30_MICRON_CAMBRIA_MICROTRAC",
            "10_MICRON_CAMBRIA_MICROTRAC",
            "2_MICRON_CAMBRIA_MICROTRAC",
            "D10_CAMBRIA_MICROTRAC",
            "D50_CAMBRIA_MICROTRAC",
            "D90_CAMBRIA_MICROTRAC",
        ]

    base_headers = ["LOT", "BAG_NUMBERS", "PO_OR_BOL"]

    color_headers = base_headers.copy()
    color_headers.extend(color_labels)

    psd_headers = base_headers.copy()
    psd_headers.extend(psd_labels)

    return color_labels, psd_labels, color_headers, psd_headers


def find_parameters(material_type, test):
    """
    Determine proper parameters based on material type (Grit or Powder) and test type
    (Color or PSD)

    Parameters
    ----------
    material_type : str
    test : str

    Returns
    -------
    parameters : list(str)

    """

    if material_type == "Grit" and test == "Color":
        parameters = ["L_AVERAGE", "A_AVERAGE", "B_AVERAGE"]
    elif material_type == "Grit" and test == "PSD":
        parameters = ["PSD"]
    elif material_type == "Powder" and test == "Color":
        parameters = [
            "L_AVERAGE_WET",
            "A_AVERAGE_WET",
            "B_AVERAGE_WET",
        ]
    elif material_type == "Powder" and test == "PSD":
        parameters = [
            "45_MICRON_CAMBRIA_MICROTRAC",
            "30_MICRON_CAMBRIA_MICROTRAC",
            "10_MICRON_CAMBRIA_MICROTRAC",
            "2_MICRON_CAMBRIA_MICROTRAC",
        ]

    return parameters


def add_value_labels(ax, spacing=5):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """
    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        space = spacing
        va = "bottom"

        if y_value < 0:
            space *= -1
            va = "top"

        label = "{:.1f}".format(y_value)

        ax.annotate(
            label,
            (x_value, y_value),
            xytext=(0, space),
            textcoords="offset points",
            ha="center",
            va=va,
        )


def format_headers(df):
    """
    Replace spaces with underscores and capitalize the column headers

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    df : DataFrame

    """

    df.columns = df.columns.str.replace(" ", "_").str.upper()
    return df


def clean_proficient_data(df, headers):
    """
    Remove duplicate tests from specified columns.

    Parameters
    ----------
    df : DataFrame
    headers : list(str)

    Returns
    -------
    DataFrame

    """

    return df[headers].dropna().drop_duplicates(["LOT", "BAG_NUMBERS"])


def calc_lot_averages(df_rm, df_prof, headers, labels):
    """
    Create dataframe of lot averages for each label.

    Parameters
    ----------
    df_rm : DataFrame
    df_prof : DataFrame
    headers : list(str)
    labels : list(str)

    Returns
    -------
    df_fill_blanks : DataFrame

    """

    df_lot_avgs = df_prof.groupby("LOT")[labels].mean()

    df_fill_blanks = df_rm.merge(
        right=df_lot_avgs,
        left_on="LOT",
        right_on="LOT",
        how="left",
        suffixes=("", "_x"),
    )

    return df_fill_blanks


def merge_tests_and_averages(df_rm, df_prof, df_blanks, labels, test):
    """
    Merge dataframe with test results and update with dataframe with averages to fill in
    blank values.

    Parameters
    ----------
    df_rm : DataFrame
    df_prof : DataFrame
    df_blanks : DataFrame
    labels : list(str)
    test : str

    Returns
    -------
    df_merged : DataFrame

    """

    df_merged = df_rm.merge(
        right=df_prof,
        left_on=["LOT", "BAG"],
        right_on=["LOT", "BAG_NUMBERS"],
        how="left",
    )
    df_merged[f"{test.upper()}_RESULT_SOURCE"] = "Calculated"
    df_merged.loc[
        df_merged[labels[0]].notna(), f"{test.upper()}_RESULT_SOURCE"
    ] = "Tested"
    df_merged.update(df_blanks, overwrite=False)

    return df_merged


def set_inventory_dtypes(df):
    """
    Set the data types of columns in the dataframe

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    df : DataFrame

    """

    df = df.fillna(0)
    df["PHYSICAL_FORMAT"] = df["PHYSICAL_FORMAT"].astype("category")
    df["ITEM_DESCRIPTION"] = df["ITEM_DESCRIPTION"].astype("category")
    df["ITEM"] = df["ITEM"].astype("int64")
    df["LOT"] = df["LOT"].astype("string")
    df["BAG"] = df["BAG"].astype("int16")
    df["LOT_NUMBER"] = df["LOT_NUMBER"].astype("string")
    df["QTY_KG"] = df["QTY_KG"].astype("float64")
    df["QTY_LB"] = df["QTY_LB"].astype("float64")
    df["LOCATION"] = df["LOCATION"].astype("category")
    df["LOCATOR"] = df["LOCATOR"].astype("string")
    df["DATE_RECEIVED"] = pd.to_datetime(df["DATE_RECEIVED"])
    df["LAST_CHANGE_DATE"] = pd.to_datetime(df["LAST_CHANGE_DATE"])
    df["QA_STATUS"] = df["QA_STATUS"].astype("category")
    df["LOG_MESSAGE"] = df["LOG_MESSAGE"].astype("string")
    return df


def set_proficient_dtypes(df):
    """
    Set the data types of columns in the dataframe

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    df : DataFrame

    """

    df = df.fillna(0)
    df["PART"] = df["PART"].astype("category")
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["LOT"] = df["LOT"].astype("string")
    df["BAG_NUMBERS"] = df["BAG_NUMBERS"].astype("int16")
    return df


def run_tab1(df_rm):
    """
    Create tab for inventory breakdown

    Parameters
    ----------
    df_rm : DataFrame

    Returns
    -------
    material : str

    """

    material = st.selectbox("Material", sorted(df_rm["ITEM_DESCRIPTION"].unique()))

    status = st.multiselect(
        "Status",
        sorted(df_rm["QA_STATUS"].unique()),
        default=df_rm["QA_STATUS"].unique(),
    )

    df_rm = df_rm.query("QA_STATUS in @status and ITEM_DESCRIPTION == @material").copy()

    path = "https://raw.githubusercontent.com/bradsby/RMMR/main/Mgmt%20Review%20Log%20Message%20Key.csv"

    log_message_key = pd.read_csv(path)

    df_rm = df_rm.merge(
        right=log_message_key,
        left_on="LOG_MESSAGE",
        right_on="LOG_MESSAGE",
        how="left",
    )

    df_rm_summary = (
        df_rm.groupby(["ITEM_DESCRIPTION", "ALIAS"], as_index=False)
        .agg({"LOT_NUMBER": "count", "QTY_KG": "sum"})
        .rename(columns={"LOT_NUMBER": "BAG_COUNT", "QTY_KG": "QUANTITY_KG"})
        .sort_values("QUANTITY_KG", ascending=False)
    )

    sns.set_context("paper")

    fig = sns.catplot(
        data=df_rm_summary,
        x="ALIAS",
        y="BAG_COUNT",
        kind="bar",
    )

    plt.title(material)
    plt.xlabel("LOG_MESSAGE")
    plt.xticks(rotation=90)

    ax = fig.facet_axis(0, 0)
    for p in ax.patches:
        ax.text(
            p.get_x() + p.get_width() / 2,
            p.get_height() + p.get_width() / 2,
            "{0:.0f}".format(p.get_height()),
            color="black",
            rotation="horizontal",
            size="large",
            ha="center",
        )

    plt.tight_layout()

    st.pyplot(fig)

    now = dt.datetime.now().strftime("%y%m%d")
    output = uniquify(f"InventoryPlot.{material}.{now}.png")

    img = BytesIO()
    plt.savefig(img, format="png")

    st.download_button(
        label="Download plot as png",
        data=img,
        file_name=output,
        mime="image/png",
    )

    return material, df_rm


def run_tab2(df_rm, df_prof):
    """
    Run tab for merging inventory with Proficient data

    Parameters
    ----------
    df_rm : DataFrame
    df_prof : DataFrame

    Returns
    -------
    df_final : DataFrame

    """

    color_labels, psd_labels, color_headers, psd_headers = find_headers(df_prof)

    df_prof["BAG_NUMBERS"] = pd.to_numeric(df_prof["BAG_NUMBERS"], errors="coerce")

    df_color = clean_proficient_data(df_prof, color_headers)
    df_psd = clean_proficient_data(df_prof, psd_headers)

    df_prof_color_fill_blanks = calc_lot_averages(
        df_rm, df_color, color_headers, color_labels
    )
    df_prof_psd_fill_blanks = calc_lot_averages(df_rm, df_psd, psd_headers, psd_labels)

    df_final = merge_tests_and_averages(
        df_rm, df_color, df_prof_color_fill_blanks, color_labels, "Color"
    )
    df_final = merge_tests_and_averages(
        df_final, df_psd, df_prof_psd_fill_blanks, psd_labels, "PSD"
    )

    df_final = df_final.drop(
        ["LOT_NUMBER", "PO_OR_BOL_x", "BAG_NUMBERS_x", "PO_OR_BOL_y", "BAG_NUMBERS_y"],
        axis=1,
    )

    now = dt.datetime.now().strftime("%y%m%d")
    output = f"PROFxINV.{now}.csv"
    output = uniquify(output)

    st.dataframe(df_final)

    st.download_button(
        label="Download data as CSV",
        data=convert_df(df_final),
        file_name=output,
        mime="text/csv",
    )

    return df_final


def run_tab3(df_final):
    """
    Run tab for clustering data.

    Parameters
    ----------
    df_final : DataFrame

    Returns
    -------
    df_clusters : DataFrame
    parameters : list(str)

    """

    labels = ["LOT", "BAG", "LOCATION", "LOCATOR", "QTY_KG", "LOG_MESSAGE"]
    material_type = st.selectbox("Material Type", ["Grit", "Powder"])
    test = st.selectbox("QA Test", ["Color", "PSD"])
    log_message = st.multiselect(
        "LOG_MESSAGE",
        sorted(df_final["LOG_MESSAGE"].unique()),
        default=df_final["LOG_MESSAGE"].unique(),
    )

    set_max_dist_limit = st.checkbox("Set Max Distance Limit")

    if set_max_dist_limit:
        max_dist_limit = st.number_input("Max Distance", value=0.5)

    if st.checkbox("Calculate", key=1):
        df_clusters = df_final[df_final["LOG_MESSAGE"].isin(log_message)].copy()

        final_labels = labels.copy()

        parameters = find_parameters(material_type, test)

        final_labels.append(f"{test.upper()}_RESULT_SOURCE")
        final_labels.extend(parameters)

        df_clusters = df_clusters[final_labels].dropna()
        test_df = df_clusters[parameters]

        inertias = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, n_init="auto")
            kmeans.fit(test_df)
            inertias.append(kmeans.inertia_)

        number_of_clusters = KneeLocator(
            range(1, 11),
            inertias,
            curve="convex",
            direction="decreasing",
        ).elbow

        max_dist_from_centroid = 100
        if set_max_dist_limit:
            while max_dist_from_centroid > max_dist_limit:
                clustering = KMeans(n_clusters=number_of_clusters, n_init="auto").fit(
                    test_df
                )

                cluster_centers = clustering.cluster_centers_
                classes = clustering.labels_

                df_clusters["CLUSTER"] = classes
                for i, x in enumerate(parameters):
                    df_clusters[f"CLUSTER_CENTER_{x}"] = [
                        cluster_centers[j][i] for j in classes
                    ]

                df_clusters["DISTANCE_FROM_CLUSTER_CENTER"] = [
                    distance.euclidean(
                        df_clusters[parameters].values.tolist()[i],
                        cluster_centers[j],
                    )
                    for i, j in enumerate(classes)
                ]

                max_dist_from_centroid = df_clusters[
                    "DISTANCE_FROM_CLUSTER_CENTER"
                ].max()
                number_of_clusters += 1

        clustering = KMeans(n_clusters=number_of_clusters, n_init="auto").fit(test_df)

        cluster_centers = clustering.cluster_centers_
        classes = clustering.labels_

        df_clusters["CLUSTER"] = classes
        for i, x in enumerate(parameters):
            df_clusters[f"CLUSTER_CENTER_{x}"] = [
                cluster_centers[j][i] for j in classes
            ]

        df_clusters["DISTANCE_FROM_CLUSTER_CENTER"] = [
            distance.euclidean(
                df_clusters[parameters].values.tolist()[i],
                cluster_centers[j],
            )
            for i, j in enumerate(classes)
        ]

        plot_args = {"x": "CLUSTER", "y": "size", "kind": "bar"}

        if test == "Color":
            df_clusters["WITHIN_LIMIT"] = (
                df_clusters["DISTANCE_FROM_CLUSTER_CENTER"] <= 0.50
            )

            plot_args["hue"] = "WITHIN_LIMIT"
            plot_args["hue_order"] = [True, False]
            plot_args["palette"] = ["C2", "C3"]

            cluster_counts = df_clusters.groupby(
                ["CLUSTER", "WITHIN_LIMIT"], as_index=False
            ).size()

        else:
            cluster_counts = df_clusters.groupby("CLUSTER", as_index=False).size()

        plot_args["data"] = cluster_counts

        st.metric("Clusters", len(df_clusters["CLUSTER"].unique()))

        fig = sns.catplot(**plot_args)

        ax = fig.facet_axis(0, 0)
        for p in ax.patches:
            ax.text(
                p.get_x() + p.get_width() / 2,
                p.get_height() + p.get_width() / 2,
                "{0:.0f}".format(p.get_height()),
                color="black",
                rotation="horizontal",
                size="large",
                ha="center",
            )

        st.pyplot(fig)

        now = dt.datetime.now().strftime("%y%m%d")
        output = f"Clusters.{now}.csv"
        output = uniquify(output)

        st.download_button(
            label="Download data as CSV",
            data=convert_df(df_clusters),
            file_name=output,
            mime="text/csv",
        )

    return df_clusters, parameters


def run_tab4(df, parameters, material):
    """
    Run tab to make 3D plot of clusters.

    Parameters
    ----------
    df_clusters : DataFrame
    parameters : list(str)
    material : str

    Returns
    -------
    None.

    """

    x = st.selectbox("X", parameters, index=0)
    y = st.selectbox("Y", parameters, index=1)
    z = st.selectbox("Z", parameters, index=2)

    fig = px.scatter_3d(
        data_frame=df,
        x=x,
        y=y,
        z=z,
        color="CLUSTER",
        title=material,
    )

    make_spec_box = st.checkbox("Specification Box")
    if make_spec_box:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader(f"{x}")
            x_specs = [
                st.number_input(label="Upper", key=f"{x} Upper"),
                st.number_input(label="Lower", key=f"{x} Lower"),
            ]
        with col2:
            st.subheader(f"{y}")
            y_specs = [
                st.number_input(label="Upper", key=f"{y} Upper"),
                st.number_input(label="Lower", key=f"{y} Lower"),
            ]
        with col3:
            st.subheader(f"{z}")
            z_specs = [
                st.number_input(label="Upper", key=f"{z} Upper"),
                st.number_input(label="Lower", key=f"{z} Lower"),
            ]

        cube_data = {
            "x": [
                x_specs[0],
                x_specs[1],
                x_specs[0],
                x_specs[1],
                x_specs[0],
                x_specs[1],
                x_specs[0],
                x_specs[1],
            ],
            "y": [
                y_specs[1],
                y_specs[1],
                y_specs[0],
                y_specs[0],
                y_specs[1],
                y_specs[1],
                y_specs[0],
                y_specs[0],
            ],
            "z": [
                z_specs[1],
                z_specs[1],
                z_specs[1],
                z_specs[1],
                z_specs[0],
                z_specs[0],
                z_specs[0],
                z_specs[0],
            ],
            "opacity": 0.1,
            "color": "green",
            "alphahull": 1,
            "hoverinfo": "skip",
            "name": "Spec Limit",
            "hovertemplate": "b=%{x:.2f}<br>" + "a=%{y:.2f}<br>" + "L=%{z:.2f}",
        }

        fig.add_trace(go.Mesh3d(cube_data))

    fig.update_scenes(xaxis_autorange="reversed")
    fig.update_layout(title_x=0.5)
    fig.update_yaxes(automargin=True)

    if st.checkbox("Calculate", key=2):
        st.plotly_chart(fig, theme="streamlit")


def main():
    """
    Main gui

    Returns
    -------
    None.

    """

    path1 = st.sidebar.file_uploader(
        "Upload RM Inventory data", type="csv", accept_multiple_files=False
    )

    path_3 = st.sidebar.file_uploader(
        "Upload Proficient data", type="txt", accept_multiple_files=False
    )

    tab_names = [
        "Status-Reason Breakdown",
        "Proficient & Inventory Merger",
        "Data Clustering",
        "3D Plotting of Cluster",
    ]

    tab1, tab2, tab3, tab4 = st.tabs(tab_names)

    with tab1:
        if path1:
            df_rm = pd.read_csv(path1, thousands=",")

            df_rm = format_headers(df_rm)

            material, df_rm = run_tab1(df_rm)

        elif not path_3:
            st.warning("Upload RM Inventory data.")

    with tab2:
        if path1 and path_3:
            df_prof = pd.read_csv(
                path_3, sep="\t", parse_dates=["Date"], quoting=csv.QUOTE_NONE
            )

            df_prof = format_headers(df_prof)

            df_final = run_tab2(df_rm, df_prof)

        elif not path1 and not path_3:
            st.warning("Upload RM Inventory and Proficient data.")
        elif not path1:
            st.warning("Upload RM Inventory data.")
        elif not path_3:
            st.warning("Upload Proficient data.")

    with tab3:
        df_clusters = None
        if path1 and path_3:
            try:
                df_clusters, parameters = run_tab3(df_final)

            except Exception as e:
                st.error(e)

        elif not path1 and not path_3:
            st.warning("Upload RM Inventory and Proficient data.")

        elif not path1:
            st.warning("Upload RM Inventory data.")

        elif not path_3:
            st.warning("Upload Proficient data.")

    with tab4:
        if path1 and path_3:
            try:
                run_tab4(df_clusters, parameters, material)

            except Exception as e:
                st.error(e)

        elif not path1 and not path_3:
            st.warning("Upload RM Inventory and Proficient data.")
        elif not path1:
            st.warning("Upload RM Inventory data.")
        elif not path_3:
            st.warning("Upload Proficient data.")


if __name__ == "__main__":
    main()
