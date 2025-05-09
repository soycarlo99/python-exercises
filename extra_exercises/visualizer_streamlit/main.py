"""
Data Visualization Explorer

A Streamlit application for exploring and visualizing data from various file formats.
Users can upload data files, select columns, and generate different types of visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
from matplotlib.figure import Figure


def main():
    """Main function to run the Streamlit application."""
    # Set page configuration
    st.set_page_config(
        page_title="Data Visualization Explorer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Application title and description
    st.title("📊 Data Visualization Explorer")
    st.markdown("""
    Upload your data file and explore it through various visualizations.
    This tool supports CSV, Excel, TSV, and more file formats.
    """)

    # Initialize session state for data and chart size
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    if 'chart_size' not in st.session_state:
        st.session_state.chart_size = "medium"

    # Sidebar for global settings
    with st.sidebar:
        st.header("Global Settings")
        chart_size = st.radio(
            "Chart Size", 
            ["small", "medium", "large"],
            index=1,
            key="global_chart_size",
            horizontal=True
        )
        st.session_state.chart_size = chart_size

    # File upload section
    uploaded_file = st.file_uploader(
        "Upload your data file",
        type=["csv", "xlsx", "xls", "txt", "tsv", "json"]
    )

    if uploaded_file is not None:
        # Process the uploaded file
        try:
            df = load_data(uploaded_file)
            st.session_state.data = df
            
            # Show success message
            st.success(f"File loaded successfully: {uploaded_file.name}")
            
            # Display application sections
            data_explorer(df)
            visualization_section(df)
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    else:
        st.info("Please upload a data file to begin.")
        # If no file is uploaded, show example visualization options
        if st.checkbox("Show example visualizations"):
            # Create sample data
            df = create_sample_data()
            st.session_state.data = df
            st.write("Using sample data for demonstration:")
            st.dataframe(df.head())
            
            # Display application sections with sample data
            data_explorer(df)
            visualization_section(df)


def load_data(file):
    """
    Load data from various file formats.
    
    Args:
        file: The uploaded file object
    
    Returns:
        pandas.DataFrame: The loaded data
    """
    file_name = file.name.lower()
    
    if file_name.endswith('.csv'):
        return pd.read_csv(file)
    
    elif file_name.endswith('.xlsx') or file_name.endswith('.xls'):
        return pd.read_excel(file)
    
    elif file_name.endswith('.tsv') or file_name.endswith('.txt'):
        # Try to detect delimiter
        content = file.getvalue().decode('utf-8')
        if '\t' in content[:1000]:
            return pd.read_csv(file, sep='\t')
        elif ',' in content[:1000]:
            return pd.read_csv(file, sep=',')
        else:
            return pd.read_csv(file, delim_whitespace=True)
    
    elif file_name.endswith('.json'):
        return pd.read_json(file)
    
    else:
        raise ValueError(f"Unsupported file format: {file_name}")


def create_sample_data():
    """
    Create sample data for demonstration purposes.
    
    Returns:
        pandas.DataFrame: Sample data
    """
    # Create date range
    date_range = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Create numeric data with some correlation
    np.random.seed(42)
    base = np.random.normal(0, 1, 100)
    
    data = {
        'Date': date_range,
        'Temperature': base * 5 + 20,
        'Humidity': base * -2 + 60,
        'Pressure': base * 10 + 1000,
        'Wind_Speed': np.abs(base * 3 + 8),
        'Precipitation': np.abs(base * 0.5),
        'Category': np.random.choice(['A', 'B', 'C', 'D'], 100),
        'Value': np.random.exponential(10, 100)
    }
    
    return pd.DataFrame(data)


def data_explorer(df):
    """
    Display data exploration tools and information.
    
    Args:
        df: pandas.DataFrame containing the data
    """
    st.header("Data Explorer")
    
    # Basic information about the data
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Data Types", len(df.dtypes.unique()))
    
    # Data preview
    with st.expander("Data Preview", expanded=True):
        num_rows = st.slider("Number of rows to display", 5, 50, 10)
        st.dataframe(df.head(num_rows))
    
    # Column information
    with st.expander("Column Information"):
        col_info = pd.DataFrame({
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(col_info)
    
    # Descriptive statistics
    with st.expander("Descriptive Statistics"):
        # Only calculate statistics for numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty:
            st.dataframe(numeric_df.describe())
        else:
            st.info("No numeric columns found for statistics.")


def visualization_section(df):
    """
    Display the visualization section with various chart options.
    
    Args:
        df: pandas.DataFrame containing the data
    """
    st.header("Data Visualization")
    
    # Organize visualizations by category
    basic_tab, advanced_tab, statistical_tab = st.tabs([
        "Basic Visualizations", 
        "Advanced Visualizations", 
        "Statistical Visualizations"
    ])
    
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    
    # Basic Visualizations
    with basic_tab:
        st.subheader("Basic Visualizations")
        
        basic_viz_options = {
            "Line Chart": len(numeric_cols) > 0,
            "Bar Chart": len(numeric_cols) > 0,
            "Histogram": len(numeric_cols) > 0,
            "Scatter Plot": len(numeric_cols) >= 2,
            "Pie Chart": len(numeric_cols) > 0 and len(categorical_cols) > 0,
            "Box Plot": len(numeric_cols) > 0
        }
        
        # Display checkboxes for basic visualizations
        st.write("Select visualizations to display:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_line = st.checkbox("Line Chart", value=True)
            show_histogram = st.checkbox("Histogram", value=False)
        
        with col2:
            show_bar = st.checkbox("Bar Chart", value=False)
            show_scatter = st.checkbox("Scatter Plot", value=False)
        
        with col3:
            show_pie = st.checkbox("Pie Chart", value=False)
            show_box = st.checkbox("Box Plot", value=False)
        
        # Display selected visualizations
        if any([show_line, show_bar, show_histogram, show_scatter, show_pie, show_box]):
            # Line chart
            if show_line and basic_viz_options["Line Chart"]:
                display_line_chart(df, numeric_cols, datetime_cols)
            
            # Bar chart
            if show_bar and basic_viz_options["Bar Chart"]:
                display_bar_chart(df, numeric_cols, categorical_cols)
            
            # Histogram
            if show_histogram and basic_viz_options["Histogram"]:
                display_histogram(df, numeric_cols)
            
            # Scatter plot
            if show_scatter and basic_viz_options["Scatter Plot"]:
                display_scatter_plot(df, numeric_cols)
            
            # Pie chart
            if show_pie and basic_viz_options["Pie Chart"]:
                display_pie_chart(df, numeric_cols, categorical_cols)
            
            # Box plot
            if show_box and basic_viz_options["Box Plot"]:
                display_box_plot(df, numeric_cols, categorical_cols)
            
        else:
            st.info("Select at least one visualization to display.")
    
    # Advanced Visualizations
    with advanced_tab:
        st.subheader("Advanced Visualizations")
        
        advanced_viz_options = {
            "Heatmap": len(numeric_cols) >= 2,
            "Correlation Matrix": len(numeric_cols) >= 2,
            "Pair Plot": len(numeric_cols) >= 3,
            "Violin Plot": len(numeric_cols) > 0 and len(categorical_cols) > 0,
            "Area Chart": len(numeric_cols) > 0,
            "Bubble Chart": len(numeric_cols) >= 3
        }
        
        # Display checkboxes for advanced visualizations
        st.write("Select visualizations to display:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_heatmap = st.checkbox("Heatmap", value=False)
            show_pair = st.checkbox("Pair Plot", value=False)
        
        with col2:
            show_corr = st.checkbox("Correlation Matrix", value=True)
            show_violin = st.checkbox("Violin Plot", value=False)
        
        with col3:
            show_area = st.checkbox("Area Chart", value=False)
            show_bubble = st.checkbox("Bubble Chart", value=False)
        
        # Display selected visualizations
        if any([show_heatmap, show_corr, show_pair, show_violin, show_area, show_bubble]):
            # Heatmap
            if show_heatmap and advanced_viz_options["Heatmap"]:
                display_heatmap(df, numeric_cols)
            
            # Correlation matrix
            if show_corr and advanced_viz_options["Correlation Matrix"]:
                display_correlation_matrix(df, numeric_cols)
            
            # Pair plot
            if show_pair and advanced_viz_options["Pair Plot"]:
                display_pair_plot(df, numeric_cols, categorical_cols)
            
            # Violin plot
            if show_violin and advanced_viz_options["Violin Plot"]:
                display_violin_plot(df, numeric_cols, categorical_cols)
            
            # Area chart
            if show_area and advanced_viz_options["Area Chart"]:
                display_area_chart(df, numeric_cols, datetime_cols)
            
            # Bubble chart
            if show_bubble and advanced_viz_options["Bubble Chart"]:
                display_bubble_chart(df, numeric_cols)
            
        else:
            st.info("Select at least one visualization to display.")
    
    # Statistical Visualizations
    with statistical_tab:
        st.subheader("Statistical Visualizations")
        
        statistical_viz_options = {
            "Distribution Plot": len(numeric_cols) > 0,
            "Q-Q Plot": len(numeric_cols) > 0,
            "Residual Plot": len(numeric_cols) >= 2,
            "Time Series": len(datetime_cols) > 0 and len(numeric_cols) > 0
        }
        
        # Display checkboxes for statistical visualizations
        st.write("Select visualizations to display:")
        col1, col2 = st.columns(2)
        
        with col1:
            show_dist = st.checkbox("Distribution Plot", value=True)
            show_qq = st.checkbox("Q-Q Plot", value=False)
        
        with col2:
            show_residual = st.checkbox("Residual Plot", value=False)
            show_time = st.checkbox("Time Series Analysis", value=False)
        
        # Display selected visualizations
        if any([show_dist, show_qq, show_residual, show_time]):
            # Distribution plot
            if show_dist and statistical_viz_options["Distribution Plot"]:
                display_distribution_plot(df, numeric_cols)
            
            # Q-Q plot
            if show_qq and statistical_viz_options["Q-Q Plot"]:
                display_qq_plot(df, numeric_cols)
            
            # Residual plot
            if show_residual and statistical_viz_options["Residual Plot"]:
                display_residual_plot(df, numeric_cols)
            
            # Time series analysis
            if show_time and statistical_viz_options["Time Series"]:
                display_time_series(df, numeric_cols, datetime_cols)
            
        else:
            st.info("Select at least one visualization to display.")


def get_chart_dimensions():
    """Get figure dimensions based on the selected chart size."""
    chart_size = st.session_state.chart_size
    
    if chart_size == "small":
        return {
            "basic": (5, 3),
            "square": (5, 4),
            "wide": (6, 3),
            "double_wide": (8, 3),
            "tall": (5, 6),
            "panel": (6, 6)
        }
    elif chart_size == "medium":
        return {
            "basic": (7, 4),
            "square": (7, 5),
            "wide": (8, 5),
            "double_wide": (9, 4),
            "tall": (7, 7),
            "panel": (8, 8)
        }
    else:  # large
        return {
            "basic": (10, 6),
            "square": (10, 8),
            "wide": (12, 8),
            "double_wide": (14, 6),
            "tall": (10, 10),
            "panel": (12, 10)
        }


def create_chart_container(chart_type="basic"):
    """Create a container with appropriate width based on chart size."""
    chart_size = st.session_state.chart_size
    
    if chart_size == "small":
        container_widths = {"basic": 0.5, "square": 0.5, "wide": 0.7, "double_wide": 0.8, "tall": 0.5, "panel": 0.7}
    elif chart_size == "medium":
        container_widths = {"basic": 0.7, "square": 0.7, "wide": 0.8, "double_wide": 0.9, "tall": 0.7, "panel": 0.8}
    else:
        container_widths = {"basic": 0.95, "square": 0.95, "wide": 0.98, "double_wide": 0.98, "tall": 0.95, "panel": 0.98}
    
    width = container_widths.get(chart_type, 0.7)
    
    if width < 1.0:
        col1, col2, col3 = st.columns([0.5 * (1 - width), width, 0.5 * (1 - width)])
        return col2
    else:
        return st


# Basic Visualization Functions

def display_line_chart(df, numeric_cols, datetime_cols):
    """Display a line chart."""
    st.subheader("Line Chart")
    
    # UI controls
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if datetime_cols:
            x_axis = st.selectbox("X-axis (Line Chart)", datetime_cols, key="line_x")
        else:
            x_axis = st.selectbox("X-axis (Line Chart)", numeric_cols, key="line_x")
    
    with col2:
        y_axes = st.multiselect(
            "Y-axis (Line Chart)", 
            numeric_cols,
            default=[numeric_cols[0]] if numeric_cols else [],
            key="line_y"
        )
    
    # Create a container for the chart
    container = create_chart_container("basic")
    
    # Plot if selections are made
    if y_axes:
        dimensions = get_chart_dimensions()
        fig, ax = plt.subplots(figsize=dimensions["basic"])
        
        for y in y_axes:
            ax.plot(df[x_axis], df[y], marker='o', linestyle='-', label=y)
        
        ax.set_xlabel(x_axis)
        ax.set_ylabel("Value")
        ax.set_title(f"Line Chart of {', '.join(y_axes)} by {x_axis}")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Format x-axis if it's a datetime
        if x_axis in datetime_cols:
            plt.gcf().autofmt_xdate()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        with container:
            st.pyplot(fig)
        
        # Download option
        st.download_button(
            "Download Line Chart",
            data=get_figure_as_bytes(fig),
            file_name="line_chart.png",
            mime="image/png"
        )
    else:
        st.info("Please select at least one column for the Y-axis.")


def display_bar_chart(df, numeric_cols, categorical_cols):
    """Display a bar chart."""
    st.subheader("Bar Chart")
    
    # UI controls
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if categorical_cols:
            x_axis = st.selectbox("X-axis (Bar Chart)", categorical_cols, key="bar_x")
        else:
            x_axis = st.selectbox("X-axis (Bar Chart)", numeric_cols, key="bar_x")
    
    with col2:
        y_axis = st.selectbox("Y-axis (Bar Chart)", numeric_cols, key="bar_y")
    
    with col3:
        agg_func = st.selectbox(
            "Aggregation Function",
            ["sum", "mean", "median", "count", "min", "max"],
            index=1,
            key="bar_agg"
        )
    
    # Additional options
    orientation = st.radio("Orientation", ["Vertical", "Horizontal"], key="bar_orient")
    show_values = st.checkbox("Show Values on Bars", value=True, key="bar_values")
    
    # Create a container for the chart
    container = create_chart_container("basic")
    
    # Prepare data
    if x_axis in categorical_cols:
        if agg_func == "sum":
            plot_data = df.groupby(x_axis)[y_axis].sum().reset_index()
        elif agg_func == "mean":
            plot_data = df.groupby(x_axis)[y_axis].mean().reset_index()
        elif agg_func == "median":
            plot_data = df.groupby(x_axis)[y_axis].median().reset_index()
        elif agg_func == "count":
            plot_data = df.groupby(x_axis)[y_axis].count().reset_index()
        elif agg_func == "min":
            plot_data = df.groupby(x_axis)[y_axis].min().reset_index()
        elif agg_func == "max":
            plot_data = df.groupby(x_axis)[y_axis].max().reset_index()
    else:
        # If no categorical columns, create bins for numeric x-axis
        plot_data = df.copy()
    
    # Create plot
    dimensions = get_chart_dimensions()
    fig, ax = plt.subplots(figsize=dimensions["basic"])
    
    if orientation == "Vertical":
        bars = ax.bar(plot_data[x_axis], plot_data[y_axis], color='skyblue')
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        if len(plot_data[x_axis]) > 10:
            plt.xticks(rotation=45, ha='right')
    else:  # Horizontal
        bars = ax.barh(plot_data[x_axis], plot_data[y_axis], color='skyblue')
        ax.set_ylabel(x_axis)
        ax.set_xlabel(y_axis)
    
    # Add value labels if requested
    if show_values:
        if orientation == "Vertical":
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + (plot_data[y_axis].max() * 0.01),
                    f'{height:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=8
                )
        else:  # Horizontal
            for bar in bars:
                width = bar.get_width()
                ax.text(
                    width + (plot_data[y_axis].max() * 0.01),
                    bar.get_y() + bar.get_height() / 2,
                    f'{width:.2f}',
                    ha='left',
                    va='center',
                    fontsize=8
                )
    
    ax.set_title(f"Bar Chart of {agg_func.capitalize()}({y_axis}) by {x_axis}")
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    with container:
        st.pyplot(fig)
    
    # Download option
    st.download_button(
        "Download Bar Chart",
        data=get_figure_as_bytes(fig),
        file_name="bar_chart.png",
        mime="image/png"
    )


def display_histogram(df, numeric_cols):
    """Display a histogram."""
    st.subheader("Histogram")
    
    # UI controls
    col1, col2 = st.columns([1, 1])
    
    with col1:
        column = st.selectbox("Column (Histogram)", numeric_cols, key="hist_col")
    
    with col2:
        bins = st.slider("Number of Bins", 5, 100, 20, key="hist_bins")
    
    # Additional options
    kde = st.checkbox("Show Density Curve (KDE)", value=True, key="hist_kde")
    cumulative = st.checkbox("Cumulative Histogram", value=False, key="hist_cum")
    
    # Create a container for the chart
    container = create_chart_container("basic")
    
    # Create plot
    dimensions = get_chart_dimensions()
    fig, ax = plt.subplots(figsize=dimensions["basic"])
    
    if kde:
        sns.histplot(
            df[column],
            bins=bins,
            kde=True,
            cumulative=cumulative,
            color='skyblue',
            edgecolor='black',
            alpha=0.7,
            ax=ax
        )
    else:
        ax.hist(
            df[column],
            bins=bins,
            cumulative=cumulative,
            color='skyblue',
            edgecolor='black',
            alpha=0.7
        )
    
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    
    if cumulative:
        ax.set_title(f"Cumulative Histogram of {column}")
    else:
        ax.set_title(f"Histogram of {column}")
    
    ax.grid(True, linestyle='--', alpha=0.7)
    
    with container:
        st.pyplot(fig)
    
    # Download option
    st.download_button(
        "Download Histogram",
        data=get_figure_as_bytes(fig),
        file_name="histogram.png",
        mime="image/png"
    )


def display_scatter_plot(df, numeric_cols):
    """Display a scatter plot."""
    st.subheader("Scatter Plot")
    
    # UI controls
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        x_axis = st.selectbox("X-axis (Scatter Plot)", numeric_cols, key="scatter_x")
    
    with col2:
        y_axis = st.selectbox(
            "Y-axis (Scatter Plot)",
            [col for col in numeric_cols if col != x_axis],
            key="scatter_y"
        )
    
    with col3:
        color_by = st.selectbox(
            "Color by (optional)",
            ["None"] + df.columns.tolist(),
            key="scatter_color"
        )
    
    # Additional options
    size_by = st.selectbox(
        "Size by (optional)",
        ["None"] + numeric_cols,
        key="scatter_size"
    )
    
    # Create a container for the chart
    container = create_chart_container("square")
    
    # Create plot
    dimensions = get_chart_dimensions()
    fig, ax = plt.subplots(figsize=dimensions["square"])
    
    # Basic scatter plot
    if color_by == "None" and size_by == "None":
        ax.scatter(df[x_axis], df[y_axis], alpha=0.7)
    
    # Scatter plot with color
    elif color_by != "None" and size_by == "None":
        if df[color_by].dtype in ['object', 'category']:
            for category in df[color_by].unique():
                subset = df[df[color_by] == category]
                ax.scatter(
                    subset[x_axis],
                    subset[y_axis],
                    alpha=0.7,
                    label=category
                )
            ax.legend(title=color_by)
        else:
            scatter = ax.scatter(
                df[x_axis],
                df[y_axis],
                c=df[color_by],
                alpha=0.7,
                cmap='viridis'
            )
            plt.colorbar(scatter, label=color_by)
    
    # Scatter plot with size
    elif color_by == "None" and size_by != "None":
        # Normalize size
        size = 10 + (df[size_by] - df[size_by].min()) / (df[size_by].max() - df[size_by].min()) * 200
        ax.scatter(df[x_axis], df[y_axis], s=size, alpha=0.7)
    
    # Scatter plot with color and size
    else:
        # Normalize size
        size = 10 + (df[size_by] - df[size_by].min()) / (df[size_by].max() - df[size_by].min()) * 200
        
        if df[color_by].dtype in ['object', 'category']:
            for category in df[color_by].unique():
                subset = df[df[color_by] == category]
                subset_size = 10 + (subset[size_by] - df[size_by].min()) / (df[size_by].max() - df[size_by].min()) * 200
                ax.scatter(
                    subset[x_axis],
                    subset[y_axis],
                    s=subset_size,
                    alpha=0.7,
                    label=category
                )
            ax.legend(title=color_by)
        else:
            scatter = ax.scatter(
                df[x_axis],
                df[y_axis],
                c=df[color_by],
                s=size,
                alpha=0.7,
                cmap='viridis'
            )
            plt.colorbar(scatter, label=color_by)
    
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(f"Scatter Plot of {y_axis} vs {x_axis}")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    with container:
        st.pyplot(fig)
    
    # Download option
    st.download_button(
        "Download Scatter Plot",
        data=get_figure_as_bytes(fig),
        file_name="scatter_plot.png",
        mime="image/png"
    )


def display_pie_chart(df, numeric_cols, categorical_cols):
    """Display a pie chart."""
    st.subheader("Pie Chart")
    
    # UI controls
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if categorical_cols:
            category_col = st.selectbox("Category Column", categorical_cols, key="pie_cat")
        else:
            category_col = st.selectbox(
                "Category Column",
                numeric_cols,
                key="pie_cat"
            )
    
    with col2:
        value_col = st.selectbox("Value Column", numeric_cols, key="pie_val")
    
    # Additional options
    agg_func = st.selectbox(
        "Aggregation Function",
        ["sum", "mean", "median", "count"],
        index=0,
        key="pie_agg"
    )
    
    max_slices = st.slider(
        "Maximum number of slices (others will be grouped)",
        3,
        15,
        8,
        key="pie_slices"
    )
    
    # Create a container for the chart
    container = create_chart_container("square")
    
    # Calculate aggregated data
    if agg_func == "sum":
        agg_data = df.groupby(category_col)[value_col].sum()
    elif agg_func == "mean":
        agg_data = df.groupby(category_col)[value_col].mean()
    elif agg_func == "median":
        agg_data = df.groupby(category_col)[value_col].median()
    else:  # count
        agg_data = df.groupby(category_col)[value_col].count()
    
    # Sort and limit slices
    agg_data = agg_data.sort_values(ascending=False)
    
    if len(agg_data) > max_slices:
        top_data = agg_data.iloc[:max_slices-1]
        others = pd.Series(
            agg_data.iloc[max_slices-1:].sum(),
            index=["Others"]
        )
        plot_data = pd.concat([top_data, others])
    else:
        plot_data = agg_data
    
    # Create plot
    dimensions = get_chart_dimensions()
    fig, ax = plt.subplots(figsize=dimensions["square"])
    
    # Create explode array (explode largest slice)
    explode = [0] * len(plot_data)
    explode[0] = 0.1
    
    wedges, texts, autotexts = ax.pie(
        plot_data,
        autopct='%1.1f%%',
        explode=explode,
        shadow=True,
        startangle=90,
        textprops={'fontsize': 10}
    )
    
    ax.set_title(f"Pie Chart of {agg_func.capitalize()}({value_col}) by {category_col}")
    ax.legend(
        wedges,
        plot_data.index,
        title=category_col,
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    
    with container:
        st.pyplot(fig)
    
    # Download option
    st.download_button(
        "Download Pie Chart",
        data=get_figure_as_bytes(fig),
        file_name="pie_chart.png",
        mime="image/png"
    )


def display_box_plot(df, numeric_cols, categorical_cols):
    """Display a box plot."""
    st.subheader("Box Plot")
    
    # UI controls
    col1, col2 = st.columns([1, 1])
    
    with col1:
        y_axis = st.selectbox("Value Column (Box Plot)", numeric_cols, key="box_y")
    
    with col2:
        if categorical_cols:
            x_axis = st.selectbox(
                "Category Column (optional)",
                ["None"] + categorical_cols,
                key="box_x"
            )
        else:
            x_axis = "None"
    
    # Additional options
    show_points = st.checkbox("Show Individual Points", value=False, key="box_points")
    
    # Create a container for the chart
    container = create_chart_container("basic")
    
    # Create plot
    dimensions = get_chart_dimensions()
    fig, ax = plt.subplots(figsize=dimensions["basic"])
    
    if x_axis != "None":
        # Box plot with categories
        if show_points:
            sns.boxplot(x=x_axis, y=y_axis, data=df, ax=ax)
            sns.stripplot(
                x=x_axis,
                y=y_axis,
                data=df,
                ax=ax,
                color='black',
                alpha=0.4,
                size=3
            )
        else:
            sns.boxplot(x=x_axis, y=y_axis, data=df, ax=ax)
        
        ax.set_title(f"Box Plot of {y_axis} by {x_axis}")
        
        if len(df[x_axis].unique()) > 6:
            plt.xticks(rotation=45, ha='right')
    else:
        # Simple box plot
        if show_points:
            sns.boxplot(y=y_axis, data=df, ax=ax)
            sns.stripplot(
                y=y_axis,
                data=df,
                ax=ax,
                color='black',
                alpha=0.4,
                size=3
            )
        else:
            sns.boxplot(y=y_axis, data=df, ax=ax)
        
        ax.set_title(f"Box Plot of {y_axis}")
    
    ax.grid(True, linestyle='--', alpha=0.7)
    
    with container:
        st.pyplot(fig)
    
    # Download option
    st.download_button(
        "Download Box Plot",
        data=get_figure_as_bytes(fig),
        file_name="box_plot.png",
        mime="image/png"
    )


# Advanced Visualization Functions

def display_heatmap(df, numeric_cols):
    """Display a heatmap of numeric data."""
    st.subheader("Heatmap")
    
    # UI controls
    cols = st.multiselect(
        "Select columns for heatmap",
        numeric_cols,
        default=numeric_cols[:min(5, len(numeric_cols))],
        key="heatmap_cols"
    )
    
    # Additional options
    col1, col2 = st.columns([1, 1])
    
    with col1:
        cmap = st.selectbox(
            "Color Map",
            ["viridis", "plasma", "inferno", "magma", "cividis", "RdBu_r", "coolwarm"],
            index=5,
            key="heatmap_cmap"
        )
    
    with col2:
        show_values = st.checkbox("Show Values", value=True, key="heatmap_vals")
    
    # Create a container for the chart
    container = create_chart_container("wide")
    
    # Plot if columns are selected
    if cols:
        # Sample data if too large
        plot_df = df[cols]
        if len(plot_df) > 100:
            st.info(f"Data has been sampled to 100 rows (from {len(plot_df)}) for visualization.")
            plot_df = plot_df.sample(100, random_state=42)
        
        # Create plot
        dimensions = get_chart_dimensions()
        fig, ax = plt.subplots(figsize=dimensions["wide"])
        
        sns.heatmap(
            plot_df,
            annot=show_values,
            fmt=".2f" if show_values else "",
            cmap=cmap,
            linewidths=0.5,
            ax=ax
        )
        
        ax.set_title("Heatmap of Selected Columns")
        plt.xticks(rotation=45, ha='right')
        
        with container:
            st.pyplot(fig)
        
        # Download option
        st.download_button(
            "Download Heatmap",
            data=get_figure_as_bytes(fig),
            file_name="heatmap.png",
            mime="image/png"
        )
    else:
        st.info("Please select at least one column for the heatmap.")


def display_correlation_matrix(df, numeric_cols):
    """Display a correlation matrix of numeric columns."""
    st.subheader("Correlation Matrix")
    
    # UI controls
    cols = st.multiselect(
        "Select columns for correlation matrix",
        numeric_cols,
        default=numeric_cols[:min(8, len(numeric_cols))],
        key="corr_cols"
    )
    
    # Additional options
    col1, col2 = st.columns([1, 1])
    
    with col1:
        corr_method = st.selectbox(
            "Correlation Method",
            ["pearson", "kendall", "spearman"],
            index=0,
            key="corr_method"
        )
    
    with col2:
        cmap = st.selectbox(
            "Color Map",
            ["coolwarm", "RdBu_r", "viridis", "plasma"],
            index=0,
            key="corr_cmap"
        )
    
    show_values = st.checkbox("Show Correlation Values", value=True, key="corr_vals")
    
    # Create a container for the chart
    container = create_chart_container("square")
    
    # Plot if columns are selected
    if cols:
        # Calculate correlation matrix
        corr_matrix = df[cols].corr(method=corr_method)
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Create plot
        dimensions = get_chart_dimensions()
        fig, ax = plt.subplots(figsize=dimensions["square"])
        
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=show_values,
            fmt=".2f" if show_values else "",
            cmap=cmap,
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=0.5,
            ax=ax
        )
        
        ax.set_title(f"{corr_method.capitalize()} Correlation Matrix")
        
        with container:
            st.pyplot(fig)
        
        # Download option
        st.download_button(
            "Download Correlation Matrix",
            data=get_figure_as_bytes(fig),
            file_name="correlation_matrix.png",
            mime="image/png"
        )
        
        # Show correlation table
        with st.expander("View Correlation Table"):
            st.dataframe(corr_matrix)
    else:
        st.info("Please select at least two columns for the correlation matrix.")


def display_pair_plot(df, numeric_cols, categorical_cols):
    """Display a pair plot of numeric columns."""
    st.subheader("Pair Plot")
    
    # UI controls
    # Limit columns to prevent overcrowded plots
    max_numeric = min(5, len(numeric_cols))
    
    cols = st.multiselect(
        "Select columns for pair plot (max 5 recommended)",
        numeric_cols,
        default=numeric_cols[:min(3, len(numeric_cols))],
        key="pair_cols"
    )
    
    # Additional options
    if categorical_cols:
        hue_col = st.selectbox(
            "Color by (optional)",
            ["None"] + categorical_cols,
            key="pair_hue"
        )
    else:
        hue_col = "None"
    
    # Create a container for the chart
    container = create_chart_container("panel")
    
    # Plot if columns are selected
    if cols:
        if len(cols) > 5:
            st.warning("Too many columns may make the pair plot difficult to read.")
        
        # Sample data if too large
        if len(df) > 1000:
            st.info(f"Data has been sampled to 1000 rows (from {len(df)}) for visualization.")
            plot_df = df.sample(1000, random_state=42)
        else:
            plot_df = df
        
        # Create figure
        if hue_col != "None":
            # Limit to 5 categories to avoid too many colors
            if len(plot_df[hue_col].unique()) > 5:
                top_categories = plot_df[hue_col].value_counts().nlargest(5).index
                plot_df = plot_df[plot_df[hue_col].isin(top_categories)]
                st.info(f"Limited to top 5 categories of {hue_col} for clarity.")
            
            fig = sns.pairplot(
                plot_df,
                vars=cols,
                hue=hue_col,
                diag_kind="kde",
                corner=True,
                plot_kws={"alpha": 0.6}
            )
        else:
            fig = sns.pairplot(
                plot_df,
                vars=cols,
                diag_kind="kde",
                corner=True,
                plot_kws={"alpha": 0.6}
            )
        
        # Adjust figure size based on chart size setting
        dimensions = get_chart_dimensions()
        fig.fig.set_size_inches(dimensions["tall"][0], dimensions["tall"][1])
        
        fig.fig.suptitle("Pair Plot of Selected Variables", y=1.02, fontsize=16)
        fig.fig.subplots_adjust(top=0.95)
        
        with container:
            st.pyplot(fig)
        
        # Download option
        st.download_button(
            "Download Pair Plot",
            data=get_figure_as_bytes(fig),
            file_name="pair_plot.png",
            mime="image/png"
        )
    else:
        st.info("Please select at least two columns for the pair plot.")


def display_violin_plot(df, numeric_cols, categorical_cols):
    """Display a violin plot."""
    st.subheader("Violin Plot")
    
    # UI controls
    col1, col2 = st.columns([1, 1])
    
    with col1:
        y_axis = st.selectbox("Value Column (Violin Plot)", numeric_cols, key="violin_y")
    
    with col2:
        if categorical_cols:
            x_axis = st.selectbox("Category Column", categorical_cols, key="violin_x")
        else:
            st.info("No categorical columns available for violin plot x-axis.")
            return
    
    # Additional options
    col1, col2 = st.columns([1, 1])
    
    with col1:
        split = st.checkbox("Split Violin", value=False, key="violin_split")
    
    with col2:
        show_points = st.checkbox("Show Data Points", value=True, key="violin_points")
    
    # Create a container for the chart
    container = create_chart_container("basic")
    
    # Create plot
    dimensions = get_chart_dimensions()
    fig, ax = plt.subplots(figsize=dimensions["basic"])
    
    # Limit categories if too many
    if len(df[x_axis].unique()) > 10:
        top_categories = df[x_axis].value_counts().nlargest(10).index
        plot_df = df[df[x_axis].isin(top_categories)]
        st.info(f"Limited to top 10 categories of {x_axis} for clarity.")
    else:
        plot_df = df
    
    if show_points:
        sns.violinplot(
            x=x_axis,
            y=y_axis,
            data=plot_df,
            split=split,
            inner="quartile",
            ax=ax
        )
        sns.stripplot(
            x=x_axis,
            y=y_axis,
            data=plot_df,
            color="black",
            alpha=0.3,
            size=3,
            ax=ax
        )
    else:
        sns.violinplot(
            x=x_axis,
            y=y_axis,
            data=plot_df,
            split=split,
            inner="quartile",
            ax=ax
        )
    
    ax.set_title(f"Violin Plot of {y_axis} by {x_axis}")
    
    if len(plot_df[x_axis].unique()) > 6:
        plt.xticks(rotation=45, ha='right')
    
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    with container:
        st.pyplot(fig)
    
    # Download option
    st.download_button(
        "Download Violin Plot",
        data=get_figure_as_bytes(fig),
        file_name="violin_plot.png",
        mime="image/png"
    )


def display_area_chart(df, numeric_cols, datetime_cols):
    """Display an area chart."""
    st.subheader("Area Chart")
    
    # UI controls
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if datetime_cols:
            x_axis = st.selectbox("X-axis (Area Chart)", datetime_cols, key="area_x")
        else:
            x_axis = st.selectbox("X-axis (Area Chart)", numeric_cols, key="area_x")
    
    with col2:
        y_axes = st.multiselect(
            "Y-axis (Area Chart)",
            numeric_cols,
            default=[numeric_cols[0]] if numeric_cols else [],
            key="area_y"
        )
    
    # Additional options
    stacked = st.checkbox("Stacked Area Chart", value=False, key="area_stacked")
    normalize = st.checkbox("Normalize (100% stacked)", value=False, key="area_norm")
    
    # Create a container for the chart
    container = create_chart_container("basic")
    
    # Plot if selections are made
    if y_axes:
        dimensions = get_chart_dimensions()
        fig, ax = plt.subplots(figsize=dimensions["basic"])
        
        # Prepare data
        plot_df = df[[x_axis] + y_axes].sort_values(x_axis)
        
        if normalize and stacked:
            # Calculate percentages for 100% stacked
            for i in range(len(y_axes)):
                if i == 0:
                    plot_df[f"{y_axes[i]}_norm"] = plot_df[y_axes[i]]
                else:
                    plot_df[f"{y_axes[i]}_norm"] = plot_df[y_axes[i]] + plot_df[f"{y_axes[i-1]}_norm"]
            
            # Normalize to 100%
            total = plot_df[f"{y_axes[-1]}_norm"]
            for i in range(len(y_axes)):
                plot_df[f"{y_axes[i]}_norm"] = 100 * plot_df[f"{y_axes[i]}_norm"] / total
            
            # Plot normalized stacked area
            for i in range(len(y_axes)-1, -1, -1):
                if i == 0:
                    ax.fill_between(
                        plot_df[x_axis],
                        0,
                        plot_df[f"{y_axes[i]}_norm"],
                        label=y_axes[i],
                        alpha=0.7
                    )
                else:
                    ax.fill_between(
                        plot_df[x_axis],
                        plot_df[f"{y_axes[i-1]}_norm"],
                        plot_df[f"{y_axes[i]}_norm"],
                        label=y_axes[i],
                        alpha=0.7
                    )
            
            ax.set_ylabel("Percentage (%)")
            ax.set_title(f"100% Stacked Area Chart by {x_axis}")
            
        elif stacked:
            # Regular stacked area chart
            ax.stackplot(
                plot_df[x_axis],
                [plot_df[y] for y in y_axes],
                labels=y_axes,
                alpha=0.7
            )
            ax.set_ylabel("Value")
            ax.set_title(f"Stacked Area Chart by {x_axis}")
            
        else:
            # Regular area chart (not stacked)
            for y in y_axes:
                ax.fill_between(
                    plot_df[x_axis],
                    plot_df[y],
                    alpha=0.3,
                    label=y
                )
                ax.plot(plot_df[x_axis], plot_df[y], lw=2)
            
            ax.set_ylabel("Value")
            ax.set_title(f"Area Chart by {x_axis}")
        
        ax.set_xlabel(x_axis)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Format x-axis if it's a datetime
        if x_axis in datetime_cols:
            plt.gcf().autofmt_xdate()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        with container:
            st.pyplot(fig)
        
        # Download option
        st.download_button(
            "Download Area Chart",
            data=get_figure_as_bytes(fig),
            file_name="area_chart.png",
            mime="image/png"
        )
    else:
        st.info("Please select at least one column for the Y-axis.")


def display_bubble_chart(df, numeric_cols):
    """Display a bubble chart."""
    st.subheader("Bubble Chart")
    
    # Need at least 3 numeric columns for bubble chart
    if len(numeric_cols) < 3:
        st.info("Bubble chart requires at least 3 numeric columns (X, Y, and Size).")
        return
    
    # UI controls
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        x_axis = st.selectbox("X-axis (Bubble Chart)", numeric_cols, key="bubble_x")
    
    with col2:
        y_axis = st.selectbox(
            "Y-axis (Bubble Chart)",
            [col for col in numeric_cols if col != x_axis],
            key="bubble_y"
        )
    
    with col3:
        size_col = st.selectbox(
            "Bubble Size",
            [col for col in numeric_cols if col not in [x_axis, y_axis]],
            key="bubble_size"
        )
    
    # Additional options
    color_by = st.selectbox(
        "Color by (optional)",
        ["None"] + df.columns.tolist(),
        key="bubble_color"
    )
    
    # Create a container for the chart
    container = create_chart_container("square")
    
    # Create plot
    dimensions = get_chart_dimensions()
    fig, ax = plt.subplots(figsize=dimensions["square"])
    
    # Normalize size for bubbles
    size = 20 + (df[size_col] - df[size_col].min()) / (df[size_col].max() - df[size_col].min()) * 500
    
    # Basic bubble chart
    if color_by == "None":
        scatter = ax.scatter(
            df[x_axis],
            df[y_axis],
            s=size,
            alpha=0.6,
            edgecolors='w',
            linewidth=0.5
        )
        
        # Add size legend
        handles, labels = create_size_legend(size_col, df[size_col], size)
        ax.legend(handles, labels, title=size_col, loc="upper left", bbox_to_anchor=(1, 1))
        
    # Bubble chart with color
    else:
        if df[color_by].dtype in ['object', 'category']:
            for category in df[color_by].unique():
                subset = df[df[color_by] == category]
                subset_size = 20 + (subset[size_col] - df[size_col].min()) / (df[size_col].max() - df[size_col].min()) * 500
                ax.scatter(
                    subset[x_axis],
                    subset[y_axis],
                    s=subset_size,
                    alpha=0.6,
                    label=category,
                    edgecolors='w',
                    linewidth=0.5
                )
            
            # Add category legend
            ax.legend(title=color_by, loc="upper left", bbox_to_anchor=(1, 1))
            
            # Add size legend
            add_size_legend(ax, size_col, df[size_col])
            
        else:
            scatter = ax.scatter(
                df[x_axis],
                df[y_axis],
                s=size,
                c=df[color_by],
                alpha=0.6,
                cmap='viridis',
                edgecolors='w',
                linewidth=0.5
            )
            
            # Add colorbar
            cbar = plt.colorbar(scatter, label=color_by)
            
            # Add size legend
            handles, labels = create_size_legend(size_col, df[size_col], size)
            ax.legend(handles, labels, title=size_col, loc="upper left", bbox_to_anchor=(1, 1))
    
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(f"Bubble Chart of {y_axis} vs {x_axis} (Size: {size_col})")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Make room for legends
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    
    with container:
        st.pyplot(fig)
    
    # Download option
    st.download_button(
        "Download Bubble Chart",
        data=get_figure_as_bytes(fig),
        file_name="bubble_chart.png",
        mime="image/png"
    )


# Statistical Visualization Functions

def display_distribution_plot(df, numeric_cols):
    """Display a distribution plot."""
    st.subheader("Distribution Plot")
    
    # UI controls
    column = st.selectbox("Column (Distribution)", numeric_cols, key="dist_col")
    
    # Additional options
    col1, col2 = st.columns([1, 1])
    
    with col1:
        plot_type = st.selectbox(
            "Plot Type",
            ["KDE", "Histogram with KDE", "ECDF"],
            index=1,
            key="dist_type"
        )
    
    with col2:
        if plot_type.startswith("Histogram"):
            bins = st.slider("Number of Bins", 5, 100, 25, key="dist_bins")
    
    # Create a container for the chart
    container = create_chart_container("basic")
    
    # Create plot
    dimensions = get_chart_dimensions()
    fig, ax = plt.subplots(figsize=dimensions["basic"])
    
    if plot_type == "KDE":
        sns.kdeplot(
            df[column],
            fill=True,
            color="skyblue",
            ax=ax
        )
        ax.set_title(f"Kernel Density Estimate of {column}")
        
    elif plot_type == "Histogram with KDE":
        sns.histplot(
            df[column],
            bins=bins,
            kde=True,
            color="skyblue",
            edgecolor="black",
            alpha=0.7,
            ax=ax
        )
        ax.set_title(f"Histogram with KDE of {column}")
        
    else:  # ECDF
        sns.ecdfplot(
            df[column],
            ax=ax
        )
        ax.set_title(f"Empirical Cumulative Distribution of {column}")
    
    ax.set_xlabel(column)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add descriptive statistics as text
    stats_text = (
        f"Mean: {df[column].mean():.2f}\n"
        f"Median: {df[column].median():.2f}\n"
        f"Std Dev: {df[column].std():.2f}\n"
        f"Min: {df[column].min():.2f}\n"
        f"Max: {df[column].max():.2f}"
    )
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=props
    )
    
    with container:
        st.pyplot(fig)
    
    # Download option
    st.download_button(
        "Download Distribution Plot",
        data=get_figure_as_bytes(fig),
        file_name="distribution_plot.png",
        mime="image/png"
    )


def display_qq_plot(df, numeric_cols):
    """Display a Q-Q plot."""
    st.subheader("Q-Q Plot")
    
    # UI controls
    column = st.selectbox("Column (Q-Q Plot)", numeric_cols, key="qq_col")
    
    # Create a container for the chart
    container = create_chart_container("basic")
    
    # Create plot
    dimensions = get_chart_dimensions()
    fig, ax = plt.subplots(figsize=dimensions["basic"])
    
    # Calculate z-scores
    from scipy import stats
    
    # Get data and remove NaNs
    data = df[column].dropna()
    
    # Create QQ plot
    stats.probplot(data, dist="norm", plot=ax)
    
    ax.set_title(f"Q-Q Plot of {column}")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    with container:
        st.pyplot(fig)
    
    # Download option
    st.download_button(
        "Download Q-Q Plot",
        data=get_figure_as_bytes(fig),
        file_name="qq_plot.png",
        mime="image/png"
    )
    
    # Show explanation
    with st.expander("What is a Q-Q Plot?"):
        st.markdown("""
        A Q-Q (Quantile-Quantile) plot compares the distribution of your data against a theoretical distribution (usually normal).
        
        - If the points roughly follow the diagonal line, the data follows the theoretical distribution.
        - Deviations from the line indicate departures from the distribution (skewness, outliers, etc.).
        
        Q-Q plots are useful for checking if data is normally distributed, which is an assumption in many statistical tests.
        """)


def display_residual_plot(df, numeric_cols):
    """Display a residual plot."""
    st.subheader("Residual Plot")
    
    # UI controls
    col1, col2 = st.columns([1, 1])
    
    with col1:
        x_axis = st.selectbox("X-axis (Independent)", numeric_cols, key="residual_x")
    
    with col2:
        y_axis = st.selectbox(
            "Y-axis (Dependent)",
            [col for col in numeric_cols if col != x_axis],
            key="residual_y"
        )
    
    # Create a container for the chart
    container = create_chart_container("double_wide")
    
    # Create plot
    from scipy import stats
    
    # Get data
    x = df[x_axis].values
    y = df[y_axis].values
    
    # Remove NaNs
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Calculate predicted values and residuals
    y_pred = intercept + slope * x
    residuals = y - y_pred
    
    # Create figure with multiple axes
    dimensions = get_chart_dimensions()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=dimensions["double_wide"])
    
    # Plot regression on first axis
    ax1.scatter(x, y, alpha=0.6)
    ax1.plot(x, y_pred, 'r-', lw=2)
    ax1.set_xlabel(x_axis)
    ax1.set_ylabel(y_axis)
    ax1.set_title(f"Linear Regression: {y_axis} vs {x_axis}")
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Display regression statistics
    stats_text = (
        f"y = {slope:.4f}x + {intercept:.4f}\n"
        f"R² = {r_value**2:.4f}\n"
        f"p-value = {p_value:.4e}\n"
        f"Std Err = {std_err:.4f}"
    )
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax1.text(
        0.05,
        0.95,
        stats_text,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=props
    )
    
    # Plot residuals on second axis
    ax2.scatter(y_pred, residuals, alpha=0.6)
    ax2.axhline(y=0, color='r', linestyle='-')
    ax2.set_xlabel("Predicted Values")
    ax2.set_ylabel("Residuals")
    ax2.set_title("Residual Plot")
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    with container:
        st.pyplot(fig)
    
    # Download option
    st.download_button(
        "Download Residual Plot",
        data=get_figure_as_bytes(fig),
        file_name="residual_plot.png",
        mime="image/png"
    )
    
    # Show explanation
    with st.expander("What is a Residual Plot?"):
        st.markdown("""
        A residual plot shows the difference between the observed values and the predicted values from a regression model.
        
        - Ideally, residuals should be randomly scattered around the horizontal line at y=0.
        - Patterns in the residuals (curves, fans, etc.) indicate that the regression model is not capturing some aspect of the data.
        
        What to look for:
        - **Random scatter**: Good! The model is appropriate.
        - **Funnel shape**: Indicates heteroscedasticity (unequal variance).
        - **Curve pattern**: Suggests a non-linear relationship that your linear model doesn't capture.
        - **Clustering**: May indicate that there are subgroups in your data.
        """)


def display_time_series(df, numeric_cols, datetime_cols):
    """Display time series analysis."""
    st.subheader("Time Series Analysis")
    
    if not datetime_cols:
        st.info("No datetime columns available for time series analysis.")
        return
    
    # UI controls
    col1, col2 = st.columns([1, 1])
    
    with col1:
        time_col = st.selectbox("Time Column", datetime_cols, key="time_col")
    
    with col2:
        value_col = st.selectbox("Value Column", numeric_cols, key="time_val")
    
    # Additional options
    analysis_type = st.selectbox(
        "Analysis Type",
        ["Time Series Plot", "Seasonal Decomposition", "Moving Average", "Autocorrelation"],
        key="time_analysis"
    )
    
    # Prepare data
    time_data = df[[time_col, value_col]].dropna().sort_values(time_col)
    
    if len(time_data) == 0:
        st.warning("No valid data for time series analysis.")
        return
    
    # Different analyses
    if analysis_type == "Time Series Plot":
        display_simple_time_series(time_data, time_col, value_col)
    
    elif analysis_type == "Moving Average":
        display_moving_average(time_data, time_col, value_col)
    
    elif analysis_type == "Seasonal Decomposition":
        display_seasonal_decomposition(time_data, time_col, value_col)
    
    elif analysis_type == "Autocorrelation":
        display_autocorrelation(time_data, value_col)


def display_simple_time_series(time_data, time_col, value_col):
    """Display a simple time series plot."""
    # Create a container for the chart
    container = create_chart_container("basic")
    
    # Create plot
    dimensions = get_chart_dimensions()
    fig, ax = plt.subplots(figsize=dimensions["basic"])
    
    ax.plot(time_data[time_col], time_data[value_col], marker='o', linestyle='-', alpha=0.7)
    
    ax.set_xlabel(time_col)
    ax.set_ylabel(value_col)
    ax.set_title(f"Time Series Plot of {value_col}")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Format x-axis dates
    plt.gcf().autofmt_xdate()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    with container:
        st.pyplot(fig)
    
    # Download option
    st.download_button(
        "Download Time Series Plot",
        data=get_figure_as_bytes(fig),
        file_name="time_series_plot.png",
        mime="image/png"
    )


def display_moving_average(time_data, time_col, value_col):
    """Display a time series with moving average."""
    # UI control for window size
    window_size = st.slider(
        "Moving Average Window Size",
        3,
        30,
        7,
        key="ma_window"
    )
    
    # Create a container for the chart
    container = create_chart_container("basic")
    
    # Calculate moving average
    time_data['Moving Average'] = time_data[value_col].rolling(window=window_size).mean()
    
    # Create plot
    dimensions = get_chart_dimensions()
    fig, ax = plt.subplots(figsize=dimensions["basic"])
    
    ax.plot(time_data[time_col], time_data[value_col], 'o-', alpha=0.5, label='Original')
    ax.plot(time_data[time_col], time_data['Moving Average'], 'r-', linewidth=2, label=f'{window_size}-point Moving Average')
    
    ax.set_xlabel(time_col)
    ax.set_ylabel(value_col)
    ax.set_title(f"Time Series with Moving Average of {value_col}")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Format x-axis dates
    plt.gcf().autofmt_xdate()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    with container:
        st.pyplot(fig)
    
    # Download option
    st.download_button(
        "Download Moving Average Plot",
        data=get_figure_as_bytes(fig),
        file_name="moving_average_plot.png",
        mime="image/png"
    )


def display_seasonal_decomposition(time_data, time_col, value_col):
    """Display seasonal decomposition of time series."""
    # Check if we have enough data
    if len(time_data) < 10:
        st.warning("Not enough data points for seasonal decomposition.")
        return
    
    # UI control for period
    period = st.slider(
        "Seasonality Period (number of observations per season)",
        2,
        min(30, len(time_data) // 2),
        7,
        key="decomp_period"
    )
    
    # Create a container for the chart
    container = create_chart_container("tall")
    
    try:
        # Import statsmodels
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Set time column as index for seasonal decomposition
        ts_data = time_data.set_index(time_col)[value_col]
        
        # Perform decomposition
        result = seasonal_decompose(ts_data, model='additive', period=period)
        
        # Create plot
        dimensions = get_chart_dimensions()
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=dimensions["tall"])
        
        result.observed.plot(ax=ax1)
        ax1.set_ylabel('Observed')
        ax1.set_title(f"Seasonal Decomposition of {value_col}")
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        result.trend.plot(ax=ax2)
        ax2.set_ylabel('Trend')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        result.seasonal.plot(ax=ax3)
        ax3.set_ylabel('Seasonal')
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        result.resid.plot(ax=ax4)
        ax4.set_ylabel('Residual')
        ax4.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        with container:
            st.pyplot(fig)
        
        # Download option
        st.download_button(
            "Download Seasonal Decomposition Plot",
            data=get_figure_as_bytes(fig),
            file_name="seasonal_decomposition_plot.png",
            mime="image/png"
        )
        
    except Exception as e:
        st.error(f"Error in seasonal decomposition: {str(e)}")
        st.info("Try adjusting the period or check your data for uniformity.")


def display_autocorrelation(time_data, value_col):
    """Display autocorrelation and partial autocorrelation plots."""
    # Check if we have enough data
    if len(time_data) < 10:
        st.warning("Not enough data points for autocorrelation analysis.")
        return
    
    # Create a container for the chart
    container = create_chart_container("tall")
    
    try:
        # Import statsmodels
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        
        # UI control for lags
        max_lags = st.slider(
            "Maximum Lags",
            5,
            min(40, len(time_data) // 2),
            20,
            key="acf_lags"
        )
        
        # Create figure
        dimensions = get_chart_dimensions()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=dimensions["panel"])
        
        # Plot autocorrelation
        plot_acf(time_data[value_col].dropna(), lags=max_lags, ax=ax1)
        ax1.set_title(f"Autocorrelation of {value_col}")
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot partial autocorrelation
        plot_pacf(time_data[value_col].dropna(), lags=max_lags, ax=ax2)
        ax2.set_title(f"Partial Autocorrelation of {value_col}")
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        with container:
            st.pyplot(fig)
        
        # Download option
        st.download_button(
            "Download Autocorrelation Plot",
            data=get_figure_as_bytes(fig),
            file_name="autocorrelation_plot.png",
            mime="image/png"
        )
        
        # Show explanation
        with st.expander("What are Autocorrelation Plots?"):
            st.markdown("""
            **Autocorrelation Function (ACF)** shows the correlation between a time series and its lagged values.
            
            - Significant spikes indicate time dependencies in your data.
            - In seasonal data, you'll see repeating patterns in the ACF.
            
            **Partial Autocorrelation Function (PACF)** shows the direct correlation between a time series and its lagged values, after removing the effect of intermediate lags.
            
            - Useful for identifying the appropriate order in ARIMA models.
            - The last significant spike often suggests the AR term in ARIMA modeling.
            
            These plots are valuable tools in time series analysis for model identification.
            """)
        
    except Exception as e:
        st.error(f"Error in autocorrelation analysis: {str(e)}")


# Helper Functions

def get_figure_as_bytes(fig):
    """Convert a matplotlib figure to bytes for download."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    return buf


def create_size_legend(column_name, data, sizes):
    """Create a custom legend for bubble sizes."""
    # Create legend handles
    min_val = data.min()
    max_val = data.max()
    mid_val = (min_val + max_val) / 2
    
    # Get corresponding sizes
    min_size = 20
    max_size = 520  # 20 + 500
    mid_size = (min_size + max_size) / 2
    
    # Create handles
    handles = [
        plt.scatter([], [], s=min_size, color='gray', alpha=0.6),
        plt.scatter([], [], s=mid_size, color='gray', alpha=0.6),
        plt.scatter([], [], s=max_size, color='gray', alpha=0.6)
    ]
    
    labels = [
        f"{min_val:.2f}",
        f"{mid_val:.2f}",
        f"{max_val:.2f}"
    ]
    
    return handles, labels


def add_size_legend(ax, column_name, data):
    """Add a custom legend for bubble sizes to a plot."""
    handles, labels = create_size_legend(column_name, data, None)
    legend = ax.legend(
        handles,
        labels,
        title=column_name,
        loc="lower left",
        bbox_to_anchor=(1, 0),
        frameon=True,
        fontsize=8
    )
    return legend


if __name__ == "__main__":
    main()
