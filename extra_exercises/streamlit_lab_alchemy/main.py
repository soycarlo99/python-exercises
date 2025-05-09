import datetime
import logging
import os
import random
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Page configuration
st.set_page_config(
    page_title="SQL Data Processing Workflow", page_icon="📊", layout="wide"
)

# Initialize session state if needed
if "log_file" not in st.session_state:
    # Setup logging
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create log file with timestamp
    log_file = os.path.join(
        log_dir, f"process_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    # Store the log file path in session state
    st.session_state.log_file = log_file

    # Configure logging
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info("Application started")

# Get the log file path from session state
log_file = st.session_state.log_file

# Database setup
Base = declarative_base()


class DataEntry(Base):
    """Database seeder"""

    __tablename__ = "data_entries"

    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    value = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return f"<DataEntry(name='{self.name}', value={self.value})>"


def setup_database(db_user, db_password, db_host, db_port, db_name):
    """Create the database and tables"""
    logging.info("Setting up database connection to PostgreSQL")

    # Create connection string
    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    # Create engine
    engine = create_engine(db_url)

    # Create tables
    Base.metadata.create_all(engine)

    logging.info(f"Connected to PostgreSQL database: {db_name}")
    return engine


def generate_sample_data(session, num_entries=10):
    """Generate some sample data for processing"""
    logging.info(f"Generating {num_entries} sample data entries")

    sample_names = ["Temperature", "Pressure", "Humidity", "Voltage", "Current"]

    for _ in range(num_entries):
        entry = DataEntry(
            name=random.choice(sample_names), value=round(random.uniform(0, 100), 2)
        )
        session.add(entry)

    session.commit()
    logging.info("Sample data generation complete")


def process_data(session):
    """Process the data in the database"""
    logging.info("Starting data processing")

    # fake process
    entries = session.query(DataEntry).all()

    results = {}
    for entry in entries:
        if entry.name not in results:
            results[entry.name] = []
        results[entry.name].append(entry.value)

    stats = {}
    for name, values in results.items():
        if values:
            stats[name] = {
                "count": len(values),
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }

    logging.info(f"Processed {len(entries)} entries into statistics")
    return stats


def send_email_notification(
    stats, sender_email, receiver_email, app_password, log_file_path
):
    """Send email notification with processing results and log file attachment"""
    logging.info("Preparing email notification with log file attachment")
    logging.info(f"Using log file path: {log_file_path}")

    # Validate that password exists
    if not app_password:
        error_msg = "Email password not provided!"
        logging.error(error_msg)
        st.error(error_msg)
        return False

    # Create message
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "Data Processing Complete"

    # Email body
    body = "Data processing has completed successfully.\n\n"
    body += "Processing Statistics:\n"
    for name, stat in stats.items():
        body += f"\n{name}:\n"
        body += f"  Count: {stat['count']}\n"
        body += f"  Average: {stat['avg']:.2f}\n"
        body += f"  Min: {stat['min']:.2f}\n"
        body += f"  Max: {stat['max']:.2f}\n"

    body += f"\nLog file is attached to this email."

    # Attach text part
    message.attach(MIMEText(body, "plain"))

    # Attach log file
    try:
        if os.path.exists(log_file_path):
            with open(log_file_path, "rb") as file:
                log_attachment = MIMEApplication(
                    file.read(), Name=os.path.basename(log_file_path)
                )
                # Add header with filename
                log_attachment["Content-Disposition"] = (
                    f'attachment; filename="{os.path.basename(log_file_path)}"'
                )
                message.attach(log_attachment)
                logging.info(
                    f"Log file {os.path.basename(log_file_path)} attached to email"
                )
        else:
            logging.error(f"Log file not found at path: {log_file_path}")
            st.error(f"Log file not found at path: {log_file_path}")
            # Continue without attachment
    except Exception as e:
        logging.error(f"Failed to attach log file: {str(e)}")
        st.error(f"Failed to attach log file: {str(e)}")
        # Continue without attachment

    try:
        # Create SMTP session
        logging.info("Connecting to email server")
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()

        # Using app password for authentication
        server.login(sender_email, app_password)
        server.send_message(message)
        server.quit()

        logging.info("Email notification sent successfully")
        return True
    except Exception as e:
        logging.error(f"Failed to send email: {str(e)}")
        st.error(f"Failed to send email: {str(e)}")
        return False


# New functions for data visualization
def plot_average_values(stats):
    """Create a bar chart of average values for each category"""
    plt.figure(figsize=(10, 6))
    names = list(stats.keys())
    avg_values = [stat["avg"] for stat in stats.values()]

    bars = plt.bar(names, avg_values, color="skyblue")

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{height:.2f}",
            ha="center",
            va="bottom",
        )

    plt.title("Average Values by Category", fontsize=16)
    plt.xlabel("Category", fontsize=14)
    plt.ylabel("Average Value", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()

    return plt


def plot_min_max_range(stats):
    """Create a line chart showing min, max, and average values"""
    plt.figure(figsize=(10, 6))

    names = list(stats.keys())
    x = range(len(names))

    min_values = [stat["min"] for stat in stats.values()]
    max_values = [stat["max"] for stat in stats.values()]
    avg_values = [stat["avg"] for stat in stats.values()]

    plt.plot(x, min_values, "bo-", label="Minimum")
    plt.plot(x, avg_values, "go-", label="Average")
    plt.plot(x, max_values, "ro-", label="Maximum")

    # Add shaded area between min and max
    plt.fill_between(x, min_values, max_values, alpha=0.2, color="gray")

    plt.xticks(x, names, rotation=45)
    plt.title("Value Range by Category", fontsize=16)
    plt.xlabel("Category", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()

    return plt


def plot_data_distribution(stats):
    """Create a pie chart showing the distribution of data points"""
    plt.figure(figsize=(10, 6))

    names = list(stats.keys())
    counts = [stat["count"] for stat in stats.values()]

    # Explode the largest slice
    explode = [0] * len(names)
    max_index = counts.index(max(counts))
    explode[max_index] = 0.1

    plt.pie(
        counts,
        labels=names,
        explode=explode,
        autopct="%1.1f%%",
        shadow=True,
        startangle=140,
        colors=plt.cm.Paired.colors,
    )

    plt.axis("equal")  # Equal aspect ratio ensures the pie chart is circular
    plt.title("Distribution of Data Points by Category", fontsize=16)
    plt.tight_layout()

    return plt


def plot_radar_chart(stats):
    """Create a radar chart comparing all metrics"""
    plt.figure(figsize=(10, 8))

    # Number of variables
    categories = list(stats.keys())
    N = len(categories)

    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop

    # Normalize data for radar chart (0-1 scale)
    max_avg = max(stat["avg"] for stat in stats.values())
    normalized_avgs = [stat["avg"] / max_avg for stat in stats.values()]
    normalized_avgs += normalized_avgs[:1]  # Close the loop

    # Draw the plot
    ax = plt.subplot(111, polar=True)

    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, size=12)

    # Draw the chart
    ax.plot(angles, normalized_avgs, "o-", linewidth=2)
    ax.fill(angles, normalized_avgs, alpha=0.25)

    # Add value annotations
    for i, angle in enumerate(angles[:-1]):
        original_value = list(stats.values())[i]["avg"]
        plt.annotate(
            f"{original_value:.2f}",
            xy=(angle, normalized_avgs[i]),
            xytext=(angle, normalized_avgs[i] + 0.1),
            ha="center",
        )

    # Set y-axis to start from center
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["25%", "50%", "75%"], color="grey", size=10)
    plt.ylim(0, 1)

    plt.title("Radar Chart of Average Values (Normalized)", size=16)
    plt.tight_layout()

    return plt


# Streamlit app
def main():
    st.title("📊 SQL Data Processing Workflow")
    st.markdown(
        """
    This application:
    1. Connects to a PostgreSQL database
    2. Generates and processes sample data
    3. Sends email notifications with results
    4. Visualizes the processed data
    """
    )

    # Display the current log file path
    st.info(f"Current log file: {st.session_state.log_file}")

    # Sidebar for configuration
    st.sidebar.header("Configuration")

    # Email configuration section in sidebar
    st.sidebar.subheader("Email Settings")
    sender_email = st.sidebar.text_input("Sender Email", "alcorel.solutions@gmail.com")
    receiver_email = st.sidebar.text_input("Recipient Email", "carlodelos90@gmail.com")
    app_password = st.sidebar.text_input(
        "Email App Password",
        "",
        type="password",
        help="For Gmail, use an App Password. Set as environment variable or enter here.",
    )

    # If no app password in input, try environment variable
    if not app_password:
        app_password = os.environ.get("Email__Password")

    # Database configuration section in sidebar
    st.sidebar.subheader("Database Configuration")
    db_user = st.sidebar.text_input("Database Username", "postgres")
    db_password = st.sidebar.text_input(
        "Database Password", "password", type="password"
    )
    db_host = st.sidebar.text_input("Database Host", "localhost")
    db_port = st.sidebar.text_input("Database Port", "5432")
    db_name = st.sidebar.text_input("Database Name", "woodwine")

    # Sample data configuration
    st.sidebar.subheader("Sample Data")
    num_entries = st.sidebar.slider("Number of sample entries", 5, 100, 10)

    # Use radio buttons to choose between Process and Results
    view_mode = st.radio("Select View", ["Process Data", "Results"])

    if view_mode == "Process Data":
        st.header("Process Data")
        st.info(
            "Configure settings in the sidebar, then click the button below to start processing."
        )

        # Process button
        if st.button("Start Data Processing"):
            if not receiver_email:
                st.error("Please enter a recipient email address in the sidebar")
                return

            if not app_password:
                st.error(
                    "Email App Password is required. Either set it as an environment variable (Email__Password) or enter it in the sidebar."
                )
                return

            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # Setup database connection
                status_text.text("Connecting to database...")
                progress_bar.progress(10)

                engine = setup_database(db_user, db_password, db_host, db_port, db_name)
                Session = sessionmaker(bind=engine)
                session = Session()

                status_text.text("Connected to database successfully")
                progress_bar.progress(30)

                # Generate sample data
                status_text.text("Generating sample data...")
                generate_sample_data(session, num_entries)
                progress_bar.progress(50)

                # Process data
                status_text.text("Processing data...")
                stats = process_data(session)
                progress_bar.progress(70)

                # Store stats in session state for the results tab
                st.session_state.stats = stats
                st.session_state.processing_complete = True

                # Send email notification
                status_text.text("Sending email notification...")
                # Use the log file path from session state
                if send_email_notification(
                    stats,
                    sender_email,
                    receiver_email,
                    app_password,
                    st.session_state.log_file,
                ):
                    progress_bar.progress(100)
                    status_text.text("Process completed successfully!")
                    st.success(f"Email notification sent to {receiver_email}")
                    st.info(f"Log file created: {st.session_state.log_file}")

                    # Switch to results tab
                    st.write("View results in the Results view")
                else:
                    progress_bar.progress(90)
                    status_text.text("Failed to send email notification")
                    st.warning(
                        "Process completed but failed to send email notification"
                    )

            except Exception as e:
                st.error(f"Error occurred: {str(e)}")
                logging.error(f"Error in data processing workflow: {str(e)}")

    elif view_mode == "Results":
        st.header("Processing Results")

        if (
            "processing_complete" in st.session_state
            and st.session_state.processing_complete
        ):
            stats = st.session_state.stats

            # Create tabs for text data and visualizations
            tabs = st.tabs(["Text Data", "Visualizations"])

            # Tab 1: Text Data (existing implementation)
            with tabs[0]:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Statistics by Category")
                    for name, stat in stats.items():
                        st.write(f"**{name}**: {stat['avg']:.2f}")
                        st.write(f"Range: {stat['max'] - stat['min']:.2f}")
                        st.write(f"Count: {stat['count']}")
                        st.write("---")

                with col2:
                    st.subheader("Detailed View")
                    for name, stat in stats.items():
                        st.write(f"**{name}:**")
                        metrics_data = {
                            "Metric": ["Count", "Average", "Minimum", "Maximum"],
                            "Value": [
                                stat["count"],
                                f"{stat['avg']:.2f}",
                                f"{stat['min']:.2f}",
                                f"{stat['max']:.2f}",
                            ],
                        }
                        for i in range(len(metrics_data["Metric"])):
                            st.write(
                                f"{metrics_data['Metric'][i]}: {metrics_data['Value'][i]}"
                            )
                        st.write("---")

            # Tab 2: Visualizations
            with tabs[1]:
                # Check if we have at least 2 categories for visualizations
                if len(stats) >= 2:
                    st.subheader("Data Visualizations")

                    # Create a DataFrame from the stats for easier visualization
                    viz_data = {
                        "Category": [],
                        "Count": [],
                        "Average": [],
                        "Min": [],
                        "Max": [],
                        "Range": [],
                    }

                    for name, stat in stats.items():
                        viz_data["Category"].append(name)
                        viz_data["Count"].append(stat["count"])
                        viz_data["Average"].append(stat["avg"])
                        viz_data["Min"].append(stat["min"])
                        viz_data["Max"].append(stat["max"])
                        viz_data["Range"].append(stat["max"] - stat["min"])

                    # Display the visualization options
                    st.write("### Choose Visualizations to Display")
                    col1, col2 = st.columns(2)

                    with col1:
                        show_bar = st.checkbox("Bar Chart - Average Values", value=True)
                        show_range = st.checkbox(
                            "Line Chart - Value Ranges", value=True
                        )

                    with col2:
                        show_pie = st.checkbox(
                            "Pie Chart - Data Distribution", value=True
                        )
                        show_radar = st.checkbox("Radar Chart - Comparison", value=True)

                    # Display selected visualizations
                    if show_bar:
                        st.write("### Average Values by Category")
                        fig_bar = plot_average_values(stats)
                        st.pyplot(fig_bar)

                    if show_range:
                        st.write("### Min, Max and Average Values")
                        fig_range = plot_min_max_range(stats)
                        st.pyplot(fig_range)

                    if show_pie:
                        st.write("### Distribution of Data Points")
                        fig_pie = plot_data_distribution(stats)
                        st.pyplot(fig_pie)

                    if show_radar and len(stats) >= 3:
                        st.write("### Radar Comparison of Categories")
                        fig_radar = plot_radar_chart(stats)
                        st.pyplot(fig_radar)
                    elif show_radar:
                        st.info(
                            "Radar chart requires at least 3 categories to display properly."
                        )

                    # Add a data table with all statistics
                    st.write("### Complete Data Table")
                    df = pd.DataFrame(viz_data)
                    st.dataframe(df.set_index("Category"))

                else:
                    st.warning(
                        "At least 2 data categories are required to generate visualizations. Please process more data."
                    )
        else:
            st.info("No processing results available. Run the processing first.")


if __name__ == "__main__":
    main()
