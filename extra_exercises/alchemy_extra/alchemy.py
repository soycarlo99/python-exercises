"""
SQL Data Processing Workflow with PostgreSQL
-------------------------------------------
This script:
1. Creates/connects to a PostgreSQL database using SQLAlchemy
2. Performs some sample data processing
3. Logs all activities to a log file
4. Sends an email notification with log file attachment when complete
5. Can be scheduled via Windows Task Scheduler
"""

import datetime
import logging
import os
import random
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# import pandas as pd
# import psycopg2
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Setup logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(
    log_dir, f"process_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

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


def setup_database():
    """Create the database and tables"""
    logging.info("Setting up database connection to PostgreSQL")

    # PostgreSQL connection parameters
    db_user = "postgres"
    db_password = "password"
    db_host = "localhost"
    db_port = "5432"
    db_name = "woodwine"

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

    # Get all data
    entries = session.query(DataEntry).all()

    # Process by category (name)
    results = {}
    for entry in entries:
        if entry.name not in results:
            results[entry.name] = []
        results[entry.name].append(entry.value)

    # Calculate statistics
    stats = {}
    for name, values in results.items():
        stats[name] = {
            "count": len(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    logging.info(f"Processed {len(entries)} entries into statistics")
    return stats


def send_email_notification(stats):
    """Send email notification with processing results and log file attachment"""
    logging.info("Preparing email notification with log file attachment")

    # Email configuration
    sender_email = "alcorel.solutions@gmail.com"
    receiver_email = "carlodelos90@gmail.com"
    app_password = os.environ.get("Email__Password")

    # Validate that password exists
    if not app_password:
        error_msg = "Email password environment variable (Email__Password) not found!"
        logging.error(error_msg)
        print(error_msg)
        return

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
        with open(log_file, "rb") as file:
            log_attachment = MIMEApplication(
                file.read(), Name=os.path.basename(log_file)
            )
            # Add header with filename
            log_attachment["Content-Disposition"] = (
                f'attachment; filename="{os.path.basename(log_file)}"'
            )
            message.attach(log_attachment)
            logging.info(f"Log file {os.path.basename(log_file)} attached to email")
    except Exception as e:
        logging.error(f"Failed to attach log file: {str(e)}")
        print(f"Failed to attach log file: {str(e)}")

    try:
        # Create SMTP session
        logging.info("Connecting to email server")
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()

        # Using app password for authentication
        server.login(sender_email, app_password)
        server.send_message(message)
        server.quit()

        logging.info("Email notification with log attachment sent successfully")
        print("Email notification with log attachment sent successfully")
    except Exception as e:
        logging.error(f"Failed to send email: {str(e)}")
        print(f"Failed to send email: {str(e)}")


def main():
    """Main execution function"""
    logging.info("Starting data processing workflow")

    try:
        # Setup database
        engine = setup_database()
        Session = sessionmaker(bind=engine)
        session = Session()

        # Generate sample data
        generate_sample_data(session)

        # Process data
        stats = process_data(session)

        # Send email notification
        send_email_notification(stats)

        logging.info("Data processing workflow completed successfully")
        print(f"Process completed. Log file: {log_file}")

    except Exception as e:
        logging.error(f"Error in data processing workflow: {str(e)}")
        print(f"Error occurred. Check log file: {log_file}")


if __name__ == "__main__":
    main()
