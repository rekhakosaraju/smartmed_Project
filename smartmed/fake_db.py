import sqlite3
from faker import Faker
import random

# Initialize Faker
fake = Faker()

# Connect to the SQLite database (it will create the file if it doesn't exist)
conn = sqlite3.connect('hospital.db')
cursor = conn.cursor()

# Create the tables
cursor.execute('''
CREATE TABLE IF NOT EXISTS doctors (
    doctor_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    specialty TEXT NOT NULL
)
''')

# Create the availability table with additional columns
cursor.execute('''
CREATE TABLE IF NOT EXISTS availability (
    availability_id INTEGER PRIMARY KEY AUTOINCREMENT,
    doctor_id INTEGER,
    available_date DATE,
    time_slot TEXT,
    slot_booked BOOLEAN NOT NULL DEFAULT 0,
    patient_name TEXT,
    FOREIGN KEY (doctor_id) REFERENCES doctors (doctor_id)
)
''')

# Insert sample data into doctors table (30 doctors with various specialties)
doctors = [
    ('Dr. John Smith', 'Cardiology'),
    ('Dr. Emma Brown', 'Cardiology'),
    ('Dr. Liam Davis', 'Cardiology'),
    ('Dr. Alice Johnson', 'Dermatology'),
    ('Dr. Sophia Hernandez', 'Dermatology'),
    ('Dr. Mia Anderson', 'Dermatology'),
    ('Dr. Mark Lee', 'Pediatrics'),
    ('Dr. Ethan Thomas', 'Pediatrics'),
    ('Dr. Lily Adams', 'Pediatrics'),
    ('Dr. Noah Martinez', 'Orthopedics'),
    ('Dr. Aiden Moore', 'Orthopedics'),
    ('Dr. Wyatt Nelson', 'Orthopedics'),
    ('Dr. Olivia Garcia', 'Gynecology'),
    ('Dr. Harper Jackson', 'Gynecology'),
    ('Dr. Amelia Harris', 'Gynecology'),
    ('Dr. Lucas Wilson', 'Neurology'),
    ('Dr. James Lewis', 'Neurology'),
    ('Dr. Victoria Wright', 'Neurology'),
    ('Dr. Benjamin Scott', 'General Surgery'),
    ('Dr. Aria Gonzalez', 'General Surgery'),
    ('Dr. Scarlett Torres', 'General Surgery'),
    ('Dr. Elijah Robinson', 'Psychiatry'),
    ('Dr. Grace Allen', 'Psychiatry'),
    ('Dr. David Baker', 'Psychiatry'),
    ('Dr. Jacob King', 'Gastroenterology'),
    ('Dr. Isabella Clark', 'Gastroenterology'),
    ('Dr. Michael Brown', 'Gastroenterology'),
    ('Dr. John Doe', 'Family Medicine'),
    ('Dr. Jane Doe', 'Family Medicine'),
    ('Dr. Sam Smith', 'Family Medicine'),
]

cursor.executemany('INSERT INTO doctors (name, specialty) VALUES (?, ?)', doctors)

# Insert availability for each doctor
availability = []
time_slots = ['09:00 AM - 10:00 AM', '10:30 AM - 11:30 AM', '01:00 PM - 02:00 PM', '03:00 PM - 04:00 PM']
for doctor_id in range(1, 31):  # For each of the 30 doctors
    for day in range(1, 31):  # For November 1 to 30
        date = f'2024-11-{day:02d}'
        # Randomly select time slots
        for _ in range(random.randint(1, 3)):  # Each doctor can have 1 to 3 time slots per day
            slot = random.choice(time_slots)
            # Decide if the slot is booked or not
            slot_booked = random.choice([0, 1])  # 0 for not booked, 1 for booked
            patient_name = fake.name() if slot_booked else None  # Assign a patient name if booked

            availability.append((doctor_id, date, slot, slot_booked, patient_name))

cursor.executemany('INSERT INTO availability (doctor_id, available_date, time_slot, slot_booked, patient_name) VALUES (?, ?, ?, ?, ?)', availability)

# Commit the changes and close the connection
conn.commit()
conn.close()

print("Database created and populated successfully with 30 doctors and availability records.")
