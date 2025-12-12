# data_generator.py
import pandas as pd
import random
from datetime import datetime, timedelta

# Base start time for clustering (Dec 2025, as per current date)
base_time = datetime(2025, 12, 9, 0, 0)
data = []

# Syslog-inspired severity levels (0-7) with weights: more normal, rare critical
severities = [
    (0, 'Emergency', 0.01),  # System unusable (e.g., total outage)
    (1, 'Alert', 0.02),      # Immediate action (e.g., neighbor drop)
    (2, 'Critical', 0.05),   # Critical conditions
    (3, 'Error', 0.10),      # Errors
    (4, 'Warning', 0.20),    # Warnings
    (5, 'Notice', 0.25),     # Normal but significant
    (6, 'Info', 0.25),       # Informational
    (7, 'Debug', 0.12)       # Debug
]
severities_flat = [s for s, _, w in severities for _ in range(int(w * 1000))]  # Weighted choices

# Facilities (common for network devices)
facilities = ['kernel', 'user', 'mail', 'local0', 'local1', 'daemon', 'auth']

# Hostnames (telco devices: RAN, Core, etc.)
hostnames = ['RAN-Cell{:05d}'.format(random.randint(10000, 99999)), 'Core-Router01', 'Optical-Switch02', 'Edge-FW03']

# Alert types (expanded for telecom realism)
alert_types = [
    'Connection Slow', 'Downtime', 'Anomaly in KPI', 'ISIS Neighbor Drop', 'BGP Peer Down',
    'High Latency', 'Packet Loss', 'Capacity Exhaustion', 'Hardware Fault', 'Diameter Failure',
    'Cell Outage', 'Config Mismatch', 'Fiber Cut', 'Interface Flap'
]

# KPI examples (random values for realism)
kpi_examples = ['Latency: {}ms (threshold: 100ms)'.format(random.randint(50, 500)),
                'Packet Loss: {}%'.format(random.uniform(0, 10)),
                'Drop Rate: {}%'.format(random.uniform(0, 15)),
                'Throughput: {}Mbps'.format(random.randint(100, 1000))]

# Description templates (Cisco/syslog style, varied)
description_templates = [
    'Network complexity due to high KPIs and logs. KPI: {}.',
    'Root cause: Configuration error in RAN module. Interface {}.',
    'Predictive maintenance needed for core network. Check {}.',
    'Alarm: Drop in neighbors, check interfaces {}.',
    '%LINK-3-UPDOWN: Interface {}, changed state to down.',
    '%OSPF-5-ADJCHG: Process 1, Nbr {} on Vlan100 from FULL to DOWN, NeighborDown: dead timer expired',
    'High CPU utilization on {}. KPI anomaly detected.',
    'Fiber cut detected on transport link {}.',
    'Diameter peer failure in core network.'
]

# Recommended actions (telco-aligned)
recommended_actions = [
    'Run diagnostics on device.',
    'Check logs for anomalies.',
    'Initiate self-healing agent.',
    'Analyze with SemCom for intent.',
    'Rollback config change.',
    'Clear adjacency and restart process.',
    'Escalate to L2 support.',
    'Schedule hardware replacement.'
]

# Generate 100 rows with clustered timestamps (e.g., bursts of events)
current_time = base_time
for i in range(100):
    # Cluster: Add random minutes, occasional bursts
    if random.random() < 0.1:  # 10% chance for burst (simulate outage cascade)
        delta = timedelta(minutes=random.randint(0, 5))
    else:
        delta = timedelta(minutes=random.randint(0, 60))
    current_time += delta
    timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')
    
    severity_id, severity_name, _ = random.choice(severities)
    facility = random.choice(facilities)
    hostname = random.choice(hostnames)
    alert_type = random.choice(alert_types)
    kpi = random.choice(kpi_examples)
    interface = f'{"Gig" if random.random() < 0.7 else "Te"}{random.randint(0,7)}/{random.randint(0,47)}'
    nbr = f'{random.randint(192, 223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}'
    description_template = random.choice(description_templates)
    description = description_template.format(kpi, interface, hostname, interface, nbr, hostname, interface)
    recommended_action = random.choice(recommended_actions)
    
    # PRI calculation (facility * 8 + severity)
    pri = (facilities.index(facility) * 8 if facility in facilities else 16) + severity_id
    
    data.append({
        'timestamp': timestamp,
        'pri': pri,
        'severity': severity_name,
        'facility': facility,
        'hostname': hostname,
        'alert_type': alert_type,
        'kpi_value': kpi,
        'description': description,
        'recommended_action': recommended_action
    })

df = pd.DataFrame(data)
df.to_csv('data/simulated_logs.csv', index=False)
print(f"Generated {len(df)} realistic telecom logs. Sample:\n{df.head()}")