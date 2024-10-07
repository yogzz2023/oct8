import numpy as np
import math
import csv
import pandas as pd
from scipy.stats import chi2
import time
r = []
el = []
az = []

r = []
el = []
az = []

class CVFilter:
    def __init__(self):
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.Sp = np.zeros((6, 1))  # Predicted state vector
        self.Pp = np.eye(6)  # Predicted state covariance matrix
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.prev_Time = 0
        self.Q = np.eye(6)
        self.Phi = np.eye(6)
        self.Z = np.zeros((3, 1)) 
        self.Z1 = np.zeros((3, 1))  # Measurement vector
        self.Z2 = np.zeros((3, 1)) 
        self.first_rep_flag = False
        self.second_rep_flag = False
        self.gate_threshold = 9.21  # 95% confidence interval for Chi-square distribution with 3 degrees of freedom

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        if not self.first_rep_flag:
            self.Z1 = np.array([[x], [y], [z]])
            self.Sf[0] = x
            self.Sf[1] = y
            self.Sf[2] = z
            self.Meas_Time = time
            self.prev_Time = self.Meas_Time
            self.first_rep_flag = True
        elif self.first_rep_flag and not self.second_rep_flag:
            self.Z2 = np.array([[x], [y], [z]])
            self.prev_Time = self.Meas_Time
            self.Meas_Time = time
            dt = self.Meas_Time - self.prev_Time
            self.vx = (self.Z1[0] - self.Z2[0]) / dt
            self.vy = (self.Z1[1] - self.Z2[1]) / dt
            self.vz = (self.Z1[2] - self.Z2[2]) / dt
            self.Meas_Time = time
            self.second_rep_flag = True
        else:
            self.Z = np.array([[x], [y], [z]])
            self.prev_Time = self.Meas_Time
            self.Meas_Time = time

    def predict_step(self, current_time):
        dt = current_time - self.prev_Time
        T_2 = (dt * dt) / 2.0
        T_3 = (dt * dt * dt) / 3.0
        self.Phi[0, 3] = dt
        self.Phi[1, 4] = dt
        self.Phi[2, 5] = dt              
        self.Q[0, 0] = T_3
        self.Q[1, 1] = T_3
        self.Q[2, 2] = T_3
        self.Q[0, 3] = T_2
        self.Q[1, 4] = T_2
        self.Q[2, 5] = T_2
        self.Q[3, 0] = T_2
        self.Q[4, 1] = T_2
        self.Q[5, 2] = T_2
        self.Q[3, 3] = dt
        self.Q[4, 4] = dt
        self.Q[5, 5] = dt
        self.Q = self.Q * self.plant_noise
        self.Sp = np.dot(self.Phi, self.Sf)
        self.Pp = np.dot(np.dot(self.Phi, self.Pf), self.Phi.T) + self.Q
        self.Meas_Time = current_time

    def update_step(self, Z):
        Inn = Z - np.dot(self.H, self.Sp)
        S = np.dot(self.H, np.dot(self.Pp, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pp, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sp + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pp)


def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            mr = float(row[7])  # MR column
            ma = float(row[8])  # MA column
            me = float(row[9])  # ME column
            mt = float(row[10])  # MT column
            md = float(row[11])
            x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
            measurements.append((mr, ma, me, mt, md, x, y, z))
    return measurements


def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z


def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    el = math.atan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
    az = math.atan2(y, x)

    if x > 0.0:
        az = np.pi / 2 - az
    else:
        az = 3 * np.pi / 2 - az

    az = az * 180 / np.pi

    if az < 0.0:
        az = 360 + az

    if az > 360:
        az = az - 360

    return r, az, el


def form_measurement_groups(measurements, max_time_diff=0.050):
    measurement_groups = []
    current_group = []
    base_time = measurements[0][3]

    for measurement in measurements:
        if measurement[3] - base_time <= max_time_diff:
            current_group.append(measurement)
        else:
            measurement_groups.append(current_group)
            current_group = [measurement]
            base_time = measurement[3]

    if current_group:
        measurement_groups.append(current_group)

    return measurement_groups


def form_clusters_via_association(tracks, reports, kalman_filter, chi2_threshold):
    association_list = []
    cov_inv = np.linalg.inv(kalman_filter.Pp[:3, :3])  # 3x3 covariance matrix for position only

    for i, track in enumerate(tracks):
        for j, report in enumerate(reports):
            distance = mahalanobis_distance(track, report, cov_inv)
            if distance < chi2_threshold:
                association_list.append((i, j))

    clusters = []
    while association_list:
        cluster_tracks = set()
        cluster_reports = set()
        stack = [association_list.pop(0)]
        
        while stack:
            track_idx, report_idx = stack.pop()
            cluster_tracks.add(track_idx)
            cluster_reports.add(report_idx)
            new_assoc = [(t, r) for t, r in association_list if t == track_idx or r == report_idx]
            for assoc in new_assoc:
                if assoc not in stack:
                    stack.append(assoc)
            association_list = [assoc for assoc in association_list if assoc not in new_assoc]
        
        clusters.append((list(cluster_tracks), [reports[r] for r in cluster_reports]))

    return clusters


def mahalanobis_distance(track, report, cov_inv):
    residual = np.array(report) - np.array(track)
    distance = np.dot(np.dot(residual.T, cov_inv), residual)
    return distance


def select_best_report(cluster_tracks, cluster_reports, kalman_filter):
    cov_inv = np.linalg.inv(kalman_filter.Pp[:3, :3])

    best_report = None
    best_track_idx = None
    max_weight = -np.inf

    for i, track in enumerate(cluster_tracks):
        for j, report in enumerate(cluster_reports):
            residual = np.array(report) - np.array(track)
            weight = np.exp(-0.5 * np.dot(np.dot(residual.T, cov_inv), residual))
            if weight > max_weight:
                max_weight = weight
                best_report = report
                best_track_idx = i

    return best_track_idx, best_report

def select_initiation_mode(mode):
    if mode == '3-state':
        return 3
    elif mode == '5-state':
        return 5
    elif mode == '7-state':
        return 7
    else:
        raise ValueError("Invalid mode selected.")
    
    
def doppler_correlation(doppler_1, doppler_2, doppler_threshold):
    return abs(doppler_1 - doppler_2) < doppler_threshold

def correlation_check(track, measurement, doppler_threshold, range_threshold):
    last_measurement = track['measurements'][-1][0]
    last_cartesian = sph2cart(last_measurement[0], last_measurement[1], last_measurement[2])
    measurement_cartesian = sph2cart(measurement[0], measurement[1], measurement[2])
    distance = np.linalg.norm(np.array(measurement_cartesian) - np.array(last_cartesian))
    
    doppler_correlated = doppler_correlation(measurement[4], last_measurement[4], doppler_threshold)
    range_satisfied = distance < range_threshold
    
    return doppler_correlated and range_satisfied

def initialize_filter_state(kalman_filter, x, y, z, vx, vy, vz, time):
    kalman_filter.initialize_filter_state(x, y, z, vx, vy, vz, time)

def perform_jpda(tracks, reports, kalman_filter):
    clusters = form_clusters_via_association(tracks, reports, kalman_filter, kalman_filter.gate_threshold)
    best_reports = []
    for cluster_tracks, cluster_reports in clusters:
        best_track_idx, best_report = select_best_report(cluster_tracks, cluster_reports, kalman_filter)
        if best_report is not None:
            best_reports.append((best_track_idx, best_report))
    return best_reports

def check_track_timeout(tracks, current_time, poss_timeout=2.0, firm_tent_timeout=5.0):
    tracks_to_remove = []
    for track_id, track in enumerate(tracks):
        last_measurement_time = track['measurements'][-1][0][3]  # Assuming the time is at index 3
        time_since_last_measurement = current_time - last_measurement_time
        
        if track['current_state'] == 'Poss1' and time_since_last_measurement > poss_timeout:
            tracks_to_remove.append(track_id)
        elif track['current_state'] in ['Tentative1', 'Firm'] and time_since_last_measurement > firm_tent_timeout:
            tracks_to_remove.append(track_id)
    
    return tracks_to_remove

def main():
    file_path = 'ttk.csv'
    measurements = read_measurements_from_csv(file_path)

    kalman_filter = CVFilter()
    measurement_groups = form_measurement_groups(measurements, max_time_diff=0.050)

    tracks = []
    track_id_list = []
    filter_states = []

    doppler_threshold = 100
    range_threshold = 100
    firm_threshold = 3
    mode = '3-state'

    firm_threshold = select_initiation_mode(mode)
    # Initialize variables outside the loop
    miss_counts = {}
    hit_counts = {}
    firm_ids = set()
    state_map = {}
    state_transition_times = {}
    progression_states = {
        3: ['Poss1', 'Tentative1', 'Firm'],
        5: ['Poss1', 'Poss2', 'Tentative1', 'Tentative2', 'Firm'],
        7: ['Poss1', 'Poss2', 'Tentative1', 'Tentative2', 'Tentative3', 'Firm']
    }[firm_threshold]

    last_check_time = 0
    check_interval = 0.0005  # 0.5 ms

    for group_idx, group in enumerate(measurement_groups):
        print(f"Processing measurement group {group_idx + 1}...")
        
        current_time = group[0][3]  # Assuming the time is at index 3 of each measurement
        
        # Periodic checking
        if current_time - last_check_time >= check_interval:
            tracks_to_remove = check_track_timeout(tracks, current_time)
            for track_id in reversed(tracks_to_remove):
                print(f"Removing track {track_id} due to timeout")
                del tracks[track_id]
                track_id_list[track_id]['state'] = 'free'
                if track_id in firm_ids:
                    firm_ids.remove(track_id)
                if track_id in state_map:
                    del state_map[track_id]
                if track_id in hit_counts:
                    del hit_counts[track_id]
                if track_id in miss_counts:
                    del miss_counts[track_id]
            last_check_time = current_time

        if len(group) == 1:  # Single measurement
            measurement = group[0]
            assigned = False
            for track_id, track in enumerate(tracks):
                if correlation_check(track, measurement, doppler_threshold, range_threshold):
                    current_state = state_map.get(track_id, None)
                    if current_state == 'Poss1':
                        initialize_filter_state(kalman_filter, *sph2cart(*measurement[:3]), 0, 0, 0, measurement[3])
                    elif current_state == 'Tentative1':
                        last_measurement = track['measurements'][-1][0]
                        dt = measurement[3] - last_measurement[3]
                        vx = (sph2cart(*measurement[:3])[0] - sph2cart(*last_measurement[:3])[0]) / dt
                        vy = (sph2cart(*measurement[:3])[1] - sph2cart(*last_measurement[:3])[1]) / dt
                        vz = (sph2cart(*measurement[:3])[2] - sph2cart(*last_measurement[:3])[2]) / dt
                        initialize_filter_state(kalman_filter, *sph2cart(*measurement[:3]), vx, vy, vz, measurement[3])
                    elif current_state == 'Firm':
                        kalman_filter.predict_step(measurement[3])
                        kalman_filter.update_step(np.array(sph2cart(*measurement[:3])).reshape(3, 1))
                    
                    track['measurements'].append((measurement, current_state))
                    hit_counts[track_id] = hit_counts.get(track_id, 0) + 1
                    assigned = True
                    break

            if not assigned:
                new_track_id = next((i for i, t in enumerate(track_id_list) if t['state'] == 'free'), None)
                if new_track_id is None:
                    new_track_id = len(track_id_list)
                    track_id_list.append({'id': new_track_id, 'state': 'occupied'})
                else:
                    track_id_list[new_track_id]['state'] = 'occupied'
                
                tracks.append({
                    'track_id': new_track_id,
                    'measurements': [(measurement, 'Poss1')],
                    'current_state': 'Poss1'
                })
                state_map[new_track_id] = 'Poss1'
                state_transition_times[new_track_id] = {'Poss1': current_time}
                hit_counts[new_track_id] = 1
                initialize_filter_state(kalman_filter, *sph2cart(*measurement[:3]), 0, 0, 0, measurement[3])

        else:  # Multiple measurements
            reports = [sph2cart(*m[:3]) for m in group]
            best_reports = perform_jpda([track['measurements'][-1][0][:3] for track in tracks], reports, kalman_filter)

            for track_id, best_report in best_reports:
                current_state = state_map.get(track_id, None)
                if current_state == 'Poss1':
                    initialize_filter_state(kalman_filter, *best_report, 0, 0, 0, group[0][3])
                elif current_state == 'Tentative1':
                    last_measurement = tracks[track_id]['measurements'][-1][0]
                    dt = group[0][3] - last_measurement[3]
                    vx = (best_report[0] - sph2cart(*last_measurement[:3])[0]) / dt
                    vy = (best_report[1] - sph2cart(*last_measurement[:3])[1]) / dt
                    vz = (best_report[2] - sph2cart(*last_measurement[:3])[2]) / dt
                    initialize_filter_state(kalman_filter, *best_report, vx, vy, vz, group[0][3])
                elif current_state == 'Firm':
                    kalman_filter.predict_step(group[0][3])
                    kalman_filter.update_step(np.array(best_report).reshape(3, 1))
                
                tracks[track_id]['measurements'].append((cart2sph(*best_report) + (group[0][3], group[0][4]), current_state))
                hit_counts[track_id] = hit_counts.get(track_id, 0) + 1

            # Handle unassigned measurements
            assigned_reports = set(best_report for _, best_report in best_reports)
            for report in reports:
                if tuple(report) not in assigned_reports:
                    new_track_id = next((i for i, t in enumerate(track_id_list) if t['state'] == 'free'), None)
                    if new_track_id is None:
                        new_track_id = len(track_id_list)
                        track_id_list.append({'id': new_track_id, 'state': 'occupied'})
                    else:
                        track_id_list[new_track_id]['state'] = 'occupied'
                    
                    tracks.append({
                        'track_id': new_track_id,
                        'measurements': [(cart2sph(*report) + (group[0][3], group[0][4]), 'Poss1')],
                        'current_state': 'Poss1'
                    })
                    state_map[new_track_id] = 'Poss1'
                    state_transition_times[new_track_id] = {'Poss1': current_time}
                    hit_counts[new_track_id] = 1
                    initialize_filter_state(kalman_filter, *report, 0, 0, 0, group[0][3])

        # Update states based on hit counts
        for track_id, track in enumerate(tracks):
            if hit_counts[track_id] >= firm_threshold and state_map[track_id] != 'Firm':
                state_map[track_id] = 'Firm'
                firm_ids.add(track_id)
                state_transition_times.setdefault(track_id, {})['Firm'] = current_time
            elif hit_counts[track_id] == 2 and state_map[track_id] != 'Tentative1':
                state_map[track_id] = 'Tentative1'
                state_transition_times.setdefault(track_id, {})['Tentative1'] = current_time
            elif hit_counts[track_id] == 1 and track_id not in state_map:
                state_map[track_id] = 'Poss1'
                state_transition_times.setdefault(track_id, {})['Poss1'] = current_time
            track['current_state'] = state_map[track_id]

    # Print summary and prepare data for CSV
    csv_data = []
    for track_id, track in enumerate(tracks):
        print(f"Track {track_id}:")
        print(f"  Current State: {track['current_state']}")
        print(f"  State Transition Times:")
        for state, time in state_transition_times.get(track_id, {}).items():
            print(f"    {state}: {time}")
        print("  Measurement History:")
        for state in ['Poss1', 'Tentative1', 'Firm']:
            measurements = [m for m, s in track['measurements'] if s == state][:3]
            print(f"    {state}: {measurements}")
        print(f"  Track Status: {track_id_list[track_id]['state']}")
        print(f"  SF: {hit_counts.get(track_id, 0)}")
        print(f"  SP: {miss_counts.get(track_id, 0)}")
        print(f"  PF: N/A")  # You might need to implement logic for PF
        print(f"  PP: N/A")  # You might need to implement logic for PP
        print()

        # Prepare data for CSV
        csv_data.append({
            'Track ID': track_id,
            'Current State': track['current_state'],
            'Poss1 Time': state_transition_times.get(track_id, {}).get('Poss1', ''),
            'Tentative1 Time': state_transition_times.get(track_id, {}).get('Tentative1', ''),
            'Firm Time': state_transition_times.get(track_id, {}).get('Firm', ''),
            'Poss1 Measurements': str([m for m, s in track['measurements'] if s == 'Poss1'][:3]),
            'Tentative1 Measurements': str([m for m, s in track['measurements'] if s == 'Tentative1'][:3]),
            'Firm Measurements': str([m for m, s in track['measurements'] if s == 'Firm'][:3]),
            'Track Status': track_id_list[track_id]['state'],
            'SF': hit_counts.get(track_id, 0),
            'SP': miss_counts.get(track_id, 0),
            'PF': 'N/A',  # You might need to implement logic for PF
            'PP': 'N/A'   # You might need to implement logic for PP
        })

    # Write to CSV
    csv_file_path = 'track_summary.csv'
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['Track ID', 'Current State', 'Poss1 Time', 'Tentative1 Time', 'Firm Time',
                      'Poss1 Measurements', 'Tentative1 Measurements', 'Firm Measurements',
                      'Track Status', 'SF', 'SP', 'PF', 'PP']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_data:
            writer.writerow(row)

    print(f"Track summary has been written to {csv_file_path}")

if __name__ == "__main__":
    main()