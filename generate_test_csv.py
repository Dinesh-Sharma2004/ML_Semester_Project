import pandas as pd
import numpy as np

def generate_synthetic_data(n_students=50, n_weeks=10):
    """
    Generates a synthetic longitudinal dataset for the
    Smart Hostel (SHS-Opt) project.
    """
    print(f"Generating data for {n_students} students...")
    
    personas = {
        'early_bird_focused': {
            'prev_gpa_mean': 3.5, 'gpa_noise': 0.1,
            'sleep_hours': 7.5, 'sleep_var': 0.5,
            'study_hours': 25, 'library_days': 4,
            'wifi_hours': 3, 'wifi_start': 7,
            'stress': 2.0, 'mood': 4.0, 'steps': 8000
        },
        'night_owl_socializer': {
            'prev_gpa_mean': 3.0, 'gpa_noise': 0.3,
            'sleep_hours': 6.0, 'sleep_var': 1.5,
            'study_hours': 10, 'library_days': 1,
            'wifi_hours': 6, 'wifi_start': 11,
            'stress': 3.5, 'mood': 3.0, 'steps': 6000
        },
        'sporadic_at_risk': {
            'prev_gpa_mean': 2.4, 'gpa_noise': 0.4,
            'sleep_hours': 5.5, 'sleep_var': 2.5,
            'study_hours': 5, 'library_days': 0,
            'wifi_hours': 8, 'wifi_start': 14,
            'stress': 4.5, 'mood': 2.0, 'steps': 4000
        },
        'balanced_all_rounder': {
            'prev_gpa_mean': 3.2, 'gpa_noise': 0.2,
            'sleep_hours': 7.0, 'sleep_var': 0.8,
            'study_hours': 15, 'library_days': 2,
            'wifi_hours': 4, 'wifi_start': 9,
            'stress': 2.5, 'mood': 3.5, 'steps': 7000
        }
    }

    student_data = []
    
    for i in range(n_students):
        student_id = f'test_s_{2000 + i}' 
        persona_name = np.random.choice(list(personas.keys()))
        persona = personas[persona_name]
        
        base_prev_gpa = np.clip(np.random.normal(persona['prev_gpa_mean'], 0.2), 2.0, 4.0)
        department = np.random.choice(['CS', 'ECE', 'MECH', 'CIVIL'])
        year = np.random.choice([1, 2, 3, 4])
        

        week = 1
        
        sleep_hours = np.clip(np.random.normal(persona['sleep_hours'], persona['sleep_var']), 3, 10)
        sleep_var = np.clip(np.random.normal(persona['sleep_var'], 0.2), 0.2, 3)
        self_study_hours = np.clip(np.random.normal(persona['study_hours'], 5), 0, 40)
        library_days = np.clip(np.random.normal(persona['library_days'], 1), 0, 7)
        wifi_hours = np.clip(np.random.normal(persona['wifi_hours'], 2), 1, 12)
        wifi_start = np.clip(np.random.normal(persona['wifi_start'], 2), 0, 23)
        stress = np.clip(np.random.normal(persona['stress'], 0.5), 1, 5)
        mood = np.clip(np.random.normal(persona['mood'], 0.5), 1, 5)
        steps = np.clip(np.random.normal(persona['steps'], 1500), 1000, 20000)

        student_data.append({
            'student_id': student_id,
            'department': department,
            'year': year,
            'prev_gpa': base_prev_gpa,
            'avg_sleep_hours': sleep_hours,
            'sleep_variability': sleep_var,
            'self_reported_study_hours': self_study_hours,
            'library_days_per_week': library_days,
            'avg_daily_online_hours': wifi_hours,
            'typical_connection_window_start': wifi_start,
            'stress_level': stress,
            'mood_score': mood,
            'avg_steps_per_day': steps,
            'cafeteria_veg_pref': np.random.choice([0, 1]),
            'roommate_conflicts_reported': 1 if (stress > 4.5 and np.random.rand() < 0.2) else 0,
        })

    return pd.DataFrame(student_data)

if __name__ == "__main__":
    numerical_cols = [
        'prev_gpa', 'avg_sleep_hours', 'sleep_variability', 
        'self_reported_study_hours', 'library_days_per_week', 
        'avg_daily_online_hours', 'typical_connection_window_start',
        'stress_level', 'mood_score', 'avg_steps_per_day'
    ]
    categorical_cols = [
        'department', 'year', 'cafeteria_veg_pref', 
        'roommate_conflicts_reported'
    ]

    all_required_cols = ['student_id'] + numerical_cols + categorical_cols
    

    df_test = generate_synthetic_data(n_students=50)
    

    df_test_final = df_test[all_required_cols]
    

    output_filename = "test_upload_data.csv"
    df_test_final.to_csv(output_filename, index=False)
    
    print(f"✅ Successfully generated '{output_filename}' with {len(df_test_final)} students.")
    print("You can now drag and drop this file into the dashboard.")