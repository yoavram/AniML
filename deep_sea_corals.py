import pandas as pd
import numpy as np

# @title 🌊 Generate "Messy" Deep Sea Coral Dataset (Hard Mode)
np.random.seed(42)
n_samples = 1500

species = []
depths = []
temps = []
salinities = []
oxygens = []

for i in range(n_samples):
    sp = np.random.choice(['Stylophora_pistillata', 'Dendrophyllia_sp', 'Desmophyllum_pertusum'])
    
    # Overlapping Niches!
    if sp == 'Stylophora_pistillata':
        # Shallow: 0-60m
        d = np.random.normal(30, 15) 
        t = np.random.normal(26, 2.0)
        o2 = np.random.normal(5.0, 0.8)
    
    elif sp == 'Dendrophyllia_sp':
        # Twilight Zone: 40-120m (Significant overlap with Shallow)
        d = np.random.normal(80, 25) 
        t = np.random.normal(24, 2.5) # Temps overlap with shallow
        o2 = np.random.normal(4.0, 0.8)
        
    elif sp == 'Desmophyllum_pertusum':
        # Deepish: 100m+ (Overlap with Dendro)
        d = np.random.normal(150, 40)
        t = np.random.normal(22, 2.0)
        o2 = np.random.normal(3.0, 1.0)

    # Add Sensor Glitches (Outliers)
    if np.random.random() < 0.05: # 5% of data is "bad"
        d = d * np.random.uniform(0.5, 1.5)
        t = t + np.random.normal(0, 5)

    # Ensure physics make sense (no negative depth/oxygen)
    d = abs(d)
    o2 = max(0, o2)
    s = np.random.normal(40.5, 0.5) # Salinity is barely useful (noise variable)

    species.append(sp)
    depths.append(round(d, 1))
    temps.append(round(t, 2))
    salinities.append(round(s, 3))
    oxygens.append(round(o2, 2))

df_coral = pd.DataFrame({
    'Species_ID': species,
    'Depth_m': depths,
    'Temperature_C': temps,
    'Salinity_PPS': salinities,
    'Dissolved_Oxygen_mlL': oxygens
})

df_coral.to_csv('deep_sea_corals.csv', index=False)
print("✅ Created 'Hard Mode' dataset. Overlap enabled.")