import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

# --- CONSTANTS ---
GALLONS_PER_BARREL = 42
BTU_PER_GALLON = 12194
BTU_PER_MJ = 947.817
BIOMASS_LHV = 18.6
EFFICIENCY = 0.50
DAYS_PER_YEAR = 365
KG_PER_TONNE = 1000
SQMI_TO_SQKM = 2.58999

FACILITY_SCALES = {
    "150,000 barrels/day (Large)": 150000,
    "10,000 barrels/day (Small)": 10000
}

FACILITY_LOCATIONS = {
    "Kansas City, MO": 29095,
    "Pittsburgh, PA": 42003
}

# DATA FILES (local paths in Colab)
BIOMASS_FILES = [
    "billionton_23_agri_mm-med.csv",
    "billionton_23_forestry_mm-med.csv",
    "billionton_23_wastes_mm-med.csv"
]
ADJACENCY_FILE = "county_adjacency_matrix.csv"


def calculate_required_biomass(barrels_per_day):
    """Calculate biomass required in tonnes/year for a given facility scale"""
    energy_output_btu_day = barrels_per_day * GALLONS_PER_BARREL * BTU_PER_GALLON
    energy_output_mj_day = energy_output_btu_day / BTU_PER_MJ
    energy_input_mj_day = energy_output_mj_day / EFFICIENCY
    biomass_mass_kg_day = energy_input_mj_day / BIOMASS_LHV
    biomass_mass_tonnes_year = biomass_mass_kg_day * DAYS_PER_YEAR / KG_PER_TONNE
    return biomass_mass_tonnes_year


def load_biomass_and_area_data():
    """
    Load biomass CSV files which contain BOTH production AND area data (Sqmi).
    Aggregate biomass by FIPS and use the area from the first occurrence.
    """
    print("\n" + "="*70)
    print("LOADING BIOMASS AND AREA DATA")
    print("="*70)
    
    all_data = []
    
    for i, filename in enumerate(BIOMASS_FILES, 1):
        source = filename.split('_')[2]  # Extract agri/forestry/wastes
        print(f"\n{i}. Loading {source} from {filename}...")
        
        try:
            df = pd.read_csv(filename)
            print(f"   Columns: {list(df.columns)[:10]}...")
            
            # Find required columns
            fips_col = next((c for c in df.columns if 'fips' in c.lower()), None)
            prod_col = next((c for c in df.columns if 'production' in c.lower()), None)
            area_col = next((c for c in df.columns if 'sqmi' in c.lower()), None)
            
            if not fips_col or not prod_col or not area_col:
                print(f"   ✗ Missing required columns")
                print(f"   Need: FIPS, Production, and Sqmi")
                continue
            
            # Extract columns
            df = df[[fips_col, prod_col, area_col]].copy()
            df.columns = ['FIPS', 'biomass', 'sqmi']
            
            # Convert to numeric
            df['FIPS'] = pd.to_numeric(df['FIPS'], errors='coerce')
            df['biomass'] = pd.to_numeric(df['biomass'], errors='coerce')
            df['sqmi'] = pd.to_numeric(df['sqmi'], errors='coerce')
            df = df.dropna()
            df['FIPS'] = df['FIPS'].astype(int)
            
            print(f"   ✓ {len(df)} records, {df['biomass'].sum():,.0f} tonnes/year")
            all_data.append(df)
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_data:
        return None
    
    # Combine all sources
    print(f"\nCombining all biomass sources...")
    df_combined = pd.concat(all_data, ignore_index=True)
    
    # Group by FIPS: sum biomass, take first sqmi (they should all be the same for a given FIPS)
    df_result = df_combined.groupby('FIPS', as_index=False).agg({
        'biomass': 'sum',
        'sqmi': 'first'
    })
    
    # Convert area to sq km
    df_result['land_area_sq_km'] = df_result['sqmi'] * SQMI_TO_SQKM
    df_result.rename(columns={'biomass': 'biomass_tonnes'}, inplace=True)
    
    print(f"✓ Final dataset:")
    print(f"  {len(df_result)} counties")
    print(f"  Total biomass: {df_result['biomass_tonnes'].sum():,.0f} tonnes/year")
    print(f"  Total land area: {df_result['land_area_sq_km'].sum():,.0f} sq km")
    print(f"  Avg county size: {df_result['land_area_sq_km'].mean():,.1f} sq km")
    
    return df_result[['FIPS', 'biomass_tonnes', 'land_area_sq_km']]


def load_adjacency_matrix():
    """Load county adjacency matrix and build NetworkX graph"""
    print("\n" + "="*70)
    print("LOADING ADJACENCY MATRIX")
    print("="*70)
    
    try:
        df_adj = pd.read_csv(ADJACENCY_FILE, index_col=0)
        print(f"✓ Matrix shape: {df_adj.shape}")
        
        # Convert index and columns to integers
        df_adj.index = df_adj.index.astype(int)
        df_adj.columns = df_adj.columns.astype(int)
        
        # Build graph from adjacency matrix
        G = nx.Graph()
        
        for county_i in df_adj.index:
            for county_j in df_adj.columns:
                if county_i < county_j and df_adj.at[county_i, county_j] == 1:
                    G.add_edge(county_i, county_j)
        
        print(f"✓ Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def bfs_accumulate_biomass(start_fips, required_tonnes, df_data, graph):
    """
    BFS from starting county (facility location) through adjacent counties.
    Accumulate biomass and land area until requirement is met.
    
    Args:
        start_fips: FIPS code of facility location
        required_tonnes: Biomass requirement in tonnes/year
        df_data: DataFrame with FIPS, biomass_tonnes, land_area_sq_km
        graph: NetworkX graph of county adjacencies
    
    Returns:
        (total_area, total_biomass, num_counties, county_list)
    """
    
    # Check if start county exists
    if start_fips not in df_data['FIPS'].values:
        print(f"    ⚠ Starting FIPS {start_fips} not in data")
        return 0, 0, 0, []
    
    if start_fips not in graph:
        print(f"    ⚠ Starting FIPS {start_fips} not in graph")
        return 0, 0, 0, []
    
    # Create lookup dictionary
    data_dict = df_data.set_index('FIPS').to_dict('index')
    
    # Initialize BFS
    queue = deque([start_fips])
    visited = {start_fips}
    
    total_biomass = 0.0
    total_area = 0.0
    counties_used = []
    
    print(f"    Starting BFS from FIPS {start_fips}...")
    
    iteration = 0
    while queue and total_biomass < required_tonnes:
        current_fips = queue.popleft()
        iteration += 1
        
        if current_fips in data_dict:
            county_data = data_dict[current_fips]
            biomass = float(county_data['biomass_tonnes'])
            area = float(county_data['land_area_sq_km'])
            
            total_biomass += biomass
            total_area += area
            counties_used.append(current_fips)
            
            # Debug output for first few counties
            if iteration <= 5:
                print(f"      County {current_fips}: +{biomass:,.0f} tonnes, +{area:,.1f} sq km")
            
            # Check if requirement met
            if total_biomass >= required_tonnes:
                print(f"    ✓ Requirement met after {iteration} counties")
                break
            
            # Add adjacent counties to queue
            if current_fips in graph:
                for neighbor in graph.neighbors(current_fips):
                    if neighbor not in visited and neighbor in data_dict:
                        visited.add(neighbor)
                        queue.append(neighbor)
    
    if total_biomass < required_tonnes:
        print(f"    ⚠ Could not fully meet requirement")
        print(f"      Collected: {total_biomass:,.0f} / {required_tonnes:,.0f} tonnes ({total_biomass/required_tonnes*100:.1f}%)")
    
    return total_area, total_biomass, len(counties_used), counties_used


def run_analysis():
    """Main analysis routine"""
    print("\n" + "="*80)
    print(" BIOFUEL FACILITY LAND AREA ANALYSIS ".center(80, "="))
    print("="*80)
    
    # Step 1: Calculate requirements
    print("\n" + "="*70)
    print("STEP 1: CALCULATE BIOMASS REQUIREMENTS")
    print("="*70)
    
    requirements = {}
    for scale_name, barrels_per_day in FACILITY_SCALES.items():
        req_tonnes = calculate_required_biomass(barrels_per_day)
        requirements[scale_name] = req_tonnes
        print(f"  {scale_name}: {req_tonnes:,.0f} tonnes/year")
    
    # Step 2: Load data (biomass files contain area!)
    df_data = load_biomass_and_area_data()
    if df_data is None:
        print("\n✗ Failed to load data")
        return
    
    graph = load_adjacency_matrix()
    if graph is None:
        print("\n✗ Failed to load adjacency data")
        return
    
    # Step 3: Verify facility locations
    print("\n" + "="*70)
    print("STEP 2: VERIFY FACILITY LOCATIONS")
    print("="*70)
    
    for location, fips in FACILITY_LOCATIONS.items():
        in_data = fips in df_data['FIPS'].values
        in_graph = fips in graph
        
        if in_data:
            row = df_data[df_data['FIPS'] == fips].iloc[0]
            print(f"\n{location} (FIPS {fips}):")
            print(f"  In data: {in_data}, In graph: {in_graph}")
            print(f"  Biomass: {row['biomass_tonnes']:,.0f} tonnes/year")
            print(f"  Area: {row['land_area_sq_km']:,.1f} sq km")
        else:
            print(f"\n{location} (FIPS {fips}): NOT FOUND")
    
    # Step 4: Run BFS analysis
    print("\n" + "="*70)
    print("STEP 3: RUN BFS ANALYSIS")
    print("="*70)
    
    results = {}
    
    for location, fips in FACILITY_LOCATIONS.items():
        print(f"\n{location} (FIPS {fips}):")
        print("-" * 60)
        
        location_results = {}
        
        for scale_name, req_tonnes in requirements.items():
            area, biomass, num_counties, county_list = bfs_accumulate_biomass(
                fips, req_tonnes, df_data, graph
            )
            
            pct_met = (biomass / req_tonnes * 100) if req_tonnes > 0 else 0
            
            print(f"\n  {scale_name}:")
            print(f"    Required:        {req_tonnes:>15,.0f} tonnes/year")
            print(f"    Available:       {biomass:>15,.0f} tonnes/year ({pct_met:.1f}%)")
            print(f"    Land Area:       {area:>15,.0f} sq km")
            print(f"    Counties Used:   {num_counties:>15,}")
            
            location_results[scale_name] = {
                'area': area,
                'biomass': biomass,
                'counties': num_counties,
                'percentage': pct_met
            }
        
        results[location] = location_results
    
    # Step 5: Visualize
    print("\n" + "="*70)
    print("STEP 4: GENERATE VISUALIZATION")
    print("="*70)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Land Area Required to Meet Biofuel Facility Demand\n(BFS from Facility County Through Adjacent Counties)', 
                 fontsize=14, fontweight='bold')
    
    colors = ['#2E86AB', '#A23B72']
    
    for idx, (location, data) in enumerate(results.items()):
        ax = axes[idx]
        
        scales = list(data.keys())
        areas = [data[s]['area'] for s in scales]
        counties = [data[s]['counties'] for s in scales]
        
        x_pos = np.arange(len(scales))
        bars = ax.bar(x_pos, areas, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel('Facility Scale', fontsize=11, fontweight='bold')
        ax.set_ylabel('Land Area (sq km)', fontsize=11, fontweight='bold')
        ax.set_title(f'{location}', fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(scales, rotation=15, ha='right', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        for i, (bar, area, county_count) in enumerate(zip(bars, areas, counties)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{area:,.0f} sq km\n({county_count} counties)',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("\n✓ ANALYSIS COMPLETE!")
    print("="*70)
    
    return results


if __name__ == "__main__":
    run_analysis()
