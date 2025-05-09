import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Load the data
df = pd.read_excel("HR_Employee_Data.xlsx")

# Convert percentage strings to floats if needed
if isinstance(df['satisfaction_level'].iloc[0], str):
    df['satisfaction_level'] = df['satisfaction_level'].str.rstrip('%').astype(float) / 100
    df['last_evaluation'] = df['last_evaluation'].str.rstrip('%').astype(float) / 100

# Create a new figure with a larger size for better visibility
fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

# Create a sample of the data to avoid overcrowding (if dataset is large)
# Adjust the sample size based on your needs
sample_size = min(1000, len(df))
df_sample = df.sample(sample_size, random_state=42)

# Extract the data for plotting
x = df_sample['satisfaction_level']
y = df_sample['last_evaluation']
z = df_sample['average_montly_hours']

# Color mapping based on whether the employee left
colors = np.where(df_sample['left'] == 1, 'red', 'blue')
sizes = np.where(df_sample['left'] == 1, 80, 30)  # Make departing employees larger

# Create the 3D scatter plot with varying point sizes
scatter = ax.scatter(x, y, z, c=colors, s=sizes, alpha=0.6)

# Add department information as text annotations for selected points
# Choose a few interesting points (high workload, low satisfaction, etc.)
interesting_points = df_sample[
    ((df_sample['satisfaction_level'] < 0.3) & 
     (df_sample['last_evaluation'] > 0.7) & 
     (df_sample['left'] == 1)) |
    ((df_sample['average_montly_hours'] > 300) & 
     (df_sample['left'] == 1))
].sample(min(10, len(df_sample)), random_state=42)

for idx, row in interesting_points.iterrows():
    ax.text(row['satisfaction_level'], row['last_evaluation'], 
            row['average_montly_hours'], row['Department'], 
            color='black', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))

# Add a reference surface showing the "danger zone" for attrition
# Calculate a grid of points
x_grid = np.linspace(0, 1, 20)
y_grid = np.linspace(0, 1, 20)
X, Y = np.meshgrid(x_grid, y_grid)

# Create a surface representing high attrition risk
# This equation approximates regions with high attrition based on satisfaction and evaluation
Z = 250 + 50 * (1 - X) + 30 * Y  # Higher hours for low satisfaction employees
ax.plot_surface(X, Y, Z, alpha=0.15, color='red', 
                linewidth=0, antialiased=True, shade=True)

# Add axis labels with a larger font size
ax.set_xlabel('Satisfaction Level', fontsize=14, labelpad=10)
ax.set_ylabel('Performance Evaluation', fontsize=14, labelpad=10)
ax.set_zlabel('Monthly Hours', fontsize=14, labelpad=10)

# Set the viewing angle for optimal visualization
ax.view_init(elev=20, azim=240)

# Add a title
plt.suptitle('3D Employee Attrition Analysis: Satisfaction, Performance, and Workload', 
             fontsize=16, y=0.95)

# Add an explanatory subtitle
plt.figtext(0.5, 0.01, 
            'Red points represent employees who left. '
            'The translucent red surface indicates the high-risk "burnout zone".',
            ha='center', fontsize=12)

# Add a legend for the colors
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
               markersize=10, label='Stayed'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
               markersize=10, label='Left')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

# Rotate the plot to have a clear view
for angle in range(0, 360, 5):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(0.001)

plt.tight_layout()
plt.savefig('hr_3d_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# Optional: Create a second 3D plot showing another interesting relationship
fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

# For this visualization, we'll explore the relationship between:
# 1. Years at company, 2. Number of projects, 3. Satisfaction
x2 = df_sample['time_spend_company']
y2 = df_sample['number_project'] 
z2 = df_sample['satisfaction_level']

# Create a colormap based on workload (hours * projects)
workload = df_sample['average_montly_hours'] * df_sample['number_project']
normalized_workload = (workload - workload.min()) / (workload.max() - workload.min())
colors = plt.cm.plasma(normalized_workload)

# Create the scatter plot with the colormap
scatter = ax.scatter(x2, y2, z2, c=colors, s=50, alpha=0.7)

# Add a colorbar to show the workload scale
color_map = cm.ScalarMappable(cmap=plt.cm.plasma)
color_map.set_array(workload)
cbar = plt.colorbar(color_map)
cbar.set_label('Workload Intensity (Hours × Projects)', fontsize=12)

# Add axis labels
ax.set_xlabel('Years at Company', fontsize=14, labelpad=10)
ax.set_ylabel('Number of Projects', fontsize=14, labelpad=10)
ax.set_zlabel('Satisfaction Level', fontsize=14, labelpad=10)

# Show median satisfaction as a transparent plane for reference
median_satisfaction = df['satisfaction_level'].median()
x_grid = np.linspace(min(x2), max(x2), 10)
y_grid = np.linspace(min(y2), max(y2), 10)
X2, Y2 = np.meshgrid(x_grid, y_grid)
Z2 = np.ones(X2.shape) * median_satisfaction
ax.plot_surface(X2, Y2, Z2, alpha=0.2, color='gray', linewidth=0)

# Add a title
plt.suptitle('3D Employee Satisfaction Analysis: Tenure, Projects, and Workload', 
             fontsize=16, y=0.95)

# Add annotations for clusters
# Find high workload with low satisfaction
overworked = df_sample[
    (df_sample['number_project'] > 5) & 
    (df_sample['satisfaction_level'] < 0.3)
].sample(min(5, len(df_sample)), random_state=42)

for idx, row in overworked.iterrows():
    ax.text(row['time_spend_company'], row['number_project'], 
            row['satisfaction_level'], f"Overworked\n{row['Department']}", 
            color='red', fontsize=9, bbox=dict(facecolor='white', alpha=0.5))

# Animate the rotation
for angle in range(0, 360, 5):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(0.001)

plt.tight_layout()
plt.savefig('hr_3d_satisfaction_tenure_projects.png', dpi=300, bbox_inches='tight')
plt.show()
