import matplotlib.pyplot as plt
import numpy as np

# Define risk categories and their impact levels
risk_categories = [
    "Incomplete/Unbalanced Dataset",
    "Poor Audio Quality",
    "Speech Model Generalization",
    "Text Model Misclassification",
    "GPU Shortage",
    "Slow Model Training",
    "Web App Performance Issues",
    "Team Communication Breakdown",
    "Version Control Conflicts",
    "Research Paper Rejection",
    "Plagiarism Concerns"
]

impact_levels = [
    "Low", "Medium", "High"
]

# Assign risk levels (0 = Low, 1 = Medium, 2 = High)
risk_levels = np.array([2, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1])

# Create color mapping for heatmap (Low: Green, Medium: Yellow, High: Red)
colors = np.array(["green", "yellow", "red"])

# Generate heatmap
plt.figure(figsize=(10, 6))
plt.barh(risk_categories, [1]*len(risk_categories), color=colors[risk_levels])
plt.xlabel("Risk Impact Level")
plt.title("Risk Assessment Heatmap")
plt.xticks([])  # Remove x-axis labels since we're using colors for impact
plt.yticks(fontsize=10)

# Add color legend
plt.legend(handles=[
    plt.Rectangle((0, 0), 1, 1, color="green", label="Low"),
    plt.Rectangle((0, 0), 1, 1, color="yellow", label="Medium"),
    plt.Rectangle((0, 0), 1, 1, color="red", label="High")
], loc="lower right")

plt.show()
