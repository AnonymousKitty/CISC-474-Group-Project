import json
import matplotlib.pyplot as plt
import numpy as np

def plot_scores(test_path):
    with open(f'{test_path}/scores_final.json', 'r') as f:
        final_scores = json.load(f)
    
    try:
        with open(f'{test_path}/scores.json', 'r') as f:
            progress_scores = json.load(f)
    except FileNotFoundError:
        progress_scores = None
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    maps = [k for k in final_scores.keys() if k != "total"]
    scores = [final_scores[m] for m in maps]
    plt.bar(maps, scores)
    plt.title('Final Performance by Map')
    plt.ylabel('Coverage Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    if progress_scores:
        plt.subplot(1, 2, 2)
        timesteps = np.arange(len(progress_scores["total"]))  
        for map_name in maps:
            plt.plot(timesteps, progress_scores[map_name], label=map_name)
        # plt.plot(timesteps, progress_scores["total"], 'k--', label='Total')
        plt.title('Training Progress')
        plt.xlabel('Evaluation Points')
        plt.ylabel('Coverage Score')
        plt.legend()
        plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'{test_path}/performance_plot.png')
    plt.show()

# Usage
# plot_scores("./test_1_1")
# plot_scores("./test_1_2")
# plot_scores("./test_1_3")
# plot_scores("./test_2_1")
plot_scores("/Users/jordancapello/Documents/Year 4 Courses/Semester 2/Cisc 474/CISC-474-Group-Project/test_2_3")