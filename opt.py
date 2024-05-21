import numpy as np
import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_grid(obstacles, grid_size):
    grid = np.ones(grid_size, dtype=bool)  # True means free
    for (x, ymin, ymax) in obstacles:
        grid[x, ymin:ymax+1] = False  # False means blocked
    return grid

def heuristic(a, b):
    return abs(b[0] - a[0]) + abs(b[1] - a[1])

def astar(grid, start, goal):
    if not grid[start[0], start[1]] or not grid[goal[0], goal[1]]:
        return None  # Start or goal is blocked

    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []

    heapq.heappush(oheap, (fscore[start], start))
    
    while oheap:
        current = heapq.heappop(oheap)[1]
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return len(path) - 1, path  # Return path length
        
        close_set.add(current)
        for i, j in neighbors:
            neighbor = (current[0] + i, current[1] + j)
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1] and grid[neighbor]:
                tentative_g_score = gscore[current] + 1
                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                    continue
                if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [h[1] for h in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
    
    return None

def solve_tsp_with_paths(distances, points):
    n = len(distances)
    dp = [[(float('inf'), [])] * n for _ in range(1 << n)]
    dp[1][0] = (0, [points[0]])  # Start from the first node

    for mask in range(1 << n):
        for u in range(n):
            if mask & (1 << u):
                for v in range(n):
                    if not mask & (1 << v) and distances[u][v]:
                        new_mask = mask | (1 << v)
                        new_cost = dp[mask][u][0] + distances[u][v][0]
                        new_path = dp[mask][u][1] + [points[v]]
                        if new_cost < dp[new_mask][v][0]:
                            dp[new_mask][v] = (new_cost, new_path)

    optimal = min((dp[(1 << n) - 1][i][0] + (distances[i][0][0] if distances[i][0] else float('inf')), dp[(1 << n) - 1][i][1]) for i in range(n))
    return optimal

def visualize_path(grid, path, obstacles):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid.T, cmap="gray_r", origin="lower")  # Transpose grid to align x-y with matplot

    # Draw obstacles correctly based on grid indexing
    for x, ymin, ymax in obstacles:
        ax.add_patch(patches.Rectangle((ymin, x), ymax-ymin+1, 1, linewidth=1, edgecolor='red', facecolor='red'))

    # Plot the path
    y, x = zip(*path)  # Transpose path coordinates for plotting
    ax.plot(x, y, 'go-', linewidth=2, markersize=12, label='Path')
    ax.scatter(x, y, color='red', s=100, label='Points')

    ax.set_xlim(-0.5, grid.shape[1]-0.5)
    ax.set_ylim(-0.5, grid.shape[0]-0.5)
    ax.set_xticks(np.arange(0, grid.shape[1], 1))
    ax.set_yticks(np.arange(0, grid.shape[0], 1))
    ax.grid(True)
    plt.legend()
    plt.show()

grid_size = (10, 10)
obstacles = [(2, 1, 3), (5, 1, 3)]  # Correctly defined obstacles
grid = create_grid(obstacles, grid_size)
points = [(0, 0), (7, 8), (8, 9), (1, 2)]  # Points including start

# Validate points are not in obstacles
for point in points:
    if not grid[point[0], point[1]]:
        print(f"Point {point} is inside an obstacle or out of bounds.")
        exit(1)

# Calculate distance matrix using A*
dist_matrix = [[None if i == j else astar(grid, points[i], points[j]) for j in range(len(points))] for i in range(len(points))]

# Check all paths calculated
for i in range(len(dist_matrix)):
    for j in range(len(dist_matrix[i])):
        if i != j and dist_matrix[i][j] is None:
            print(f"Error: No path between {points[i]} and {points[j]}")
            exit(1)

# Solve TSP
min_path_cost, path = solve_tsp_with_paths(dist_matrix, points)
print("Minimum path cost to visit all points and return to start:", min_path_cost)
print("Path:", path)

# Visualize the path
visualize_path(grid, path, obstacles)
