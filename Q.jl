using Random
using PrettyTables

# Define the Gridworld struct
struct Gridworld
    grid::Array{Int, 2}
    start::Tuple{Int, Int}
    goal::Tuple{Int, Int}
end

# This function below is to create the gridworld for the agent 
function create_grid_world()
    grid = [
        0 2 0 0 0 0;
        0 2 0 3 2 0;
        0 2 0 0 2 0;
        0 2 3 0 2 0;
        0 2 0 0 2 0;
        0 0 0 3 2 1  # The Goal State is represented by 1
    ]
    start = (1, 1)                       # Starting Position
    row,col = size(grid)
    goal = (0,0)
    for i in 1:row
        for j in 1:col
            if grid[i,j] == 1
                goal = (i,j)             # This calculates the goal index so only the grid needs to be changed
            end
        end 
    end
    return Gridworld(grid, start, goal)  # Ensure you return the correct type
end

# This function converts (row, column) to state index
function state_to_index(row, col, num_cols)
    return (row - 1) * num_cols + col # This creates a flatenned array where each index coresponds to an corrdinate on the grid
end

# This function gives the next state based on current state and action
function get_next_state(state, action, num_rows, num_cols)
    row, col = divrem(state - 1, num_cols)  # Convert state index to row, col where the quotient is the Row index and the remainder is the column index. Subtract 1 for base 0 indexing.
# The next state is then found
    row += 1
    col += 1
# The next random action is assesed and implemented
    if action == 1  # Move Up
        row -= 1
    elseif action == 2  # Move Down
        row += 1
    elseif action == 3  # Move Left
        col -= 1
    elseif action == 4  # Move Right
        col += 1
    end

    # Ensure it's within bounds
    # Clamp ensures that if row or column is less than 1 it becomes 1 and if it is greater than the maximum it is the maximum
    row = clamp(row, 1, num_rows)
    col = clamp(col, 1, num_cols)

    return state_to_index(row, col, num_cols)
end

# Q-learning function to train the agent
function q_learning(environment, num_episodes, α, γ, ϵ, ϵ_min, decay_rate)
    num_rows, num_cols = size(environment.grid)                                       # This gives the size of the grid
    num_actions = 4                                                                   # The only actions are up, down, left, and right.
    Q = zeros(Float64, num_rows * num_cols, num_actions)                              # Initialize Q-table

    # Q-learning loop
    for episode in 1:num_episodes                                                     # The more episodes there are the more accurate the policy, but the tradeoff is efficency
        ϵ = max(ϵ_min, ϵ * decay_rate)                                                # Decay epsilon but ensure it doesn't go below 0.1
        state = state_to_index(environment.start[1], environment.start[2], num_cols)  # Start state converted to flattened array
        done = false
        while done == false
            # Epsilon-greedy action selection
            if rand() < ϵ                     # rand() generates a random number between 0 and 1
                action = rand(1:4)            # If ϵ is less than the random number explore randomly
            else
                action = argmax(Q[state, :])  # Exploit (select action with highest Q-value)
            end

            # Get next state and reward
            next_state = get_next_state(state, action, num_rows, num_cols)
            # The code below gives the reward\
            next_state_row, next_state_cols = divrem(next_state - 1, num_cols)  # Coordinates are derived from the flattened array.
            if (environment.grid[next_state_row + 1, next_state_cols + 1] == 1) # reward for goal. +1 for base 1 indexing.
                reward = 100 # Large reward for finishing the maze
            elseif (environment.grid[next_state_row + 1, next_state_cols + 1] == 2) # reward for walls.
                reward = -75
            elseif (environment.grid[next_state_row + 1, next_state_cols + 1] == 3) # reward for trap.
                reward = -100
            else
                reward = -1  # Reward for everything else
            end
            # Q-value update rule
            Q[state, action] += α * (reward + γ * maximum(Q[next_state, :]) - Q[state, action])   # This updates the Q table by using the general equation for Q-learning
            state = next_state                                                                    # Update the state to the next state
            done = (state == state_to_index(environment.goal[1], environment.goal[2], num_cols))  # Check if goal is reached
        end

        # Periodically display the progress (every 100 episodes). A Tool to check if system is converging.
        if episode % 100 == 0
            percentage = (episode / num_episodes) * 100
            print("\rTraining progress: $(round(percentage, digits=1))% ($episode/$num_episodes episodes)")
            flush(stdout)
        end
    end
    println()  # Move to the next line after progress display
    return Q
end

# Extract the optimal policy from the Q-values
function extract_policy(Q, num_rows, num_cols)
    policy = zeros(num_rows,num_cols)
    for i in 1:num_rows
        for j in 1:num_cols
            state = (i - 1) * num_cols + j
            best_action = argmax(Q[state, :]) # Best action is determined by the largest Q-value
            policy[i,j] = best_action
        end
    end
    return policy
end 

function visualize_policy(policy, num_rows, num_cols, env_grid)
    grid = []
    for i in 1:num_rows
        row = []
        for j in 1:num_cols
            if env_grid[i,j] == 1
                push!(row, "X")  # Mark the goal with 'X'
            elseif env_grid[i,j] == 2
                push!(row, "W")  # Mark the walls with 'W'
            elseif env_grid[i,j] == 3
                push!(row, "T")  # Mark the traps with 'T'
            else
                action = policy[i,j]
                if action == 1
                    push!(row, "↑")
                elseif action == 2
                    push!(row, "↓")
                elseif action == 3
                    push!(row, "←")
                else
                    push!(row, "→")
                end
            end
        end
        push!(grid, row)
    end
      for row in grid
      println(join(row, " "))
      end
end
# Creating the environment
env = create_grid_world()      # Creates the environment
goal_row, goal_col = env.goal  # Extracts the goal row and column

# Set Q-learning parameters
num_episodes = 10000           # Number of episodes
α = 0.01                       # Learning rate   (Small learning rate means the agent learns slower but more effectivly; Large learning rate means the agent learns quickly but less effectivly)
γ = 0.95                       # Discount factor (Closer to 1 future rewards become more important; closer to 0 current rewards become more important)
ϵ = 1.0                        # Initial exploration rate (Must be 1 or less than 1)
ϵ_min = 0.1                    # Minimum exploration rate (Must not be 0)
decay_rate = 0.999             # Decay factor (Closer to 1 more exploration; closer to 0 more exploitation)

# Train the agent using Q-learning
Q = q_learning(env, num_episodes, α, γ, ϵ, ϵ_min, decay_rate)

pretty_table(Q, header=["Move Up", "Move Down", "Move Left", "Move Right"]) # Prints out the Q-table from (1,1) to (x,x)
policy = extract_policy(Q, size(env.grid, 1), size(env.grid, 2))            # Gets the policy from the Q-table
println("Optimal Policy: ")                                                 
pretty_table(policy)                                                        # Prints the policy
# Visualize the policy
visualize_policy(policy, size(env.grid, 1), size(env.grid, 2), env.grid)    # Prints a visualization of the actions the agent would take in anystate 
