using Random
using Flux
using BSON

# Define the Gridworld struct
struct Gridworld
    grid::Array{Int, 2}
    start::Tuple{Int, Int}
    goal::Tuple{Int, Int}
end

mutable struct ReplayBuffer
    buffer::Vector{Tuple{Int, Int, Float32, Int}}  # buffer tuple: state, action, reward, next_state
    capacity::Int
end

# This function below is to create the gridworld for the agent 
function create_grid_world()
    grid = [
        0 2 0 0 0;
        0 2 0 2 0;
        0 2 0 2 0;
        0 0 0 2 1                        # The Goal State = 1, Walls = 2, Traps = 3
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

function ReplayBuffer(capacity)
    return ReplayBuffer([], capacity)    # This initiallized the struct
end

function add_experience!(buffer::ReplayBuffer, experience)
    if length(buffer.buffer) >= buffer.capacity
        popfirst!(buffer.buffer)         # Remove the oldest experience
    end
    push!(buffer.buffer, experience)
end

function sample_experiences(buffer::ReplayBuffer, batch_size)
    return rand(buffer.buffer, min(batch_size, length(buffer.buffer)))
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
    if action == 1      # Move Up
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

# This function builds the Q-learning nerual network using a hyperbolic tangent activation function.
function QNN(state,actions)
    model = Chain(
            Dense(state, 64, tanh; init=Flux.glorot_uniform()), 
            Dense(64, 64, tanh; init=Flux.glorot_uniform()),
            Dense(64, actions, identity; init=Flux.glorot_uniform()))
    return(model)
end

# Q-learning function to train the agent
function q_learning(environment, num_episodes, α, γ, ϵ, ϵ_min, decay_rate)
    num_rows, num_cols = size(environment.grid)                                        # This gives the size of the grid
    num_actions = 4                                                                    # The only actions are up, down, left, and right.
    # Q-learning loop
    state = state_to_index(environment.start[1], environment.start[2], num_cols)       # Start state converted to flattened array value
    Q = QNN(1,num_actions)                                                             # might need to make state from 0 to 1
    optim = Flux.setup(Adam(α), Q)                                                     # This is the optimizer

    replay_buffer = ReplayBuffer(50000)                                                # Set buffer size
    batch_size = 64                                                                    # Number of experiences to sample

    for episode in 1:num_episodes
        ϵ = max(ϵ_min, ϵ * decay_rate)                                                 # This decays the ϵ value until it is 0.1
        state = state_to_index(environment.start[1], environment.start[2], num_cols)   # This gets the state (grid location) in index form
        done = false

        while !done
            normalized_state = Float32(2 * (state - 1) / (num_rows * num_cols - 1) - 1)# Normalizes the state for neural network

            # Epsilon-greedy action selection
            if rand() < ϵ                                                              # rand() gives random number between 0 and 1
                action = rand(1:4)                                                     # If ϵ < rand(), Explore with random action
            else
                action = argmax(Q([normalized_state]))                                 # If ϵ > rand(), Exploit with largest Q value
            end

            # Get next state and reward
            next_state = get_next_state(state, action, num_rows, num_cols)             # Gets next state
            normalized_next_state = Float32(2*(next_state-1)/(num_rows*num_cols-1)-1)  # Normalizes next state
            
            # Compute reward
            next_state_row, next_state_col = divrem(next_state - 1, num_cols)
            if environment.grid[next_state_row + 1, next_state_col + 1] == 1           # Reward for achieving goal
                reward = 0.1
            elseif environment.grid[next_state_row + 1, next_state_col + 1] == 2       # Reward for wall
                reward = -0.5
            elseif environment.grid[next_state_row + 1, next_state_col + 1] == 3       # Reward for trap
                reward = -0.5
            else                                                                       # Reward for everything else (negative to deter wandering)
                reward = -0.01
            end

            add_experience!(replay_buffer, (state, action, reward, next_state))        # Store experience in replay buffer


            # Train using a mini-batch
            if length(replay_buffer.buffer) >= batch_size
                minibatch = sample_experiences(replay_buffer, batch_size)

                for (s, a, r, s_next) in minibatch
                    norm_s = Float32(2 * (s - 1) / (num_rows * num_cols - 1) - 1)           # Normalizes state
                    norm_s_next = Float32(2 * (s_next - 1) / (num_rows * num_cols - 1) - 1) # Normalizes next state

                    y_target = r + γ * maximum(Q([norm_s_next]))                            # calculates the Q-value for Update
                    target = Q([norm_s])
                    target = copy(target)                                                   # Avoid modifying the original array directly
                    target[a] = y_target                                                    # Adds new Q-value to action list
                    Flux.train!((m, x, y) -> Flux.Losses.mse(m(x)[a], y),                   # Trains Neural network using MSE as loss function         
                        Q, [([norm_s], [Float32.(target[a])])], optim)
                end
            end

            state = next_state
            done = (state == state_to_index(environment.goal[1], environment.goal[2], num_cols)) # Calculates if the agent has found the goal
        end
        # Periodically display the progress. A Tool to check if system is converging.
        if episode % 1 == 0
            percentage = (episode / num_episodes) * 100
            print("\rTraining progress: $(round(percentage, digits=1))% ($episode/$num_episodes epochs)")
            flush(stdout)
        end
    end
    return Q
end
 
# Function to save the model
function save_model(model, filename="model.bson")
    BSON.@save filename model
    println()
    println("Model saved to $filename")
end

# Function to load the model
function load_model(filename="model.bson")
    BSON.@load filename model
    println("Model loaded from $filename")
    return model
end

# Function to construct a visual represention of agent's actions in grid at any state
function visualize_policy(Model, num_rows, num_cols, env_grid)
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
                index = state_to_index(i, j, num_cols)
                normalized_index = Float32(2 * (index - 1) / (num_rows * num_cols - 1) - 1)
                action = argmax(Model([normalized_index]))
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
num_episodes = 35000           # Number of epochs for training the nerual network
γ = 0.96                       # Discount factor (Closer to 1 future rewards become more important; closer to 0 current rewards become more important)
ϵ = 1.0                        # Initial exploration rate (Must be 1 or less than 1)
α = 1E-4                       # This is the learning rate for the neural network
ϵ_min = 0.1                    # Minimum exploration rate (Must not be 0)
decay_rate = 0.995             # Decay factor (Closer to 1 more exploration; closer to 0 more exploitation)

# Train the agent using Q-learning
println("Used save model? (y/n): ")
user_input = readline()
if user_input == "n"    
    Q = q_learning(env, num_episodes, α, γ, ϵ, ϵ_min, decay_rate)
    save_model(Q, "trained_model.bson")
elseif user_input == "y"
    Q = load_model("trained_model.bson")
else
    println("Error: Input Not Recongnized")
end

# Visualize the policy
visualize_policy(Q, size(env.grid, 1), size(env.grid, 2), env.grid)    # Prints a visualization of the actions the agent would take in anystate 

