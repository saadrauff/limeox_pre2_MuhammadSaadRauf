
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path

# ==========================================
# TASK 14: REINFORCEMENT LEARNING
# Algorithm: Q-Learning
# ==========================================

# Create output directory
OUTPUT_DIR = Path(__file__).parent / "model_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Environment settings
GRID_SIZE = 5

START_STATE = (0, 0)
GOAL_STATE = (4, 4)

OBSTACLES = [
    (1, 1),
    (2, 1),
    (3, 3)
]

# Available actions
ACTIONS = {
    0: (-1, 0),  # Up
    1: (1, 0),   # Down
    2: (0, -1),  # Left
    3: (0, 1)    # Right
}

ACTION_NAMES = {
    0: "Up",
    1: "Down",
    2: "Left",
    3: "Right"
}

# Q-Learning parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95

EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01

EPISODES = 2000
MAX_STEPS = 100

# Initialize Q-table
Q_TABLE = np.zeros(
    (GRID_SIZE * GRID_SIZE, len(ACTIONS))
)


# ==========================================
# ENVIRONMENT FUNCTIONS
# ==========================================

def state_to_index(state):

    return state[0] * GRID_SIZE + state[1]


def take_action(state, action):

    row, col = state

    move_row, move_col = ACTIONS[action]

    new_row = row + move_row
    new_col = col + move_col

    # Check grid boundaries
    if not (
        0 <= new_row < GRID_SIZE
        and 0 <= new_col < GRID_SIZE
    ):

        return state, -10, False

    new_state = (new_row, new_col)

    # Check obstacles
    if new_state in OBSTACLES:

        return state, -10, False

    # Check goal
    if new_state == GOAL_STATE:

        return new_state, 100, True

    # Normal movement
    return new_state, -1, False


# ==========================================
# Q-LEARNING TRAINING
# ==========================================

def train_agent():

    global EPSILON

    rewards_history = []

    for episode in range(EPISODES):

        state = START_STATE

        total_reward = 0

        for step in range(MAX_STEPS):

            state_index = state_to_index(state)

            # Epsilon-greedy action selection
            if np.random.random() < EPSILON:

                # Explore
                action = np.random.randint(4)

            else:

                # Exploit learned knowledge
                action = np.argmax(
                    Q_TABLE[state_index]
                )

            # Perform action
            next_state, reward, done = take_action(
                state, action
            )

            next_state_index = state_to_index(
                next_state
            )

            # Q-Learning update formula
            old_value = Q_TABLE[
                state_index, action
            ]

            next_max = np.max(
                Q_TABLE[next_state_index]
            )

            new_value = (
                old_value
                + LEARNING_RATE
                * (
                    reward
                    + DISCOUNT_FACTOR * next_max
                    - old_value
                )
            )

            Q_TABLE[
                state_index, action
            ] = new_value

            state = next_state

            total_reward += reward

            if done:

                break

        # Reduce exploration over time
        EPSILON = max(
            MIN_EPSILON,
            EPSILON * EPSILON_DECAY
        )

        rewards_history.append(total_reward)

        if (episode + 1) % 200 == 0:

            print(
                f"Episode {episode + 1}/{EPISODES} "
                f"| Reward: {total_reward} "
                f"| Epsilon: {EPSILON:.4f}"
            )

    return rewards_history


# ==========================================
# SAVE Q-TABLE
# ==========================================

def save_q_table():

    columns = list(ACTION_NAMES.values())

    with open(
        OUTPUT_DIR / "q_table.csv",
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            ["State"] + columns
        )

        for i, values in enumerate(Q_TABLE):

            state = (
                f"({i // GRID_SIZE}, "
                f"{i % GRID_SIZE})"
            )

            writer.writerow(
                [state] + list(values)
            )

    print("\nQ-Table saved successfully!")


# ==========================================
# SAVE TRAINING REWARDS AND GRAPH
# ==========================================

def save_rewards(rewards):

    with open(
        OUTPUT_DIR / "training_rewards.csv",
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            ["Episode", "Total_Reward"]
        )

        for episode, reward in enumerate(
            rewards, start=1
        ):

            writer.writerow(
                [episode, reward]
            )

    # Plot training performance
    plt.figure(figsize=(10, 5))

    plt.plot(
        rewards,
        alpha=0.4,
        label="Episode Rewards"
    )

    # Calculate 100-episode moving average
    window = 100

    if len(rewards) >= window:

        moving_avg = np.convolve(
            rewards,
            np.ones(window) / window,
            mode="valid"
        )

        plt.plot(
            range(window - 1, len(rewards)),
            moving_avg,
            linewidth=2,
            label="100 Episode Moving Average"
        )

    plt.xlabel("Episodes")

    plt.ylabel("Total Reward")

    plt.title(
        "Q-Learning Training Performance"
    )

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "training_rewards.png"
    )

    plt.close()

    print("Training rewards and graph saved!")


# ==========================================
# TEST TRAINED AGENT
# ==========================================

def test_agent():

    state = START_STATE

    path = [state]

    total_reward = 0

    for step in range(MAX_STEPS):

        state_index = state_to_index(state)

        # Select best learned action
        action = np.argmax(
            Q_TABLE[state_index]
        )

        next_state, reward, done = take_action(
            state, action
        )

        total_reward += reward

        state = next_state

        path.append(state)

        if done:

            break

    success = state == GOAL_STATE

    print("\n========== TEST RESULTS ==========")

    print("Start:", START_STATE)

    print("Goal:", GOAL_STATE)

    print("\nLearned Path:")

    for position in path:

        print(position)

    print("\nTotal Steps:", len(path) - 1)

    print("Total Reward:", total_reward)

    if success:

        print("Result: SUCCESS - Goal Reached!")

    else:

        print("Result: Goal Not Reached")

    # Save learned path
    with open(
        OUTPUT_DIR / "learned_path.txt",
        "w"
    ) as file:

        file.write("Q-Learning Agent Path\n")

        file.write(f"Start: {START_STATE}\n")

        file.write(f"Goal: {GOAL_STATE}\n")

        file.write(f"Path: {path}\n")

        file.write(
            f"Steps: {len(path) - 1}\n"
        )

        file.write(
            f"Reward: {total_reward}\n"
        )

        file.write(
            f"Success: {success}\n"
        )

    print("\nLearned path saved!")


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    print("=" * 45)

    print("REINFORCEMENT LEARNING - Q-LEARNING")

    print("=" * 45)

    print("\nStarting Q-Learning Training...\n")

    rewards = train_agent()

    print("\nTraining Completed!")

    save_q_table()

    save_rewards(rewards)

    test_agent()

    print("\nAll results saved in model_results!")

    print("\nTask 14 Completed!")
