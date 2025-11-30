import pygame
import random
import numpy as np
from collections import deque

# --- 設定參數 ---
BLOCK_SIZE = 20
SPEED = 50  # 訓練時可以調高這個數字 (例如 200) 加快速度
# 顏色定義
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

# --- 遊戲環境 (Environment) ---
class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # 初始化 Pygame
        pygame.init()
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake Q-Learning')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # 初始化遊戲狀態
        self.direction = 1  # 0: Left, 1: Right, 2: Up, 3: Down
        
        self.head = [self.w/2, self.h/2]
        self.snake = [self.head, 
                      [self.head[0]-BLOCK_SIZE, self.head[1]],
                      [self.head[0]-(2*BLOCK_SIZE), self.head[1]]]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        return self.get_state()

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = [x, y]
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. 處理使用者輸入 (允許隨時退出)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. 移動
        self._move(action)
        self.snake.insert(0, list(self.head))
        
        # 3. 檢查遊戲結束
        reward = 0
        game_over = False
        
        # 如果撞牆、撞自己、或者在原地繞太久(防呆)
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10  # 死亡懲罰
            return reward, game_over, self.score

        # 4. 放置新食物或移動尾巴
        if self.head == self.food:
            self.score += 1
            reward = 10   # 吃到蘋果獎勵
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. 更新 UI
        self._update_ui()
        self.clock.tick(SPEED)
        
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # 撞牆
        if pt[0] > self.w - BLOCK_SIZE or pt[0] < 0 or pt[1] > self.h - BLOCK_SIZE or pt[1] < 0:
            return True
        # 撞身體
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt[0], pt[1], BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt[0]+4, pt[1]+4, 12, 12))
        
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE))
        
        text = pygame.font.SysFont('arial', 25).render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # action 是相對動作: [直走, 右轉, 左轉]
        # clock_wise = [Right, Down, Left, Up]
        clock_wise = [1, 3, 0, 2] 
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # 直走
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # 右轉
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # 左轉

        self.direction = new_dir

        x = self.head[0]
        y = self.head[1]
        if self.direction == 1: # Right
            x += BLOCK_SIZE
        elif self.direction == 0: # Left
            x -= BLOCK_SIZE
        elif self.direction == 3: # Down
            y += BLOCK_SIZE
        elif self.direction == 2: # Up
            y -= BLOCK_SIZE
            
        self.head = [x, y]
    
    # --- 獲取狀態 (State) ---
    # 這是 AI 的眼睛，將複雜畫面簡化成 11 個布林值
    def get_state(self):
        head = self.snake[0]
        
        # 建立四個方向的測試點
        point_l = [head[0] - BLOCK_SIZE, head[1]]
        point_r = [head[0] + BLOCK_SIZE, head[1]]
        point_u = [head[0], head[1] - BLOCK_SIZE]
        point_d = [head[0], head[1] + BLOCK_SIZE]
        
        # 當前方向
        dir_l = self.direction == 0
        dir_r = self.direction == 1
        dir_u = self.direction == 2
        dir_d = self.direction == 3

        state = [
            # 1. 前方有危險
            (dir_r and self.is_collision(point_r)) or 
            (dir_l and self.is_collision(point_l)) or 
            (dir_u and self.is_collision(point_u)) or 
            (dir_d and self.is_collision(point_d)),

            # 2. 右邊有危險
            (dir_u and self.is_collision(point_r)) or 
            (dir_d and self.is_collision(point_l)) or 
            (dir_l and self.is_collision(point_u)) or 
            (dir_r and self.is_collision(point_d)),

            # 3. 左邊有危險
            (dir_d and self.is_collision(point_r)) or 
            (dir_u and self.is_collision(point_l)) or 
            (dir_r and self.is_collision(point_u)) or 
            (dir_l and self.is_collision(point_d)),
            
            # 4. 移動方向
            dir_l, dir_r, dir_u, dir_d,
            
            # 5. 食物位置
            self.food[0] < head[0],  # 食物在左
            self.food[0] > head[0],  # 食物在右
            self.food[1] < head[1],  # 食物在上
            self.food[1] > head[1]   # 食物在下
        ]
        
        # 將布林轉為 0 或 1
        return np.array(state, dtype=int)

# --- Q-Learning Agent ---
class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # 隨機性 (Exploration rate)
        self.gamma = 0.9 # 折扣因子 (Discount rate) - 重視未來的程度
        self.learning_rate = 0.1 # 學習率
        self.q_table = {} # 用字典當作 Q-Table (State -> Actions)

    def get_state_key(self, state):
        return str(state)

    def get_action(self, state):
        # 隨機探索 (Epsilon-Greedy)
        self.epsilon = 80 - self.n_games # 隨著遊戲次數增加，減少隨機亂走
        final_move = [0,0,0]
        
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state_key = self.get_state_key(state)
            if state_key not in self.q_table:
                self.q_table[state_key] = [0, 0, 0]
            
            prediction = self.q_table[state_key]
            move = np.argmax(prediction)
            final_move[move] = 1
            
        return final_move

    def train(self, state, action, reward, next_state, done):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = [0, 0, 0]
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = [0, 0, 0]

        # Q-Learning 核心公式 (Bellman Equation)
        # Q_new = Q + lr * (Reward + gamma * max(Q_next) - Q)
        
        action_idx = np.argmax(action)
        old_value = self.q_table[state_key][action_idx]
        
        next_max = np.max(self.q_table[next_state_key])
        
        # 如果遊戲結束，未來獎勵為 0
        if done:
            target = reward
        else:
            target = reward + self.gamma * next_max
            
        new_value = old_value + self.learning_rate * (target - old_value)
        
        # 更新表
        self.q_table[state_key][action_idx] = new_value

# --- 主程式 ---
def train():
    agent = Agent()
    game = SnakeGameAI()
    
    # 紀錄最高分
    record = 0
    
    while True:
        # 1. 獲取舊狀態
        state_old = game.get_state()
        
        # 2. 決定動作
        final_move = agent.get_action(state_old)
        
        # 3. 執行動作並獲取結果
        reward, done, score = game.play_step(final_move)
        state_new = game.get_state()
        
        # 4. 訓練 (更新 Q-Table)
        agent.train(state_old, final_move, reward, state_new, done)
        
        if done:
            game.reset()
            agent.n_games += 1
            
            if score > record:
                record = score
                # 可以把模型存起來
                # agent.model.save() 
            
            print(f'Game: {agent.n_games}, Score: {score}, Record: {record}')

if __name__ == '__main__':
    train()