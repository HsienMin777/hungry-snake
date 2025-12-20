# save as multi_snake_dqn.py
import pygame
import random
import numpy as np
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import math
import os

# ----------------- Hyperparams -----------------
BLOCK_SIZE = 20
SPEED = 100       # 速度設為 100 以加快訓練節奏
WINDOW_W = 640
WINDOW_H = 480

GAME_DURATION = 300  # 遊戲時間 5 分鐘 (秒)

GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 128
MEMORY_SIZE = 20000
TARGET_UPDATE_FREQ = 1000   
TRAIN_START = 500          
TRAIN_EVERY = 4
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 10000          

# 若設為 True，會嘗試讀取舊模型，適合接續訓練或測試
# 若想從頭訓練，請改為 False
LOAD_MODEL = True 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# ----------------- 環境 (含復活機制) -----------------
class MultiSnakeGameAI:
    def __init__(self, w=WINDOW_W, h=WINDOW_H, render=True):
        self.w = w
        self.h = h
        self.render = render
        if self.render:
            pygame.init()
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake DQN: Hunter S2 (Target: S1 Head)')
            self.font = pygame.font.SysFont('arial', 20)
            self.clock = pygame.time.Clock()
        self.reset_all()

    def reset_all(self):
        """重置整個遊戲局，包含分數"""
        self.score1 = 0
        self.score2 = 0
        self.respawn(1)
        self.respawn(2)
        
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.start_time = time.time()
        
        return self.get_state(1), self.get_state(2)

    def respawn(self, snake_id):
        """
        讓特定的蛇復活 (重置位置與長度)，保留分數
        """
        if snake_id == 1:
            self.direction1 = 1 # Right
            self.head1 = [self.w//4, self.h//2]
            self.snake1 = [list(self.head1),
                           [self.head1[0]-BLOCK_SIZE, self.head1[1]],
                           [self.head1[0]-2*BLOCK_SIZE, self.head1[1]]]
        else:
            self.direction2 = 0 # Left
            self.head2 = [3*self.w//4, self.h//2]
            self.snake2 = [list(self.head2),
                           [self.head2[0]+BLOCK_SIZE, self.head2[1]],
                           [self.head2[0]+2*BLOCK_SIZE, self.head2[1]]]

    def _place_food(self):
        while True:
            x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE )*BLOCK_SIZE
            y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE )*BLOCK_SIZE
            candidate = [x,y]
            if candidate not in self.snake1 and candidate not in self.snake2:
                self.food = candidate
                break

    def play_step(self, action1_idx, action2_idx):
        """
        回傳: state1, state2, reward1, reward2, dead1, dead2
        """
        self.frame_iteration += 1
        
        # 處理 pygame 事件
        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        # 1) 計算兩蛇的新方向與 new heads
        next_head1 = self._calc_next_head(action1_idx, 1)
        next_head2 = self._calc_next_head(action2_idx, 2)
        
        # === 偵測 Snake 1 是否撞上 Snake 2 的身體 (用於獎勵 S2) ===
        snake2_obstacle_body = self.snake2[:-1] if len(self.snake2) > 0 else [] 

        # 檢查 Snake 1 的下一顆頭是否撞上 Snake 2 的身體
        snake1_hit_snake2_body = next_head1 in snake2_obstacle_body
        # =========================================================================

        # 2) 檢查碰撞 (死亡判定)
        c1 = self._will_collision(next_head1, snake_id=1, next_head_other=next_head2)
        c2 = self._will_collision(next_head2, snake_id=2, next_head_other=next_head1)
        
        # 額外判斷是否為頭對頭碰撞 (用於 S2 獲勝規則)
        head_on_crash = (next_head1 == next_head2) 

        reward1 = 0.0
        reward2 = 0.0
        dead1 = False
        dead2 = False

        # 3) 處理死亡與復活
        if c1 or c2:
            if c1 and c2: # 雙死
                if head_on_crash:
                    # 🎯 S2 頭對頭獲勝邏輯
                    reward1 = -100.0  # S1 強烈懲罰
                    reward2 = 100.0    # S2 獲得小獎勵 (比起阻擋，頭對頭風險較大，給少一點)
                    self.score2 += 1  # S2 分數增加
                else:
                    # 雙方撞到身體，或撞到牆/自己導致的雙死
                    reward1 = -10.0
                    reward2 = -10.0
                
                dead1 = True
                dead2 = True
                self.respawn(1)
                self.respawn(2)
                
            elif c1: # 只有 1 死
                reward1 = -50.0
                dead1 = True
                self.respawn(1)
                
                # === 🎯 獎勵 Snake 2：如果 Snake 1 撞上 S2 身體，給予獎勵 ===
                if snake1_hit_snake2_body:
                    reward2 = 200.0   # 高獎勵，鼓勵阻擋行為
                    reward1 = -100.0  # S1 受到更大的懲罰
                    self.score2 += 1  # 阻擋成功，Snake 2 加分
                
                # =================================================================
                
            elif c2: # 只有 2 死
                reward2 = -50.0
                dead2 = True
                self.respawn(2)

        # 4) 移動處理 (如果沒有死才移動)
        if not dead1:
            self.snake1.insert(0, list(next_head1))
            self.head1 = list(next_head1)
        
        if not dead2:
            self.snake2.insert(0, list(next_head2))
            self.head2 = list(next_head2)

        # 5) 吃食物 與 身體長度處理 
        ate1 = False
        ate2 = False
        
        if not dead1 and self.head1 == self.food:
            ate1 = True
        if not dead2 and self.head2 == self.food:
            ate2 = True

        if not dead2:
            # 1. 取得目標位置 (Snake 1 的頭)
            # 注意：要用 self.head1 還是 next_head1 取決於你希望它預判還是追蹤
            # 這裡用 next_head1 代表追蹤最新位置
            target_pos = next_head1 

            # 2. 計算「移動前」與目標的距離
            dist_old = self._manhattan(self.head2, target_pos)

            # 3. 計算「移動後」與目標的距離
            dist_new = self._manhattan(next_head2, target_pos)

            

        # --- 處理 Snake 1 (正常邏輯：吃食物變長) ---
        if ate1:
            reward1 = 100.0  # S1 吃食物給高獎勵
            reward2 = -100.0 # S1 吃食物給 S2 高懲罰 (競爭目標)
            self.score1 += 1
            self._place_food()
            # Snake 1 吃到食物，不執行 pop() -> 變長
        else:
            if not dead1: 
                self.snake1.pop() # 沒吃，正常縮尾巴

        # --- 處理 Snake 2 (特殊邏輯：阻擋變長，吃食物不變長) ---
        
        
        if not dead2:
            # 判斷 S2 是否應該變長
            if snake1_hit_snake2_body:
                # [關鍵] 阻擋成功！雖然沒吃食物，但不執行 pop() -> 變長
                pass 
            else:
                # 沒阻擋成功 (即使吃到食物)，強制 pop() -> 不變長
                self.snake2.pop()

        # 畫面更新
        if self.render:
            self._update_ui()
            self.clock.tick(SPEED)

        return self.get_state(1), self.get_state(2), reward1, reward2, dead1, dead2
    
    def _calc_next_head(self, action_idx, snake_id):
        if snake_id == 1:
            curr_dir = self.direction1
            head = self.head1[:]
        else:
            curr_dir = self.direction2
            head = self.head2[:]

        clock_wise = [1, 3, 0, 2] # R, D, L, U
        idx = clock_wise.index(curr_dir)
        
        if action_idx == 0:   # straight
            new_dir = clock_wise[idx]
        elif action_idx == 1: # right turn
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:                 # left turn
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        if snake_id == 1: self.direction1 = new_dir
        else: self.direction2 = new_dir

        x, y = head[0], head[1]
        if new_dir == 1: x += BLOCK_SIZE
        elif new_dir == 0: x -= BLOCK_SIZE
        elif new_dir == 3: y += BLOCK_SIZE
        elif new_dir == 2: y -= BLOCK_SIZE

        return [x, y]

    def _will_collision(self, next_head, snake_id, next_head_other=None):
        # 1. 撞牆
        if next_head[0] > self.w - BLOCK_SIZE or next_head[0] < 0 or next_head[1] > self.h - BLOCK_SIZE or next_head[1] < 0:
            return True

        # 2. 準備身體障礙物
        if snake_id == 1:
            self_body = self.snake1[:-1] if len(self.snake1) > 0 else []
            other_body = self.snake2[:-1] if len(self.snake2) > 0 else []
        else:
            self_body = self.snake2[:-1] if len(self.snake2) > 0 else []
            other_body = self.snake1[:-1] if len(self.snake1) > 0 else []

        obstacles = self_body + other_body

        # 3. 撞到頭 (Head-on)
        if next_head_other is not None and next_head == next_head_other:
            return True

        if next_head in obstacles:
            return True
            
        return False

    def get_state(self, snake_id):
        if snake_id == 1:
            head = self.snake1[0]
            direction = self.direction1
        else:
            head = self.snake2[0]
            direction = self.direction2

        point_l = [head[0] - BLOCK_SIZE, head[1]]
        point_r = [head[0] + BLOCK_SIZE, head[1]]
        point_u = [head[0], head[1] - BLOCK_SIZE]
        point_d = [head[0], head[1] + BLOCK_SIZE]

        dir_l = (direction == 0)
        dir_r = (direction == 1)
        dir_u = (direction == 2)
        dir_d = (direction == 3)

        s1_dir_l = (self.direction1 == 0)
        s1_dir_r = (self.direction1 == 1)
        s1_dir_u = (self.direction1 == 2)
        s1_dir_d = (self.direction1 == 3)

        # ------------------- 核心感知邏輯 (修改後) -------------------
        if snake_id == 1:
            # Snake 1: 目標是食物
            target_flags = [
                self.food[0] < head[0], # Food Left
                self.food[0] > head[0], # Food Right
                self.food[1] < head[1], # Food Up
                self.food[1] > head[1]  # Food Down
            ]
        else:
            # Snake 2: 目標是 Snake 1 的頭 (追殺)
            s1_head = self.snake1[0] 
            target_flags = [
                s1_head[0] < head[0], # S1 Head Left
                s1_head[0] > head[0], # S1 Head Right
                s1_head[1] < head[1], # S1 Head Up
                s1_head[1] > head[1]  # S1 Head Down
            ]
        # -----------------------------------------------------------

        state = [
            # 1. Danger Straight
            (dir_r and self._will_collision(point_r, snake_id)) or 
            (dir_l and self._will_collision(point_l, snake_id)) or 
            (dir_u and self._will_collision(point_u, snake_id)) or 
            (dir_d and self._will_collision(point_d, snake_id)),

            # 2. Danger Right
            (dir_u and self._will_collision(point_r, snake_id)) or 
            (dir_d and self._will_collision(point_l, snake_id)) or 
            (dir_l and self._will_collision(point_u, snake_id)) or 
            (dir_r and self._will_collision(point_d, snake_id)),

            # 3. Danger Left
            (dir_d and self._will_collision(point_r, snake_id)) or 
            (dir_u and self._will_collision(point_l, snake_id)) or 
            (dir_r and self._will_collision(point_u, snake_id)) or 
            (dir_l and self._will_collision(point_d, snake_id)),

            # 4-7. Current Direction
            dir_l, dir_r, dir_u, dir_d,

            # 8-11. Target Direction (Food for S1, Enemy Head for S2)
            target_flags[0], target_flags[1], target_flags[2], target_flags[3],

            s1_dir_l, s1_dir_r, s1_dir_u, s1_dir_d
        ]
        return np.array(state, dtype=np.int32)

    def _manhattan(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def _update_ui(self):
        self.display.fill((0,0,0))
        
        # Draw Snake 1 (Blue)
        for pt in self.snake1:
            pygame.draw.rect(self.display, (0,0,255), pygame.Rect(pt[0], pt[1], BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, (0,100,255), pygame.Rect(pt[0]+4, pt[1]+4, 12, 12))
        
        # Draw Snake 2 (Green)
        for pt in self.snake2:
            pygame.draw.rect(self.display, (0,255,0), pygame.Rect(pt[0], pt[1], BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, (0,200,0), pygame.Rect(pt[0]+4, pt[1]+4, 12, 12))
            
        # Draw Food (Red)
        pygame.draw.rect(self.display, (200,0,0), pygame.Rect(self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE))
        
        # UI Text
        time_left = max(0, int(GAME_DURATION - (time.time() - self.start_time)))
        text = self.font.render(f"Time: {time_left}s | S1(Food): {self.score1} | S2(Blocks): {self.score2}", True, (255, 255, 255))
        self.display.blit(text, [0, 0])
        
        pygame.display.flip()

# ----------------- DQN 模型 / Memory -----------------
class DQNNet(nn.Module):
    def __init__(self, input_dim=15, output_dim=3):
        super(DQNNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity=MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

# ----------------- 獨立 Agent 類別 -----------------
class Agent:
    def __init__(self, input_dim=15):
        self.policy_net = DQNNet(input_dim).to(DEVICE)
        self.target_net = DQNNet(input_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayBuffer()
        
    def select_action(self, state, eps):
        if random.random() < eps:
            return random.randrange(3)
        else:
            state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                q = self.policy_net(state_t)
                return q.max(1)[1].item()

    def optimize(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*transitions)

        # 轉換為 Tensor
        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32, device=DEVICE)
        action_batch = torch.tensor(batch.action, dtype=torch.int64, device=DEVICE).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=DEVICE)
        done_batch = torch.tensor(batch.done, dtype=torch.float32, device=DEVICE).unsqueeze(1)

        # Q(s, a)
        q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Target Q = r + gamma * max Q(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
            expected_q_values = reward_batch + (1 - done_batch) * GAMMA * next_q_values

        loss = F.mse_loss(q_values, expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping 避免梯度爆炸
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

# ----------------- 訓練主程式 (時間制) -----------------
def train_timed_dual_dqn():
    env = MultiSnakeGameAI(render=True)
    
    # 建立兩個獨立的 Agent
    agent1 = Agent()
    agent2 = Agent()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. 將資料夾路徑與檔名結合
    # 這會變成 D:\文件\桌面\Coding\side_proj\hungry_snake\App\checkpoint.pth
    checkpoint_path1 = os.path.join(script_dir, "agent1_dqn.pth")
    checkpoint_path2 = os.path.join(script_dir, "agent2_dqn.pth")
    

    # 嘗試載入先前的模型 (可選)
    if os.path.exists(checkpoint_path1):
        try:
            agent1.policy_net.load_state_dict(torch.load(checkpoint_path1, map_location=DEVICE))
            agent1.target_net.load_state_dict(agent1.policy_net.state_dict())
            print(f"Loaded Agent 1 model from: {checkpoint_path1}")
        except: pass
    if os.path.exists(checkpoint_path2):
        try:
            agent2.policy_net.load_state_dict(torch.load(checkpoint_path2, map_location=DEVICE))
            agent2.target_net.load_state_dict(agent2.policy_net.state_dict())
            print(f"Loaded Agent 2 model from: {checkpoint_path2}")
        except: pass
    
    total_steps = 0
    start_time = time.time()
    
    print("Game Started! Duration: 5 Minutes...")

    # 取得初始狀態
    state1, state2 = env.reset_all()

    # 根據開關決定 Epsilon 起始值
    if LOAD_MODEL:
        current_eps = 0.05  # 直接讓它變聰明，只保留 5% 隨機性來微調
        print("Mode: Continued Training / Testing (Smart Mode)")
    else:
        current_eps = 0.5   # 或是 1.0，從頭學
        print("Mode: New Training (Exploration Mode)")

    while True:
        # 計算經過時間
        elapsed_time = time.time() - start_time
        if elapsed_time > GAME_DURATION:
            print("Time's up! Game Over.")
            break
            
        # 計算 Epsilon decay (隨時間減少)
        # 假設前 4 分鐘 epsilon 從 1.0 降到 0.05
        if not LOAD_MODEL:
            progress = min(1.0, elapsed_time / (GAME_DURATION * 0.8))
            current_eps = EPS_START - (EPS_START - EPS_END) * progress
        else:
            current_eps = 0.05 # 鎖定低隨機率

        # 1. 選擇動作
        action1 = agent1.select_action(state1, current_eps)
        action2 = agent2.select_action(state2, current_eps)

        # 2. 執行一步
        next_state1, next_state2, reward1, reward2, dead1, dead2 = env.play_step(action1, action2)

        # 3. 儲存記憶 (個別儲存)
        # 注意: 如果 dead=True，這一步對該 Agent 來說是 Done
        agent1.memory.push(state1, action1, reward1, next_state1, float(dead1))
        agent2.memory.push(state2, action2, reward2, next_state2, float(dead2))

        # 4. 更新狀態
        state1 = next_state1
        state2 = next_state2

        # 5. 訓練模型
        total_steps += 1
        if total_steps > TRAIN_START and total_steps % TRAIN_EVERY == 0:
            agent1.optimize()
            agent2.optimize()

        # 6. 更新 Target Net
        if total_steps % TARGET_UPDATE_FREQ == 0:
            agent1.target_net.load_state_dict(agent1.policy_net.state_dict())
            agent2.target_net.load_state_dict(agent2.policy_net.state_dict())

        # Debug 資訊 (每 1000 步顯示一次)
        if total_steps % 1000 == 0:
            print(f"Time: {int(elapsed_time)}s | Steps: {total_steps} | Score: {env.score1}-{env.score2} | Eps: {current_eps:.2f}")

    # 結束後儲存
    torch.save(agent1.policy_net.state_dict(), "agent1_dqn.pth")
    torch.save(agent2.policy_net.state_dict(), "agent2_dqn.pth")
    print("Models saved.")
    pygame.quit()

if __name__ == "__main__":
    train_timed_dual_dqn()