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

# ----------------- Hyperparams -----------------
BLOCK_SIZE = 20
SPEED = 150        # 畫面更新速度 (訓練時可設高)
WINDOW_W = 640
WINDOW_H = 480

GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 128
MEMORY_SIZE = 20000
TARGET_UPDATE_FREQ = 1000   # steps
TRAIN_START = 500          # 收集多少 transitions 才開始 train
TRAIN_EVERY = 4
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 20000          # 減少 epsilon 的步數

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# ----------------- 環境 (含若干修正) -----------------
class MultiSnakeGameAI:
    def __init__(self, w=WINDOW_W, h=WINDOW_H, render=True):
        self.w = w
        self.h = h
        self.render = render
        if self.render:
            pygame.init()
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake DQN: Shared Network (1 vs 1)')
            self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # 方向：0=Left,1=Right,2=Up,3=Down (與你原本順序略調整)
        self.direction1 = 1  # Snake1 向右
        self.direction2 = 0  # Snake2 向左

        # 頭位置
        self.head1 = [self.w//4, self.h//2]
        self.head2 = [3*self.w//4, self.h//2]

        # 建構身體（list of [x,y]）
        self.snake1 = [list(self.head1),
                       [self.head1[0]-BLOCK_SIZE, self.head1[1]],
                       [self.head1[0]-2*BLOCK_SIZE, self.head1[1]]]
        self.snake2 = [list(self.head2),
                       [self.head2[0]+BLOCK_SIZE, self.head2[1]],
                       [self.head2[0]+2*BLOCK_SIZE, self.head2[1]]]

        self.score1 = 0
        self.score2 = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

        # 回傳兩者初始 state
        return self.get_state(1), self.get_state(2)

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
        action indices: 0=straight,1=right,2=left (relative)
        回傳: reward1, reward2, done, score1, score2, state1, state2
        """
        self.frame_iteration += 1
        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        # 1) 計算兩蛇的新方向與 new heads（但還沒插入到 body）
        next_head1 = self._calc_next_head(action1_idx, 1)
        next_head2 = self._calc_next_head(action2_idx, 2)

        # 2) 檢查碰撞（根據 next_head 對同時移動做正確判定）
        c1 = self._will_collision(next_head1, snake_id=1, next_head_other=next_head2)
        c2 = self._will_collision(next_head2, snake_id=2, next_head_other=next_head1)

        # 若兩頭同時到同一格 → 視為 head-on collision (雙方死亡) or 視為平手吃到食物?
        head_on = (next_head1 == next_head2)

        reward1 = 0.0
        reward2 = 0.0
        done = False

        # 3) 如果有碰撞或 frame iteration 太久，結束
                # 3) 如果有碰撞或 frame iteration 太久，結束
        max_iter = 100 * min(len(self.snake1), len(self.snake2))
        if c1 or c2 or self.frame_iteration > max_iter:
            done = True
            if c1: 
                reward1 = -10.0
            if c2: reward2 = -10.0
            # 若 head_on 造成雙方同時撞，兩邊都 -10
            return reward1, reward2, done, self.score1, self.score2, self.get_state(1), self.get_state(2)
        if c1 or c2 or self.frame_iteration > max_iter:
            done = True
            # 預設都先為 0（接下來依情況設定 +/-）
            reward1 = 0.0
            reward2 = 0.0

            # 兩方同時撞（head-on 或兩方同時撞到別處）→ 雙方都懲罰 -10（不給 +10）
            if c1 and c2:
                reward1 = -10.0
                reward2 = -10.0
            else:
                # 若只有 snake1 發生碰撞
                if c1:
                    reward1 = -10.0
                    # 若 snake1 的 next_head 撞到的是 snake2 的身體（排除 snake2 的 head），給 snake2 +10
                    if next_head1 in self.snake2[1:]:
                        reward2 = 10.0
                # 若只有 snake2 發生碰撞
                if c2:
                    reward2 = -10.0
                    # 若 snake2 的 next_head 撞到的是 snake1 的身體（排除 snake1 的 head），給 snake1 +10
                    if next_head2 in self.snake1[1:]:
                        reward1 = 10.0

            return reward1, reward2, done, self.score1, self.score2, self.get_state(1), self.get_state(2)

        # 4) 先把 new head 插入 body 的最前面（同時動作）
        self.snake1.insert(0, list(next_head1))
        self.snake2.insert(0, list(next_head2))
        self.head1 = list(next_head1)
        self.head2 = list(next_head2)

        ate1 = (self.head1 == self.food)
        ate2 = (self.head2 == self.food)

        # 5) 同時吃到食物的處理
        if ate1 and ate2:
            # 給兩邊 reward、都不 pop，直接放新食物
            reward1 += 10.0
            reward2 += 10.0
            self.score1 += 1
            self.score2 += 1
            self._place_food()
        elif ate1:
            reward1 += 10.0
            self.score1 += 1
            # snake2 沒吃到，pop tail (維持長度)
            self.snake2.pop()
            self._place_food()
        elif ate2:
            reward2 += 10.0
            self.score2 += 1
            self.snake1.pop()
            self._place_food()
        else:
            # 正常移動：尾巴都 pop（同時）
            self.snake1.pop()
            self.snake2.pop()

        # 6) reward shaping: 距離食物變化
        # 使用曼哈頓距離或歐式距離作為補充獎勵
        d1_before = self._manhattan(self.snake1[1], self.food) if len(self.snake1)>1 else self._manhattan(self.head1, self.food)
        d1_after = self._manhattan(self.head1, self.food)
        d2_before = self._manhattan(self.snake2[1], self.food) if len(self.snake2)>1 else self._manhattan(self.head2, self.food)
        d2_after = self._manhattan(self.head2, self.food)

        if d1_after < d1_before: reward1 += 0.1
        else: reward1 -= 0.05

        if d2_after < d2_before: reward2 += 0.1
        else: reward2 -= 0.05

        # 畫面更新
        if self.render:
            self._update_ui()
            self.clock.tick(SPEED)

        return reward1, reward2, done, self.score1, self.score2, self.get_state(1), self.get_state(2)

    def _calc_next_head(self, action_idx, snake_id):
        # action: 0 straight, 1 right, 2 left (相對)
        if snake_id == 1:
            curr_dir = self.direction1
            head = self.head1[:]
        else:
            curr_dir = self.direction2
            head = self.head2[:]

        # 周向 order: Right(1), Left(0), Up(2), Down(3) -- we will use relative turning rules
        # 我們定義：順時針序列 = [1,3,0,2] 與之前相同，但簡化：只處理 relative turn
        clock_wise = [1, 3, 0, 2]
        idx = clock_wise.index(curr_dir)
        if action_idx == 0:    # straight
            new_dir = clock_wise[idx]
        elif action_idx == 1:  # right turn
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:                  # left turn
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        # update direction now
        if snake_id == 1:
            self.direction1 = new_dir
        else:
            self.direction2 = new_dir

        x, y = head[0], head[1]
        if new_dir == 1: x += BLOCK_SIZE
        elif new_dir == 0: x -= BLOCK_SIZE
        elif new_dir == 3: y += BLOCK_SIZE
        elif new_dir == 2: y -= BLOCK_SIZE

        return [x, y]

    def _will_collision(self, next_head, snake_id, next_head_other=None):
        # 撞牆
        if next_head[0] > self.w - BLOCK_SIZE or next_head[0] < 0 or next_head[1] > self.h - BLOCK_SIZE or next_head[1] < 0:
            return True

        # 建立障礙：注意尾巴會被 pop（如果該蛇沒有吃食物）
        # 但這裡我們大致上把目前的 body(除了最後一節) 當障礙
        if snake_id == 1:
            other_body = self.snake2[:]  # 包含 head
            self_body = self.snake1[:]
        else:
            other_body = self.snake1[:]
            self_body = self.snake2[:]

        # 除去自己的尾巴（因為若不會吃食物，尾巴會消失） -> 減少誤判
        if len(self_body) > 0:
            self_body_to_check = self_body[:-1]
        else:
            self_body_to_check = []

        # 對方 body 除了對方尾巴（也可能會被 pop）
        if len(other_body) > 0:
            other_body_to_check = other_body[:-1]
        else:
            other_body_to_check = []

        obstacles = self_body_to_check + other_body_to_check

        # 如果對方下一步 head 要移到這格，也視為撞 (head-on)
        if next_head_other is not None and next_head == next_head_other:
            return True

        if next_head in obstacles:
            return True
        return False

    def get_state(self, snake_id):
        # 回傳 11 維 binary 狀態向量（與你原本類似）
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

        state = [
            int((dir_r and self._will_collision(point_r, snake_id)) or 
                (dir_l and self._will_collision(point_l, snake_id)) or 
                (dir_u and self._will_collision(point_u, snake_id)) or 
                (dir_d and self._will_collision(point_d, snake_id))),

            int((dir_u and self._will_collision(point_r, snake_id)) or 
                (dir_d and self._will_collision(point_l, snake_id)) or 
                (dir_l and self._will_collision(point_u, snake_id)) or 
                (dir_r and self._will_collision(point_d, snake_id))),

            int((dir_d and self._will_collision(point_r, snake_id)) or 
                (dir_u and self._will_collision(point_l, snake_id)) or 
                (dir_r and self._will_collision(point_u, snake_id)) or 
                (dir_l and self._will_collision(point_d, snake_id))),

            int(dir_l), int(dir_r), int(dir_u), int(dir_d),

            int(self.food[0] < head[0]),
            int(self.food[0] > head[0]),
            int(self.food[1] < head[1]),
            int(self.food[1] > head[1])
        ]
        return np.array(state, dtype=np.float32)  # DQN 用 float32

    def _manhattan(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def _update_ui(self):
        self.display.fill((0,0,0))
        # snake1 blue
        for i,pt in enumerate(self.snake1):
            pygame.draw.rect(self.display, (0,0,255), pygame.Rect(pt[0], pt[1], BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, (0,100,255), pygame.Rect(pt[0]+4, pt[1]+4, 12, 12))
        # snake2 green
        for i,pt in enumerate(self.snake2):
            pygame.draw.rect(self.display, (0,255,0), pygame.Rect(pt[0], pt[1], BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, (0,200,0), pygame.Rect(pt[0]+4, pt[1]+4, 12, 12))
        pygame.draw.rect(self.display, (200,0,0), pygame.Rect(self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE))
        pygame.display.flip()

# ----------------- DQN 模型 / Memory -----------------
class DQNNet(nn.Module):
    def __init__(self, input_dim=11, output_dim=3):
        super(DQNNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity=MEMORY_SIZE):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

# ----------------- 共享 Agent（簡化） -----------------
class SharedDQNAgent:
    def __init__(self, policy_net, target_net, memory):
        self.policy = policy_net
        self.target = target_net
        self.memory = memory
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.steps_done = 0

    def select_action(self, state, eps):
        # state: numpy array
        sample = random.random()
        if sample < eps:
            return random.randrange(3)
        else:
            self.policy.eval()
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                q = self.policy(s)
                return int(q.max(1)[1].item())

    def optimize(self):
        if len(self.memory) < BATCH_SIZE:
            return None

        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*transitions)

        state_batch = torch.tensor(np.vstack(batch.state), dtype=torch.float32, device=DEVICE)
        action_batch = torch.tensor(batch.action, dtype=torch.int64, device=DEVICE).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        next_batch = torch.tensor(np.vstack(batch.next_state), dtype=torch.float32, device=DEVICE)
        done_batch = torch.tensor(batch.done, dtype=torch.float32, device=DEVICE).unsqueeze(1)

        # Q(s,a)
        q_values = self.policy(state_batch).gather(1, action_batch)

        # max_a' Q_target(next, a')
        with torch.no_grad():
            next_q = self.target(next_batch).max(1)[0].unsqueeze(1)
            expected_q = reward_batch + (1.0 - done_batch) * GAMMA * next_q

        loss = F.mse_loss(q_values, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

# ----------------- 訓練主程式 -----------------
def train_shared_dqn(policy_net=None, target_net=None, optimizer=None, start_episode=0, num_episodes=2000, render=False):
    env = MultiSnakeGameAI(render=render)
    
    # 如果沒給就建立新的
    if policy_net is None:
        policy_net = DQNNet().to(DEVICE)
    if target_net is None:
        target_net = DQNNet().to(DEVICE)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
    if optimizer is None:
        optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)
    
    memory = ReplayBuffer()
    agent = SharedDQNAgent(policy_net, target_net, memory)
    
    total_steps = 0
    eps = EPS_START

    for ep in range(start_episode+1, start_episode + num_episodes + 1):
        state1, state2 = env.reset()
        done = False
        ep_reward1 = 0.0
        ep_reward2 = 0.0
        steps = 0

        while not done:
            eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * total_steps / EPS_DECAY)

            a1 = agent.select_action(state1, eps)
            a2 = agent.select_action(state2, eps)

            r1, r2, done, s1_score, s2_score, next1, next2 = env.play_step(a1, a2)

            memory.push(state1, a1, r1, next1, float(done))
            memory.push(state2, a2, r2, next2, float(done))

            state1 = next1
            state2 = next2

            ep_reward1 += r1
            ep_reward2 += r2
            steps += 1
            total_steps += 1

            if total_steps > TRAIN_START and total_steps % TRAIN_EVERY == 0:
                agent.optimize()

            if total_steps % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {ep} | Steps: {steps} | Total Steps: {total_steps} | R1: {ep_reward1:.2f} | R2: {ep_reward2:.2f} | Eps: {eps:.3f} | Memory: {len(memory)}")
        
        # 每 100 episode 存一次 checkpoint
        if ep % 100 == 0:
            torch.save({
                "model_state": policy_net.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "episode": ep,
            }, "checkpoint.pth")

    torch.save(policy_net.state_dict(), "shared_dqn_final.pth")
    print("Training finished.")

if __name__ == "__main__":
    policy_net = DQNNet().to(DEVICE)
    target_net = DQNNet().to(DEVICE)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.001)

    start_episode = 0

    # 嘗試載入 checkpoint
    try:
        checkpoint = torch.load("checkpoint.pth", map_location=DEVICE)
        policy_net.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_episode = checkpoint["episode"]

        target_net.load_state_dict(policy_net.state_dict())

        print(f"成功載入模型，從 episode {start_episode} 繼續訓練")
    except Exception as e:
        print("沒有找到 checkpoint，從 0 開始訓練")
        print("錯誤訊息：", e)

    # 開始訓練
    train_shared_dqn(
        policy_net=policy_net,
        target_net=target_net,
        optimizer=optimizer,
        start_episode=start_episode,
        num_episodes=1000,
        render=True
    )
