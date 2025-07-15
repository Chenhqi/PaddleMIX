# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import BatchSampler, DataLoader, Dataset

BATCH_NUM = 20
BATCH_SIZE = 16
EPOCH_NUM = 4

IMAGE_SIZE = 784
CLASS_NUM = 10

# define a random dataset
class RandomDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([IMAGE_SIZE]).astype("float32")
        label = np.random.randint(0, CLASS_NUM - 1, (1,)).astype("int64")
        return image, label

    def __len__(self):
        return self.num_samples


# dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)

# class SimpleNet(nn.Layer):
#     def __init__(self):
#         super().__init__()
#         self.fc = nn.Linear(IMAGE_SIZE, CLASS_NUM)

#     def forward(self, image, label=None):
#         return self.fc(image)

# simple_net = SimpleNet()
# opt = paddle.optimizer.SGD(learning_rate=1e-3,
#                             parameters=simple_net.parameters())
def read_prompt_list(prompt_list_path):
    with open(prompt_list_path, "r") as f:
        prompt_list = json.load(f)
    prompt_list = [prompt["prompt_en"] for prompt in prompt_list]
    return prompt_list


class myDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.all_prompts = read_prompt_list(
            "/root/paddlejob/workspace/env_run/chq/PaddleMIX/ppdiffusers/examples/Wan2.1/VBench_full_info.json"
        )
        # print(f"[RANK {get_rank()}] Loaded {len(self.all_prompts)} prompts")

    def __getitem__(self, idx):
        # print(type(self.all_prompts[idx]))
        return self.all_prompts[idx], idx

    def __len__(self):
        return len(self.all_prompts)


dataset = myDataset(100)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2)

for i, prompt in enumerate(loader):
    print(prompt)
# for i, (img, label) in enumerate(loader):
#     print(i, img.shape)
# for e in range(EPOCH_NUM):
#     for i, (image, label) in enumerate(loader()):
#         out = simple_net(image)
#         loss = F.cross_entropy(out, label)
#         avg_loss = paddle.mean(loss)
#         avg_loss.backward()
#         opt.minimize(avg_loss)
#         simple_net.clear_gradients()
#         print("Epoch {} batch {}: loss = {}".format(e, i, np.mean(loss.numpy())))
