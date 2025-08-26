# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

__all__ = ['register', "get_reward_manager_cls"]

REWARD_MANAGER_REGISTRY = {}

def register(name):
    """Decorator to register a reward manager class with a given name.

    Args:
        name: `(str)`
            The name of the reward manager.
    """
    def decorator(cls):
        if name in REWARD_MANAGER_REGISTRY and REWARD_MANAGER_REGISTRY[name] != cls:
            raise ValueError(f"Reward manager {name} has already been registered: {REWARD_MANAGER_REGISTRY[name]} vs {cls}")
        REWARD_MANAGER_REGISTRY[name] = cls
        return cls
    return decorator

def get_reward_manager_cls(name):
    """Get the reward manager class with a given name.

    Args:
        name: `(str)`
            The name of the reward manager.

    Returns:
        `(type)`: The reward manager class.
    """
    # 🔍 调试：检查注册表
    print(f"🔍🔍 [Registry] 请求reward manager: {name}")
    print(f"🔍🔍 [Registry] 已注册的managers: {list(REWARD_MANAGER_REGISTRY.keys())}")

    if name not in REWARD_MANAGER_REGISTRY:
        raise ValueError(f"Unknown reward manager: {name}. Available: {list(REWARD_MANAGER_REGISTRY.keys())}")

    selected_cls = REWARD_MANAGER_REGISTRY[name]
    print(f"🔍🔍 [Registry] 选择的manager类: {selected_cls}")
    return selected_cls