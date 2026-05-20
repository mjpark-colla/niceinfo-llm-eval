#!/bin/bash
# 새 GCP 인스턴스에서 Docker + NVIDIA Container Toolkit 자동 설치
# 사용: bash setup.sh
set -e

echo '=== Docker 설치 ==='
sudo apt-get update -qq
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --batch --yes --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu jammy stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update -qq
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER

echo '=== NVIDIA Container Toolkit 설치 ==='
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
sudo apt-get update -qq
sudo apt-get install -y nvidia-container-toolkit=1.17.8-1 nvidia-container-toolkit-base=1.17.8-1 libnvidia-container-tools=1.17.8-1 libnvidia-container1=1.17.8-1

echo '=== Docker NVIDIA runtime 설정 ==='
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo '=== Test ==='
sudo docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi | head -5

echo '=== 완료. 새 셸 시작하면 sudo 없이 docker 사용 가능 ==='
