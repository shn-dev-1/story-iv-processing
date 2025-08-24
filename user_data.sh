#!/bin/bash
set -e

# Install ECS agent
yum update -y
yum install -y ecs-init amazon-efs-utils

# Start ECS
systemctl enable --now ecs

# Configure ECS cluster
echo "ECS_CLUSTER=${cluster_name}" >> /etc/ecs/ecs.config

# Install NVIDIA Container Runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | tee /etc/yum.repos.d/nvidia-docker.repo

yum install -y nvidia-container-runtime

# Restart Docker to use NVIDIA runtime
systemctl restart docker

# Install NVIDIA drivers and tools (if not already in AMI)
yum install -y nvidia-driver nvidia-utils

# Configure Docker to use NVIDIA runtime by default
cat > /etc/docker/daemon.json <<EOF
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF

# Restart Docker again
systemctl restart docker

# Start ECS service
systemctl start ecs

# Wait for ECS agent to be ready
until curl -s http://localhost:51678/v1/metadata; do
    echo "Waiting for ECS agent..."
    sleep 5
done

echo "ECS agent is ready"
