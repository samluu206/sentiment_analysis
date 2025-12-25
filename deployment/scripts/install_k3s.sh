#!/bin/bash

set -e

echo "========================================="
echo "Installing K3s on AWS EC2"
echo "========================================="

if command -v k3s &> /dev/null; then
    echo "K3s is already installed"
    k3s --version
    exit 0
fi

echo "Step 1: Update system packages"
sudo apt-get update -y
sudo apt-get upgrade -y

echo "Step 2: Install required dependencies"
sudo apt-get install -y curl wget git

echo "Step 3: Install K3s (lightweight Kubernetes)"
curl -sfL https://get.k3s.io | sh -s - \
    --write-kubeconfig-mode 644 \
    --disable traefik \
    --node-name k3s-node

echo "Step 4: Verify K3s installation"
sudo systemctl status k3s --no-pager || true

echo "Step 5: Configure kubectl access"
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $(id -u):$(id -g) ~/.kube/config
export KUBECONFIG=~/.kube/config

echo "Step 6: Wait for K3s to be ready"
sleep 10
kubectl wait --for=condition=Ready nodes --all --timeout=60s

echo "Step 7: Verify cluster"
kubectl get nodes
kubectl get pods -A

echo "========================================="
echo "K3s installation complete!"
echo "========================================="
echo ""
echo "Add this to your ~/.bashrc or ~/.profile:"
echo "export KUBECONFIG=~/.kube/config"
echo ""
echo "To check cluster status: kubectl get nodes"
