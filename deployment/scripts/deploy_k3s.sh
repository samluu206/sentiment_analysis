#!/bin/bash

set -e

echo "========================================="
echo "Deploying Sentiment Analysis to K3s"
echo "========================================="

export KUBECONFIG=~/.kube/config

echo "Step 1: Check K3s cluster status"
kubectl get nodes

echo "Step 2: Load Docker image into K3s (if not already loaded)"
if [ -f ~/sentiment-analyzer.tar ]; then
    echo "Loading Docker image..."
    sudo k3s ctr images import ~/sentiment-analyzer.tar
    echo "Image loaded successfully"
else
    echo "Warning: sentiment-analyzer.tar not found in home directory"
    echo "Assuming image is already loaded or will be pulled from registry"
fi

echo "Step 3: Apply Kubernetes manifests"
echo "Creating namespace first..."
kubectl apply -f ../../k8s/namespace.yaml

echo "Waiting for namespace to be ready..."
sleep 3

echo "Applying ConfigMap..."
kubectl apply -f ../../k8s/configmap.yaml

echo "Applying deployments and services..."
kubectl apply -f ../../k8s/api-deployment.yaml
kubectl apply -f ../../k8s/api-service.yaml
kubectl apply -f ../../k8s/gradio-deployment.yaml
kubectl apply -f ../../k8s/gradio-service.yaml

echo "Step 4: Wait for deployments to be ready"
echo "This may take several minutes as init containers download the model..."
kubectl wait --for=condition=available --timeout=600s \
    deployment/sentiment-api \
    deployment/sentiment-gradio \
    -n sentiment-analysis || true

echo "Step 5: Check deployment status"
kubectl get all -n sentiment-analysis

echo "Step 6: Get service endpoints"
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
API_PORT=$(kubectl get svc sentiment-api -n sentiment-analysis -o jsonpath='{.spec.ports[0].nodePort}')
GRADIO_PORT=$(kubectl get svc sentiment-gradio -n sentiment-analysis -o jsonpath='{.spec.ports[0].nodePort}')

echo "========================================="
echo "Deployment complete!"
echo "========================================="
echo ""
echo "Access the services:"
echo "  FastAPI: http://${NODE_IP}:${API_PORT}"
echo "  Swagger UI: http://${NODE_IP}:${API_PORT}/docs"
echo "  Gradio: http://${NODE_IP}:${GRADIO_PORT}"
echo ""
echo "For external access, use your EC2 public IP:"
echo "  FastAPI: http://<EC2_PUBLIC_IP>:${API_PORT}"
echo "  Gradio: http://<EC2_PUBLIC_IP>:${GRADIO_PORT}"
echo ""
echo "Make sure to open ports ${API_PORT} and ${GRADIO_PORT} in EC2 security group!"
echo ""
echo "To check logs:"
echo "  kubectl logs -f deployment/sentiment-api -n sentiment-analysis"
echo "  kubectl logs -f deployment/sentiment-gradio -n sentiment-analysis"
