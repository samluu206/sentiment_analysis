# Deployment Infrastructure

This directory contains all infrastructure-as-code and deployment scripts for the sentiment analysis application on AWS EC2 with K3s (lightweight Kubernetes).

## Directory Structure

```
deployment/
├── k8s/                          # Kubernetes manifests
│   ├── namespace.yaml            # Namespace definition
│   ├── configmap.yaml            # Configuration
│   ├── api-deployment.yaml       # FastAPI deployment
│   ├── api-service.yaml          # FastAPI service (NodePort 30800)
│   ├── gradio-deployment.yaml    # Gradio deployment
│   ├── gradio-service.yaml       # Gradio service (NodePort 30786)
│   └── kustomization.yaml        # Kustomize config
├── terraform/                    # Terraform IaC
│   ├── main.tf                   # Main configuration
│   ├── variables.tf              # Variable definitions
│   ├── outputs.tf                # Output values
│   ├── terraform.tfvars.example  # Example variables
│   └── .gitignore                # Terraform gitignore
├── scripts/                      # Deployment scripts
│   ├── install_k3s.sh            # K3s installation script
│   ├── build_and_export.sh       # Docker image build & export
│   └── deploy_k3s.sh             # Application deployment
├── DEPLOYMENT.md                 # Comprehensive deployment guide
└── README.md                     # This file
```

## Quick Reference

### Prerequisites

- AWS CLI configured
- Terraform >= 1.0
- Docker installed
- EC2 key pair

### Deploy in 5 Steps

```bash
cd deployment

terraform -chdir=terraform init && terraform -chdir=terraform apply

ssh -i your-key.pem ubuntu@<EC2_IP> "bash -s" < scripts/install_k3s.sh

./scripts/build_and_export.sh
scp -i your-key.pem sentiment-analyzer.tar.gz ubuntu@<EC2_IP>:~/

ssh -i your-key.pem ubuntu@<EC2_IP>
gunzip sentiment-analyzer.tar.gz && sudo k3s ctr images import sentiment-analyzer.tar
cd <repo>/deployment && ./scripts/deploy_k3s.sh
```

### Access Services

- **FastAPI**: `http://<EC2_IP>:30800`
- **Swagger UI**: `http://<EC2_IP>:30800/docs`
- **Gradio**: `http://<EC2_IP>:30786`

## Documentation

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for:
- Detailed setup instructions
- Cost estimation (free tier options)
- Troubleshooting guide
- Cost optimization tips
- Monitoring and logging

## Architecture

### Infrastructure

- **Cloud Provider**: AWS
- **Compute**: EC2 (t2.medium recommended, t2.micro for testing)
- **Orchestration**: K3s (lightweight Kubernetes)
- **Container Runtime**: containerd (included with K3s)
- **Networking**: VPC, Security Groups, NodePort services

### Application Stack

- **API**: FastAPI + Uvicorn (port 30800)
- **Frontend**: Gradio web interface (port 30786)
- **Model**: RoBERTa sentiment classifier (640MB)
- **Model Loading**: Init containers download from Hugging Face Hub

### Resource Limits

Per pod:
- **Requests**: 1 GB RAM, 500m CPU
- **Limits**: 1.5 GB RAM, 1000m CPU
- **Init Container**: 512 MB RAM, 250m CPU

## Cost Estimation

### Free Tier (12 months)

- **t2.micro**: FREE (750 hrs/month) - Limited performance
- **t2.medium**: ~$35/month - Recommended
- **Storage**: 30 GB FREE
- **Data Transfer**: 100 GB outbound FREE

### Optimization

- Stop instances when not in use: **Save 70-90%**
- Use Spot instances: **Save up to 90%**
- Deploy single service: **Save 50% resources**

## Security

- SSH access restricted to specific IPs (configurable)
- Security groups limit inbound traffic
- No root access required for deployment
- Secrets managed via Kubernetes ConfigMaps (consider Secrets for production)

## Scaling

### Vertical Scaling (Upgrade Instance)

```bash
terraform apply -var="instance_type=t2.large"
```

### Horizontal Scaling (Add Replicas)

Edit `k8s/*-deployment.yaml`:

```yaml
spec:
  replicas: 3
```

Then:

```bash
kubectl apply -f k8s/
```

### Multi-Node Cluster (Advanced)

Add worker nodes to K3s cluster. See [K3s docs](https://docs.k3s.io/quick-start).

## Monitoring

```bash
kubectl get all -n sentiment-analysis

kubectl logs -f deployment/sentiment-api -n sentiment-analysis

kubectl top nodes
kubectl top pods -n sentiment-analysis
```

## Cleanup

```bash
cd deployment/terraform
terraform destroy
```

## Support

- **Deployment Guide**: [DEPLOYMENT.md](DEPLOYMENT.md)
- **Project Documentation**: [../README.md](../README.md)
- **K3s Docs**: https://docs.k3s.io/
- **Terraform Docs**: https://www.terraform.io/docs/
