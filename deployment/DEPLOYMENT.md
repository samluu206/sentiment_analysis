# AWS EC2 + K3s Deployment Guide

This guide walks you through deploying the sentiment analysis application to AWS EC2 with K3s (lightweight Kubernetes).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Cost Estimation](#cost-estimation)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Deployment Steps](#deployment-steps)
- [Accessing Services](#accessing-services)
- [Monitoring and Troubleshooting](#monitoring-and-troubleshooting)
- [Cost Optimization Tips](#cost-optimization-tips)
- [Cleanup](#cleanup)

## Prerequisites

### Local Machine

- AWS CLI configured with credentials
- Terraform >= 1.0 (for infrastructure provisioning)
- Docker installed (for building images)
- SSH client

### AWS Account

- Active AWS account with free tier eligibility (recommended)
- EC2 key pair created in your target region
- Default VPC (or custom VPC with internet gateway)
- IAM permissions for EC2, VPC, and Security Groups

## Cost Estimation

### Free Tier (12 months)

- **t2.micro**: 750 hours/month FREE
  - 1 vCPU, 1 GB RAM
  - **Limitation**: May struggle with 640MB model + PyTorch
  - **Recommendation**: Use for testing only

- **t2.medium** (not free): ~$0.0464/hour = **$35/month**
  - 2 vCPU, 4 GB RAM
  - **Recommended** for production deployment
  - Comfortable for model + both services

- **Storage**: 30 GB EBS FREE (gp3)
- **Data Transfer**: 100 GB outbound FREE

### Cost Optimization

1. **Stop instances when not in use**: Save 70-90% of costs
2. **Use t2.micro for demos**: Free but limited performance
3. **Spot instances**: Save up to 90% (not covered in this guide)
4. **Schedule auto-shutdown**: Use AWS Lambda to stop instances at night

## Quick Start

```bash
cd deployment

terraform -chdir=terraform init
terraform -chdir=terraform plan
terraform -chdir=terraform apply

export EC2_IP=$(terraform -chdir=terraform output -raw instance_public_ip)

ssh -i your-key.pem ubuntu@$EC2_IP

bash install_k3s.sh

exit

./scripts/build_and_export.sh  

scp -i your-key.pem sentiment-analyzer.tar.gz ubuntu@$EC2_IP:~/

ssh -i your-key.pem ubuntu@$EC2_IP
gunzip sentiment-analyzer.tar.gz
sudo k3s ctr images import sentiment-analyzer.tar

git clone <your-repo-url>
cd <repo-name>/deployment
./scripts/deploy_k3s.sh
```

Access your services at:
- FastAPI: `http://<EC2_IP>:30800`
- Gradio: `http://<EC2_IP>:30786`

## Detailed Setup

### Step 1: Provision EC2 Instance with Terraform

1. Navigate to terraform directory:

```bash
cd deployment/terraform
```

2. Copy example variables:

```bash
cp terraform.tfvars.example terraform.tfvars
```

3. Edit `terraform.tfvars` with your values:

```hcl
aws_region = "us-east-1"
environment = "dev"
instance_type = "t2.medium"
key_name = "your-ec2-key-pair"
vpc_id = "vpc-xxxxx"
subnet_id = "subnet-xxxxx"
allowed_ssh_cidrs = ["YOUR_IP/32"]
root_volume_size = 20
use_elastic_ip = false
```

To find your VPC and Subnet IDs:

```bash
aws ec2 describe-vpcs --query 'Vpcs[?IsDefault==`true`].VpcId' --output text

aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=<YOUR_VPC_ID>" \
  --query 'Subnets[0].SubnetId' \
  --output text

curl -s https://checkip.amazonaws.com
```

4. Initialize and apply Terraform:

```bash
terraform init
terraform plan
terraform apply
```

5. Save the outputs:

```bash
terraform output
```

Example output:

```
instance_public_ip = "54.123.45.67"
ssh_command = "ssh -i your-key.pem ubuntu@54.123.45.67"
api_url = "http://54.123.45.67:30800"
gradio_url = "http://54.123.45.67:30786"
```

### Step 2: Install K3s on EC2

1. SSH into the instance:

```bash
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>
```

2. Transfer and run the K3s installation script:

Option A: Clone your repository on EC2:

```bash
git clone <your-repo-url>
cd <repo-name>/deployment/scripts
bash install_k3s.sh
```

Option B: Download script directly:

```bash
wget https://raw.githubusercontent.com/<your-repo>/main/deployment/scripts/install_k3s.sh
bash install_k3s.sh
```

3. Verify installation:

```bash
kubectl get nodes
kubectl get pods -A
```

### Step 3: Build and Transfer Docker Image

1. On your local machine, build the Docker image:

```bash
cd /path/to/project
bash deployment/scripts/build_and_export.sh
```

This creates `sentiment-analyzer.tar.gz` (~500-800MB compressed).

2. Transfer to EC2:

```bash
scp -i your-key.pem sentiment-analyzer.tar.gz ubuntu@<EC2_IP>:~/
```

3. On EC2, load the image into K3s:

```bash
gunzip sentiment-analyzer.tar.gz
sudo k3s ctr images import sentiment-analyzer.tar
```

4. Verify image is loaded:

```bash
sudo k3s ctr images ls | grep sentiment
```

### Step 4: Deploy Application

1. On EC2, navigate to the project:

```bash
cd <repo-name>/deployment/scripts
```

2. Run deployment script:

```bash
bash deploy_k3s.sh
```

3. Monitor deployment:

```bash
kubectl get pods -n sentiment-analysis -w
```

Wait for pods to show `Running` status. This takes 5-10 minutes as init containers download the model.

## Accessing Services

After deployment completes:

1. **FastAPI** (REST API):
   - URL: `http://<EC2_PUBLIC_IP>:30800`
   - Swagger UI: `http://<EC2_PUBLIC_IP>:30800/docs`
   - Health check: `curl http://<EC2_PUBLIC_IP>:30800/health`

2. **Gradio** (Web Demo):
   - URL: `http://<EC2_PUBLIC_IP>:30786`

3. Test the API:

```bash
curl -X POST "http://<EC2_PUBLIC_IP>:30800/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing!"}'
```

## Monitoring and Troubleshooting

### Check Pod Status

```bash
kubectl get pods -n sentiment-analysis

kubectl describe pod <pod-name> -n sentiment-analysis
```

### View Logs

```bash
kubectl logs -f deployment/sentiment-api -n sentiment-analysis

kubectl logs -f deployment/sentiment-gradio -n sentiment-analysis

kubectl logs <pod-name> -c model-downloader -n sentiment-analysis
```

### Common Issues

#### 1. Pods stuck in `Init:0/1`

Model download is in progress. Check init container logs:

```bash
kubectl logs <pod-name> -c model-downloader -n sentiment-analysis -f
```

#### 2. `ImagePullBackOff` error

Image not found in K3s. Re-import:

```bash
sudo k3s ctr images import ~/sentiment-analyzer.tar
```

#### 3. Pod `CrashLoopBackOff`

Check application logs:

```bash
kubectl logs <pod-name> -n sentiment-analysis --previous
```

#### 4. Out of Memory (OOMKilled)

Instance has insufficient RAM. Options:
- Upgrade to larger instance (t2.medium or t2.large)
- Reduce resource limits in deployment YAML
- Deploy only one service (API or Gradio)

#### 5. Can't access services from browser

Check security group allows ports 30800 and 30786:

```bash
aws ec2 describe-security-groups \
  --group-ids <security-group-id> \
  --query 'SecurityGroups[0].IpPermissions'
```

### Resource Usage

Monitor cluster resources:

```bash
kubectl top nodes
kubectl top pods -n sentiment-analysis
```

## Cost Optimization Tips

### 1. Stop Instances When Not in Use

```bash
aws ec2 stop-instances --instance-ids <instance-id>

aws ec2 start-instances --instance-ids <instance-id>
```

### 2. Use t2.micro for Testing

Edit `terraform.tfvars`:

```hcl
instance_type = "t2.micro"
```

Then:

```bash
terraform apply
```

Note: Performance will be limited with t2.micro.

### 3. Schedule Auto-Shutdown

Create a Lambda function to stop instances at night:

```python
import boto3

def lambda_handler(event, context):
    ec2 = boto3.client('ec2')
    ec2.stop_instances(InstanceIds=['i-xxxxx'])
    return 'Instances stopped'
```

Set CloudWatch Events to trigger at specific times.

### 4. Use Spot Instances (Advanced)

Save up to 90% by using EC2 Spot instances. Modify Terraform:

```hcl
resource "aws_spot_instance_request" "k3s_spot" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = var.instance_type
  spot_price    = "0.02"
}
```

### 5. Deploy Only Essential Services

If you only need the API, comment out Gradio deployment:

```bash
kubectl delete deployment sentiment-gradio -n sentiment-analysis
kubectl delete service sentiment-gradio -n sentiment-analysis
```

## Cleanup

To avoid ongoing costs, destroy all resources:

```bash
cd deployment/terraform
terraform destroy
```

Or manually:

```bash
aws ec2 terminate-instances --instance-ids <instance-id>
```

Verify cleanup:

```bash
aws ec2 describe-instances --instance-ids <instance-id>
```

## Next Steps

1. **Set up CI/CD**: Automate deployments with GitHub Actions
2. **Add monitoring**: Integrate Prometheus + Grafana
3. **Domain and HTTPS**: Configure Route53 + Cert Manager
4. **Horizontal scaling**: Add more nodes to K3s cluster
5. **Database**: Add PostgreSQL for logging predictions
6. **Load balancing**: Set up Application Load Balancer

## Support

For issues or questions:
- Check logs: `kubectl logs -n sentiment-analysis`
- Review K3s docs: https://docs.k3s.io/
- AWS troubleshooting: https://docs.aws.amazon.com/ec2/
